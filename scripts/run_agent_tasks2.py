"""Complete the four-task agent evaluation: cross-cohort routing and
classifier-vs-retrieval discrepancy resolution, run live over a tool-calling LLM.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from breastpath_copilot.agents.clients import OpenAICompatClient
from breastpath_copilot.agents.runner import AgentRunner
from breastpath_copilot.agents.tasks import (CrossCohortRoutingTask, CrossCohortRoutingInput,
                                             CrossCohortRoutingGold, DiscrepancyTask, DiscrepancyInput,
                                             DiscrepancyGold)
from breastpath_copilot.agents.tool_registry import build_default_registry, AgentContext
from breastpath_copilot.core.fm_cache import FMEmbeddingCache
from breastpath_copilot.core.retrieval_index import MoERetrievalIndex
from breastpath_copilot.models.moe import MoEClassifier, MoEConfig
from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest

CACHE = "artifacts/fm_cache"; MAN = Path("artifacts/manifests")
FMS = ["uni", "virchow2", "phikon_v2_ft"]
CKPT = "artifacts/agent/moe_bracs_3class.pt"
INDEX = "artifacts/agent/retrieval_index.joblib"
I2L = {0: "benign", 1: "atypical", 2: "malignant"}
NEI = {"benign": "benign", "atypia": "atypical", "atypical": "atypical", "malignant": "malignant"}


def load_model(dev):
    cfg = MoEConfig(routing="soft_topk", shared_dim=512, top_k=2)
    model = MoEClassifier({"uni": 1024, "virchow2": 2560, "phikon_v2_ft": 1024}, num_classes=3, config=cfg)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")); model.eval().to(dev)
    return model


def align(cohort):
    caches = {f: FMEmbeddingCache(CACHE, cohort, f) for f in FMS}
    base = caches[FMS[0]].load_manifest(); ids = base["image_id"].astype(str).tolist()
    surv, sl = None, {}
    for f in FMS:
        rows, vecs = caches[f].slice_by_image_ids(ids)
        rid = rows["image_id"].astype(str).tolist(); sl[f] = (rid, np.asarray(vecs))
        surv = set(rid) if surv is None else surv & set(rid)
    surv = [i for i in ids if i in surv]
    emb = {}
    for f in FMS:
        rid, vecs = sl[f]; pos = {i: k for k, i in enumerate(rid)}
        emb[f] = vecs[[pos[i] for i in surv]].astype(np.float32)
    return surv, emb


def preds_gates(model, emb, dev, bs=256):
    n = len(emb[FMS[0]]); P, G = [], []
    for s in range(0, n, bs):
        ed = {f: torch.tensor(emb[f][s:s + bs], device=dev) for f in FMS}
        with torch.no_grad():
            logits, out = model(ed, magnifications=None)
        P.append(torch.softmax(logits, 1).cpu().numpy()); G.append(out.gate_weights.cpu().numpy())
    return np.concatenate(P), np.concatenate(G)


def make_ctx(cohort):
    return AgentContext(cache_root=CACHE, cohort=cohort, fm_names=FMS, moe_routing="soft_topk",
                        moe_shared_dim=512, moe_top_k=2, moe_classifier_num_classes=3,
                        moe_checkpoint_path=CKPT,
                        retrieval_index_path=INDEX if Path(INDEX).exists() else None)


def run_routing(model, client, dev, per=6, max_steps=6):
    reg = build_default_registry(); task = CrossCohortRoutingTask()
    cohorts = ["bracs", "bach", "breakhis"]; rng = np.random.default_rng(0)
    n_ok = n_cohort = n_expert = n_done = 0
    for coh in cohorts:
        ids, emb = align(coh)
        # BRACS is the only cohort with a train/val/test split. Restrict it to
        # the official test split so routing never samples cases the MoE (whose
        # gate weights the agent must read back) was trained on. BACH and
        # BreaKHis have no split and are used in full.
        if coh == "bracs":
            bm = pd.read_parquet(MAN / "bracs.parquet")
            test_ids = set(bm[bm["split"].astype(str) == "test"]["image_id"].astype(str))
            keep = [k for k, i in enumerate(ids) if str(i) in test_ids]
            if keep:
                ids = [ids[k] for k in keep]
                emb = {f: v[keep] for f, v in emb.items()}
        _, G = preds_gates(model, emb, dev)
        pick = rng.choice(len(ids), size=min(per, len(ids)), replace=False)
        runner = AgentRunner(client=client, task=task, registry=reg, context=make_ctx(coh), max_steps=max_steps)
        for i in pick:
            iid = ids[i]; dom = FMS[int(G[i].argmax())]
            inp = CrossCohortRoutingInput(image_id=iid, candidate_cohorts=cohorts)
            gold = CrossCohortRoutingGold(image_id=iid, correct_cohort=coh, expected_dominant_expert=dom)
            try:
                res = runner.run(inp, gold=gold, case_id=iid)
            except Exception as e:
                print(f"  routing {iid} ERR {e}", flush=True); continue
            n_done += 1
            if res.success:
                n_ok += 1; bd = (res.decision or {}).get("_eval_breakdown", {})
                n_cohort += int(bool(bd.get("cohort_correct")))
                n_expert += int(bool(bd.get("expert_correct")))
    print(f"\n[routing] done={n_done} completed={n_ok} cohort_acc={n_cohort}/{n_ok} "
          f"expert_acc={n_expert}/{n_ok}", flush=True)
    return f"task=routing completed={n_ok}/{n_done} cohort={n_cohort} expert={n_expert}\n"


def run_discrepancy(model, client, dev, n=24, max_steps=10, traj_dir=None):
    idx = MoERetrievalIndex.load(INDEX)
    cache = FMEmbeddingCache(CACHE, "bracs", "uni"); man = cache.load_manifest()
    sch = get_scheme("3class"); bm = remap_bracs_manifest(pd.read_parquet(MAN / "bracs.parquet"), sch)
    test = bm[bm["split"].astype(str) == "test"]
    ids, emb = align("bracs")
    P, _ = preds_gates(model, emb, dev); pos = {i: k for k, i in enumerate(ids)}
    tmap = dict(zip(man["image_id"].astype(str), man["patient_id"].astype(str)))
    reg = build_default_registry()
    # The gold label here comes from the 3-class scheme (benign/atypical/
    # malignant) -- the prompt's allowed vocabulary must match exactly, or
    # every case whose gold is "atypical" or "malignant" is unwinnable by
    # construction (see DiscrepancyTask.__init__ docstring).
    task = DiscrepancyTask(label_options=list(I2L.values()))
    runner = AgentRunner(client=client, task=task, registry=reg, context=make_ctx("bracs"),
                         max_steps=max_steps, trajectory_dir=Path(traj_dir) if traj_dir else None)
    cases = []
    for _, r in test.iterrows():
        iid = str(r["image_id"])
        if iid not in pos:
            continue
        clf = I2L[int(P[pos[iid]].argmax())]
        rows, vecs = cache.slice_by_image_ids([iid])
        if vecs.size == 0:
            continue
        nb = idx.query(vecs[0], top_k=5, query_patient_id=tmap.get(iid), query_image_id=iid)
        if not nb:
            continue
        labs = [NEI.get(str(x.label), "benign") for x in nb]
        retr = max(set(labs), key=labs.count)
        if retr != clf:                                    # disagreement
            cases.append((iid, str(r["label"]), clf, retr))
    rng = np.random.default_rng(0)
    sel = [cases[i] for i in rng.choice(len(cases), size=min(n, len(cases)), replace=False)] if cases else []
    n_ok = n_lab = n_src = n_done = 0
    t0 = time.monotonic()
    for k, (iid, gold_lab, clf, retr) in enumerate(sel, 1):
        inp = DiscrepancyInput(image_id=iid, cohort="bracs", classifier_prediction=clf, retrieval_prediction=retr)
        gold = DiscrepancyGold(image_id=iid, label=gold_lab, classifier_prediction=clf, retrieval_prediction=retr)
        try:
            res = runner.run(inp, gold=gold, case_id=iid)
        except Exception as e:
            print(f"  [{k}/{len(sel)}] disc {iid} ERR {e}", flush=True); continue
        n_done += 1
        ok_str = "?"
        if res.success:
            n_ok += 1; bd = (res.decision or {}).get("_eval_breakdown", {})
            n_lab += int(bool(bd.get("label_correct"))); n_src += int(bool(bd.get("source_correct")))
            ok_str = f"label={bool(bd.get('label_correct'))} source={bool(bd.get('source_correct'))}"
        else:
            ok_str = f"truncated (err={res.error})"
        elapsed = time.monotonic() - t0
        eta = elapsed / k * (len(sel) - k)
        print(f"  [{k}/{len(sel)}] {iid}: {ok_str}  "
              f"(running: completed={n_ok}/{n_done}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m)", flush=True)
    print(f"\n[discrepancy] disagreements_found={len(cases)} run={n_done} completed={n_ok} "
          f"label_acc={n_lab}/{n_ok} source_acc={n_src}/{n_ok}", flush=True)
    return f"task=discrepancy disagreements={len(cases)} completed={n_ok}/{n_done} label={n_lab} source={n_src}\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="qwen3.6:35b")
    ap.add_argument("--per", type=int, default=6); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--only", choices=["both", "routing", "discrepancy"], default="both")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="per-turn completion budget; thinking-enabled models (e.g. "
                    "qwen3.6) can exhaust the default 1024 on their reasoning trace "
                    "alone before emitting report_decision, which truncates the run "
                    "with an empty completion -- raise this for harder tasks")
    ap.add_argument("--traj-dir", default=None,
                    help="if set, persist every run's full tool-call trajectory as JSON here")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(dev)
    client = OpenAICompatClient(model_name=args.model, max_tokens=args.max_tokens)
    out = Path("artifacts/q1_results/agent_tasks2.txt")
    with open(out, "a") as f:
        f.write(f"### {args.model} (max_steps={args.max_steps}) ###\n")
        if args.only in ("both", "routing"):
            f.write(run_routing(model, client, dev, per=args.per, max_steps=args.max_steps))
        if args.only in ("both", "discrepancy"):
            f.write(run_discrepancy(model, client, dev, n=args.n, max_steps=args.max_steps,
                                    traj_dir=args.traj_dir))
    print(f"appended -> {out}", flush=True)


if __name__ == "__main__":
    main()
