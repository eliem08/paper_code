"""Real agentic evaluation: run the TriageTask over BRACS test
cases with a tool-using Ollama LLM that calls the MoE classifier as a tool.

Reports triage accuracy, safe-error rate (never calling a 'refer' case
'benign'), flag-for-review rate, and mean tool calls per case. The agent path
is: LLM -> predict_case (MoE) -> [optional retrieve] -> report_decision.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from breastpath_copilot.agents.clients import OpenAICompatClient
from breastpath_copilot.agents.runner import AgentRunner
from breastpath_copilot.agents.tasks import (TriageTask, TriageInput, TriageGold,
                                             EvidenceReportTask, EvidenceReportInput, EvidenceReportGold)
from breastpath_copilot.agents.tool_registry import build_default_registry, AgentContext
from breastpath_copilot.training.bracs_schemes import get_scheme, remap_bracs_manifest

CACHE = "artifacts/fm_cache"; MAN = Path("artifacts/manifests")
FMS = ["uni", "virchow2", "phikon_v2_ft"]
CKPT = "artifacts/agent/moe_bracs_3class.pt"
INDEX = "artifacts/agent/retrieval_index.joblib"
# BRACS 3-class label -> unified vocab so TriageTask.map_label_to_triage works
# (benign->benign, atypia->suspicious, dcis/invasive->refer).
COARSE_TO_UNIFIED = {"benign": "benign", "atypical": "atypia", "malignant": "invasive",
                     "0_N": "normal"}


class EvidenceReport3Class(EvidenceReportTask):
    """Evidence-report task in the engine's 3-class space (benign/atypical/
    malignant). Requires retrieve_neighbors + non-empty evidence_image_ids."""

    def system_prompt(self) -> str:
        return (
            "You are an evidence-grounded breast-pathology reporting agent. For "
            "the given case you MUST: (1) call predict_case to get the MoE "
            "verdict and per-expert gate weights; (2) call retrieve_neighbors "
            "(top_k 3-5) to gather similar prior cases; (3) call report_decision "
            "exactly once with decision in {benign, atypical, malignant}, a "
            "confidence in [0,1], a one-sentence rationale, and "
            "evidence_image_ids = the neighbour ids you actually used "
            "(MUST be non-empty). Reports without evidence are rejected.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6:35b")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--task", choices=["triage", "evidence"], default="triage")
    args = ap.parse_args()

    sch = get_scheme("3class")
    m = remap_bracs_manifest(pd.read_parquet(MAN / "bracs.parquet"), sch)
    test = m[m["split"].astype(str) == "test"].reset_index(drop=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(test), size=min(args.n, len(test)), replace=False)
    cases = test.iloc[idx]

    idx_on = Path(INDEX).exists()
    ctx = AgentContext(cache_root=CACHE, cohort="bracs", fm_names=FMS,
                       moe_routing="soft_topk", moe_shared_dim=512, moe_top_k=2,
                       moe_classifier_num_classes=3, moe_checkpoint_path=CKPT,
                       retrieval_index_path=INDEX if idx_on else None)
    registry = build_default_registry()
    task = TriageTask() if args.task == "triage" else EvidenceReport3Class()
    runner = AgentRunner(client=OpenAICompatClient(model_name=args.model), task=task,
                         registry=registry, context=ctx, max_steps=args.max_steps,
                         trajectory_dir=Path("artifacts/agent/traj"))

    n_done = n_ok = n_correct = n_tools = 0
    n_flag = n_refer = n_refer_safe = 0           # triage
    n_evid = n_evid_rel = n_retrieved = 0          # evidence
    t0 = time.monotonic()
    total = len(cases)
    for k, (_, r) in enumerate(cases.iterrows()):
        iid = str(r["image_id"]); coarse = str(r["label"])
        if args.task == "triage":
            inp = TriageInput(image_id=iid, cohort="bracs")
            gold = TriageGold(image_id=iid, cohort="bracs", label=COARSE_TO_UNIFIED.get(coarse, "invasive"))
        else:
            inp = EvidenceReportInput(image_id=iid, cohort="bracs", top_k_neighbors=5)
            gold = EvidenceReportGold(image_id=iid, label=coarse)
        try:
            res = runner.run(inp, gold=gold, case_id=iid)
        except Exception as e:
            print(f"  [{k+1}/{total}] {iid} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        n_done += 1; n_tools += res.trajectory.step_count
        ok_str = "truncated"
        if res.success:
            n_ok += 1
            d = res.decision or {}; bd = d.get("_eval_breakdown", {})
            if res.correct:
                n_correct += 1
            ok_str = f"correct={bool(res.correct)}"
            if args.task == "triage":
                if bd.get("flag_for_review"):
                    n_flag += 1
                if bd.get("expected_triage") == "refer":
                    n_refer += 1
                    if bd.get("predicted") in ("refer", "suspicious"):
                        n_refer_safe += 1
            else:
                ev = d.get("evidence_image_ids") or []
                if ev:
                    n_evid += 1
                    if bd.get("evidence_overlap_ok", True) and bd.get("label_correct"):
                        n_evid_rel += 1
        elapsed = time.monotonic() - t0
        eta = elapsed / (k + 1) * (total - k - 1)
        print(f"  [{k+1}/{total}] {iid}: {ok_str}  "
              f"(running: completed={n_ok}/{n_done}, acc={n_correct}/{n_ok if n_ok else 1}, "
              f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m)", flush=True)

    out = Path("artifacts/q1_results/agent_triage_eval.txt")
    print(f"\n===== AGENT {args.task.upper()} EVAL (retrieval={'on' if idx_on else 'off'}) =====", flush=True)
    print(f"model={args.model} cases={n_done} completed={n_ok} steps/case={n_tools/max(1,n_done):.2f}", flush=True)
    if args.task == "triage":
        print(f"triage acc {n_correct}/{n_ok} ({(n_correct/n_ok if n_ok else 0):.2%}) | "
              f"refer-safe {n_refer_safe}/{n_refer} | flagged {n_flag}", flush=True)
        line = (f"task=triage model={args.model} retrieval={'on' if idx_on else 'off'} cases={n_done} "
                f"completed={n_ok} acc={(n_correct/n_ok if n_ok else 0):.4f} "
                f"refer_safe={(n_refer_safe/n_refer if n_refer else 0):.4f} flagged={n_flag} "
                f"mean_steps={n_tools/max(1,n_done):.2f}\n")
    else:
        print(f"label acc {n_correct}/{n_ok} ({(n_correct/n_ok if n_ok else 0):.2%}) | "
              f"evidence-provided {n_evid}/{n_ok} | grounded-correct {n_evid_rel}/{n_ok}", flush=True)
        line = (f"task=evidence model={args.model} retrieval={'on' if idx_on else 'off'} cases={n_done} "
                f"completed={n_ok} acc={(n_correct/n_ok if n_ok else 0):.4f} "
                f"evidence_provided={(n_evid/n_ok if n_ok else 0):.4f} "
                f"grounded_correct={(n_evid_rel/n_ok if n_ok else 0):.4f} "
                f"mean_steps={n_tools/max(1,n_done):.2f}\n")
    with open(out, "a") as f:
        f.write(line)
    print(f"appended -> {out}", flush=True)


if __name__ == "__main__":
    main()
