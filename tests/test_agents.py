"""End-to-end agent harness tests using MockClient and dummy FM caches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from breastpath_copilot.agents import (  # noqa: E402
    AgentRunner,
    CrossCohortRoutingGold,
    CrossCohortRoutingInput,
    CrossCohortRoutingTask,
    DiscrepancyGold,
    DiscrepancyInput,
    DiscrepancyTask,
    EvidenceReportGold,
    EvidenceReportInput,
    EvidenceReportTask,
    MockClient,
    ScriptedTurn,
    TriageGold,
    TriageInput,
    TriageTask,
    build_default_registry,
    triage_accuracy,
    trajectory_summary_stats,
)
from breastpath_copilot.agents.tasks import map_label_to_triage  # noqa: E402
from breastpath_copilot.agents.tool_registry import (  # noqa: E402
    FINAL_TOOL_NAME,
    AgentContext,
)
from breastpath_copilot.core.fm_cache import FMEmbeddingCache  # noqa: E402


def _seed_caches(tmp_path: Path, image_ids: list[str], dims: dict[str, int]) -> Path:
    """Create cached embeddings for the requested image_ids across N fake FMs."""
    cache_root = tmp_path / "cache"
    for fm_name, dim in dims.items():
        cache = FMEmbeddingCache(cache_root, "breakhis", fm_name)
        df = pd.DataFrame(
            {
                "image_id": image_ids,
                "image_path": ["/dummy"] * len(image_ids),
                "label": ["benign", "invasive", "dcis"][: len(image_ids)],
                "subtype": ["fibroadenoma", "ductal_carcinoma", "ductal_carcinoma_in_situ"][
                    : len(image_ids)
                ],
                "magnification": ["40X", "400X", "200X"][: len(image_ids)],
                "patient_id": [f"p{i}" for i in range(len(image_ids))],
                "cohort": ["breakhis"] * len(image_ids),
            }
        )
        vecs = np.random.default_rng(dim).standard_normal((len(image_ids), dim)).astype(np.float32)
        cache.write(df, vecs, model_id=f"dummy/{fm_name}", embedding_dim=dim)
    return cache_root


def _make_context(cache_root: Path, fm_names: list[str]) -> AgentContext:
    return AgentContext(
        cache_root=str(cache_root),
        cohort="breakhis",
        fm_names=fm_names,
        moe_routing="soft_topk",
        moe_shared_dim=16,
        moe_top_k=2,
        moe_classifier_num_classes=5,
    )


# --------------------------------------------------------------------------- #
# Tool registry sanity
# --------------------------------------------------------------------------- #

def test_default_registry_exposes_expected_tools():
    reg = build_default_registry()
    assert set(reg.names()) == {
        "list_cases",
        "get_case_metadata",
        "predict_case",
        "retrieve_neighbors",
        FINAL_TOOL_NAME,
    }


def test_default_registry_emits_openai_function_schemas():
    reg = build_default_registry()
    tools = reg.to_openai_tools()
    assert all(t["type"] == "function" for t in tools)
    names = {t["function"]["name"] for t in tools}
    assert FINAL_TOOL_NAME in names
    # JSON schema parameters exist for every tool
    for t in tools:
        assert "parameters" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"


# --------------------------------------------------------------------------- #
# Triage label mapping
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "label,expected",
    [
        ("normal", "benign"),
        ("benign", "benign"),
        ("atypia", "suspicious"),
        ("dcis", "refer"),
        ("invasive", "refer"),
        ("unknown_label_x", "refer"),  # conservative default
    ],
)
def test_map_label_to_triage(label: str, expected: str):
    assert map_label_to_triage(label) == expected


# --------------------------------------------------------------------------- #
# Full triage loop end-to-end
# --------------------------------------------------------------------------- #

def test_triage_run_calls_predict_then_reports_decision(tmp_path: Path):
    image_ids = ["img_a", "img_b", "img_c"]
    cache_root = _seed_caches(tmp_path, image_ids, {"fm_a": 8, "fm_b": 16})
    ctx = _make_context(cache_root, ["fm_a", "fm_b"])

    client = MockClient(
        [
            ScriptedTurn.call("predict_case", image_id="img_b"),
            ScriptedTurn.call(
                FINAL_TOOL_NAME,
                decision="refer",
                confidence=0.85,
                rationale="invasive carcinoma indicated by MoE classifier",
                evidence_image_ids=[],
                flag_for_review=True,
            ),
        ]
    )
    runner = AgentRunner(
        client=client, task=TriageTask(), registry=build_default_registry(), context=ctx
    )
    gold = TriageGold(image_id="img_b", cohort="breakhis", label="invasive")
    result = runner.run(TriageInput(image_id="img_b", cohort="breakhis"), gold=gold, case_id="img_b")

    assert result.success is True
    assert result.correct is True
    assert result.decision is not None
    assert result.decision["decision"] == "refer"
    assert result.trajectory.step_count == 2
    assert not result.trajectory.truncated
    # Two tool calls (predict + report), both successful.
    assert len(result.trajectory.tool_results) == 2
    assert all(tr.ok for tr in result.trajectory.tool_results)


def test_triage_run_truncates_when_no_final_call(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    ctx = _make_context(cache_root, ["fm_a"])

    # Model never calls report_decision; runner should hit max_steps.
    client = MockClient(
        [ScriptedTurn.call("list_cases", cohort="breakhis", limit=3)] * 5
    )
    runner = AgentRunner(
        client=client,
        task=TriageTask(),
        registry=build_default_registry(),
        context=ctx,
        max_steps=3,
    )
    result = runner.run(
        TriageInput(image_id="img_a", cohort="breakhis"),
        gold=TriageGold(image_id="img_a", cohort="breakhis", label="benign"),
        case_id="img_a",
    )
    assert result.success is False
    assert result.trajectory.truncated is True
    assert result.trajectory.step_count == 3


def test_triage_run_text_only_response_is_truncation(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    ctx = _make_context(cache_root, ["fm_a"])
    client = MockClient([ScriptedTurn.text("I think it's probably benign.")])
    runner = AgentRunner(
        client=client, task=TriageTask(), registry=build_default_registry(), context=ctx
    )
    result = runner.run(
        TriageInput(image_id="img_a", cohort="breakhis"),
        gold=TriageGold(image_id="img_a", cohort="breakhis", label="benign"),
        case_id="img_a",
    )
    assert result.success is False
    assert result.trajectory.truncated is True


def test_triage_run_records_tool_error_when_id_unknown(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    ctx = _make_context(cache_root, ["fm_a"])
    client = MockClient(
        [
            ScriptedTurn.call("get_case_metadata", image_id="img_does_not_exist"),
            ScriptedTurn.call(
                FINAL_TOOL_NAME,
                decision="benign",
                confidence=0.4,
                rationale="no data found",
            ),
        ]
    )
    runner = AgentRunner(
        client=client, task=TriageTask(), registry=build_default_registry(), context=ctx
    )
    result = runner.run(
        TriageInput(image_id="img_does_not_exist", cohort="breakhis"),
        gold=TriageGold(image_id="img_does_not_exist", cohort="breakhis", label="benign"),
        case_id="img_x",
    )
    # The metadata tool returns ok=False; runner still records it and continues.
    metadata_results = [tr for tr in result.trajectory.tool_results if tr.name == "get_case_metadata"]
    assert metadata_results and metadata_results[0].ok is False
    assert result.success is True


def test_triage_run_persists_trajectory_to_disk(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    ctx = _make_context(cache_root, ["fm_a"])
    client = MockClient(
        [
            ScriptedTurn.call(
                FINAL_TOOL_NAME,
                decision="benign",
                confidence=0.9,
                rationale="trivial",
            )
        ]
    )
    out_dir = tmp_path / "trajectories"
    runner = AgentRunner(
        client=client,
        task=TriageTask(),
        registry=build_default_registry(),
        context=ctx,
        trajectory_dir=out_dir,
    )
    runner.run(
        TriageInput(image_id="img_a", cohort="breakhis"),
        gold=TriageGold(image_id="img_a", cohort="breakhis", label="benign"),
        case_id="img_a",
    )
    files = list(out_dir.glob("triage_mock_img_a.json"))
    assert files and files[0].read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_triage_accuracy_metrics_over_a_small_cohort(tmp_path: Path):
    image_ids = ["img_a", "img_b", "img_c"]
    cache_root = _seed_caches(tmp_path, image_ids, {"fm_a": 8})
    ctx = _make_context(cache_root, ["fm_a"])

    cases = [
        (image_ids[0], "benign", "benign", True),    # correct
        (image_ids[1], "invasive", "suspicious", False),  # incorrect but SAFE on refer
        (image_ids[2], "dcis", "refer", True),       # correct
    ]
    results = []
    for img, gold_label, pred_decision, _correct in cases:
        client = MockClient(
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision=pred_decision,
                    confidence=0.7,
                    rationale="-",
                    flag_for_review=(pred_decision == "suspicious"),
                )
            ]
        )
        runner = AgentRunner(
            client=client, task=TriageTask(), registry=build_default_registry(), context=ctx
        )
        results.append(
            runner.run(
                TriageInput(image_id=img, cohort="breakhis"),
                gold=TriageGold(image_id=img, cohort="breakhis", label=gold_label),
                case_id=img,
            )
        )

    metrics = triage_accuracy(results)
    assert metrics["n"] == 3
    assert metrics["success_rate"] == 1.0
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    # One 'refer' case predicted 'suspicious' (still safe) + one 'refer' correctly -> 100% safe.
    assert metrics["safe_on_refer"] == pytest.approx(1.0)
    assert metrics["flag_rate"] == pytest.approx(1 / 3)


def _evidence_runner(cache_root: Path, scripted_turns: list[ScriptedTurn]) -> AgentRunner:
    return AgentRunner(
        client=MockClient(scripted_turns),
        task=EvidenceReportTask(),
        registry=build_default_registry(),
        context=_make_context(cache_root, ["fm_a"]),
    )


# --------------------------------------------------------------------------- #
# EvidenceReportTask
# --------------------------------------------------------------------------- #

def test_evidence_report_credits_label_plus_evidence(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a", "img_b", "img_c"], {"fm_a": 8})
    turns = [
        ScriptedTurn.call("predict_case", image_id="img_b"),
        ScriptedTurn.call("retrieve_neighbors", image_id="img_b", top_k=5),
        ScriptedTurn.call(
            FINAL_TOOL_NAME,
            decision="invasive",
            confidence=0.9,
            rationale="Classifier + 3 similar prior cases agree.",
            evidence_image_ids=["img_a", "img_c"],
        ),
    ]
    runner = _evidence_runner(cache_root, turns)
    gold = EvidenceReportGold(
        image_id="img_b", label="invasive", acceptable_evidence_ids=["img_a", "img_x"]
    )
    result = runner.run(
        EvidenceReportInput(image_id="img_b", cohort="breakhis"),
        gold=gold,
        case_id="img_b",
    )
    assert result.success is True
    assert result.correct is True
    bd = result.decision["_eval_breakdown"]  # type: ignore[index]
    assert bd["label_correct"] is True
    assert bd["evidence_provided"] is True
    assert bd["evidence_overlap_ok"] is True
    assert bd["evidence_count"] == 2


def test_evidence_report_rejects_decision_without_evidence(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    turns = [
        ScriptedTurn.call(
            FINAL_TOOL_NAME,
            decision="benign",
            confidence=0.9,
            rationale="lazy",
            evidence_image_ids=[],
        )
    ]
    runner = _evidence_runner(cache_root, turns)
    result = runner.run(
        EvidenceReportInput(image_id="img_a", cohort="breakhis"),
        gold=EvidenceReportGold(image_id="img_a", label="benign"),
        case_id="img_a",
    )
    assert result.success is True
    # Decision label is correct, but no evidence -> evaluator marks NOT correct.
    assert result.correct is False
    assert result.decision["_eval_breakdown"]["evidence_provided"] is False


def test_evidence_report_acceptable_evidence_overlap_required_when_set(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a", "img_b"], {"fm_a": 8})
    turns = [
        ScriptedTurn.call(
            FINAL_TOOL_NAME,
            decision="benign",
            confidence=0.9,
            rationale="agree",
            evidence_image_ids=["img_b"],  # not in acceptable set
        )
    ]
    runner = _evidence_runner(cache_root, turns)
    result = runner.run(
        EvidenceReportInput(image_id="img_a", cohort="breakhis"),
        gold=EvidenceReportGold(
            image_id="img_a", label="benign", acceptable_evidence_ids=["only_acceptable"]
        ),
        case_id="img_a",
    )
    assert result.correct is False
    assert result.decision["_eval_breakdown"]["evidence_overlap_ok"] is False


# --------------------------------------------------------------------------- #
# CrossCohortRoutingTask
# --------------------------------------------------------------------------- #

def test_cross_cohort_routing_credits_correct_cohort_and_expert(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8, "fm_b": 16})
    runner = AgentRunner(
        client=MockClient(
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision="breakhis",
                    confidence=0.95,
                    rationale="prefix breakhis_ + gate weight maxed on uni",
                    dominant_expert="uni",
                )
            ]
        ),
        task=CrossCohortRoutingTask(),
        registry=build_default_registry(),
        context=_make_context(cache_root, ["fm_a", "fm_b"]),
    )
    result = runner.run(
        CrossCohortRoutingInput(
            image_id="img_a", candidate_cohorts=["breakhis", "bach", "bracs"]
        ),
        gold=CrossCohortRoutingGold(
            image_id="img_a", correct_cohort="breakhis", expected_dominant_expert="uni"
        ),
        case_id="img_a",
    )
    assert result.correct is True
    bd = result.decision["_eval_breakdown"]  # type: ignore[index]
    assert bd["cohort_correct"] is True
    assert bd["expert_correct"] is True


def test_cross_cohort_routing_wrong_cohort_fails(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    runner = AgentRunner(
        client=MockClient(
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision="bach",
                    confidence=0.6,
                    rationale="wrong guess",
                )
            ]
        ),
        task=CrossCohortRoutingTask(),
        registry=build_default_registry(),
        context=_make_context(cache_root, ["fm_a"]),
    )
    result = runner.run(
        CrossCohortRoutingInput(image_id="img_a", candidate_cohorts=["breakhis", "bach"]),
        gold=CrossCohortRoutingGold(image_id="img_a", correct_cohort="breakhis"),
        case_id="img_a",
    )
    assert result.correct is False
    assert result.decision["_eval_breakdown"]["cohort_correct"] is False


def test_cross_cohort_routing_skips_expert_check_when_gold_unspecified(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    runner = AgentRunner(
        client=MockClient(
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision="breakhis",
                    confidence=0.9,
                    rationale="-",
                    dominant_expert=None,
                )
            ]
        ),
        task=CrossCohortRoutingTask(),
        registry=build_default_registry(),
        context=_make_context(cache_root, ["fm_a"]),
    )
    gold = CrossCohortRoutingGold(image_id="img_a", correct_cohort="breakhis")
    result = runner.run(
        CrossCohortRoutingInput(image_id="img_a", candidate_cohorts=["breakhis", "bracs"]),
        gold=gold,
        case_id="img_a",
    )
    assert result.correct is True
    assert result.decision["_eval_breakdown"]["expert_correct"] is None


# --------------------------------------------------------------------------- #
# DiscrepancyTask
# --------------------------------------------------------------------------- #

def test_discrepancy_credits_correct_label_plus_correct_source(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    runner = AgentRunner(
        client=MockClient(
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision="dcis",
                    confidence=0.8,
                    rationale="classifier predicted dcis and matches gold",
                    trust_source="classifier",
                )
            ]
        ),
        task=DiscrepancyTask(),
        registry=build_default_registry(),
        context=_make_context(cache_root, ["fm_a"]),
    )
    gold = DiscrepancyGold(
        image_id="img_a",
        label="dcis",
        classifier_prediction="dcis",
        retrieval_prediction="invasive",
    )
    result = runner.run(
        DiscrepancyInput(
            image_id="img_a",
            cohort="breakhis",
            classifier_prediction="dcis",
            retrieval_prediction="invasive",
        ),
        gold=gold,
        case_id="img_a",
    )
    assert result.correct is True
    bd = result.decision["_eval_breakdown"]  # type: ignore[index]
    assert bd["label_correct"] is True
    assert bd["source_correct"] is True
    assert bd["gold_correct_source"] == "classifier"


def test_discrepancy_wrong_source_fails(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    runner = AgentRunner(
        client=MockClient(
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision="invasive",
                    confidence=0.7,
                    rationale="trusted retrieval but classifier was right",
                    trust_source="retrieval",
                )
            ]
        ),
        task=DiscrepancyTask(),
        registry=build_default_registry(),
        context=_make_context(cache_root, ["fm_a"]),
    )
    gold = DiscrepancyGold(
        image_id="img_a",
        label="dcis",
        classifier_prediction="dcis",
        retrieval_prediction="invasive",
    )
    result = runner.run(
        DiscrepancyInput(
            image_id="img_a",
            cohort="breakhis",
            classifier_prediction="dcis",
            retrieval_prediction="invasive",
        ),
        gold=gold,
        case_id="img_a",
    )
    assert result.correct is False
    bd = result.decision["_eval_breakdown"]  # type: ignore[index]
    assert bd["source_correct"] is False
    assert bd["gold_correct_source"] == "classifier"


def test_discrepancy_neither_correct_must_escalate(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    runner = AgentRunner(
        client=MockClient(
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision="dcis",
                    confidence=0.4,
                    rationale="neither source agrees with gold; escalating",
                    trust_source="neither",
                    flag_for_review=True,
                )
            ]
        ),
        task=DiscrepancyTask(),
        registry=build_default_registry(),
        context=_make_context(cache_root, ["fm_a"]),
    )
    gold = DiscrepancyGold(
        image_id="img_a",
        label="dcis",
        classifier_prediction="benign",
        retrieval_prediction="invasive",
    )
    result = runner.run(
        DiscrepancyInput(
            image_id="img_a",
            cohort="breakhis",
            classifier_prediction="benign",
            retrieval_prediction="invasive",
        ),
        gold=gold,
        case_id="img_a",
    )
    bd = result.decision["_eval_breakdown"]  # type: ignore[index]
    assert bd["gold_correct_source"] == "neither"
    assert bd["source_correct"] is True
    assert result.correct is True


# --------------------------------------------------------------------------- #
# Aggregate metrics for new tasks
# --------------------------------------------------------------------------- #

def test_evidence_report_metrics_aggregate(tmp_path: Path):
    from breastpath_copilot.agents import evidence_report_metrics

    cache_root = _seed_caches(tmp_path, ["img_a", "img_b"], {"fm_a": 8})
    # Two runs: one fully correct, one wrong label.
    runs = []
    for img, label, pred, ev in [
        ("img_a", "benign", "benign", ["img_b"]),
        ("img_b", "invasive", "benign", ["img_a"]),
    ]:
        runner = _evidence_runner(
            cache_root,
            [
                ScriptedTurn.call(
                    FINAL_TOOL_NAME,
                    decision=pred,
                    confidence=0.8,
                    rationale="-",
                    evidence_image_ids=ev,
                )
            ],
        )
        runs.append(
            runner.run(
                EvidenceReportInput(image_id=img, cohort="breakhis"),
                gold=EvidenceReportGold(image_id=img, label=label),
                case_id=img,
            )
        )
    m = evidence_report_metrics(runs)
    assert m["n"] == 2
    assert m["joint_accuracy"] == pytest.approx(0.5)
    assert m["evidence_provided_rate"] == pytest.approx(1.0)


def test_cohort_routing_metrics_aggregate(tmp_path: Path):
    from breastpath_copilot.agents import cohort_routing_metrics

    cache_root = _seed_caches(tmp_path, ["img_a", "img_b"], {"fm_a": 8})
    runs = []
    for img, gold_c, pred_c, exp in [
        ("img_a", "breakhis", "breakhis", "uni"),
        ("img_b", "bach", "bracs", "uni"),  # wrong cohort
    ]:
        runner = AgentRunner(
            client=MockClient(
                [
                    ScriptedTurn.call(
                        FINAL_TOOL_NAME,
                        decision=pred_c,
                        confidence=0.7,
                        rationale="-",
                        dominant_expert=exp,
                    )
                ]
            ),
            task=CrossCohortRoutingTask(),
            registry=build_default_registry(),
            context=_make_context(cache_root, ["fm_a"]),
        )
        runs.append(
            runner.run(
                CrossCohortRoutingInput(
                    image_id=img, candidate_cohorts=["breakhis", "bach", "bracs"]
                ),
                gold=CrossCohortRoutingGold(
                    image_id=img, correct_cohort=gold_c, expected_dominant_expert="uni"
                ),
                case_id=img,
            )
        )
    m = cohort_routing_metrics(runs)
    assert m["n"] == 2
    assert m["cohort_accuracy"] == pytest.approx(0.5)
    assert m["expert_accuracy"] == pytest.approx(1.0)
    assert m["joint_accuracy"] == pytest.approx(0.5)


def test_discrepancy_metrics_aggregate(tmp_path: Path):
    from breastpath_copilot.agents import discrepancy_metrics

    cache_root = _seed_caches(tmp_path, ["img_a", "img_b"], {"fm_a": 8})
    runs = []
    cases = [
        # correct label + correct source
        ("img_a", "dcis", "dcis", "invasive", "dcis", "classifier"),
        # neither correct - agent flags + says neither
        ("img_b", "dcis", "benign", "invasive", "dcis", "neither"),
    ]
    for img, gold_lbl, cls_pred, ret_pred, agent_decision, agent_trust in cases:
        runner = AgentRunner(
            client=MockClient(
                [
                    ScriptedTurn.call(
                        FINAL_TOOL_NAME,
                        decision=agent_decision,
                        confidence=0.7,
                        rationale="-",
                        trust_source=agent_trust,
                        flag_for_review=(agent_trust == "neither"),
                    )
                ]
            ),
            task=DiscrepancyTask(),
            registry=build_default_registry(),
            context=_make_context(cache_root, ["fm_a"]),
        )
        runs.append(
            runner.run(
                DiscrepancyInput(
                    image_id=img,
                    cohort="breakhis",
                    classifier_prediction=cls_pred,
                    retrieval_prediction=ret_pred,
                ),
                gold=DiscrepancyGold(
                    image_id=img,
                    label=gold_lbl,
                    classifier_prediction=cls_pred,
                    retrieval_prediction=ret_pred,
                ),
                case_id=img,
            )
        )
    m = discrepancy_metrics(runs)
    assert m["n"] == 2
    assert m["joint_accuracy"] == pytest.approx(1.0)
    assert m["neither_escalation_rate"] == pytest.approx(1.0)


def test_trajectory_summary_stats_smoke(tmp_path: Path):
    cache_root = _seed_caches(tmp_path, ["img_a"], {"fm_a": 8})
    ctx = _make_context(cache_root, ["fm_a"])
    client = MockClient(
        [
            ScriptedTurn.call("list_cases", limit=2),
            ScriptedTurn.call(FINAL_TOOL_NAME, decision="benign", confidence=0.5, rationale="-"),
        ]
    )
    runner = AgentRunner(
        client=client, task=TriageTask(), registry=build_default_registry(), context=ctx
    )
    result = runner.run(
        TriageInput(image_id="img_a", cohort="breakhis"),
        gold=TriageGold(image_id="img_a", cohort="breakhis", label="benign"),
    )
    stats = trajectory_summary_stats([result.trajectory])
    assert stats["n"] == 1
    assert stats["mean_step_count"] == 2
    assert stats["mean_tool_calls"] == 2
    assert stats["tool_success_rate"] == pytest.approx(1.0)
    assert stats["truncation_rate"] == 0.0
