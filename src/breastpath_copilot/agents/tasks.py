"""Agent task definitions for the Q1 article's agentic evaluation.

Four tasks, jointly cover the three co-equal contributions of the paper:

1. :class:`TriageTask`           -- classifier-driven decision under uncertainty.
2. :class:`EvidenceReportTask`   -- structured report grounded in retrieval.
3. :class:`CrossCohortRoutingTask` -- gate-aware cohort selection (tests MoE
   routing literacy).
4. :class:`DiscrepancyTask`      -- classifier-vs-retrieval disagreement resolution.

This file ships :class:`TriageTask` end-to-end so the harness can be exercised
immediately. The remaining three tasks have full input/output schemas and
docstrings; their ``evaluate`` and ``system_prompt`` methods raise
``NotImplementedError`` until we fill them in. That keeps the public surface
visible without committing to half-specified evaluation rubrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Common
# --------------------------------------------------------------------------- #

UNIFIED_LABELS = ("normal", "benign", "atypia", "dcis", "invasive")


@dataclass
class AgentTask(ABC):
    """Common interface for all four agent tasks."""

    name: str

    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @abstractmethod
    def user_prompt(self, task_input: BaseModel) -> str:
        ...

    @abstractmethod
    def evaluate(self, decision: dict[str, Any], gold: BaseModel) -> tuple[bool, dict[str, Any]]:
        """Return ``(correct, breakdown)``. ``breakdown`` carries per-criterion details."""


# --------------------------------------------------------------------------- #
# Triage task (fully implemented)
# --------------------------------------------------------------------------- #

class TriageInput(BaseModel):
    image_id: str
    cohort: str


class TriageGold(BaseModel):
    image_id: str
    cohort: str
    label: str  # unified vocabulary
    subtype: str | None = None
    patient_id: str | None = None


class TriageDecision(BaseModel):
    decision: str = Field(description="One of: benign, suspicious, refer")
    confidence: float
    rationale: str = ""
    evidence_image_ids: list[str] = Field(default_factory=list)
    flag_for_review: bool = False


_TRIAGE_DECISION_MAP = {
    "benign": {"normal", "benign"},
    "suspicious": {"atypia"},
    "refer": {"dcis", "invasive"},
}


def map_label_to_triage(label: str) -> str:
    for triage, labels in _TRIAGE_DECISION_MAP.items():
        if label in labels:
            return triage
    return "refer"  # conservative default for unknown labels


class TriageTask(AgentTask):
    """Decide whether a case is benign / suspicious (needs second look) / refer.

    The mapping from the unified five-class vocabulary to the three-way triage
    decision intentionally collapses atypia (FEA/ADH) into ``suspicious`` and
    DCIS/invasive into ``refer`` -- early-detection-aligned.
    """

    def __init__(self) -> None:
        self.name = "triage"

    def system_prompt(self) -> str:
        return (
            "You are an assistant pathology triage agent. Your job is to "
            "decide whether a breast-tissue case is benign (no action), "
            "suspicious (needs a second look by a pathologist), or refer "
            "(clearly malignant or pre-invasive: needs a referral).\n\n"
            "Available tools:\n"
            "- list_cases / get_case_metadata: inspect what is in the cohort.\n"
            "- predict_case: run the MoE classifier; returns class "
            "probabilities, gate weights over foundation-model experts, and "
            "chosen experts (use these for auditability).\n"
            "- retrieve_neighbors: fetch similar prior cases.\n"
            "- report_decision: submit your final decision. Call this "
            "exactly once when you are confident; it terminates the task.\n\n"
            "Rules:\n"
            "1. Always call predict_case before reporting a decision.\n"
            "2. If predicted probabilities are diffuse (max < 0.6), prefer "
            "'suspicious' and set flag_for_review=true.\n"
            "3. Decision must be one of: benign, suspicious, refer.\n"
            "4. Confidence is your subjective certainty in [0, 1]."
        )

    def user_prompt(self, task_input: BaseModel) -> str:
        assert isinstance(task_input, TriageInput)
        return (
            f"Triage case image_id={task_input.image_id} from cohort="
            f"{task_input.cohort}. Use the available tools, then submit your "
            "decision via report_decision."
        )

    def evaluate(
        self, decision: dict[str, Any], gold: BaseModel
    ) -> tuple[bool, dict[str, Any]]:
        assert isinstance(gold, TriageGold)
        expected = map_label_to_triage(gold.label)
        predicted = str(decision.get("decision", "")).strip().lower()
        correct = predicted == expected
        # Partial credit: 'suspicious' on a 'refer' case is safer than 'benign'.
        is_safe = predicted in {"suspicious", "refer"} if expected == "refer" else None
        return correct, {
            "expected_triage": expected,
            "gold_label": gold.label,
            "predicted": predicted,
            "is_safe": is_safe,
            "confidence": float(decision.get("confidence", 0.0) or 0.0),
            "flag_for_review": bool(decision.get("flag_for_review", False)),
        }


# --------------------------------------------------------------------------- #
# Evidence report task (stub)
# --------------------------------------------------------------------------- #

class EvidenceReportInput(BaseModel):
    image_id: str
    cohort: str
    top_k_neighbors: int = 5


class EvidenceReportGold(BaseModel):
    image_id: str
    label: str
    """Optional ids of acceptable supporting neighbors; if non-empty, the
    evaluator credits the agent when at least one provided
    ``evidence_image_ids`` overlaps with this set."""
    acceptable_evidence_ids: list[str] = Field(default_factory=list)


class EvidenceReportTask(AgentTask):
    """Produce a structured report grounded in retrieved prior cases.

    The agent is expected to use both the classifier (``predict_case``) and
    retrieval (``retrieve_neighbors``) and submit a decision *plus* a list
    of ``evidence_image_ids`` referencing supporting prior cases.
    """

    def __init__(self) -> None:
        self.name = "evidence_report"

    def system_prompt(self) -> str:
        return (
            "You are an evidence-grounded pathology reporting agent. For each "
            "case, you must:\n"
            "  1. Call predict_case to obtain the MoE classifier's verdict "
            "and per-expert gate weights.\n"
            "  2. Call retrieve_neighbors with a sensible top_k (3-7) to "
            "gather similar prior cases.\n"
            "  3. Optionally call get_case_metadata on the most informative "
            "neighbors to inspect their labels.\n"
            "  4. Call report_decision exactly once with:\n"
            "       - decision: predicted unified label "
            "(normal / benign / atypia / dcis / invasive)\n"
            "       - confidence in [0, 1]\n"
            "       - rationale: a 1-3 sentence justification grounded in the "
            "neighbors and the classifier output\n"
            "       - evidence_image_ids: ids of the neighbors you actually "
            "used as evidence (MUST be non-empty)\n"
            "Reports without evidence_image_ids will be rejected."
        )

    def user_prompt(self, task_input: BaseModel) -> str:
        assert isinstance(task_input, EvidenceReportInput)
        return (
            f"Produce a structured pathology report for image_id="
            f"{task_input.image_id} from cohort={task_input.cohort}. "
            f"Use top_k={task_input.top_k_neighbors} for retrieval. "
            f"Submit your final report via report_decision."
        )

    def evaluate(
        self, decision: dict[str, Any], gold: BaseModel
    ) -> tuple[bool, dict[str, Any]]:
        assert isinstance(gold, EvidenceReportGold)
        predicted = str(decision.get("decision", "")).strip().lower()
        evidence_ids = list(decision.get("evidence_image_ids") or [])
        label_correct = predicted == gold.label
        evidence_provided = len(evidence_ids) > 0
        overlap_ok = True
        if gold.acceptable_evidence_ids:
            overlap_ok = bool(set(evidence_ids) & set(gold.acceptable_evidence_ids))
        correct = bool(label_correct and evidence_provided and overlap_ok)
        return correct, {
            "label_correct": label_correct,
            "evidence_provided": evidence_provided,
            "evidence_overlap_ok": overlap_ok,
            "evidence_count": len(evidence_ids),
            "predicted": predicted,
            "gold_label": gold.label,
        }


# --------------------------------------------------------------------------- #
# Cross-cohort routing task (stub)
# --------------------------------------------------------------------------- #

class CrossCohortRoutingInput(BaseModel):
    image_id: str
    candidate_cohorts: list[str]


class CrossCohortRoutingGold(BaseModel):
    image_id: str
    correct_cohort: str
    """If set, the evaluator also requires that the agent identified this
    foundation-model expert as the one with the highest gate weight."""
    expected_dominant_expert: str | None = None


class CrossCohortRoutingTask(AgentTask):
    """Identify a case's source cohort and the dominant MoE expert.

    This task tests two things at once: (a) cohort discovery via the
    metadata tool, and (b) gate-weight literacy -- the agent must call
    predict_case and inspect ``gate_weights`` / ``expert_names`` to name the
    dominant FM expert. Useful for paper claims about MoE auditability.
    """

    def __init__(self) -> None:
        self.name = "cross_cohort_routing"

    def system_prompt(self) -> str:
        return (
            "You are a cohort-routing agent for a multi-cohort pathology "
            "workflow. Given a candidate image_id and a list of candidate "
            "cohorts, identify which cohort it belongs to and report which "
            "foundation-model expert dominates the MoE routing for this "
            "case.\n\n"
            "Procedure:\n"
            "  1. Call get_case_metadata with the image_id (the metadata "
            "tool searches the currently-configured cohort; switch your "
            "AgentContext to each candidate cohort or rely on the unified "
            "image_id prefix convention -- breakhis_*, bach_*, bracs_*).\n"
            "  2. Call predict_case to obtain gate_weights and expert_names.\n"
            "  3. Call report_decision with:\n"
            "       - decision: cohort name (one of the candidate cohorts)\n"
            "       - confidence\n"
            "       - dominant_expert: the expert_name with the largest "
            "weight in gate_weights\n"
            "       - rationale: short explanation."
        )

    def user_prompt(self, task_input: BaseModel) -> str:
        assert isinstance(task_input, CrossCohortRoutingInput)
        return (
            f"Identify the source cohort of image_id={task_input.image_id} "
            f"from the candidate set {task_input.candidate_cohorts}, and "
            f"report which MoE expert dominates the routing for this case."
        )

    def evaluate(
        self, decision: dict[str, Any], gold: BaseModel
    ) -> tuple[bool, dict[str, Any]]:
        assert isinstance(gold, CrossCohortRoutingGold)
        predicted_cohort = str(decision.get("decision", "")).strip().lower()
        predicted_expert = (decision.get("dominant_expert") or "").strip().lower() or None
        cohort_correct = predicted_cohort == gold.correct_cohort.lower()
        expert_correct: bool | None = None
        if gold.expected_dominant_expert is not None:
            expert_correct = predicted_expert == gold.expected_dominant_expert.lower()
            correct = cohort_correct and expert_correct
        else:
            correct = cohort_correct
        return correct, {
            "predicted_cohort": predicted_cohort,
            "gold_cohort": gold.correct_cohort,
            "cohort_correct": cohort_correct,
            "predicted_dominant_expert": predicted_expert,
            "expected_dominant_expert": gold.expected_dominant_expert,
            "expert_correct": expert_correct,
        }


# --------------------------------------------------------------------------- #
# Discrepancy resolution task (stub)
# --------------------------------------------------------------------------- #

class DiscrepancyInput(BaseModel):
    image_id: str
    cohort: str
    classifier_prediction: str
    retrieval_prediction: str


class DiscrepancyGold(BaseModel):
    image_id: str
    label: str
    classifier_prediction: str
    retrieval_prediction: str

    def correct_source(self) -> str:
        """Which source matched the gold label.

        Returns ``"classifier"``, ``"retrieval"``, ``"both"``, or
        ``"neither"`` -- the agent should ideally trust the matching source
        (or escalate if neither matches).
        """
        cls_match = self.classifier_prediction == self.label
        ret_match = self.retrieval_prediction == self.label
        if cls_match and ret_match:
            return "both"
        if cls_match:
            return "classifier"
        if ret_match:
            return "retrieval"
        return "neither"


class DiscrepancyTask(AgentTask):
    """Resolve a disagreement between classifier and retrieval channels.

    Inputs supply both predictions up front; the agent must decide which
    source to trust (or flag escalation) and submit a final label.
    """

    def __init__(self, label_options: list[str] | None = None) -> None:
        self.name = "discrepancy_resolution"
        # Must match the gold-label vocabulary the caller's scheme actually
        # produces (see DiscrepancyGold.label), or evaluate()'s exact-string
        # match will systematically fail for any label the prompt omits.
        # Defaults to the historical 7-class-derived vocabulary for backward
        # compatibility with existing tests/scripts; pass the caller's own
        # scheme labels explicitly when running under a different scheme
        # (e.g. the 3-class benign/atypical/malignant triage scheme).
        self.label_options = label_options or ["normal", "benign", "atypia", "dcis", "invasive"]

    def system_prompt(self) -> str:
        options = " / ".join(self.label_options)
        return (
            "You are a discrepancy-resolution agent for a pathology workflow. "
            "You will be told what the MoE classifier predicted AND what the "
            "retrieval channel predicted for the same case -- and these will "
            "disagree. Your job is to pick which one to trust and submit a "
            "final unified label, or flag the case for human review when "
            "neither source is credible.\n\n"
            "Procedure:\n"
            "  1. Optionally call get_case_metadata, predict_case (to inspect "
            "gate weights), and retrieve_neighbors (to inspect neighbor "
            "labels) to ground your judgement.\n"
            "  2. Call report_decision with:\n"
            f"       - decision: final unified label ({options}) -- use "
            "exactly one of these tokens, verbatim\n"
            "       - confidence\n"
            "       - trust_source: 'classifier', 'retrieval', or 'neither' "
            "(if you set 'neither', also set flag_for_review=true)\n"
            "       - rationale: 1-3 sentences justifying which source you "
            "trusted and why.\n"
            "       - evidence_image_ids: neighbor ids that supported your "
            "decision, if any."
        )

    def user_prompt(self, task_input: BaseModel) -> str:
        assert isinstance(task_input, DiscrepancyInput)
        return (
            f"Case image_id={task_input.image_id} from cohort="
            f"{task_input.cohort}. The MoE classifier predicted "
            f"'{task_input.classifier_prediction}' while the retrieval "
            f"channel predicted '{task_input.retrieval_prediction}'. Decide "
            f"which source to trust and submit your final label."
        )

    def evaluate(
        self, decision: dict[str, Any], gold: BaseModel
    ) -> tuple[bool, dict[str, Any]]:
        assert isinstance(gold, DiscrepancyGold)
        predicted = str(decision.get("decision", "")).strip().lower()
        trust = (decision.get("trust_source") or "").strip().lower() or None
        flag = bool(decision.get("flag_for_review", False))
        gold_source = gold.correct_source()
        label_correct = predicted == gold.label

        source_correct: bool | None = None
        if trust is not None:
            if gold_source == "neither":
                source_correct = trust == "neither" and flag
            elif gold_source == "both":
                source_correct = trust in {"classifier", "retrieval"}
            else:
                source_correct = trust == gold_source

        correct = bool(label_correct and (trust is not None) and (source_correct or False))
        return correct, {
            "predicted": predicted,
            "gold_label": gold.label,
            "label_correct": label_correct,
            "trust_source": trust,
            "gold_correct_source": gold_source,
            "source_correct": source_correct,
            "flagged": flag,
        }
