"""Agent-facing tool registry.

The agent harness deliberately exposes a *smaller* tool surface than the full
MCP API. Each agent tool wraps one or more low-level MCP tools to give the
LLM clean, single-purpose actions (predict_case / retrieve_neighbors /
report_decision / ...) instead of forcing it to assemble multi-step pipelines
from primitive cache + MoE operations.

Tools are registered as :class:`AgentToolSpec` instances. The registry's
``to_openai_tools()`` method emits OpenAI function-calling schemas suitable
for any client that speaks the Chat Completions API (Ollama, vLLM,
OpenRouter, etc.).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel


@dataclass
class AgentToolSpec:
    """One tool the agent can call.

    Attributes
    ----------
    name
        Function name the LLM uses to call the tool.
    description
        Human-readable, included in the OpenAI tool schema. The LLM uses
        this to decide *when* to call the tool, so be concrete.
    input_model
        Pydantic model describing the call arguments. Used both for
        validation at dispatch time and for auto-generating the JSON schema.
    handler
        Callable ``(input_model_instance, context) -> Any``. The return
        value is JSON-encoded and sent back to the LLM as a tool message.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[Any, "AgentContext"], Any]

    def to_openai_tool(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        # OpenAI's "function" parameter expects a strict subset of JSON
        # schema; Pydantic emits compatible output for our use cases.
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": schema,
            },
        }

    def dispatch(self, arguments: dict[str, Any], context: "AgentContext") -> Any:
        parsed = self.input_model.model_validate(arguments)
        return self.handler(parsed, context)


@dataclass
class AgentContext:
    """Shared state passed to every tool handler during a task run.

    Tools read paths, caches, and configuration from here rather than from
    globals so the same registry can be reused with different cohorts /
    cache roots in a single Python process (important for the leave-one-
    cohort-out evaluation rotations).
    """

    cache_root: str
    cohort: str
    fm_names: list[str]
    moe_routing: str = "soft_topk"
    moe_shared_dim: int = 512
    moe_top_k: int = 2
    moe_classifier_num_classes: int = 5
    moe_checkpoint_path: str | None = None
    retrieval_index_path: str | None = None
    retrieval_index: Any = None  # injectable MoERetrievalIndex
    extras: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extras is None:
            self.extras = {}


# --------------------------------------------------------------------------- #
# Agent tool input schemas
# --------------------------------------------------------------------------- #

class _CaseQuery(BaseModel):
    """Identify a case by its unified image_id."""

    image_id: str


class _RetrievalQuery(BaseModel):
    image_id: str
    top_k: int = 5


class _ReportDecision(BaseModel):
    """Submit the final decision for the case.

    The LLM must call this exactly once per task to terminate the run.
    The schema carries optional task-specific fields so the same tool
    serves all four agent tasks:

    * ``dominant_expert`` -- used by CrossCohortRoutingTask; the name of the
      foundation-model expert that received the largest gate weight on the
      MoE forward pass for this case.
    * ``trust_source``    -- used by DiscrepancyTask; which evidence source
      the agent decided to trust when classifier and retrieval disagreed.
    """

    decision: str
    confidence: float
    rationale: str = ""
    evidence_image_ids: list[str] = []
    flag_for_review: bool = False
    dominant_expert: str | None = None
    trust_source: str | None = None


class _ListCasesQuery(BaseModel):
    cohort: str | None = None
    label: str | None = None
    limit: int = 10


# --------------------------------------------------------------------------- #
# Default handlers (small, dependency-light; agents can be tested with mocks)
# --------------------------------------------------------------------------- #

def _get_case_metadata(args: _CaseQuery, ctx: AgentContext) -> dict[str, Any]:
    """Look up a case in the FM cache manifest of the first configured FM."""
    from breastpath_copilot.core.fm_cache import FMEmbeddingCache

    if not ctx.fm_names:
        return {"ok": False, "error": "AgentContext.fm_names is empty"}
    cache = FMEmbeddingCache(ctx.cache_root, ctx.cohort, ctx.fm_names[0])
    manifest = cache.load_manifest()
    if manifest.empty or args.image_id not in set(manifest["image_id"].astype(str)):
        return {"ok": False, "error": f"unknown image_id {args.image_id!r}"}
    row = manifest[manifest["image_id"] == args.image_id].iloc[0].to_dict()
    return {"ok": True, "metadata": row}


def _list_cases(args: _ListCasesQuery, ctx: AgentContext) -> dict[str, Any]:
    from breastpath_copilot.core.fm_cache import FMEmbeddingCache

    if not ctx.fm_names:
        return {"ok": False, "error": "AgentContext.fm_names is empty"}
    cache = FMEmbeddingCache(ctx.cache_root, args.cohort or ctx.cohort, ctx.fm_names[0])
    manifest = cache.load_manifest()
    if manifest.empty:
        return {"ok": True, "image_ids": []}
    df = manifest
    if args.label is not None and "label" in df.columns:
        df = df[df["label"] == args.label]
    return {"ok": True, "image_ids": df["image_id"].astype(str).head(args.limit).tolist()}


def _predict_case(args: _CaseQuery, ctx: AgentContext) -> dict[str, Any]:
    from breastpath_copilot.api.tools import moe_forward_tool
    from breastpath_copilot.schemas import FMCacheLocator, MoEForwardRequest, MoEHeadConfig

    locators = [
        FMCacheLocator(cache_root=ctx.cache_root, cohort=ctx.cohort, fm_name=fm)
        for fm in ctx.fm_names
    ]
    req = MoEForwardRequest(
        fm_caches=locators,
        image_ids=[args.image_id],
        moe_config=MoEHeadConfig(
            routing=ctx.moe_routing,
            shared_dim=ctx.moe_shared_dim,
            top_k=ctx.moe_top_k,
        ),
        checkpoint_path=ctx.moe_checkpoint_path,
        classifier_num_classes=ctx.moe_classifier_num_classes,
    )
    result = moe_forward_tool(req)
    if not result.image_ids:
        return {"ok": False, "error": f"no cached embeddings for {args.image_id!r}"}
    logits = result.logits[0] if result.logits else None
    if logits is None:
        return {
            "ok": True,
            "image_id": result.image_ids[0],
            "gate_weights": result.gate_weights[0],
            "chosen_experts": result.chosen_experts[0],
            "expert_names": result.expert_names,
            "note": "no classifier head configured -> no class probabilities",
        }
    # Softmax for probabilities
    import math

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps) or 1.0
    probs = [e / z for e in exps]
    pred_idx = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "ok": True,
        "image_id": result.image_ids[0],
        "predicted_class_index": pred_idx,
        "class_probabilities": probs,
        "gate_weights": result.gate_weights[0],
        "chosen_experts": result.chosen_experts[0],
        "expert_names": result.expert_names,
    }


def _retrieve_neighbors(args: _RetrievalQuery, ctx: AgentContext) -> dict[str, Any]:
    """Retrieve top-k similar prior cases using a patient-disjoint MoE index.

    The agent context can supply the index in two ways:

    * ``retrieval_index`` -- a live :class:`MoERetrievalIndex` instance.
    * ``retrieval_index_path`` -- a joblib path to a saved index.

    The query embedding is fetched from the first configured FM's cache
    (consistent with how :func:`_predict_case` builds its inputs). The
    returned neighbors are filtered to be patient-disjoint from the query.
    """
    from breastpath_copilot.core.fm_cache import FMEmbeddingCache
    from breastpath_copilot.core.retrieval_index import MoERetrievalIndex

    index = ctx.retrieval_index
    if index is None and ctx.retrieval_index_path:
        index = MoERetrievalIndex.load(ctx.retrieval_index_path)
    if index is None:
        return {
            "ok": False,
            "error": "no retrieval index configured (set AgentContext.retrieval_index or .retrieval_index_path)",
            "neighbors": [],
        }
    if not ctx.fm_names:
        return {"ok": False, "error": "AgentContext.fm_names is empty", "neighbors": []}

    # Fetch query embedding + patient_id from the first FM's cache.
    cache = FMEmbeddingCache(ctx.cache_root, ctx.cohort, ctx.fm_names[0])
    manifest = cache.load_manifest()
    if manifest.empty or args.image_id not in set(manifest["image_id"].astype(str)):
        return {"ok": False, "error": f"unknown image_id {args.image_id!r}", "neighbors": []}
    row = manifest[manifest["image_id"] == args.image_id].iloc[0]
    rows_arr, vectors = cache.slice_by_image_ids([args.image_id])
    if vectors.size == 0:
        return {"ok": False, "error": "no cached embedding for query", "neighbors": []}
    query_vec = vectors[0]
    patient_id = str(row.get("patient_id", ""))
    cohort = str(row.get("cohort", ctx.cohort))

    neighbors = index.query(
        query_vec,
        top_k=int(args.top_k),
        query_patient_id=patient_id or None,
        query_image_id=args.image_id,
        query_cohort=cohort,
    )
    return {
        "ok": True,
        "image_id": args.image_id,
        "neighbors": [
            {
                "image_id": n.image_id,
                "score": n.score,
                "rank": n.rank,
                "label": n.label,
                "subtype": n.subtype,
                "cohort": n.cohort,
            }
            for n in neighbors
        ],
    }


def _report_decision(args: _ReportDecision, ctx: AgentContext) -> dict[str, Any]:
    """Mark the task as complete. The runner uses this to terminate the loop."""
    return {
        "ok": True,
        "decision": args.decision,
        "confidence": args.confidence,
        "rationale": args.rationale,
        "evidence_image_ids": args.evidence_image_ids,
        "flag_for_review": args.flag_for_review,
        "dominant_expert": args.dominant_expert,
        "trust_source": args.trust_source,
        "_is_final": True,
    }


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class AgentToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AgentToolSpec] = {}

    def register(self, spec: AgentToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> AgentToolSpec:
        if name not in self._tools:
            raise KeyError(f"Unknown agent tool {name!r}. Known: {sorted(self._tools)}")
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_tool() for t in self._tools.values()]


FINAL_TOOL_NAME = "report_decision"


def build_default_registry() -> AgentToolRegistry:
    """Return the default agent-facing tool registry used by all four tasks."""
    reg = AgentToolRegistry()
    reg.register(
        AgentToolSpec(
            name="list_cases",
            description=(
                "List cached case image_ids for the current (or given) cohort, "
                "optionally filtered by lesion label."
            ),
            input_model=_ListCasesQuery,
            handler=_list_cases,
        )
    )
    reg.register(
        AgentToolSpec(
            name="get_case_metadata",
            description=(
                "Return the manifest row for a single case (label, subtype, "
                "magnification, patient_id, cohort, image_path)."
            ),
            input_model=_CaseQuery,
            handler=_get_case_metadata,
        )
    )
    reg.register(
        AgentToolSpec(
            name="predict_case",
            description=(
                "Run the MoE classifier on a case's cached FM embeddings. "
                "Returns class probabilities, gate weights over experts, and "
                "the chosen expert indices for auditability."
            ),
            input_model=_CaseQuery,
            handler=_predict_case,
        )
    )
    reg.register(
        AgentToolSpec(
            name="retrieve_neighbors",
            description=(
                "Retrieve the top-k most similar prior cases by MoE embedding "
                "distance. Returns image_ids, labels, and similarity scores."
            ),
            input_model=_RetrievalQuery,
            handler=_retrieve_neighbors,
        )
    )
    reg.register(
        AgentToolSpec(
            name=FINAL_TOOL_NAME,
            description=(
                "Submit the final decision for this case and terminate the "
                "task. Call this exactly once."
            ),
            input_model=_ReportDecision,
            handler=_report_decision,
        )
    )
    return reg
