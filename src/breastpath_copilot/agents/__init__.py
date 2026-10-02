"""Open-source agentic harness for the Q1 pathology MCP server.

Layered as:

* :mod:`.types`         -- JSON-serializable schemas (messages, tool calls,
                           trajectories, task I/O).
* :mod:`.tool_registry` -- the agent-facing tool surface and OpenAI-style
                           function schemas auto-generated from Pydantic.
* :mod:`.clients`       -- LLM client abstraction with a :class:`MockClient`
                           for tests and an OpenAI-compatible client for
                           Ollama / vLLM / OpenRouter.
* :mod:`.tasks`         -- the four agent tasks for the Q1 evaluation
                           (triage / evidence / cross-cohort / discrepancy).
* :mod:`.runner`        -- step-budgeted tool-call loop.
* :mod:`.metrics`       -- triage accuracy, tool-call efficiency, etc.
"""

from breastpath_copilot.agents.types import (
    AgentMessage,
    AgentResult,
    ToolCall,
    ToolResult,
    Trajectory,
)
from breastpath_copilot.agents.tool_registry import (
    AgentToolSpec,
    build_default_registry,
)
from breastpath_copilot.agents.clients import (
    ChatResponse,
    LLMClient,
    MockClient,
    OpenAICompatClient,
    ScriptedTurn,
)
from breastpath_copilot.agents.tasks import (
    AgentTask,
    CrossCohortRoutingGold,
    CrossCohortRoutingInput,
    CrossCohortRoutingTask,
    DiscrepancyGold,
    DiscrepancyInput,
    DiscrepancyTask,
    EvidenceReportGold,
    EvidenceReportInput,
    EvidenceReportTask,
    TriageDecision,
    TriageGold,
    TriageInput,
    TriageTask,
)
from breastpath_copilot.agents.runner import AgentRunner
from breastpath_copilot.agents.metrics import (
    cohort_routing_metrics,
    discrepancy_metrics,
    evidence_report_metrics,
    triage_accuracy,
    trajectory_summary_stats,
)

__all__ = [
    "AgentMessage",
    "AgentResult",
    "AgentRunner",
    "AgentTask",
    "AgentToolSpec",
    "ChatResponse",
    "CrossCohortRoutingGold",
    "CrossCohortRoutingInput",
    "CrossCohortRoutingTask",
    "DiscrepancyGold",
    "DiscrepancyInput",
    "DiscrepancyTask",
    "EvidenceReportGold",
    "EvidenceReportInput",
    "EvidenceReportTask",
    "LLMClient",
    "MockClient",
    "OpenAICompatClient",
    "ScriptedTurn",
    "ToolCall",
    "ToolResult",
    "Trajectory",
    "TriageDecision",
    "TriageGold",
    "TriageInput",
    "TriageTask",
    "build_default_registry",
    "cohort_routing_metrics",
    "discrepancy_metrics",
    "evidence_report_metrics",
    "triage_accuracy",
    "trajectory_summary_stats",
]
