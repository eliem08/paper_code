"""JSON-serializable schemas for the agent harness.

Every artifact the harness produces -- tool calls, tool results, full
trajectories, and final agent decisions -- is a Pydantic model so it can be
persisted (one JSONL per task run) for offline analysis. The format mirrors
the OpenAI Chat Completions tool-use convention so it stays interoperable
with the open LLM clients we benchmark (Gemma 4, Qwen 3.6, LFM 2.5, Granite
4.1) which all expose OpenAI-compatible endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation requested by the LLM."""

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The return value of a tool call, sent back to the LLM."""

    call_id: str
    name: str
    ok: bool
    content: Any = None
    error: str | None = None
    latency_ms: float | None = None


class AgentMessage(BaseModel):
    """OpenAI Chat Completions-style message.

    Roles:
        - ``system``: harness-supplied system prompt.
        - ``user``: task input / human turn.
        - ``assistant``: LLM response; may carry ``tool_calls``.
        - ``tool``: result of a tool call (``tool_call_id`` references the
          assistant's ``ToolCall.call_id``).
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # tool function name when role == "tool"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


class Trajectory(BaseModel):
    """Complete record of one task run."""

    task_name: str
    model_name: str
    case_id: str | None = None
    cohort: str | None = None
    messages: list[AgentMessage] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    step_count: int = 0
    truncated: bool = False
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: str | None = None

    def add_message(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)


class AgentResult(BaseModel):
    """Final harness output for one case."""

    task_name: str
    model_name: str
    case_id: str | None = None
    cohort: str | None = None
    success: bool
    decision: dict[str, Any] | None = None
    gold: dict[str, Any] | None = None
    correct: bool | None = None
    trajectory: Trajectory
    error: str | None = None
