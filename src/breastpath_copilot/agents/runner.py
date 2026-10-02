"""Bounded tool-call loop that runs an :class:`AgentTask` to completion."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from breastpath_copilot.agents.clients import LLMClient
from breastpath_copilot.agents.tasks import AgentTask
from breastpath_copilot.agents.tool_registry import (
    FINAL_TOOL_NAME,
    AgentContext,
    AgentToolRegistry,
)
from breastpath_copilot.agents.types import (
    AgentMessage,
    AgentResult,
    ToolCall,
    ToolResult,
    Trajectory,
)


@dataclass
class AgentRunner:
    """Drive an :class:`LLMClient` through a tool-call loop until it submits a
    final ``report_decision`` or the step budget is exhausted.

    Parameters
    ----------
    client
        The LLM client to call.
    task
        The agent task definition (system prompt, evaluator).
    registry
        Agent-facing tool registry whose tools are exposed to the LLM.
    context
        Shared per-run state passed to every tool handler.
    max_steps
        Hard cap on assistant turns. Prevents runaway loops on misbehaving
        models; ``truncated=True`` is recorded on the trajectory when hit.
    trajectory_dir
        If set, each completed run is written as a JSON file
        ``<task>_<model>_<case_id>.json`` for offline analysis.
    """

    client: LLMClient
    task: AgentTask
    registry: AgentToolRegistry
    context: AgentContext
    max_steps: int = 8
    trajectory_dir: Path | None = None

    def run(
        self,
        task_input: BaseModel,
        gold: BaseModel | None = None,
        case_id: str | None = None,
    ) -> AgentResult:
        trajectory = Trajectory(
            task_name=self.task.name,
            model_name=self.client.model_name,
            case_id=case_id,
            cohort=self.context.cohort,
        )
        trajectory.add_message(AgentMessage(role="system", content=self.task.system_prompt()))
        trajectory.add_message(AgentMessage(role="user", content=self.task.user_prompt(task_input)))

        decision: dict[str, Any] | None = None
        error: str | None = None
        tools_schema = self.registry.to_openai_tools()

        for _ in range(self.max_steps):
            trajectory.step_count += 1
            try:
                response = self._chat_with_retry(trajectory.messages, tools_schema)
            except StopIteration as e:
                error = f"client exhausted: {e}"
                trajectory.truncated = True
                break
            except Exception as e:  # pragma: no cover - depends on client impl
                error = f"client error: {type(e).__name__}: {e}"
                break

            assistant_msg = AgentMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls or None,
            )
            trajectory.add_message(assistant_msg)

            if not response.tool_calls:
                # Model produced a final assistant text with no tool call.
                # Treat as truncation -- the protocol requires report_decision.
                trajectory.truncated = True
                break

            final_seen = False
            for call in response.tool_calls:
                result = self._dispatch(call)
                trajectory.add_tool_result(result)
                trajectory.add_message(
                    AgentMessage(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.call_id,
                        content=json.dumps(result.content, default=str)[:4000],
                    )
                )
                if call.name == FINAL_TOOL_NAME and result.ok:
                    decision = {k: v for k, v in result.content.items() if not k.startswith("_")}
                    final_seen = True
            if final_seen:
                break
        else:
            trajectory.truncated = True

        trajectory.finished_at = _utc_now()

        correct: bool | None = None
        if decision is not None and gold is not None:
            correct, breakdown = self.task.evaluate(decision, gold)
            decision = {**decision, "_eval_breakdown": breakdown}

        result = AgentResult(
            task_name=self.task.name,
            model_name=self.client.model_name,
            case_id=case_id,
            cohort=self.context.cohort,
            success=decision is not None and error is None,
            decision=decision,
            gold=gold.model_dump() if gold is not None else None,
            correct=correct,
            trajectory=trajectory,
            error=error,
        )
        if self.trajectory_dir is not None:
            self._persist(result)
        return result

    def _chat_with_retry(
        self,
        messages: list[AgentMessage],
        tools_schema: list[dict[str, Any]],
        max_attempts: int = 4,
        base_delay: float = 2.0,
    ):
        """Call ``self.client.chat`` with retries for transient failures.

        Live LLM endpoints (shared-pool rate limits, occasional malformed
        provider responses) fail transiently in ways that have nothing to do
        with the model's actual reasoning ability. Without a retry, those
        failures are indistinguishable in the results from a genuine agent
        breakdown and silently deflate the completion rate. ``StopIteration``
        (a scripted :class:`MockClient` running dry, used in tests) is never
        retried -- it is a deterministic signal, not a transient fault.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return self.client.chat(messages, tools=tools_schema)
            except StopIteration:
                raise
            except Exception as e:  # pragma: no cover - depends on client impl
                last_exc = e
                if attempt < max_attempts - 1:
                    delay = base_delay * (2**attempt)
                    print(f"    [retry {attempt + 1}/{max_attempts - 1}] "
                          f"{type(e).__name__}: {e} -- retrying in {delay:.0f}s", flush=True)
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _dispatch(self, call: ToolCall) -> ToolResult:
        start = time.perf_counter()
        try:
            spec = self.registry.get(call.name)
            content = spec.dispatch(call.arguments, self.context)
            ok = bool(content.get("ok", True)) if isinstance(content, dict) else True
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=ok,
                content=content,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        except Exception as e:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

    def _persist(self, result: AgentResult) -> None:
        assert self.trajectory_dir is not None
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        case_part = _sanitize_filename_part(result.case_id or "case")
        model_part = _sanitize_filename_part(result.model_name)
        path = self.trajectory_dir / f"{result.task_name}_{model_part}_{case_part}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _sanitize_filename_part(s: str) -> str:
    """Replace characters illegal (or special) in Windows filenames.

    Model tags like ``qwen3.6:35b`` contain a colon, which NTFS treats as an
    Alternate Data Stream separator: ``path/foo:bar.json`` silently creates a
    ``bar.json`` stream *inside* the file ``path/foo``, invisible to normal
    directory listings, ``git``, and most tooling. Replace every character
    outside ``[A-Za-z0-9._-]`` with ``-`` so trajectory files are always
    visible, ordinary files.
    """
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "-", s)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
