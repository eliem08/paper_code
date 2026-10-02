"""LLM client abstraction for the agent harness.

Two concrete clients ship:

* :class:`MockClient`           -- scripted responses; used by the test
                                   suite and for offline trajectory replays.
* :class:`OpenAICompatClient`   -- talks to any OpenAI-Chat-Completions-
                                   compatible endpoint (Ollama, vLLM, OpenRouter,
                                   any HF text-generation-inference deployment).
                                   This is how we evaluate the four open
                                   models named in the Q1 plan
                                   (Gemma 4 31B IT, Qwen 3.6 27B, LFM 2.5 1.2B,
                                   Granite 4.1 30B).

A future :class:`HFInProcessClient` (loading the model directly via
``transformers``) is intentionally left as a separate follow-up so this
module stays light-weight and dependency-optional.
"""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

from breastpath_copilot.agents.types import AgentMessage, ToolCall


@dataclass
class ChatResponse:
    """Normalised response from any LLM client."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMClient(ABC):
    """Abstract chat completion client with optional tool-calling."""

    model_name: str

    @abstractmethod
    def chat(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        ...


# --------------------------------------------------------------------------- #
# MockClient
# --------------------------------------------------------------------------- #

@dataclass
class ScriptedTurn:
    """One scripted assistant turn for :class:`MockClient`.

    Either provides ``tool_calls`` (the assistant calls tools and the runner
    continues) or ``content`` (a plain assistant message; if it is the final
    turn the runner exits).
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @classmethod
    def call(cls, name: str, **arguments: Any) -> "ScriptedTurn":
        return cls(
            tool_calls=[
                ToolCall(call_id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments)
            ]
        )

    @classmethod
    def text(cls, content: str) -> "ScriptedTurn":
        return cls(content=content)


class MockClient(LLMClient):
    """LLM client that replays a fixed list of :class:`ScriptedTurn`.

    Each ``chat`` call consumes one turn from the front of the queue.
    Raises :class:`StopIteration` when exhausted, which the runner treats as
    truncation.
    """

    def __init__(self, turns: Iterable[ScriptedTurn], model_name: str = "mock") -> None:
        self.model_name = model_name
        self._turns: list[ScriptedTurn] = list(turns)
        self._idx = 0
        # Recorded conversation history for inspection in tests.
        self.last_messages: list[AgentMessage] = []
        self.last_tools: list[dict[str, Any]] | None = None

    def chat(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.last_messages = list(messages)
        self.last_tools = list(tools) if tools is not None else None
        if self._idx >= len(self._turns):
            raise StopIteration("MockClient exhausted: no more scripted turns")
        turn = self._turns[self._idx]
        self._idx += 1
        return ChatResponse(content=turn.content, tool_calls=list(turn.tool_calls), raw=turn)


# --------------------------------------------------------------------------- #
# OpenAI-compatible client (Ollama / vLLM / OpenRouter / TGI)
# --------------------------------------------------------------------------- #

class OpenAICompatClient(LLMClient):
    """Thin wrapper over the ``openai`` SDK pointing at any compatible endpoint.

    Defaults to ``http://localhost:11434/v1`` (Ollama). For vLLM, set
    ``base_url=http://localhost:8000/v1``. For OpenRouter, set
    ``base_url=https://openrouter.ai/api/v1`` and provide an API key.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - guard for missing dep
            raise RuntimeError(
                "OpenAICompatClient requires the 'openai' package. "
                "Install via `pip install openai`."
            ) from exc
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._client = OpenAI(
            base_url=base_url or os.environ.get("AGENT_LLM_BASE_URL") or "http://localhost:11434/v1",
            api_key=api_key or os.environ.get("AGENT_LLM_API_KEY") or "ollama",
            timeout=timeout,
        )

    def _serialise_messages(self, messages: list[AgentMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role}
            if m.content is not None:
                entry["content"] = m.content
            if m.tool_call_id is not None:
                entry["tool_call_id"] = m.tool_call_id
            if m.name is not None and m.role == "tool":
                entry["name"] = m.name
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        return out

    def chat(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._serialise_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")
        response = self._client.chat.completions.create(**payload)
        msg = response.choices[0].message
        tool_calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):  # type: ignore[union-attr]
            arguments: dict[str, Any] = {}
            raw_args = tc.function.arguments or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {"_raw_arguments": raw_args}
            tool_calls.append(ToolCall(call_id=tc.id, name=tc.function.name, arguments=arguments))
        return ChatResponse(content=msg.content, tool_calls=tool_calls, raw=response)
