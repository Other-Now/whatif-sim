"""LLM providers behind one small interface.

``AnthropicProvider`` calls the Messages API with forced tool use (structured
output) and a cache breakpoint on the system prompt. ``ScriptedProvider``
replays hand-written tool inputs, for tests and for running the pipeline
without an API key. Scripted responses are NOT model output; they exist to
exercise the validators and the repair loop deterministically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

from .prompts import SYSTEM, TOOLS


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    usage: dict = field(default_factory=dict)


class Provider(Protocol):
    def call(self, messages: list[dict], tool: str) -> ToolCall: ...


def system_blocks() -> list[dict]:
    return [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]


class AnthropicProvider:
    def __init__(self, model: str | None = None, max_tokens: int = 4096):
        import anthropic  # imported lazily so the rest works without the SDK

        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model = model or os.environ.get("WHATIF_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens

    def call(self, messages: list[dict], tool: str) -> ToolCall:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks(),
            tools=TOOLS,                               # all tools, every call: keeps the prefix cacheable
            tool_choice={"type": "tool", "name": tool},
            messages=messages,
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        u = resp.usage
        usage = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        return ToolCall(block.id, block.name, block.input, usage)


class ScriptedProvider:
    """Replays ``script``: a list of tool inputs, one per call, in order.

    Also records every request so tests can check what the real API would
    have been sent (repair messages, an unchanged cacheable prefix)."""

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.requests: list[dict] = []

    def call(self, messages: list[dict], tool: str) -> ToolCall:
        self.requests.append({"system": system_blocks(), "tools": TOOLS, "tool": tool,
                              "messages": [dict(m) for m in messages]})
        if not self.script:
            raise RuntimeError("script exhausted")
        step = self.script.pop(0)
        if step["tool"] != tool:
            raise RuntimeError(f"script expected {step['tool']}, pipeline asked for {tool}")
        return ToolCall(f"toolu_{len(self.requests):03d}", tool, step["input"], {})
