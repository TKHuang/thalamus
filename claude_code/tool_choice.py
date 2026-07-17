"""Deterministic translation of client ``tool_choice`` values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ToolChoiceError(ValueError):
    """Raised when a client requests an unsupported or unavailable tool choice."""


@dataclass(frozen=True)
class ToolChoicePolicy:
    mode: str
    name: str | None = None

    def permitted_names(self, advertised_names: list[str]) -> list[str]:
        if self.mode == "none":
            return []
        if self.mode == "specific":
            return [self.name] if self.name else []
        return list(advertised_names)

    def filter_tools(self, tools: list[dict]) -> list[dict]:
        if self.mode == "none":
            return []
        if self.mode != "specific":
            return list(tools)
        return [
            tool
            for tool in tools
            if (tool.get("function") or tool).get("name") == self.name
        ]

    def instruction(self) -> str:
        if self.mode == "none":
            return "Tool choice policy: none. Do not emit a tool call in this response."
        if self.mode == "required":
            return (
                "Tool choice policy: required. Emit at least one complete call to an "
                "advertised client tool in this response."
            )
        if self.mode == "specific":
            return (
                f"Tool choice policy: specific. Call only the client tool {self.name!r} "
                "in this response."
            )
        return (
            "Tool choice policy: auto. Use plain text only when it fully satisfies the "
            "request without external action. If the user asks to create, modify, inspect, "
            "execute, verify, or otherwise act through an advertised client tool, call it "
            "in this response; a plan, future promise, or capability refusal is not a "
            "complete response."
        )


def _advertised_spelling(name: str, advertised_names: list[str]) -> str | None:
    """Return a request-advertised tool name only for an exact match."""
    return name if name in advertised_names else None


def _specific_name(value: dict[str, Any]) -> str | None:
    choice_type = value.get("type")
    if choice_type == "tool":
        name = value.get("name")
        return name if isinstance(name, str) and name else None
    if choice_type == "function":
        function = value.get("function")
        if isinstance(function, dict):
            name = function.get("name")
        else:
            name = value.get("name")
        return name if isinstance(name, str) and name else None
    return None


def resolve_tool_choice(
    value: dict[str, Any] | str | None,
    advertised_names: list[str],
) -> ToolChoicePolicy:
    """Normalize Anthropic, Chat Completions, and Responses tool choices."""
    names = [name for name in advertised_names if isinstance(name, str) and name]
    raw_type: Any = value
    if isinstance(value, dict):
        raw_type = value.get("type")

    if value is None or raw_type == "auto":
        return ToolChoicePolicy("auto")
    if raw_type == "none":
        return ToolChoicePolicy("none")
    if raw_type in {"any", "required"}:
        if not names:
            raise ToolChoiceError("tool_choice requires at least one advertised tool")
        return ToolChoicePolicy("required")
    if isinstance(value, dict) and raw_type in {"tool", "function"}:
        requested_name = _specific_name(value)
        if requested_name is None:
            raise ToolChoiceError("specific tool_choice is missing a tool name")
        advertised_name = _advertised_spelling(requested_name, names)
        if advertised_name is None:
            raise ToolChoiceError(
                f"tool_choice references unadvertised tool: {requested_name}"
            )
        return ToolChoicePolicy("specific", advertised_name)
    raise ToolChoiceError(f"unsupported tool_choice: {value!r}")
