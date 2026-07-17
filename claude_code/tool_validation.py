"""Fail-closed authorization of decoded tool-call candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

@dataclass(frozen=True)
class ToolRejection:
    """A decoded tool call that was not safe to expose to an executor."""

    raw_name: str
    reason: str


@dataclass(frozen=True)
class ToolValidationResult:
    """Immutable separation of executable and rejected decoded candidates."""

    accepted: tuple[dict, ...]
    rejected: tuple[ToolRejection, ...]


def _resolve_allowed_name(raw_name: str, allowed_names: set[str]) -> str | None:
    """Resolve Cursor's MCP leaf back to one exact request-owned function name.

    Cursor can return ``toolName`` without the MCP server namespace even when
    the client advertised a flattened ``server_tool`` function.  Exact names
    always win.  A leaf is accepted only when it identifies one unique
    advertised suffix; ambiguity and arbitrary aliases remain fail-closed.
    """
    if raw_name in allowed_names:
        return raw_name
    suffix = f"_{raw_name}"
    matches = [name for name in allowed_names if name.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {value}")


def _normalized_arguments(raw_arguments: object) -> tuple[str | None, dict | None, str | None]:
    """Serialize only a strict JSON object; preserve rejection of other values."""
    arguments = raw_arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ValueError):
            return None, None, "arguments_invalid_json"

    if not isinstance(arguments, dict):
        return None, None, "arguments_not_object"

    try:
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None, None, "arguments_invalid_json"
    return serialized, arguments, None


def _candidate_parts(candidate: object) -> tuple[str, str, object] | None:
    """Extract the minimal executor representation from supported decoded forms."""
    if isinstance(candidate, dict):
        function = candidate.get("function")
        function_data = function if isinstance(function, dict) else {}
        raw_name = function_data.get("name", candidate.get("name", ""))
        raw_arguments = candidate.get(
            "_validation_arguments",
            function_data["arguments"]
            if "arguments" in function_data
            else candidate.get("arguments"),
        )
        call_id = candidate.get("id", "")
    else:
        raw_name = getattr(candidate, "raw_name", "")
        raw_arguments = getattr(candidate, "arguments", None)
        call_id = getattr(candidate, "call_id", "")

    if not isinstance(raw_name, str) or not raw_name:
        return None
    return str(call_id or ""), raw_name, raw_arguments


def _validate_candidate(
    candidate: object,
    allowed_names: set[str],
) -> tuple[dict | None, ToolRejection | None]:
    parts = _candidate_parts(candidate)
    if parts is None:
        return None, ToolRejection(raw_name="", reason="invalid_candidate")

    call_id, raw_name, raw_arguments = parts
    resolved_name = _resolve_allowed_name(raw_name, allowed_names)
    if resolved_name is None:
        return None, ToolRejection(raw_name=raw_name, reason="unknown_tool")

    arguments, _argument_object, reason = _normalized_arguments(raw_arguments)
    if reason is not None:
        return None, ToolRejection(raw_name=raw_name, reason=reason)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": resolved_name, "arguments": arguments},
    }, None


def validate_tool_candidates(
    candidates: Iterable[object] | None,
    allowed_names: set[str],
) -> ToolValidationResult:
    """Authorize decoded calls against only names advertised by this request."""
    allowed = {name for name in allowed_names if isinstance(name, str) and name}
    accepted: list[dict] = []
    rejected: list[ToolRejection] = []

    for candidate in candidates or ():
        executable, rejection = _validate_candidate(
            candidate,
            allowed,
        )
        if executable is not None:
            accepted.append(executable)
        elif rejection is not None:
            rejected.append(rejection)

    return ToolValidationResult(accepted=tuple(accepted), rejected=tuple(rejected))
