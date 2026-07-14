from __future__ import annotations

"""Composer-2.x tool-call token parser.

Cursor's ``composer-2.5`` / ``composer-2.5-fast`` models stream their entire
output — reasoning, answer, and tool calls — through the protobuf *thinking*
field (field 25), not the content field.  Their tool calls are emitted as
DeepSeek-style **text tokens**, not Anthropic ``tool_use`` JSON:

    <|tool_calls_begin|>
      <|tool_call_begin|> <name> <|tool_sep|> <key>\\n<value> <|tool_sep|> ... <|tool_call_end|>
      <|tool_call_begin|> ... <|tool_call_end|>
    <|tool_calls_end|>

Both an ASCII form (``|`` / ``_``) and a unicode form (``｜`` U+FF5C / ``▁``
U+2581) exist; we canonicalize to ASCII first.  Reasoning precedes the answer,
separated by ``</think>`` or ``<|final|>`` (a.k.a. control tokens).

This module ports the proven parser from the ``standardagents/composer-api``
reference (``worker/cursor.ts``).  ``decodeBinaryToolCall`` is a stub there —
tool calls only ever arrive as text — so there is no binary path here either.

Output tool-call dicts use the flat ``{"name", "arguments"}`` shape that
``tool_parser.normalize_tool_calls`` accepts directly.
"""

import json
import re
from typing import NamedTuple

# ── Marker constants (canonical ASCII form) ────────────────────────────────

TOOL_CALLS_BEGIN = "<|tool_calls_begin|>"
TOOL_CALLS_END = "<|tool_calls_end|>"
TOOL_CALL_BEGIN = "<|tool_call_begin|>"
TOOL_CALL_END = "<|tool_call_end|>"
TOOL_SEP = "<|tool_sep|>"

_ASCII_MARKERS = [TOOL_CALLS_BEGIN, TOOL_CALLS_END, TOOL_CALL_BEGIN, TOOL_CALL_END, TOOL_SEP]


def _unicode_variant(marker: str) -> str:
    return marker.replace("|", "｜").replace("_", "▁")


# ASCII + unicode forms, used for partial-marker detection at chunk boundaries.
TOOL_MARKER_CANDIDATES = [m for marker in _ASCII_MARKERS for m in (marker, _unicode_variant(marker))]

# ``</think>`` or ``<|final|>`` / ``<｜final｜>`` (with optional whitespace).
_CONTROL_TOKEN_RE = re.compile(r"</think>|<\s*[|｜]\s*final\s*[|｜]\s*>")

# Canonicalizes any unicode marker variant back to ASCII.
_MARKER_CANON_RE = re.compile(
    r"<\s*[|｜]\s*"
    r"(tool[_▁]calls[_▁]begin|tool[_▁]calls[_▁]end|tool[_▁]call[_▁]begin|tool[_▁]call[_▁]end|tool[_▁]sep)"
    r"\s*[|｜]\s*>"
)

_INT_OR_FLOAT_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_KEY_VALUE_RE = re.compile(r"^([^\r\n]+)(?:\r?\n([\s\S]*))?$")
_INLINE_CALL_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*(?:\(([\s\S]*)\)|\[([\s\S]*)\])?$")
_INLINE_ARG_RE = re.compile(r"^([A-Za-z0-9_.\-]+)\s*[:=]\s*([\s\S]*)$")


def is_composer_model(model: str | None) -> bool:
    """True for Cursor Composer-2.x models.

    These stream their whole output (reasoning + answer + DeepSeek-style tool
    tokens) through the protobuf *thinking* field and require this parser.

    ``composer-1.5`` is intentionally excluded: it embeds ``<think>`` tags in
    the *content* field and is handled by ``pipeline._ThinkTagSplitter``.  It is
    also a fallback target for many models, so it must keep its existing path.
    """
    return (model or "").strip().lower().startswith("composer-2")


def canonicalize_composer_markers(text: str) -> str:
    """Rewrite unicode marker variants (``｜``/``▁``) to ASCII (``|``/``_``)."""
    return _MARKER_CANON_RE.sub(lambda m: f"<|{m.group(1).replace('▁', '_')}|>", text)


def strip_composer_control_tokens(text: str) -> str:
    """Remove any complete ``</think>`` / ``<|final|>`` control tokens."""
    return _CONTROL_TOKEN_RE.sub("", text)


def _find_control_token(value: str) -> tuple[int, int] | None:
    """First complete control token in *value* as ``(index, length)``."""
    m = _CONTROL_TOKEN_RE.search(value)
    return (m.start(), len(m.group(0))) if m else None


# Concrete control-token strings, for partial-prefix detection at chunk edges.
_CONTROL_TOKEN_CANDIDATES = ("</think>", "<|final|>", "<｜final｜>", "< | final | >")


def _control_token_prefix_len(value: str) -> int:
    """Retain a trailing partial control token or Composer marker prefix."""
    candidates = (*_CONTROL_TOKEN_CANDIDATES, *TOOL_MARKER_CANDIDATES)
    max_len = min(len(value), max(len(candidate) for candidate in candidates))
    keep = 0
    for length in range(1, max_len + 1):
        suffix = value[len(value) - length:]
        if any(candidate.startswith(suffix) and suffix != candidate for candidate in candidates):
            keep = length
    return keep


class _ControlTokenFilter:
    """Streaming filter that removes complete ``</think>`` / ``<|final|>``
    control tokens anywhere in the stream while holding back a trailing partial
    token so one split across chunks is never emitted in fragments."""

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = ""

    def push(self, delta: str) -> str:
        self._buffer += delta
        cleaned = strip_composer_control_tokens(self._buffer)
        keep = _control_token_prefix_len(cleaned)
        if keep == 0:
            self._buffer = ""
            return cleaned
        self._buffer = cleaned[len(cleaned) - keep:]
        return cleaned[: len(cleaned) - keep]

    def pending(self) -> str:
        """Return the unresolved trailing control-token candidate."""
        return self._buffer

    def flush(self) -> str:
        cleaned = strip_composer_control_tokens(self._buffer)
        self._buffer = ""
        return cleaned


def split_reasoning_and_answer(text: str) -> tuple[str, str]:
    """Split *text* into ``(reasoning, answer)`` on the last control token.

    Everything up to and including the final ``</think>`` / ``<|final|>`` is
    reasoning; the remainder is the answer.  With no control token, the whole
    string is treated as the answer (defensive: never blank out content).
    """
    matches = list(_CONTROL_TOKEN_RE.finditer(text))
    if not matches:
        return "", text
    last = matches[-1]
    reasoning = strip_composer_control_tokens(text[: last.start()])
    answer = strip_composer_control_tokens(text[last.end():]).lstrip()
    return reasoning, answer


# ── Marker search helpers ──────────────────────────────────────────────────


def _find_tool_marker(value: str, marker: str) -> tuple[int, int] | None:
    """Find ``marker`` (e.g. ``"tool_calls_begin"``) in either ASCII or unicode
    form; return ``(index, length)`` of the first match or ``None``."""
    body = marker.replace("_", "[_▁]")
    pattern = re.compile(rf"<\s*[|｜]\s*{body}\s*[|｜]\s*>")
    m = pattern.search(value)
    return (m.start(), len(m.group(0))) if m else None


def _tool_marker_prefix_index(value: str) -> int:
    """Index where a trailing substring of *value* could begin a tool marker.

    Used to hold back a possible marker split across streaming chunks so it is
    never emitted as visible text.  Returns ``-1`` when no suffix matches.
    """
    max_len = min(len(value), max(len(c) for c in TOOL_MARKER_CANDIDATES))
    for length in range(max_len, 0, -1):
        index = len(value) - length
        suffix = value[index:]
        if any(c.startswith(suffix) for c in TOOL_MARKER_CANDIDATES):
            return index
    return -1


def _is_unambiguous_tool_marker_prefix(value: str) -> bool:
    """Whether a pending prefix distinguishes a tool marker from literal '<'."""
    return len(value) >= 2 and any(
        marker.startswith(value) for marker in TOOL_MARKER_CANDIDATES
    )


# ── Tool-call body parsing ─────────────────────────────────────────────────


def _first_string(*values) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _record_from_tool_arguments(value):
    """Decode JSON strings but retain invalid argument values for validation."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _parse_composer_tool_argument(value: str):
    """Coerce a raw argument string to bool / number / json / string."""
    if not value:
        return ""
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "null":
        return None
    if _INT_OR_FLOAT_RE.match(value):
        return float(value) if "." in value else int(value)
    if (value.startswith("{") and value.endswith("}")) or (
        value.startswith("[") and value.endswith("]")
    ):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    # A JSON string literal (common in the inline ``k="v"`` form) → its value.
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _split_inline_arguments(value: str) -> list[str]:
    """Split ``a=1, b="x,y", c={...}`` on top-level commas (quote/brace aware)."""
    parts: list[str] = []
    start = 0
    quote: str | None = None
    depth = 0
    for i, char in enumerate(value):
        if quote:
            if char == quote and (i == 0 or value[i - 1] != "\\"):
                quote = None
            continue
        if char in ('"', "'"):
            quote = char
            continue
        if char in ("{", "["):
            depth += 1
        elif char in ("}", "]"):
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(value[start:i])
            start = i + 1
    parts.append(value[start:])
    return parts


def _parse_inline_tool_arguments(value: str) -> dict:
    args: dict = {}
    for part in _split_inline_arguments(value):
        m = _INLINE_ARG_RE.match(part.strip())
        if not m:
            continue
        args[m.group(1)] = _parse_composer_tool_argument(m.group(2).strip())
    return args


def _parse_inline_tool_call(value: str) -> dict | None:
    """Parse ``Name(k=v, ...)`` / ``Name[k=v]`` / bare ``Name``."""
    m = _INLINE_CALL_RE.match(value.strip())
    if not m:
        return None
    raw = m.group(2) if m.group(2) is not None else (m.group(3) or "")
    raw = raw.strip()
    return {"name": m.group(1).strip(), "arguments": _parse_inline_tool_arguments(raw) if raw else {}}


def _parse_json_tool_call_body(value: str) -> dict | None:
    """Parse a ``{"name":...,"arguments":...}`` JSON tool-call body."""
    if not (value.startswith("{") and value.endswith("}")):
        return None
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    fn = parsed.get("function") if isinstance(parsed.get("function"), dict) else None
    name = _first_string(
        parsed.get("name"),
        parsed.get("tool"),
        parsed.get("tool_name"),
        parsed.get("toolName"),
        fn.get("name") if fn else None,
    )
    if not name:
        return None
    argument_sources = (
        (parsed, "arguments"),
        (parsed, "args"),
        (parsed, "input"),
        (parsed, "parameters"),
        (parsed, "params"),
        (fn, "arguments"),
    )
    raw_args = next(
        (source[key] for source, key in argument_sources if source is not None and key in source),
        {},
    )
    return {"name": name, "arguments": _record_from_tool_arguments(raw_args)}


def _parse_tool_call_body(value: str) -> dict | None:
    """Parse the body between ``<|tool_call_begin|>`` and ``<|tool_call_end|>``.

    Tries JSON, then the token form (``name <sep> key\\nvalue <sep> ...``),
    falling back to an inline ``name(...)`` form when no separators are present.
    """
    json_body = _parse_json_tool_call_body(value.strip())
    if json_body:
        return json_body

    parts = value.split(TOOL_SEP)
    name = (parts.pop(0) if parts else "").strip()
    if not name:
        return None

    if not parts:
        inline = _parse_inline_tool_call(name)
        return inline or {"name": name, "arguments": {}}

    args: dict = {}
    for part in parts:
        trimmed = part.lstrip()
        if not trimmed:
            continue
        m = _KEY_VALUE_RE.match(trimmed)
        if not m:
            continue
        key = m.group(1).strip()
        if not key:
            continue
        args[key] = _parse_composer_tool_argument((m.group(2) or "").strip())
    return {"name": name, "arguments": args}


def parse_composer_tool_calls(value: str) -> list[dict]:
    """Parse every tool call inside a ``<|tool_calls_begin|>...<|tool_calls_end|>``
    block.  Returns a list of ``{"name", "arguments"}`` dicts (possibly empty)."""
    normalized = canonicalize_composer_markers(value)
    begin = normalized.find(TOOL_CALLS_BEGIN)
    end = normalized.rfind(TOOL_CALLS_END)
    if begin == -1 or end == -1 or end <= begin:
        return []

    body = normalized[begin + len(TOOL_CALLS_BEGIN):end]
    calls: list[dict] = []
    offset = 0
    while True:
        start = body.find(TOOL_CALL_BEGIN, offset)
        if start == -1:
            break
        content_start = start + len(TOOL_CALL_BEGIN)
        call_end = body.find(TOOL_CALL_END, content_start)
        if call_end == -1:
            break
        call = _parse_tool_call_body(body[content_start:call_end])
        if call:
            calls.append(call)
        offset = call_end + len(TOOL_CALL_END)
    return calls


# ── Streaming filter ───────────────────────────────────────────────────────

_MarkerEvent = tuple[str, object]  # ("text", str) | ("tool_call", dict)


class ComposerToolCallFilter:
    """Streaming separator: feed text deltas, get back interleaved prose text
    events and parsed tool-call events.  Buffers across deltas so a marker
    split across chunk boundaries is never leaked as visible text."""

    __slots__ = ("_buffer",)

    def __init__(self) -> None:
        self._buffer = ""

    def push(self, delta: str) -> list[_MarkerEvent]:
        self._buffer += delta
        return self._drain(force=False)

    def flush(self) -> list[_MarkerEvent]:
        return self._drain(force=True)

    def pending_tool_block(self) -> str:
        """Return an unfinished marker block or marker prefix without exposing it."""
        begin = _find_tool_marker(self._buffer, "tool_calls_begin")
        if begin is not None:
            block = self._buffer[begin[0]:]
            if _find_tool_marker(block[begin[1]:], "tool_calls_end") is None:
                return block
        prefix_index = _tool_marker_prefix_index(self._buffer)
        if prefix_index < 0:
            return ""
        prefix = self._buffer[prefix_index:]
        return prefix if _is_unambiguous_tool_marker_prefix(prefix) else ""

    def _drain(self, force: bool) -> list[_MarkerEvent]:
        events: list[_MarkerEvent] = []
        while True:
            begin = _find_tool_marker(self._buffer, "tool_calls_begin")
            if not begin:
                if not self._buffer.strip():
                    if force:
                        self._buffer = ""
                    break
                prefix_index = -1 if force else _tool_marker_prefix_index(self._buffer)
                if prefix_index != -1:
                    visible = self._buffer[:prefix_index]
                    if visible.strip():
                        events.append(("text", visible))
                    self._buffer = self._buffer[prefix_index:]
                    break
                visible = self._buffer
                if visible:
                    events.append(("text", visible))
                self._buffer = ""
                break

            begin_index, begin_len = begin
            if begin_index > 0:
                before = self._buffer[:begin_index]
                if before.strip():
                    events.append(("text", before))
                self._buffer = self._buffer[begin_index:]
                continue

            end = _find_tool_marker(self._buffer[begin_len:], "tool_calls_end")
            if not end:
                if force:
                    events.append(("text", self._buffer))
                    self._buffer = ""
                break

            block_end = begin_len + end[0] + end[1]
            for tool_call in parse_composer_tool_calls(self._buffer[:block_end]):
                events.append(("tool_call", tool_call))
            self._buffer = self._buffer[block_end:].lstrip()
        return events


class ComposerEmit(NamedTuple):
    thinking: str
    text: str
    tool_calls: list[dict]


class ComposerStreamProcessor:
    """Stateful processor for a Composer-2.x stream.

    Composer streams ``[reasoning] <control-token> [answer + tool tokens]``.
    ``feed_thinking`` drives the reasoning→answer phase split (the thinking
    field); ``feed_content`` routes content-field text straight to the answer
    pipeline.  Reasoning is surfaced as *thinking*; the answer is split into
    clean prose *text* and parsed *tool_calls*.
    """

    __slots__ = ("_phase", "_reason_buf", "_tool_filter", "_control_filter", "_answer_started")

    def __init__(self) -> None:
        self._phase = "reasoning"
        self._reason_buf = ""
        self._tool_filter = ComposerToolCallFilter()
        self._control_filter = _ControlTokenFilter()
        self._answer_started = False

    def feed_thinking(self, delta: str) -> ComposerEmit:
        if self._phase != "reasoning":
            return self._feed_answer(delta)

        self._reason_buf += delta
        marker = _find_control_token(self._reason_buf)
        if marker is None:
            return ComposerEmit("", "", [])  # still buffering reasoning

        reasoning = strip_composer_control_tokens(self._reason_buf[: marker[0]])
        remainder = self._reason_buf[marker[0] + marker[1]:].lstrip()
        self._reason_buf = ""
        self._phase = "answer"

        answer = self._feed_answer(remainder)
        return ComposerEmit(reasoning, answer.text, answer.tool_calls)

    def feed_content(self, delta: str) -> ComposerEmit:
        """Content-field text is always answer (never reasoning)."""
        return self._feed_answer(delta)

    def _feed_answer(self, text: str) -> ComposerEmit:
        # Strip any further control tokens (e.g. a stray <|final|> at the start
        # of the answer) BEFORE the tool filter, which would otherwise fragment
        # them across text events and leak the pieces.
        cleaned_in = self._control_filter.push(text)
        return self._drain_tool_events(self._tool_filter.push(cleaned_in))

    def _drain_tool_events(self, events) -> ComposerEmit:
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for kind, payload in events:
            if kind == "text":
                if payload:
                    text_parts.append(payload)  # type: ignore[arg-type]
            else:
                tool_calls.append(payload)  # type: ignore[arg-type]
        text = "".join(text_parts)
        if not self._answer_started:
            # Drop leading whitespace left behind after the control token(s).
            text = text.lstrip()
            if text:
                self._answer_started = True
        return ComposerEmit("", text, tool_calls)

    def pending_tool_block(self) -> str:
        """Return a buffered incomplete marker call for a retry prompt."""
        pending = self._tool_filter.pending_tool_block()
        if pending:
            return pending
        marker_filter = ComposerToolCallFilter()
        marker_filter.push(self._control_filter.pending())
        return marker_filter.pending_tool_block()

    def flush(self) -> ComposerEmit:
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        if self._phase == "reasoning" and self._reason_buf:
            # No control token ever appeared — treat the buffer as the answer
            # rather than discarding it (avoids a blank response).
            answer = self._feed_answer(self._reason_buf)
            if answer.text:
                text_parts.append(answer.text)
            tool_calls.extend(answer.tool_calls)
            self._reason_buf = ""
            self._phase = "answer"

        # Flush the control filter through the tool filter, then drain both.
        residual = self._control_filter.flush()
        if residual:
            self._tool_filter.push(residual)
        final = self._drain_tool_events(self._tool_filter.flush())
        if final.text:
            text_parts.append(final.text)
        tool_calls.extend(final.tool_calls)
        return ComposerEmit("", "".join(text_parts), tool_calls)
