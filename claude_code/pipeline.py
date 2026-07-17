from __future__ import annotations
"""
Main pipeline module — the heart of thalamus-py.

Pipeline flow (format-agnostic):
  1. Receive UnifiedRequest (from normalize_anthropic or normalize_openai)
  2. Inject tool prompts
  3. Call Cursor API via H2
  4. Parse tool calls from text response
  5. Assemble SSE output (Anthropic or OpenAI, chosen by original_format)
"""

import asyncio
from collections import deque
from contextvars import ContextVar
import inspect
import json
import re
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Awaitable, Callable

from utils.structured_logging import ThalamusStructuredLogger

from core.token_manager import get_cursor_access_token
from core.bearer_token import strip_cursor_user_prefix
from core.unified_request import UnifiedRequest
from utils.llm_payload_logger import (
    log_llm_request,
    log_llm_response,
    log_llm_api_call,
)
from core.protobuf_builder import (
    build_gzip_framed_protobuf_chat_request_body,
    compute_sha256_hex_digest,
    generate_obfuscated_machine_id_checksum,
)
from core.protobuf_frame_parser import CURSOR_ABORT_ERROR_CODE, ProtobufFrameParser
from core.cursor_h2_client import open_streaming_h2_request
from core.cursor_agent_client import call_cursor_agent, resumable_agent_tool_names
from core.model_context import get_model_context_length
from core.token_usage import estimate_input_tokens, input_tokens_from_remaining_context
from claude_code.tool_prompt_builder import inject_tool_prompt_into_messages
from config.system_prompt import THALAMUS_INSTRUCTION_SUPPLEMENT
from claude_code.composer_tool_parser import (
    ComposerStreamProcessor,
    ComposerToolCallFilter,
)
from claude_code.tool_protocols import (
    DecodedToolCandidate,
    ProtocolAdapter,
    ProtocolDecodeResult,
    ProtocolDecodeState,
    ProtocolFragment,
    ToolProtocol,
    classify_tool_protocol,
    create_protocol_adapter,
)
from claude_code.tool_choice import ToolChoiceError, resolve_tool_choice
from claude_code.sse_assembler import (
    StreamingAnthropicSession,
    build_unary_anthropic_response,
)
from claude_code.openai_sse_assembler import (
    StreamingOpenAISession,
    build_unary_openai_response,
)
from claude_code.openai_responses_assembler import (
    StreamingOpenAIResponsesSession,
    build_unary_openai_response as build_unary_openai_responses_response,
)
from claude_code.compatibility_trace import (
    CompatibilityTrace,
    emit_attempt_trace,
    emit_terminal_trace,
)
from claude_code.tool_validation import validate_tool_candidates
from config.fallback_config import load_fallback_config
from config.cursor_client import get_cursor_client_version

logger = ThalamusStructuredLogger.get_logger("pipeline", "DEBUG")

_TOOL_CALL_START_CALLBACK: ContextVar[
    Callable[[str, str], Awaitable[None] | None] | None
] = ContextVar("thalamus_tool_call_start_callback", default=None)


def _accepts_keyword(callback: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return True
    return keyword in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

FATAL_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(r"unable\s+to\s+reach\s+the\s+model\s+provider", re.I),
    re.compile(r"trouble\s+connecting", re.I),
    re.compile(r'code["\']?\s*:\s*["\']?unavailable', re.I),
    re.compile(r"ERROR_OPENAI", re.I),
    re.compile(r"service.*unavailable", re.I),
]

TOOL_JSON_START_MARKERS: list[str] = [
    '{"type":"tool_use"',
    '{"type": "tool_use"',
    '{"tool_calls"',
    '"tool_calls":',
    "```json",
    '{"function"',
]

_GENERIC_CAPABILITY_DENIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:this|the|current)?\s*(?:session|environment|workspace)?\s*"
        r"(?:has|have|provides?)\s+no\s+(?:available\s+)?(?:tools?|capabilit(?:y|ies))\b",
        re.I,
    ),
    re.compile(
        r"\btools?\b.{0,48}"
        r"\b(?:is|are)\s+(?:not\s+available|unavailable|disabled|missing)\b",
        re.I | re.S,
    ),
    re.compile(
        r"(?:沒有|没有|未提供|不具備|不具备|缺少)(?:任何|可用的?)?(?:工具|能力)"
    ),
)

_FILE_CAPABILITY_DENIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:no|without)\s+(?:available\s+)?(?:file\s+writing|file\s+editing|"
        r"filesystem|workspace)\s+(?:tools?|capabilit(?:y|ies)|access)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:cannot|can't|unable\s+to)\s+(?:write|create|edit|modify|access)\s+"
        r"(?:the\s+)?(?:file|filesystem|workspace)\b",
        re.I,
    ),
    re.compile(
        r"\bworkspace\s+path\b.{0,32}\b(?:unknown|not\s+(?:known|provided)|missing)\b",
        re.I | re.S,
    ),
    re.compile(
        r"(?:無法|无法|不能)(?:直接)?(?:寫入|写入|建立|创建|編輯|编辑|存取|访问).{0,16}"
        r"(?:檔案|文件|工作區|工作区)"
    ),
    re.compile(
        r"(?:工作區|工作区)(?:的)?(?:路徑|路径).{0,20}(?:未知|不明|未提供|不知道)"
    ),
    re.compile(
        r"(?:沒有|没有|未提供|缺少)(?:任何|可用的?)?.{0,16}"
        r"(?:檔案寫入|文件写入|檔案編輯|文件编辑|工作區|工作区).{0,8}(?:工具|能力|存取|访问)?"
    ),
)

_TERMINAL_CAPABILITY_DENIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:no|without)\s+(?:available\s+)?(?:terminal|shell|command)\s+"
        r"(?:tools?|capabilit(?:y|ies)|access)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:cannot|can't|unable\s+to)\s+(?:run|execute|use|access).{0,24}"
        r"\b(?:terminal|shell|command)\b",
        re.I | re.S,
    ),
    re.compile(r"(?:無法|无法|不能)(?:執行|执行|使用|存取|访问).{0,16}(?:終端|终端|命令)"),
    re.compile(
        r"(?:沒有|没有|未提供|缺少)(?:任何|可用的?)?.{0,12}"
        r"(?:終端|终端|命令列|命令行).{0,8}(?:工具|能力|存取|访问)?"
    ),
)

# Cursor can spend several minutes generating a large native tool argument
# without sending argument deltas. Emit protocol-valid, non-visible events so
# downstream harnesses do not mistake that quiet period for a dead stream.
SSE_KEEPALIVE_INTERVAL_SECONDS = 15.0
RESPONSES_TEXT_ITEM_IDLE_FLUSH_SECONDS = 0.75

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PartialStreamConsumptionError(Exception):
    """Raised when a Cursor stream fails after partial content was consumed."""

    def __init__(self, message: str, consumed: dict) -> None:
        super().__init__(message)
        self.consumed = consumed


def _looks_like_false_capability_denial(text: str, advertised_names: list[str]) -> bool:
    """Detect a tool denial only when the request inventory contradicts it."""
    if not text or not advertised_names:
        return False

    inventory = " ".join(advertised_names).lower()
    has_file_capability = any(
        token in inventory
        for token in ("write", "edit", "patch", "file", "filesystem", "workspace")
    )
    has_terminal_capability = any(
        token in inventory
        for token in ("terminal", "shell", "exec", "command", "bash")
    )

    for advertised_name in advertised_names:
        escaped_name = re.escape(advertised_name)
        exact_name_denials = (
            re.compile(
                rf"(?:沒有|没有)(?:(?:任何|可用的?)[\s`'\"「」]*|[\s`'\"「」]*)"
                rf"{escaped_name}(?:[\s`'\"「」]*(?:工具|能力))?",
                re.I,
            ),
            re.compile(rf"(?:未提供|缺少).{{0,24}}{escaped_name}", re.I | re.S),
            re.compile(
                rf"{escaped_name}.{{0,32}}(?:不可用|無法使用|无法使用|未啟用|未启用|不存在)",
                re.I | re.S,
            ),
            re.compile(
                rf"\b(?:no|without)\b.{{0,32}}\b{escaped_name}\b",
                re.I | re.S,
            ),
            re.compile(
                rf"\b{escaped_name}\b.{{0,32}}\b(?:unavailable|not\s+available|disabled|missing)\b",
                re.I | re.S,
            ),
        )
        if any(pattern.search(text) for pattern in exact_name_denials):
            return True

    if any(pattern.search(text) for pattern in _GENERIC_CAPABILITY_DENIAL_PATTERNS):
        return True
    if has_file_capability and any(
        pattern.search(text) for pattern in _FILE_CAPABILITY_DENIAL_PATTERNS
    ):
        return True
    return has_terminal_capability and any(
        pattern.search(text) for pattern in _TERMINAL_CAPABILITY_DENIAL_PATTERNS
    )


def _render_capability_denial_correction(advertised_names: list[str]) -> str:
    names = json.dumps(advertised_names, ensure_ascii=False)
    return (
        "Correction: the preceding answer incorrectly denied client capabilities. "
        f"This request advertises these callable client functions: {names}. "
        "That inventory is authoritative. Relative paths are actionable within the "
        "client's current workspace; do not ask for an absolute workspace path. "
        "Continue the original request now. If action is required, invoke one or more "
        "functions using an exact advertised name and its exact schema. Do not repeat "
        "the denial or emit another status-only preamble."
    )


def _to_api_error_body(message: str, error_type: str = "api_error") -> dict:
    return {"type": "error", "error": {"type": error_type, "message": message}}


def _extract_raw_auth_token(value: Any) -> str:
    if not value:
        return ""
    raw = value[0] if isinstance(value, list) else value
    return re.sub(r"^Bearer\s+", "", str(raw), flags=re.I).strip()


# ---------------------------------------------------------------------------
# max_tokens limiter
# ---------------------------------------------------------------------------


def _parse_max_tokens(value: Any) -> dict:
    if value is None:
        return {"ok": True, "value": None}
    try:
        n = int(value)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"Invalid max_tokens: {value}"}
    if n < 1:
        return {"ok": False, "error": f"Invalid max_tokens: {value}"}
    return {"ok": True, "value": n}


class _OutputLimiter:
    """Approximate char-budget limiter (1 token ~ 4 chars)."""

    def __init__(self, max_tokens: int | None) -> None:
        if max_tokens and max_tokens > 0:
            self.has_limit = True
            self.char_budget = max_tokens * 4
        else:
            self.has_limit = False
            self.char_budget = None
        self._emitted = 0
        self._exhausted = False

    def emit_within_limit(self, text: str) -> str:
        if not text or self._exhausted:
            return ""
        if not self.has_limit:
            return text
        remaining = self.char_budget - self._emitted
        if remaining <= 0:
            self._exhausted = True
            return ""
        out = text[:remaining]
        self._emitted += len(out)
        if len(out) < len(text) or self._emitted >= self.char_budget:
            self._exhausted = True
        return out

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    @property
    def emitted_chars(self) -> int:
        return self._emitted


# ---------------------------------------------------------------------------
# Tool-JSON-aware text forwarder
# ---------------------------------------------------------------------------


def _find_first_tool_json_start_index(full_text: str) -> int:
    if not full_text:
        return -1
    first = -1
    for marker in TOOL_JSON_START_MARKERS:
        idx = full_text.find(marker)
        if idx >= 0 and (first < 0 or idx < first):
            first = idx
    return first


class ToolJsonAwareTextForwarder:
    """Buffer streaming text deltas and stop forwarding once tool JSON begins."""

    def __init__(
        self,
        emit_text_delta: Callable[[str], str | None],
        limiter: _OutputLimiter,
    ) -> None:
        self._emit = emit_text_delta
        self._limiter = limiter
        self.full_text_seen = ""
        self._pending_buffer = ""
        self._safe_text_consumed_len = 0
        self.stopped_due_to_tool_json = False

    @staticmethod
    def _possible_marker_suffix_len(text: str) -> int:
        """Keep only a suffix that could still become a tool marker."""
        longest = 0
        for marker in TOOL_JSON_START_MARKERS:
            max_length = min(len(text), len(marker) - 1)
            for length in range(max_length, 0, -1):
                if text.endswith(marker[:length]):
                    longest = max(longest, length)
                    break
        return longest

    def _process_safe_chunk(self, chunk: str) -> str | None:
        if not chunk:
            return None
        self._safe_text_consumed_len += len(chunk)
        limited = self._limiter.emit_within_limit(chunk)
        if limited:
            return self._emit(limited)
        return None

    def on_delta(self, delta_text: str) -> str | None:
        """Feed a new text delta. Returns SSE string if text was forwarded."""
        delta = delta_text or ""
        if not delta:
            return None
        self.full_text_seen += delta
        if self.stopped_due_to_tool_json:
            return None

        self._pending_buffer += delta
        split_idx = _find_first_tool_json_start_index(self._pending_buffer)
        if split_idx >= 0:
            self.stopped_due_to_tool_json = True
            if split_idx > 0:
                safe = self._pending_buffer[:split_idx]
                self._pending_buffer = ""
                return self._process_safe_chunk(safe)
            self._pending_buffer = ""
            return None

        keep_len = self._possible_marker_suffix_len(self._pending_buffer)
        safe_flush_len = len(self._pending_buffer) - keep_len
        if safe_flush_len > 0:
            result = self._process_safe_chunk(self._pending_buffer[:safe_flush_len])
            self._pending_buffer = self._pending_buffer[safe_flush_len:]
            return result
        return None

    def flush_using_final_safe_text(self, final_safe_text: str) -> str | None:
        """Flush remaining buffered text after the stream has ended."""
        final = final_safe_text or ""

        result_parts: list[str] = []
        if not self.stopped_due_to_tool_json and self._pending_buffer:
            r = self._process_safe_chunk(self._pending_buffer)
            if r:
                result_parts.append(r)
            self._pending_buffer = ""

        if len(final) > self._safe_text_consumed_len:
            r = self._process_safe_chunk(final[self._safe_text_consumed_len:])
            if r:
                result_parts.append(r)

        return "".join(result_parts) if result_parts else None


# ---------------------------------------------------------------------------
# Text-before-JSON extraction
# ---------------------------------------------------------------------------


def extract_text_before_json(full_text: str) -> str:
    """Extract text appearing before tool-call JSON in LLM output."""
    if not full_text:
        return ""
    patterns = [
        re.compile(r'\{"type"\s*:\s*"tool_use"'),
        re.compile(r'\{[\s\S]*?"tool_calls"\s*:\s*\['),
        re.compile(r'```(?:json)?\s*\{[\s\S]*?"tool_calls"'),
        re.compile(r"<tool_call>\s*\{"),
        re.compile(r"<<function=[^>]+>>\s*\{"),
        re.compile(r'\{[\s\S]*?"function_call"\s*:\s*\{'),
    ]
    for pattern in patterns:
        m = pattern.search(full_text)
        if m and m.start() > 0:
            before = full_text[: m.start()].strip()
            if before:
                return before
    return ""


class _ProtocolStreamDecoder:
    """Decode one upstream attempt before it reaches validators or SSE callbacks."""

    def __init__(self, adapter: ProtocolAdapter) -> None:
        self._adapter = adapter
        self._state = ProtocolDecodeState()
        self._candidates: list[DecodedToolCandidate] = []
        self._visible_parts: list[str] = []
        self._thinking_parts: list[str] = []
        self._incomplete = False
        self._invalid_reason: str | None = None
        self._has_fragments = False

    def feed(self, lane: str, text: str) -> ProtocolDecodeResult:
        self._has_fragments = True
        result = self._adapter.incremental_decode(
            ProtocolFragment(sequence=self._state.next_sequence, lane=lane, text=text),
            self._state,
        )
        self._record(result)
        return result

    def finish(self) -> ProtocolDecodeResult:
        result = self._adapter.incremental_decode(None, self._state)
        self._record(result)
        return result

    def attach(self, consumed: dict, *, flush: bool = True) -> dict:
        if not self._has_fragments:
            self._decode_consumed_fields(consumed)
        if flush:
            self.finish()
        return {
            **consumed,
            "_protocol_candidates": tuple(self._candidates),
            "_protocol_incomplete": self._incomplete,
            "_protocol_invalid_reason": self._invalid_reason,
            "_protocol_visible_text": "".join(self._visible_parts),
            "_protocol_thinking_text": "".join(self._thinking_parts),
        }

    def _decode_consumed_fields(self, consumed: dict) -> None:
        for lane, field in (("text", "text"), ("reasoning", "thinking")):
            value = consumed.get(field)
            if isinstance(value, str) and value:
                self.feed(lane, value)

    def _record(self, result: ProtocolDecodeResult) -> None:
        self._state = result.state
        self._candidates.extend(result.candidates)
        self._visible_parts.append(result.visible_text)
        self._thinking_parts.append(result.thinking_text)
        self._incomplete = result.is_incomplete
        if result.invalid_reason is not None:
            self._invalid_reason = result.invalid_reason


def _parse_tool_calls_from_consumed(consumed: dict) -> tuple[list[DecodedToolCandidate], str]:
    """Prefer structured Cursor calls, then the active text protocol."""
    native_calls = consumed.get("native_tool_calls") or ()
    if native_calls:
        return [
            DecodedToolCandidate(
                call_id=call.call_id,
                raw_name=call.name,
                arguments=call.arguments,
                source_lane="native",
            )
            for call in native_calls
        ], "native"
    candidates = list(consumed.get("_protocol_candidates") or ())
    source = candidates[0].source_lane if candidates else ""
    return candidates, source


def _safe_text_for_tool_source(consumed: dict, _source: str) -> str:
    """Return adapter-clean visible text for a decoded tool call."""
    return consumed.get("_protocol_visible_text", consumed.get("text", "")) or ""


def _safe_final_text_for_stream(full_text: str, has_tool_calls: bool) -> str:
    if not has_tool_calls:
        return full_text
    return extract_text_before_json(full_text)


def _is_incomplete_composer_marker(text: str) -> bool:
    """Recognize only marker state retained by the Composer streaming parser."""
    if not text:
        return False
    marker_filter = ComposerToolCallFilter()
    marker_filter.push(text)
    return bool(marker_filter.pending_tool_block())


def _is_interrupted_tool_json(consumed: dict, is_composer: bool = False) -> bool:
    """Legacy test helper for callers that do not provide adapter outcomes."""
    text = (consumed.get("text") or "") + "\n" + (consumed.get("thinking") or "")
    if is_composer:
        interrupted_state = consumed.get("interrupted_tool_state")
        return isinstance(interrupted_state, str) and _is_incomplete_composer_marker(interrupted_state)
    return _find_first_tool_json_start_index(text) >= 0


@dataclass(frozen=True)
class StreamCallbacks:
    on_text_delta: Callable[[str], Awaitable[None] | None] | None = None
    on_thinking_delta: Callable[[str], Awaitable[None] | None] | None = None
    on_tool_call_start: (
        Callable[[str, str], Awaitable[None] | None] | None
    ) = None


@dataclass
class _SemanticLatencyTracker:
    """Measure first downstream-meaningful events independently of transport.

    ``first_chunk_latency_ms`` is an established metrics key used by callers and
    fixtures, but a protobuf/Agent event can contain only control metadata.  Do
    not treat it as time-to-first-token.  These measurements are taken where the
    pipeline can actually identify text, reasoning, or a client tool name.
    """

    started_at: float
    first_text_latency_ms: float | None = None
    first_reasoning_latency_ms: float | None = None
    first_tool_identity_latency_ms: float | None = None

    @classmethod
    def start(cls) -> "_SemanticLatencyTracker":
        return cls(started_at=time.monotonic())

    def _elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def mark_text(self, value: str) -> None:
        if value and self.first_text_latency_ms is None:
            self.first_text_latency_ms = self._elapsed_ms()

    def mark_reasoning(self, value: str) -> None:
        if value and self.first_reasoning_latency_ms is None:
            self.first_reasoning_latency_ms = self._elapsed_ms()

    def mark_tool_identity(self, call_id: str, name: str) -> None:
        if call_id and name and self.first_tool_identity_latency_ms is None:
            self.first_tool_identity_latency_ms = self._elapsed_ms()

    def metrics(self) -> dict[str, float]:
        values = {
            "first_text_latency_ms": self.first_text_latency_ms,
            "first_reasoning_latency_ms": self.first_reasoning_latency_ms,
            "first_tool_identity_latency_ms": self.first_tool_identity_latency_ms,
        }
        semantic_values = [value for value in values.values() if value is not None]
        return {
            **{
                key: value if value is not None else -1
                for key, value in values.items()
            },
            "first_semantic_latency_ms": (
                min(semantic_values) if semantic_values else -1
            ),
        }

    def attach(self, consumed: dict) -> dict:
        consumed.setdefault("metrics", {}).update(self.metrics())
        return consumed


def _format_stream_metrics(metrics: dict[str, Any]) -> str:
    """Format latency names without presenting a control chunk as a token."""

    def value(name: str) -> float:
        candidate = metrics.get(name, -1)
        return float(candidate) if isinstance(candidate, (int, float)) else -1

    return (
        f"chunks={metrics.get('chunk_count', 0)} "
        f"first_chunk_ms={value('first_chunk_latency_ms'):.0f} "
        f"first_semantic_ms={value('first_semantic_latency_ms'):.0f} "
        f"first_text_ms={value('first_text_latency_ms'):.0f} "
        f"first_reasoning_ms={value('first_reasoning_latency_ms'):.0f} "
        f"first_tool_identity_ms={value('first_tool_identity_latency_ms'):.0f}"
    )


@dataclass(frozen=True)
class AttemptConsumption:
    consumed: dict
    repair_attempted: bool


async def _consume_attempt_with_repair(
    *,
    messages: list[dict],
    open_and_consume: Callable[[list[dict]], Awaitable[dict]],
    render_repair: Callable[[dict], str],
    allow_repair: bool,
    is_composer: bool,
    is_protocol_incomplete: Callable[[dict], bool] | None = None,
    on_repair_attempt: Callable[[], None] | None = None,
    on_upstream_attempt: Callable[[dict | None, bool, str, int], None] | None = None,
) -> AttemptConsumption:
    async def consume(messages_for_call: list[dict], is_repair: bool) -> dict:
        call_start = time.monotonic()
        try:
            consumed = await open_and_consume(messages_for_call)
        except PartialStreamConsumptionError as exc:
            if on_upstream_attempt:
                on_upstream_attempt(
                    exc.consumed,
                    is_repair,
                    "protocol_incomplete",
                    int((time.monotonic() - call_start) * 1000),
                )
            raise
        except Exception:
            if on_upstream_attempt:
                on_upstream_attempt(
                    None,
                    is_repair,
                    "upstream_error",
                    int((time.monotonic() - call_start) * 1000),
                )
            raise
        if on_upstream_attempt:
            on_upstream_attempt(
                consumed,
                is_repair,
                "attempt_complete",
                int((time.monotonic() - call_start) * 1000),
            )
        return consumed

    async def repair(partial: dict) -> AttemptConsumption:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": partial.get("text", "")},
            {"role": "user", "content": render_repair(partial)},
        ]
        if on_repair_attempt:
            on_repair_attempt()
        return AttemptConsumption(
            consumed=await consume(repair_messages, True),
            repair_attempted=True,
        )

    incomplete = is_protocol_incomplete or (
        lambda consumed: _is_interrupted_tool_json(consumed, is_composer)
    )
    try:
        consumed = await consume(messages, False)
    except PartialStreamConsumptionError as exc:
        partial = exc.consumed
        if not allow_repair or not incomplete(partial):
            raise
        return await repair(partial)

    if allow_repair and incomplete(consumed):
        return await repair(consumed)
    return AttemptConsumption(consumed=consumed, repair_attempted=False)


# ---------------------------------------------------------------------------
# Fatal error detection
# ---------------------------------------------------------------------------


def _is_fatal_stream_error(error: Any) -> bool:
    text = (
        error if isinstance(error, str)
        else str(
            getattr(error, "detail", None)
            or getattr(error, "raw", None)
            or getattr(error, "message", None)
            or json.dumps(error if isinstance(error, dict) else {})
        )
    )
    return any(p.search(text) for p in FATAL_ERROR_PATTERNS)


# ---------------------------------------------------------------------------
# Cursor stream plumbing
# ---------------------------------------------------------------------------


def build_cursor_stream_params(
    token: str,
    messages: list[dict],
    model: str,
    tools: list[dict] | None = None,
) -> tuple[str, dict[str, str], bytes]:
    """Build H2 path, headers, and protobuf body for a Cursor streaming request."""
    chosen_auth = strip_cursor_user_prefix(token)
    checksum = generate_obfuscated_machine_id_checksum(chosen_auth.strip())
    session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chosen_auth))
    client_key = compute_sha256_hex_digest(chosen_auth)
    client_version = get_cursor_client_version()

    body = build_gzip_framed_protobuf_chat_request_body(
        messages, model, agent_mode=True, tools=tools
    )
    is_gzipped = body[0] == 0x01

    headers = {
        "authorization": f"Bearer {chosen_auth}",
        "connect-accept-encoding": "gzip",
        "connect-protocol-version": "1",
        "content-type": "application/connect+proto",
        "user-agent": "connect-es/1.6.1",
        "x-amzn-trace-id": f"Root={uuid.uuid4()}",
        "x-client-key": client_key,
        "x-cursor-checksum": checksum,
        "x-cursor-client-version": client_version,
        "x-cursor-config-version": str(uuid.uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-ghost-mode": "true",
        "x-request-id": str(uuid.uuid4()),
        "x-session-id": session_id,
        "Host": "api2.cursor.sh",
    }
    if is_gzipped:
        headers["connect-content-encoding"] = "gzip"

    path = "/aiserver.v1.ChatService/StreamUnifiedChatWithTools"
    return path, headers, body


class _ThinkTagSplitter:
    """Streaming splitter that separates <think>...</think> from regular text.

    Some models (e.g. composer-1.5) embed thinking inside the text field
    using <think> tags instead of the protobuf thinking field.  Claude Code
    expects thinking to arrive as proper ``thinking_delta`` SSE events, not
    as raw text — otherwise the response appears blank.
    """

    __slots__ = ("_inside_think", "_buf")

    def __init__(self) -> None:
        self._inside_think = False
        self._buf = ""

    def feed(self, chunk: str) -> tuple[str, str]:
        """Return (thinking_part, text_part) extracted from *chunk*."""
        self._buf += chunk
        thinking_out: list[str] = []
        text_out: list[str] = []

        while self._buf:
            if self._inside_think:
                end = self._buf.find("</think>")
                if end == -1:
                    thinking_out.append(self._buf)
                    self._buf = ""
                else:
                    thinking_out.append(self._buf[:end])
                    self._buf = self._buf[end + len("</think>"):]
                    self._inside_think = False
            else:
                start = self._buf.find("<think>")
                if start == -1:
                    if len(self._buf) >= 7:
                        safe = len(self._buf) - 6
                        text_out.append(self._buf[:safe])
                        self._buf = self._buf[safe:]
                    break
                else:
                    text_out.append(self._buf[:start])
                    self._buf = self._buf[start + len("<think>"):]
                    self._inside_think = True

        return "".join(thinking_out), "".join(text_out)

    def flush(self) -> tuple[str, str]:
        """Flush any remaining buffered content."""
        remaining = self._buf
        self._buf = ""
        if self._inside_think:
            self._inside_think = False
            return remaining, ""
        return "", remaining


async def consume_stream(
    stream_iterator: AsyncIterator[bytes],
    on_text_delta: Callable[[str], Any] | None = None,
    on_thinking_delta: Callable[[str], Any] | None = None,
    composer: bool = False,
) -> dict:
    """Consume a Cursor protobuf stream, accumulating text/thinking/errors.

    Tool calls are extracted from text after the stream completes (prompt
    injection approach), not from protobuf wire-level tool call fields.

    Models that embed <think>...</think> in the text field (instead of the
    protobuf thinking field) are handled transparently: the tags are stripped
    and the content is routed to on_thinking_delta.

    When ``composer`` is True the model is a Composer-2.x model, which streams
    its whole output (reasoning + answer + DeepSeek-style tool tokens) through
    the thinking field.  That stream is routed through a ComposerStreamProcessor
    that separates reasoning (→thinking), clean answer prose (→text), and tool
    calls (returned as ``composer_tool_calls``).
    """
    parser = ProtobufFrameParser()
    splitter = _ThinkTagSplitter()
    composer_proc = ComposerStreamProcessor() if composer else None
    composer_tool_calls: list[dict] = []
    native_tool_calls: dict[str, Any] = {}
    interrupted_tool_state = ""
    text = ""
    thinking = ""
    errors: list[Any] = []
    context_remaining_percent: float | None = None
    had_content = False
    has_fatal_error = False
    chunk_count = 0
    text_delta_count = 0
    thinking_delta_count = 0
    latency_tracker = _SemanticLatencyTracker.start()
    stream_start = latency_tracker.started_at
    first_chunk_latency_ms: float | None = None

    def _build_consumed_result() -> dict:
        filtered_errors = [
            e for e in errors
            if not (hasattr(e, "error_code") and e.error_code == CURSOR_ABORT_ERROR_CODE)
        ]
        return {
            "text": text,
            "thinking": thinking,
            "composer_tool_calls": composer_tool_calls,
            "native_tool_calls": list(native_tool_calls.values()),
            "interrupted_tool_state": interrupted_tool_state,
            "errors": filtered_errors,
            "context_remaining_percent": context_remaining_percent,
            "had_content": had_content,
            "has_fatal_error": has_fatal_error,
            "metrics": {
                "stream_duration_ms": (time.monotonic() - stream_start) * 1000,
                "first_chunk_latency_ms": first_chunk_latency_ms if first_chunk_latency_ms is not None else -1,
                "chunk_count": chunk_count,
                "text_delta_count": text_delta_count,
                "thinking_delta_count": thinking_delta_count,
                "protocol_error_count": len(filtered_errors),
                **latency_tracker.metrics(),
            },
        }

    def _apply_composer_emit(emit) -> None:
        nonlocal text, thinking, had_content, text_delta_count, thinking_delta_count
        if emit.thinking:
            latency_tracker.mark_reasoning(emit.thinking)
            thinking += emit.thinking
            thinking_delta_count += 1
            had_content = True
            if on_thinking_delta:
                on_thinking_delta(emit.thinking)
        if emit.text:
            latency_tracker.mark_text(emit.text)
            text += emit.text
            text_delta_count += 1
            had_content = True
            if on_text_delta:
                on_text_delta(emit.text)
        if emit.tool_calls:
            first_call = emit.tool_calls[0]
            latency_tracker.mark_tool_identity(
                str(first_call.get("id") or first_call.get("name") or "composer"),
                str(first_call.get("name") or ""),
            )
            composer_tool_calls.extend(emit.tool_calls)
            had_content = True

    try:
        async for chunk in stream_iterator:
            chunk_count += 1
            if first_chunk_latency_ms is None:
                first_chunk_latency_ms = (time.monotonic() - stream_start) * 1000
            logger.debug(f"[consume] chunk#{chunk_count} len={len(chunk)} hex_head={chunk[:20].hex()}")

            result = parser.parse(chunk)
            complete_native_call = False
            for call in result.native_tool_calls:
                latency_tracker.mark_tool_identity(call.call_id, call.name)
                previous = native_tool_calls.get(call.call_id)
                if previous is None or len(call.raw_arguments) >= len(previous.raw_arguments):
                    native_tool_calls[call.call_id] = call
                if call.is_last:
                    complete_native_call = True
                    had_content = True
            if (
                context_remaining_percent is None
                and result.context_remaining_percent is not None
            ):
                context_remaining_percent = result.context_remaining_percent
            logger.debug(
                f"[consume] chunk#{chunk_count} parsed: text_len={len(result.text)} "
                f"thinking_len={len(result.thinking)} errors={len(result.errors)}"
            )

            if result.errors:
                errors.extend(result.errors)
                for err in result.errors:
                    if _is_fatal_stream_error(err):
                        has_fatal_error = True

            if composer_proc is not None:
                if result.thinking:
                    _apply_composer_emit(composer_proc.feed_thinking(result.thinking))
                if result.text:
                    _apply_composer_emit(composer_proc.feed_content(result.text))
                if complete_native_call:
                    break
                continue

            if result.thinking:
                latency_tracker.mark_reasoning(result.thinking)
                thinking += result.thinking
                thinking_delta_count += 1
                had_content = True
                if on_thinking_delta:
                    on_thinking_delta(result.thinking)

            if result.text:
                think_part, text_part = splitter.feed(result.text)

                if think_part:
                    latency_tracker.mark_reasoning(think_part)
                    thinking += think_part
                    thinking_delta_count += 1
                    had_content = True
                    if on_thinking_delta:
                        on_thinking_delta(think_part)

                if text_part:
                    latency_tracker.mark_text(text_part)
                    text += text_part
                    text_delta_count += 1
                    had_content = True
                    if on_text_delta:
                        on_text_delta(text_part)
                    logger.debug(f"[consume] text so far ({len(text)} chars): ...{text[-200:]}")
            if complete_native_call:
                break
    except Exception as exc:
        if composer_proc is not None:
            interrupted_tool_state = composer_proc.pending_tool_block()
            if not interrupted_tool_state:
                _apply_composer_emit(composer_proc.flush())
        flush_think, flush_text = splitter.flush()
        if flush_think:
            latency_tracker.mark_reasoning(flush_think)
            thinking += flush_think
            thinking_delta_count += 1
            had_content = True
            if on_thinking_delta:
                on_thinking_delta(flush_think)
        if flush_text:
            latency_tracker.mark_text(flush_text)
            text += flush_text
            text_delta_count += 1
            had_content = True
            if on_text_delta:
                on_text_delta(flush_text)
        if had_content or interrupted_tool_state:
            raise PartialStreamConsumptionError(str(exc), _build_consumed_result()) from exc
        raise

    if composer_proc is not None:
        interrupted_tool_state = composer_proc.pending_tool_block()
        if not interrupted_tool_state:
            _apply_composer_emit(composer_proc.flush())

    flush_think, flush_text = splitter.flush()
    if flush_think:
        latency_tracker.mark_reasoning(flush_think)
        thinking += flush_think
        thinking_delta_count += 1
        had_content = True
        if on_thinking_delta:
            on_thinking_delta(flush_think)
    if flush_text:
        latency_tracker.mark_text(flush_text)
        text += flush_text
        text_delta_count += 1
        had_content = True
        if on_text_delta:
            on_text_delta(flush_text)

    return _build_consumed_result()


# ---------------------------------------------------------------------------
# Internal Cursor caller with fallback
# ---------------------------------------------------------------------------




async def _call_cursor_direct(
    messages: list[dict],
    model: str,
    tools: list[dict],
    valid_tool_names: list[str],
    auth_token: str,
    on_stream_delta: Callable[[str], Any] | None = None,
    on_thinking_delta: Callable[[str], Any] | None = None,
    compact_tools: bool = False,
    requested_model: str | None = None,
    client_format: str | None = None,
    tool_choice: dict[str, Any] | str | None = None,
) -> dict:
    """Call Cursor API with strict tool decoding, one repair, and fallback."""
    start_time = time.monotonic()
    request_id = f"cc_{uuid.uuid4().hex[:12]}"

    base_messages = list(messages)
    choice_policy = resolve_tool_choice(tool_choice, valid_tool_names)
    effective_tool_names = choice_policy.permitted_names(valid_tool_names)
    resumable_tool_names = resumable_agent_tool_names(base_messages, auth_token)
    if not effective_tool_names and resumable_tool_names:
        effective_tool_names = list(resumable_tool_names)
    effective_tools = choice_policy.filter_tools(tools)
    has_valid_tools = bool(effective_tool_names)

    fallback_cfg = load_fallback_config()
    tried_models: list[str] = []
    current_model = model
    repair_attempted = False
    capability_retry_attempted = False
    fallback_reason: str | None = None
    last_attempt_trace: CompatibilityTrace | None = None
    last_context_remaining_percent: float | None = None

    while len(tried_models) < fallback_cfg.max_attempts:
        attempt_number = len(tried_models) + 1
        protocol = classify_tool_protocol(current_model)
        adapter = create_protocol_adapter(protocol)
        use_agent_api = bool(effective_tools) or bool(resumable_tool_names)
        attempt_messages = (
            list(base_messages)
            if use_agent_api
            else inject_tool_prompt_into_messages(
                list(base_messages),
                effective_tools,
                compact_tools=compact_tools,
                adapter=adapter,
                tool_choice=tool_choice,
                advertised_tool_names=effective_tool_names,
            )
        )
        logger.info(
            f"[{request_id}] Calling Cursor direct | model={current_model} | tools={len(valid_tool_names)} | msgs={len(attempt_messages)} | attempt={len(tried_models) + 1}"
        )

        attempt_start = time.monotonic()
        is_composer = protocol == ToolProtocol.COMPOSER_MARKER_V1
        replay_safe = True

        req_payload_path = log_llm_request(
            request_id, current_model, attempt_messages,
            extra={"tools": len(valid_tool_names), "attempt": len(tried_models) + 1},
        )

        callbacks = StreamCallbacks(
            on_text_delta=on_stream_delta,
            on_thinking_delta=on_thinking_delta,
            on_tool_call_start=_TOOL_CALL_START_CALLBACK.get(),
        )

        async def open_and_consume(attempt_messages: list[dict]) -> dict:
            if use_agent_api:
                latency_tracker = _SemanticLatencyTracker.start()

                def on_agent_text(delta: str) -> Any:
                    nonlocal replay_safe
                    latency_tracker.mark_text(delta)
                    if delta:
                        replay_safe = False
                    if callbacks.on_text_delta is not None:
                        return callbacks.on_text_delta(delta)
                    return None

                def on_agent_reasoning(delta: str) -> Any:
                    nonlocal replay_safe
                    latency_tracker.mark_reasoning(delta)
                    if delta:
                        replay_safe = False
                    if callbacks.on_thinking_delta is not None:
                        return callbacks.on_thinking_delta(delta)
                    return None

                def on_agent_tool_start(call_id: str, name: str) -> Any:
                    nonlocal replay_safe
                    latency_tracker.mark_tool_identity(call_id, name)
                    replay_safe = False
                    if callbacks.on_tool_call_start is not None:
                        return callbacks.on_tool_call_start(call_id, name)
                    return None

                callback_kwargs: dict[str, Any] = {}
                if _accepts_keyword(call_cursor_agent, "on_tool_call_start"):
                    callback_kwargs["on_tool_call_start"] = on_agent_tool_start
                if _accepts_keyword(call_cursor_agent, "client_request_id"):
                    callback_kwargs["client_request_id"] = request_id
                consumed = await call_cursor_agent(
                    attempt_messages,
                    current_model,
                    effective_tools,
                    auth_token,
                    on_text_delta=on_agent_text,
                    on_thinking_delta=on_agent_reasoning,
                    **callback_kwargs,
                )
                return latency_tracker.attach(consumed)

            stream_decoder = _ProtocolStreamDecoder(create_protocol_adapter(protocol))

            def forward_protocol_result(result: ProtocolDecodeResult) -> None:
                nonlocal replay_safe
                if result.visible_text or result.thinking_text:
                    replay_safe = False
                if result.visible_text and callbacks.on_text_delta:
                    callbacks.on_text_delta(result.visible_text)
                if result.thinking_text and callbacks.on_thinking_delta:
                    callbacks.on_thinking_delta(result.thinking_text)

            def decode_text_delta(delta: str) -> None:
                forward_protocol_result(stream_decoder.feed("text", delta))

            def decode_thinking_delta(delta: str) -> None:
                forward_protocol_result(stream_decoder.feed("reasoning", delta))

            path, headers, body = build_cursor_stream_params(
                auth_token,
                attempt_messages,
                current_model,
                effective_tools,
            )
            try:
                async with open_streaming_h2_request(path, headers, body) as stream_iter:
                    consumed = await consume_stream(
                        stream_iter,
                        on_text_delta=decode_text_delta,
                        on_thinking_delta=decode_thinking_delta,
                        composer=False,
                    )
            except PartialStreamConsumptionError as exc:
                partial_result = stream_decoder.finish()
                forward_protocol_result(partial_result)
                raise PartialStreamConsumptionError(
                    str(exc), stream_decoder.attach(exc.consumed, flush=False)
                ) from exc

            final_result = stream_decoder.finish()
            forward_protocol_result(final_result)
            return stream_decoder.attach(consumed, flush=False)

        def mark_repair_attempted() -> None:
            nonlocal repair_attempted
            repair_attempted = True

        def render_repair(partial: dict) -> str:
            parsed_candidates, _ = _parse_tool_calls_from_consumed(partial)
            validation = validate_tool_candidates(
                parsed_candidates,
                allowed_names=set(effective_tool_names),
            )
            if validation.rejected and not validation.accepted:
                rejected_names = sorted(
                    {rejection.raw_name for rejection in validation.rejected if rejection.raw_name}
                )
                return (
                    "The previous native tool call was rejected because its function "
                    f"name was not in this request's client inventory: {rejected_names}. "
                    "Do not invoke another native Cursor or MCP server tool. Instead, emit "
                    "exactly one raw JSON object in this form and no other prose: "
                    '{"type":"tool_use","id":"toolu_repair","name":"<exact advertised '
                    'client function name>","input":{<arguments matching its exact schema>}}. '
                    "Choose the name only from the client functions already advertised in "
                    "the system tool manifest. Do not repeat the status preamble."
                )
            interrupted_state = partial.get("interrupted_tool_state") or "\n".join(
                value for value in (partial.get("text"), partial.get("thinking")) if value
            )
            repair_adapter = create_protocol_adapter(protocol)
            return repair_adapter.render_repair(effective_tools, interrupted_state)

        def needs_protocol_repair(value: dict) -> bool:
            if value.get("_protocol_incomplete"):
                return True
            parsed_candidates, _ = _parse_tool_calls_from_consumed(value)
            if not parsed_candidates:
                return False
            validation = validate_tool_candidates(
                parsed_candidates,
                allowed_names=set(effective_tool_names),
            )
            return bool(validation.rejected and not validation.accepted)

        upstream_attempt_number = 0

        def trace_upstream_attempt(
            upstream_consumed: dict | None,
            is_repair: bool,
            result: str,
            latency_ms: int,
        ) -> None:
            nonlocal upstream_attempt_number, last_attempt_trace
            upstream_attempt_number += 1
            parsed_candidates, candidate_source = _parse_tool_calls_from_consumed(
                upstream_consumed or {}
            )
            validation = validate_tool_candidates(
                parsed_candidates,
                allowed_names=set(effective_tool_names),
            )
            accepted_names = tuple(
                (tool_call.get("function") or {}).get("name", "")
                for tool_call in validation.accepted
                if (tool_call.get("function") or {}).get("name")
            )
            rejection_reason = (
                ",".join(sorted({rejection.reason for rejection in validation.rejected}))
                or None
            )
            trace = _build_compatibility_trace(
                request_id=request_id,
                attempt_number=f"{attempt_number}.{upstream_attempt_number}",
                requested_model=requested_model or model,
                effective_model=current_model,
                protocol=protocol,
                compact_tools=compact_tools,
                client_format=client_format,
                fallback_reason=fallback_reason,
                consumed=upstream_consumed,
                tool_candidate_source=candidate_source or None,
                candidate_count=len(parsed_candidates or []),
                accepted_tool_names=accepted_names,
                rejection_reason=rejection_reason,
                repair_attempted=is_repair,
                terminal_result=result,
                latency_ms=latency_ms,
            )
            emit_attempt_trace(trace)
            last_attempt_trace = trace

        try:
            attempt = await _consume_attempt_with_repair(
                messages=attempt_messages,
                open_and_consume=open_and_consume,
                render_repair=render_repair,
                allow_repair=(
                    has_valid_tools and not repair_attempted and not use_agent_api
                ),
                is_composer=is_composer,
                is_protocol_incomplete=needs_protocol_repair,
                on_repair_attempt=mark_repair_attempted,
                on_upstream_attempt=trace_upstream_attempt,
            )
            consumed = attempt.consumed
            repair_attempted = repair_attempted or attempt.repair_attempted
            if attempt.repair_attempted:
                logger.info(
                    f"[{request_id}] Interrupted tool JSON repair consumed | "
                    f"text_len={len(consumed['text'])}"
                )

            parsed_candidates, _ = _parse_tool_calls_from_consumed(consumed)
            visible_text = consumed.get("_protocol_visible_text", consumed.get("text", ""))
            if (
                has_valid_tools
                and not use_agent_api
                and not capability_retry_attempted
                and not parsed_candidates
                and _looks_like_false_capability_denial(visible_text, effective_tool_names)
            ):
                capability_retry_attempted = True
                logger.warn(
                    f"[{request_id}] False client-capability denial detected; "
                    "retrying once with the authoritative request tool inventory"
                )
                correction_messages = [
                    *attempt_messages,
                    {"role": "assistant", "content": visible_text},
                    {
                        "role": "user",
                        "content": _render_capability_denial_correction(effective_tool_names),
                    },
                ]
                retry = await _consume_attempt_with_repair(
                    messages=correction_messages,
                    open_and_consume=open_and_consume,
                    render_repair=render_repair,
                    allow_repair=has_valid_tools and not repair_attempted,
                    is_composer=is_composer,
                    is_protocol_incomplete=needs_protocol_repair,
                    on_repair_attempt=mark_repair_attempted,
                    on_upstream_attempt=lambda value, is_repair, result, latency: (
                        trace_upstream_attempt(value, True, result, latency)
                    ),
                )
                consumed = retry.consumed
                repair_attempted = repair_attempted or retry.repair_attempted

            last_context_remaining_percent = consumed.get("context_remaining_percent")
            if "replay_safe" in consumed:
                replay_safe = replay_safe and bool(consumed["replay_safe"])
            else:
                replay_safe = replay_safe and not bool(consumed.get("had_content"))
        except Exception as exc:
            err_msg = str(exc)
            logger.error(f"[{request_id}] Connection/stream error: {err_msg}")
            latency = int((time.monotonic() - attempt_start) * 1000)
            fallback_requested = fallback_cfg.should_fallback(err_msg)
            if fallback_requested and replay_safe:
                tried_models.append(current_model)
                next_model = fallback_cfg.select_next_model(model, tried_models)
                if next_model:
                    logger.info(
                        f"[{request_id}] Fallback: {current_model} -> {next_model}"
                    )
                    fallback_reason = "transport_error"
                    current_model = next_model
                    continue
            elif fallback_requested:
                logger.warn(
                    f"[{request_id}] Suppressing model fallback after observed work"
                )
            res_payload_path = log_llm_response(
                request_id, current_model, "", error=err_msg, latency_ms=latency,
            )
            log_llm_api_call(
                request_id, current_model, "ERROR", latency,
                req_payload_path, res_payload_path, error=err_msg,
            )
            if last_attempt_trace is not None:
                _emit_terminal_compatibility_trace(
                    last_attempt_trace,
                    "upstream_error",
                    int((time.monotonic() - start_time) * 1000),
                )
            return {
                "error": err_msg,
                "status": 503,
                "fallback_attempts": len(tried_models),
                "context_remaining_percent": last_context_remaining_percent,
                "model": current_model,
            }

        should_fallback_from_stream = (
            (consumed["errors"] and not consumed["had_content"])
            or consumed["has_fatal_error"]
        )

        if should_fallback_from_stream:
            err_detail = _first_error_detail(consumed["errors"])
            logger.warn(
                f"[{request_id}] Stream error: {err_detail} | "
                f"had_content={consumed['had_content']} "
                f"fatal={consumed['has_fatal_error']} replay_safe={replay_safe}"
            )
            fallback_requested = fallback_cfg.should_fallback(err_detail)
            if fallback_requested and replay_safe:
                tried_models.append(current_model)
                next_model = fallback_cfg.select_next_model(model, tried_models)
                if next_model:
                    logger.info(
                        f"[{request_id}] Fallback: {current_model} -> {next_model} (stream error)"
                    )
                    fallback_reason = (
                        "fatal_response" if consumed["has_fatal_error"] else "stream_error"
                    )
                    current_model = next_model
                    continue
            elif fallback_requested:
                logger.warn(
                    f"[{request_id}] Suppressing model fallback after observed work"
                )
            if last_attempt_trace is not None:
                _emit_terminal_compatibility_trace(
                    last_attempt_trace,
                    "upstream_error",
                    int((time.monotonic() - start_time) * 1000),
                )
            return {
                "error": err_detail,
                "status": 503,
                "fallback_attempts": len(tried_models),
                "context_remaining_percent": last_context_remaining_percent,
                "model": current_model,
            }

        metrics = consumed["metrics"]
        clean_text = consumed.get("_protocol_visible_text", consumed["text"])
        clean_thinking = consumed.get("_protocol_thinking_text", consumed["thinking"])
        attempt_latency = int((time.monotonic() - attempt_start) * 1000)
        logger.info(
            f"[{request_id}] Stream consumed | text_len={len(consumed['text'])} thinking_len={len(consumed['thinking'])} "
            f"errors={len(consumed['errors'])} "
            f"{_format_stream_metrics(metrics)}"
        )

        converted_tcs: list[dict] = []
        parsed_tcs: list[DecodedToolCandidate] = []
        parsed_source = ""
        if effective_tool_names:
            parsed_tcs, parsed_source = _parse_tool_calls_from_consumed(consumed)
            if parsed_tcs:
                validation = validate_tool_candidates(
                    parsed_tcs,
                    allowed_names=set(effective_tool_names),
                )
                converted_tcs = list(validation.accepted)
                if validation.rejected:
                    logger.warn(
                        f"[{request_id}] Rejected decoded tool calls: "
                        f"{[(r.raw_name, r.reason) for r in validation.rejected]}"
                    )
                logger.info(
                    f"[{request_id}] Parsed tool calls: {len(converted_tcs)} "
                    f"(composer={is_composer} source={parsed_source})"
                )

        assert last_attempt_trace is not None
        attempt_trace = last_attempt_trace

        res_payload_path = log_llm_response(
            request_id, current_model, consumed["text"],
            tool_calls=converted_tcs or None,
            error=_first_error_detail(consumed["errors"]) if consumed["errors"] else None,
            latency_ms=attempt_latency,
            extra={
                "thinking_len": len(consumed["thinking"]),
                "chunks": metrics["chunk_count"],
                # Numeric-only timing/counter telemetry.  Keep the established
                # first_chunk key for consumers while exposing semantic timing
                # under names that cannot be mistaken for token latency.
                "stream_metrics": dict(metrics),
            },
        )
        log_llm_api_call(
            request_id, current_model,
            "OK" if not consumed["errors"] else "STREAM_ERROR",
            attempt_latency, req_payload_path, res_payload_path,
            error=_first_error_detail(consumed["errors"]) if consumed["errors"] else None,
        )

        if converted_tcs:
            raw_names = [(tc.get("function") or {}).get("name", "?") for tc in converted_tcs]
            logger.info(f"[{request_id}] Tool calls: {json.dumps([{'name': n} for n in raw_names])}")
            _emit_terminal_compatibility_trace(
                attempt_trace, "tool_calls", int((time.monotonic() - start_time) * 1000)
            )

            return {
                "tool_calls": converted_tcs,
                "text": "" if compact_tools else _safe_text_for_tool_source(consumed, parsed_source),
                "thinking": clean_thinking,
                "model": current_model,
                "stats": {"passed": len(converted_tcs), "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
                "fallback_attempts": len(tried_models),
                "context_remaining_percent": last_context_remaining_percent,
            }

        logger.info(
            f"[{request_id}] Final text response | len={len(clean_text)} | "
            f"{(time.monotonic() - start_time) * 1000:.0f}ms"
        )
        _emit_terminal_compatibility_trace(
            attempt_trace,
            "final_text",
            int((time.monotonic() - start_time) * 1000),
        )
        return {
            "text": clean_text,
            "thinking": clean_thinking,
            "model": current_model,
            "fallback_attempts": len(tried_models),
            "context_remaining_percent": last_context_remaining_percent,
            "stats": {"passed": 0, "normalized": 0, "filtered": 0, "invalid_arguments_filtered": 0},
        }

    logger.error(f"[{request_id}] All fallback models exhausted")
    if last_attempt_trace is not None:
        _emit_terminal_compatibility_trace(
            last_attempt_trace,
            "fallback_exhausted",
            int((time.monotonic() - start_time) * 1000),
        )
    return {
        "error": "All available models are currently unavailable",
        "status": 503,
        "fallback_attempts": len(tried_models),
        "model": None,
    }



def _build_compatibility_trace(
    *,
    request_id: str,
    attempt_number: int | str,
    requested_model: str,
    effective_model: str,
    protocol: ToolProtocol,
    compact_tools: bool,
    client_format: str | None,
    fallback_reason: str | None,
    consumed: dict | None,
    tool_candidate_source: str | None,
    candidate_count: int,
    accepted_tool_names: tuple[str, ...],
    rejection_reason: str | None,
    repair_attempted: bool,
    terminal_result: str,
    latency_ms: int,
) -> CompatibilityTrace:
    """Build a trace record without exposing request, response, or error content.

    Compatibility ``latency_ms`` is the completed attempt duration.  It is not
    time-to-first-token; semantic first-event timings live in stream metrics.
    """
    text = (consumed or {}).get("text") or ""
    thinking = (consumed or {}).get("thinking") or ""
    return CompatibilityTrace(
        request_id=request_id,
        attempt_id=f"{request_id}:{attempt_number}",
        requested_model=requested_model,
        effective_model=effective_model,
        protocol_adapter=protocol.value,
        client_format=client_format or ("openai" if compact_tools else "anthropic"),
        fallback_reason=fallback_reason,
        text_bytes=len(text.encode("utf-8")),
        reasoning_bytes=len(thinking.encode("utf-8")),
        tool_candidate_source=tool_candidate_source,
        candidate_count=candidate_count,
        accepted_tool_names=accepted_tool_names,
        rejection_reason=rejection_reason,
        repair_attempted=repair_attempted,
        terminal_result=terminal_result,
        latency_ms=latency_ms,
    )


def _emit_terminal_compatibility_trace(
    trace: CompatibilityTrace,
    terminal_result: str,
    latency_ms: int,
) -> None:
    """Record a terminal disposition while preserving immutable attempt metadata."""
    emit_terminal_trace(
        replace(trace, terminal_result=terminal_result, latency_ms=latency_ms)
    )


def _first_error_detail(errors: list) -> str:
    if not errors:
        return "Unknown stream error"
    err = errors[0]
    if isinstance(err, str):
        return err
    return str(
        getattr(err, "detail", None)
        or getattr(err, "raw", None)
        or getattr(err, "message", None)
        or err
    )


# ---------------------------------------------------------------------------
# Streaming delta granularity — split large Cursor chunks into small SSE events
# ---------------------------------------------------------------------------

DELTA_TARGET_SIZE = 8  # chars per SSE event, simulates token-level streaming
MAX_PENDING_REASONING_EVENTS = 32
MAX_PENDING_REASONING_CHARS = 8192
MIN_EVENT_DELAY = 0.003   # seconds — fast drain when queue is backlogged
MAX_EVENT_DELAY = 0.015   # seconds — smooth pacing when stream is trickling in


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_pipeline(
    req: UnifiedRequest,
    request_id: str,
    auth_token: str = "",
) -> dict:
    """Format-agnostic pipeline entry point.

    Accepts a UnifiedRequest (from normalize_anthropic or normalize_openai).
    Returns a dict with:
      - ok (bool)
      - stream (bool, if streaming)
      - body (dict, if non-streaming)
      - stream_handler (async generator, if streaming)
      - telemetry (dict)
    """
    pipeline_start = time.monotonic()

    messages = req.messages
    tools = req.tools
    stream = req.stream
    resolved_model = req.model
    max_tokens = req.max_tokens
    original_format = req.original_format
    requested_model = req.original_model or resolved_model
    valid_tool_names = [(t.get("function") or t).get("name", "") for t in tools]

    try:
        resolve_tool_choice(req.tool_choice, valid_tool_names)
    except ToolChoiceError as exc:
        return {
            "ok": False,
            "status": 400,
            "body": _to_api_error_body(str(exc), "invalid_request_error"),
            "telemetry": {
                "request_id": request_id,
                "pipeline": "claude_code",
                "model_requested": requested_model,
                "model_used": None,
                "latency_ms": _elapsed_ms(pipeline_start),
                "stream": stream,
            },
        }

    full_system = (req.system or "") + THALAMUS_INSTRUCTION_SUPPLEMENT
    messages = [{"role": "system", "content": full_system}] + messages

    if req.metadata:
        logger.info(f"[{request_id}] CC metadata: {json.dumps(req.metadata, ensure_ascii=False)[:200]}")
    if req.thinking:
        logger.info(f"[{request_id}] CC thinking config: {req.thinking}")
    if req.context_management:
        logger.debug(f"[{request_id}] CC context_management: {req.context_management}")

    parsed_mt = _parse_max_tokens(max_tokens)
    if not parsed_mt["ok"]:
        return {
            "ok": False,
            "status": 400,
            "body": _to_api_error_body(parsed_mt["error"], "invalid_request_error"),
            "telemetry": {
                "request_id": request_id,
                "pipeline": "claude_code",
                "model_requested": requested_model,
                "model_used": None,
                "latency_ms": _elapsed_ms(pipeline_start),
                "stream": stream,
            },
        }
    max_tokens = parsed_mt["value"]

    raw_req_token = _extract_raw_auth_token(auth_token)
    if raw_req_token and ("::" in raw_req_token or raw_req_token.startswith("eyJ")):
        token = raw_req_token
    else:
        token = get_cursor_access_token()

    logger.info(
        f"[{request_id}] pipeline=claude_code format={original_format} model={resolved_model} "
        f"stream={stream} tools={len(valid_tool_names)} msgs={len(messages)} max_tokens={max_tokens or '-'}"
    )
    if valid_tool_names and len(messages) <= 5:
        logger.info(f"[{request_id}] tool names: {valid_tool_names}")
        for t in tools[:5]:
            fn = t.get("function") or t
            tname = fn.get("name", "")
            tschema = fn.get("input_schema") or fn.get("parameters") or {}
            req_params = tschema.get("required", [])
            props = list((tschema.get("properties") or {}).keys())
            logger.info(f"[{request_id}] tool schema: {tname} props={props} required={req_params}")

    base_telemetry: dict[str, Any] = {
        "request_id": request_id,
        "pipeline": "claude_code",
        "original_format": original_format,
        "model_requested": requested_model,
        "max_tokens": max_tokens,
        "stream": stream,
        "agent_mode": True,
    }

    if stream:
        return _build_streaming_result(
            req, request_id, messages, tools, valid_tool_names,
            resolved_model, max_tokens, token, original_format,
            pipeline_start, base_telemetry, requested_model,
        )

    # --- Non-streaming (unary) path ---
    return await _build_unary_result(
        req, request_id, messages, tools, valid_tool_names,
        resolved_model, max_tokens, token, original_format,
        pipeline_start, base_telemetry, requested_model,
    )


def _build_streaming_result(
    req: UnifiedRequest,
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    original_format: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
    requested_model: str | None = None,
) -> dict:
    """Build the streaming result dict with an async generator."""

    if original_format in ("openai", "openai_responses"):
        return _build_streaming_result_openai(
            request_id, messages, tools, valid_tool_names,
            resolved_model, max_tokens, token,
            pipeline_start, base_telemetry, requested_model, original_format,
            req.tool_choice, req.thinking,
        )
    return _build_streaming_result_anthropic(
        request_id, messages, tools, valid_tool_names,
        resolved_model, max_tokens, token,
        pipeline_start, base_telemetry, requested_model, original_format,
        req.tool_choice,
    )


def _build_streaming_result_anthropic(
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
    requested_model: str | None = None,
    client_format: str | None = None,
    tool_choice: dict[str, Any] | str | None = None,
) -> dict:
    message_id = f"msg_{uuid.uuid4().hex}"
    estimated_input_tokens = estimate_input_tokens(messages, tools)

    async def stream_handler() -> AsyncIterator[str]:
        session = StreamingAnthropicSession(
            message_id, resolved_model, input_tokens=estimated_input_tokens
        )
        yield session.emit_message_start()

        limiter = _OutputLimiter(max_tokens)
        sse_queue: asyncio.Queue[str | None] = asyncio.Queue()
        early_tool_call_id: str | None = None

        thinking_started = False
        thinking_ended = False
        def _enqueue_text_fragments(text: str) -> None:
            """Split text into DELTA_TARGET_SIZE chunks and enqueue as text_delta SSE."""
            for i in range(0, len(text), DELTA_TARGET_SIZE):
                fragment = text[i:i + DELTA_TARGET_SIZE]
                sse = session.emit_text_delta(fragment)
                if sse:
                    sse_queue.put_nowait(sse)

        def on_thinking_as_text(delta: str) -> None:
            nonlocal thinking_started
            if not delta:
                return
            if not thinking_started:
                thinking_started = True
                thinking_forwarder.on_delta("thinking: ")
            thinking_forwarder.on_delta(delta)

        def _emit_and_enqueue(text: str) -> str | None:
            """Emit callback for ToolJsonAwareTextForwarder — splits into fragments."""
            if text:
                _enqueue_text_fragments(text)
            return text

        forwarder = ToolJsonAwareTextForwarder(
            emit_text_delta=_emit_and_enqueue,
            limiter=limiter,
        )
        thinking_forwarder = ToolJsonAwareTextForwarder(
            emit_text_delta=_emit_and_enqueue,
            limiter=limiter,
        )

        def on_text_delta(delta: str) -> None:
            nonlocal thinking_ended
            if not delta:
                return
            if thinking_started and not thinking_ended:
                thinking_ended = True
                thinking_forwarder.on_delta("\n\n")
            forwarder.on_delta(delta)

        def on_tool_call_start(call_id: str, name: str) -> None:
            nonlocal early_tool_call_id
            if early_tool_call_id is not None or not call_id or not name:
                return
            early_tool_call_id = call_id
            sse = session.emit_tool_call_start(call_id, name)
            if sse:
                sse_queue.put_nowait(sse)

        async def run_cursor_call() -> dict:
            trace_context = (
                {"requested_model": requested_model, "client_format": client_format}
                if requested_model is not None
                else {}
            )
            choice_context = {"tool_choice": tool_choice} if tool_choice is not None else {}
            callback_token = _TOOL_CALL_START_CALLBACK.set(on_tool_call_start)
            try:
                return await _call_cursor_direct(
                    messages, resolved_model, tools, valid_tool_names, token,
                    on_stream_delta=on_text_delta,
                    on_thinking_delta=on_thinking_as_text,
                    compact_tools=False,
                    **choice_context,
                    **trace_context,
                )
            finally:
                _TOOL_CALL_START_CALLBACK.reset(callback_token)

        cursor_task = asyncio.create_task(run_cursor_call())
        next_keepalive_at = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS

        try:
            while not cursor_task.done() or not sse_queue.empty():
                try:
                    sse = sse_queue.get_nowait()
                    if sse:
                        yield sse
                        next_keepalive_at = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
                        depth = sse_queue.qsize()
                        if depth > 5:
                            await asyncio.sleep(MIN_EVENT_DELAY)
                        elif depth <= 2:
                            await asyncio.sleep(MAX_EVENT_DELAY)
                        else:
                            frac = (depth - 2) / 3.0
                            await asyncio.sleep(MAX_EVENT_DELAY - frac * (MAX_EVENT_DELAY - MIN_EVENT_DELAY))
                except asyncio.QueueEmpty:
                    now = time.monotonic()
                    if now >= next_keepalive_at:
                        yield session.emit_keepalive()
                        next_keepalive_at = now + SSE_KEEPALIVE_INTERVAL_SECONDS
                        continue
                    await asyncio.sleep(0.005)
        except BaseException:
            if not cursor_task.done():
                cursor_task.cancel()
                try:
                    await cursor_task
                except asyncio.CancelledError:
                    pass
            raise

        direct_result = cursor_task.result()
        used_model = direct_result.get("model") or resolved_model
        session.input_tokens = input_tokens_from_remaining_context(
            get_model_context_length(used_model),
            direct_result.get("context_remaining_percent"),
            estimated_input_tokens,
        )

        if direct_result.get("error"):
            logger.error(
                f"[{request_id}] pipeline=claude_code stage=error(stream) error={direct_result['error']}"
            )
            yield session.finish(stop_reason="end_turn")
            return

        tool_calls = direct_result.get("tool_calls") or []

        full_text = direct_result.get("text", "")
        final_safe = _safe_final_text_for_stream(full_text, bool(tool_calls))
        thinking_final_safe = _safe_final_text_for_stream(
            direct_result.get("thinking", ""), bool(tool_calls)
        )
        thinking_forwarder.flush_using_final_safe_text(thinking_final_safe)
        forwarder.flush_using_final_safe_text(final_safe)
        while not sse_queue.empty():
            sse = sse_queue.get_nowait()
            if sse:
                yield sse

        remaining_tool_calls = list(tool_calls)
        if early_tool_call_id is not None:
            early_call = next(
                (
                    tool_call
                    for tool_call in remaining_tool_calls
                    if tool_call.get("id") == early_tool_call_id
                ),
                None,
            )
            if early_call is not None:
                function = early_call.get("function") or {}
                yield session.finish_tool_call(function.get("arguments", "{}"))
                remaining_tool_calls.remove(early_call)

        yield session.close_open_blocks()

        if remaining_tool_calls:
            yield session.emit_tool_use_blocks(remaining_tool_calls)

        stop_reason: str
        if tool_calls:
            stop_reason = "tool_use"
        elif limiter.is_exhausted:
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"

        yield session.finish(stop_reason=stop_reason)

        latency = _elapsed_ms(pipeline_start)
        logger.info(
            f"[{request_id}] pipeline=claude_code stage=result(stream/anthropic) model={used_model} "
            f"tool_calls={len(tool_calls)} stop_reason={stop_reason} latency_ms={latency:.0f}"
        )

    return {
        "ok": True,
        "stream": True,
        "stream_handler": stream_handler,
        "telemetry": {**base_telemetry, "model_used": resolved_model},
    }


def _build_streaming_result_openai(
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
    requested_model: str | None = None,
    client_format: str | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    thinking: dict[str, Any] | None = None,
) -> dict:
    is_responses = client_format == "openai_responses"
    completion_id = (
        f"resp_{uuid.uuid4().hex[:24]}"
        if is_responses
        else f"chatcmpl-{uuid.uuid4().hex[:24]}"
    )
    estimated_input_tokens = estimate_input_tokens(messages, tools)

    async def stream_handler() -> AsyncIterator[str]:
        if is_responses:
            summary_mode = (thinking or {}).get("summary")
            session = StreamingOpenAIResponsesSession(
                completion_id,
                resolved_model,
                input_tokens=estimated_input_tokens,
                emit_reasoning_summary=summary_mode not in (None, "none"),
            )
            yield session.start()
        else:
            session = StreamingOpenAISession(
                completion_id, resolved_model, input_tokens=estimated_input_tokens
            )
            yield session.emit_role_chunk()

        limiter = _OutputLimiter(max_tokens)
        pending_text: deque[str] = deque()
        pending_reasoning: deque[str] = deque()
        pending_tool_events: deque[str] = deque()
        pending_reasoning_chars = 0
        last_text_delta_at: float | None = None
        early_tool_call_id: str | None = None
        def _emit_and_enqueue(text: str) -> str | None:
            if text:
                for i in range(0, len(text), DELTA_TARGET_SIZE):
                    pending_text.append(text[i:i + DELTA_TARGET_SIZE])
            return text

        forwarder = ToolJsonAwareTextForwarder(
            emit_text_delta=_emit_and_enqueue,
            # Apply the shared output budget when queued events are emitted so
            # reasoning that arrived first cannot lose its budget to a later
            # synchronous text callback.
            limiter=_OutputLimiter(None),
        )

        reasoning_source_chars_consumed = 0

        def on_text_delta(delta: str) -> None:
            nonlocal last_text_delta_at
            if not delta:
                return
            last_text_delta_at = time.monotonic()
            forwarder.on_delta(delta)

        def on_thinking_delta(delta: str) -> None:
            nonlocal pending_reasoning_chars, reasoning_source_chars_consumed
            if not delta:
                return
            reasoning_source_chars_consumed += len(delta)
            remaining = MAX_PENDING_REASONING_CHARS - pending_reasoning_chars
            if remaining <= 0:
                return
            buffered = delta[:remaining]
            if len(pending_reasoning) < MAX_PENDING_REASONING_EVENTS:
                pending_reasoning.append(buffered)
            else:
                pending_reasoning[-1] += buffered
            pending_reasoning_chars += len(buffered)

        def on_tool_call_start(call_id: str, name: str) -> None:
            nonlocal early_tool_call_id
            if early_tool_call_id is not None or not call_id or not name:
                return
            early_tool_call_id = call_id
            sse = session.emit_tool_call_start(call_id, name)
            if sse:
                pending_tool_events.append(sse)

        def next_pending_sse() -> str | None:
            nonlocal pending_reasoning_chars
            if pending_reasoning:
                reasoning = pending_reasoning.popleft()
                pending_reasoning_chars -= len(reasoning)
                limited = limiter.emit_within_limit(reasoning)
                if limited:
                    return session.emit_reasoning_delta(limited)
            if pending_text:
                limited = limiter.emit_within_limit(pending_text.popleft())
                if limited:
                    return session.emit_text_delta(limited)
            if pending_tool_events:
                return pending_tool_events.popleft()
            return None

        async def run_cursor_call() -> dict:
            trace_context = (
                {"requested_model": requested_model, "client_format": client_format}
                if requested_model is not None
                else {}
            )
            choice_context = {"tool_choice": tool_choice} if tool_choice is not None else {}
            callback_token = _TOOL_CALL_START_CALLBACK.set(on_tool_call_start)
            try:
                return await _call_cursor_direct(
                    messages, resolved_model, tools, valid_tool_names, token,
                    on_stream_delta=on_text_delta,
                    on_thinking_delta=on_thinking_delta,
                    compact_tools=True,
                    **choice_context,
                    **trace_context,
                )
            finally:
                _TOOL_CALL_START_CALLBACK.reset(callback_token)

        cursor_task = asyncio.create_task(run_cursor_call())
        next_keepalive_at = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS

        try:
            while (
                not cursor_task.done()
                or pending_text
                or pending_reasoning
                or pending_tool_events
            ):
                sse = next_pending_sse()
                if sse:
                    yield sse
                    next_keepalive_at = time.monotonic() + SSE_KEEPALIVE_INTERVAL_SECONDS
                    depth = (
                        len(pending_text)
                        + len(pending_reasoning)
                        + len(pending_tool_events)
                    )
                    if depth > 5:
                        await asyncio.sleep(MIN_EVENT_DELAY)
                    elif depth <= 2:
                        await asyncio.sleep(MAX_EVENT_DELAY)
                    else:
                        frac = (depth - 2) / 3.0
                        await asyncio.sleep(MAX_EVENT_DELAY - frac * (MAX_EVENT_DELAY - MIN_EVENT_DELAY))
                else:
                    now = time.monotonic()
                    if (
                        is_responses
                        and last_text_delta_at is not None
                        and session.has_open_message
                        and now - last_text_delta_at
                        >= RESPONSES_TEXT_ITEM_IDLE_FLUSH_SECONDS
                    ):
                        completed_item = session.flush_text_item()
                        if completed_item:
                            logger.debug(
                                f"[{request_id}] Responses text item completed after "
                                f"{RESPONSES_TEXT_ITEM_IDLE_FLUSH_SECONDS:.2f}s idle "
                                "while upstream remained active"
                            )
                            yield completed_item
                            last_text_delta_at = None
                            next_keepalive_at = (
                                now + SSE_KEEPALIVE_INTERVAL_SECONDS
                            )
                            continue
                    if now >= next_keepalive_at:
                        yield session.emit_keepalive()
                        next_keepalive_at = now + SSE_KEEPALIVE_INTERVAL_SECONDS
                        continue
                    await asyncio.sleep(0.005)
        except BaseException:
            if not cursor_task.done():
                cursor_task.cancel()
                try:
                    await cursor_task
                except asyncio.CancelledError:
                    pass
            raise

        direct_result = cursor_task.result()
        used_model = direct_result.get("model") or resolved_model
        session.input_tokens = input_tokens_from_remaining_context(
            get_model_context_length(used_model),
            direct_result.get("context_remaining_percent"),
            estimated_input_tokens,
        )

        if direct_result.get("error"):
            logger.error(
                f"[{request_id}] pipeline=claude_code stage=error(stream) error={direct_result['error']}"
            )
            yield session.finish(stop_reason="stop")
            return

        tool_calls = direct_result.get("tool_calls") or []
        final_thinking = direct_result.get("thinking", "")
        if len(final_thinking) > reasoning_source_chars_consumed:
            on_thinking_delta(final_thinking[reasoning_source_chars_consumed:])

        full_text = direct_result.get("text", "")
        final_safe = _safe_final_text_for_stream(full_text, bool(tool_calls))
        forwarder.flush_using_final_safe_text(final_safe)
        while pending_reasoning or pending_text or pending_tool_events:
            sse = next_pending_sse()
            if sse:
                yield sse

        remaining_tool_calls = list(tool_calls)
        if early_tool_call_id is not None:
            early_call = next(
                (
                    tool_call
                    for tool_call in remaining_tool_calls
                    if tool_call.get("id") == early_tool_call_id
                ),
                None,
            )
            if early_call is not None:
                if is_responses:
                    yield session.finish_tool_call(early_call)
                else:
                    function = early_call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)
                    yield session.emit_tool_call_args_delta(arguments)
                    session.advance_tool_call()
                remaining_tool_calls.remove(early_call)

        if remaining_tool_calls:
            yield session.emit_tool_use_blocks(remaining_tool_calls)

        stop_reason: str
        if tool_calls:
            stop_reason = "tool_calls"
        elif limiter.is_exhausted:
            stop_reason = "length"
        else:
            stop_reason = "stop"

        yield session.finish(stop_reason=stop_reason)

        latency = _elapsed_ms(pipeline_start)
        logger.info(
            f"[{request_id}] pipeline=claude_code stage=result(stream/openai) model={used_model} "
            f"tool_calls={len(tool_calls)} stop_reason={stop_reason} latency_ms={latency:.0f}"
        )

    return {
        "ok": True,
        "stream": True,
        "stream_handler": stream_handler,
        "telemetry": {**base_telemetry, "model_used": resolved_model},
    }


async def _build_unary_result(
    req: UnifiedRequest,
    request_id: str,
    messages: list[dict],
    tools: list[dict],
    valid_tool_names: list[str],
    resolved_model: str,
    max_tokens: int | None,
    token: str,
    original_format: str,
    pipeline_start: float,
    base_telemetry: dict[str, Any],
    requested_model: str | None = None,
) -> dict:
    """Build a non-streaming result."""
    trace_context = (
        {"requested_model": requested_model, "client_format": original_format}
        if requested_model is not None
        else {}
    )
    choice_context = {"tool_choice": req.tool_choice} if req.tool_choice is not None else {}
    direct_result = await _call_cursor_direct(
        messages, resolved_model, tools, valid_tool_names, token,
        compact_tools=(original_format in ("openai", "openai_responses")),
        **choice_context,
        **trace_context,
    )

    logger.info(
        f"[{request_id}] DIRECT_RESULT(unary): text_len={len(direct_result.get('text', ''))} "
        f"tool_calls={len(direct_result.get('tool_calls') or [])} error={direct_result.get('error')}"
    )

    if direct_result.get("error"):
        return {
            "ok": False,
            "status": direct_result.get("status", 500),
            "body": _to_api_error_body(direct_result["error"]),
            "telemetry": {
                **base_telemetry,
                "model_used": direct_result.get("model"),
                "latency_ms": _elapsed_ms(pipeline_start),
            },
        }

    used_model = direct_result.get("model") or resolved_model
    input_tokens = input_tokens_from_remaining_context(
        get_model_context_length(used_model),
        direct_result.get("context_remaining_percent"),
        estimate_input_tokens(messages, tools),
    )
    tool_calls = direct_result.get("tool_calls") or []
    text = direct_result.get("text", "")

    truncated = False
    if max_tokens and max_tokens > 0:
        char_budget = max_tokens * 4
        if len(text) > char_budget:
            text = text[:char_budget]
            truncated = True

    stop_reason_override = ""
    if not tool_calls and truncated:
        stop_reason_override = "max_tokens"

    telemetry = {
        **base_telemetry,
        "model_used": used_model,
        "fallback_attempts": direct_result.get("fallback_attempts", 0),
        "latency_ms": _elapsed_ms(pipeline_start),
        "stream": False,
        "output_truncated": truncated,
    }

    logger.info(
        f"[{request_id}] pipeline=claude_code stage=result model={used_model} "
        f"tool_calls={len(tool_calls)} text_len={len(text)} latency_ms={telemetry['latency_ms']:.0f}"
    )

    if original_format == "openai_responses":
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        body = build_unary_openai_responses_response(
            response_id=response_id,
            model=used_model,
            text=text,
            tool_calls=tool_calls,
            stop_reason_override=stop_reason_override,
            input_tokens=input_tokens,
        )
    elif original_format == "openai":
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        body = build_unary_openai_response(
            completion_id=completion_id,
            model=used_model,
            text=text,
            tool_calls=tool_calls,
            stop_reason_override=stop_reason_override,
            input_tokens=input_tokens,
        )
    else:
        message_id = f"msg_{uuid.uuid4().hex}"
        body = build_unary_anthropic_response(
            message_id=message_id,
            model=used_model,
            text=text,
            thinking="",
            tool_calls=tool_calls,
            stop_reason_override=stop_reason_override,
            input_tokens=input_tokens,
        )

    return {
        "ok": True,
        "stream": False,
        "body": body,
        "telemetry": telemetry,
    }


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
