"""Strict native JSON tool protocol adapters."""

from __future__ import annotations

import json
import re

from claude_code.tool_parser import extract_balanced_text
from claude_code.tool_prompt_rules import (
    CLIENT_TOOL_INVENTORY_AUTHORITY_RULE,
    POST_TOOL_OUTPUT_FORMAT_RULE,
)
from claude_code.tool_protocols import (
    DecodedToolCandidate,
    ProtocolDecodeResult,
    ProtocolDecodeState,
    ProtocolFragment,
    ToolProtocol,
)

_NATIVE_TOOL_TYPE = "tool_use"
_NATIVE_TOOL_TYPE_RE = re.compile(r'"type"\s*:\s*"tool_use"')


def _contains_native_tool_prefix(text: str) -> bool:
    """Whether an unclosed object has identified itself as native tool JSON."""
    return bool(_NATIVE_TOOL_TYPE_RE.search(text))


def _decode_complete_object(
    raw_object: str,
    source_lane: str,
) -> tuple[DecodedToolCandidate | None, str | None]:
    """Decode exactly one native tool object without JSON repair heuristics."""
    try:
        value = json.loads(raw_object)
    except json.JSONDecodeError:
        return None, "malformed_json"

    if not isinstance(value, dict):
        return None, None
    if value.get("type") != _NATIVE_TOOL_TYPE:
        return None, None

    call_id = value.get("id")
    raw_name = value.get("name")
    arguments = value.get("input")
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(raw_name, str)
        or not raw_name
        or not isinstance(arguments, dict)
    ):
        return None, "invalid_tool_shape"

    return DecodedToolCandidate(call_id, raw_name, arguments, source_lane), None


def _decode_lane(
    buffer: str,
    source_lane: str,
    *,
    flush: bool,
) -> tuple[str, str, tuple[DecodedToolCandidate, ...], bool, str | None]:
    """Consume complete JSON objects while retaining one possible partial object."""
    visible_parts: list[str] = []
    candidates: list[DecodedToolCandidate] = []
    cursor = 0

    while cursor < len(buffer):
        object_start = buffer.find("{", cursor)
        if object_start < 0:
            visible_parts.append(buffer[cursor:])
            return "", "".join(visible_parts), tuple(candidates), False, None

        visible_parts.append(buffer[cursor:object_start])
        raw_object = extract_balanced_text(buffer, object_start, "{", "}")
        if raw_object is None:
            partial = buffer[object_start:]
            if flush and not _contains_native_tool_prefix(partial):
                visible_parts.append(partial)
                return "", "".join(visible_parts), tuple(candidates), False, None
            return (
                partial,
                "".join(visible_parts),
                tuple(candidates),
                _contains_native_tool_prefix(partial),
                None,
            )

        if not _contains_native_tool_prefix(raw_object):
            visible_parts.append(raw_object)
            cursor = object_start + len(raw_object)
            continue

        candidate, invalid_reason = _decode_complete_object(raw_object, source_lane)
        if candidate is not None:
            candidates.append(candidate)
        elif invalid_reason is not None:
            return "", "".join(visible_parts), tuple(candidates), False, invalid_reason
        cursor = object_start + len(raw_object)

    return "", "".join(visible_parts), tuple(candidates), False, None


def _cross_lane_native_json(state: ProtocolDecodeState) -> bool:
    """Return whether quarantined fragments form native JSON in either lane order."""
    text_parts = state.text_buffer + state.quarantined_text
    reasoning_parts = state.reasoning_buffer + state.quarantined_reasoning
    combinations = (
        text_parts + reasoning_parts,
        reasoning_parts + text_parts,
    )
    for combined in combinations:
        if not combined:
            continue
        _, _, candidates, _, _ = _decode_lane(combined, "cross_lane", flush=True)
        if candidates or _contains_native_tool_prefix(combined):
            return True
    return False


def _quarantine_fragment(
    fragment: ProtocolFragment,
    state: ProtocolDecodeState,
) -> ProtocolDecodeResult:
    """Hold cross-lane output until EOF proves it cannot form native tool JSON."""
    if fragment.lane == "text":
        next_state = ProtocolDecodeState(
            text_buffer=state.text_buffer,
            reasoning_buffer=state.reasoning_buffer,
            quarantined_text=state.quarantined_text + fragment.text,
            quarantined_reasoning=state.quarantined_reasoning,
            next_sequence=max(state.next_sequence, fragment.sequence + 1),
        )
    else:
        next_state = ProtocolDecodeState(
            text_buffer=state.text_buffer,
            reasoning_buffer=state.reasoning_buffer,
            quarantined_text=state.quarantined_text,
            quarantined_reasoning=state.quarantined_reasoning + fragment.text,
            next_sequence=max(state.next_sequence, fragment.sequence + 1),
        )
    return ProtocolDecodeResult(state=next_state, is_incomplete=True)


class StandardJsonV1Adapter:
    """Strict decoder for native Anthropic-style ``tool_use`` JSON objects."""

    protocol = ToolProtocol.STANDARD_JSON_V1

    def render_tool_manifest(self, tools: list[dict], execution_policy: str) -> str:
        """Make native tools visible without asking for hand-authored tool JSON."""
        serialized_tools = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        return (
            f"{execution_policy}\n\n"
            "Available client tools are exposed through native function calling.\n"
            "Exact available client tool schemas:\n"
            f"{serialized_tools}\n"
            f"{CLIENT_TOOL_INVENTORY_AUTHORITY_RULE}\n"
            f"{POST_TOOL_OUTPUT_FORMAT_RULE}\n"
            "Use the exact advertised tool name and schema. If the current request "
            "requires an action, issue the native tool call in this response. Do not "
            "copy schemas or emit tool-call JSON as assistant prose."
        )

    def render_repair(self, tools: list[dict], interrupted_state: str) -> str:
        """Ask for one replacement call after a recognized partial native object."""
        del tools
        return (
            "Repair the interrupted tool call. The existing system instruction contains "
            "the client tool schemas and protocol.\n"
            f"Interrupted output: {interrupted_state}\n"
            "Start over and emit one fresh, complete tool_use JSON object. Do not continue truncated JSON."
        )

    def incremental_decode(
        self,
        fragment: ProtocolFragment | None,
        state: ProtocolDecodeState,
    ) -> ProtocolDecodeResult:
        """Decode one text/reasoning delta while never mixing their partial JSON."""
        if fragment is None:
            return self._flush(state)
        if fragment.lane not in {"text", "reasoning"}:
            return ProtocolDecodeResult(state=state, invalid_reason="invalid_lane")

        if (
            state.quarantined_text
            or state.quarantined_reasoning
            or (fragment.lane == "text" and state.reasoning_buffer)
            or (fragment.lane == "reasoning" and state.text_buffer)
        ):
            return _quarantine_fragment(fragment, state)

        text_buffer = state.text_buffer
        reasoning_buffer = state.reasoning_buffer
        if fragment.lane == "text":
            text_buffer += fragment.text
            remainder, visible_text, candidates, incomplete, invalid_reason = _decode_lane(
                text_buffer, "text", flush=False
            )
            text_buffer = remainder
            thinking_text = ""
        else:
            reasoning_buffer += fragment.text
            remainder, thinking_text, candidates, incomplete, invalid_reason = _decode_lane(
                reasoning_buffer, "reasoning", flush=False
            )
            reasoning_buffer = remainder
            visible_text = ""

        next_state = ProtocolDecodeState(
            text_buffer=text_buffer,
            reasoning_buffer=reasoning_buffer,
            quarantined_text=state.quarantined_text,
            quarantined_reasoning=state.quarantined_reasoning,
            next_sequence=max(state.next_sequence, fragment.sequence + 1),
        )
        return ProtocolDecodeResult(
            state=next_state,
            visible_text=visible_text,
            thinking_text=thinking_text,
            candidates=candidates,
            is_incomplete=incomplete,
            invalid_reason=invalid_reason,
        )

    def _flush(self, state: ProtocolDecodeState) -> ProtocolDecodeResult:
        if state.quarantined_text or state.quarantined_reasoning:
            if _cross_lane_native_json(state):
                return ProtocolDecodeResult(
                    state=ProtocolDecodeState(next_sequence=state.next_sequence),
                    is_incomplete=True,
                    invalid_reason="cross_lane_native_json",
                )
            text_source = state.text_buffer + state.quarantined_text
            reasoning_source = state.reasoning_buffer + state.quarantined_reasoning
        else:
            text_source = state.text_buffer
            reasoning_source = state.reasoning_buffer

        text_remainder, visible_text, text_candidates, text_incomplete, text_invalid = _decode_lane(
            text_source, "text", flush=True
        )
        reasoning_remainder, thinking_text, reasoning_candidates, reasoning_incomplete, reasoning_invalid = _decode_lane(
            reasoning_source, "reasoning", flush=True
        )
        return ProtocolDecodeResult(
            state=ProtocolDecodeState(
                text_buffer=text_remainder,
                reasoning_buffer=reasoning_remainder,
                next_sequence=state.next_sequence,
            ),
            visible_text=visible_text,
            thinking_text=thinking_text,
            candidates=(*text_candidates, *reasoning_candidates),
            is_incomplete=text_incomplete or reasoning_incomplete,
            invalid_reason=text_invalid or reasoning_invalid,
        )


class LegacyThinkTagV1Adapter(StandardJsonV1Adapter):
    """Composer-1.5 wrapper preserving the pipeline's existing think splitter."""

    protocol = ToolProtocol.LEGACY_THINK_TAG_V1

    def __init__(self) -> None:
        from claude_code.pipeline import _ThinkTagSplitter

        self._splitter = _ThinkTagSplitter()
        self._native_adapter = StandardJsonV1Adapter()

    def incremental_decode(
        self,
        fragment: ProtocolFragment | None,
        state: ProtocolDecodeState,
    ) -> ProtocolDecodeResult:
        """Split Composer-1.5 text tags before strict native JSON decoding."""
        if fragment is not None and fragment.lane != "text":
            pending_text = ""
            if not self._splitter._inside_think:
                tool_start = self._splitter._buf.find("{")
                if tool_start >= 0:
                    pending_text = self._splitter._buf[:tool_start + 1]
                    self._splitter._buf = self._splitter._buf[tool_start + 1:]
            if not pending_text:
                return self._native_adapter.incremental_decode(fragment, state)

            primed = self._native_adapter.incremental_decode(
                ProtocolFragment(
                    sequence=state.next_sequence,
                    lane="text",
                    text=pending_text,
                ),
                state,
            )
            routed = self._native_adapter.incremental_decode(fragment, primed.state)
            return ProtocolDecodeResult(
                state=routed.state,
                visible_text=primed.visible_text + routed.visible_text,
                thinking_text=primed.thinking_text + routed.thinking_text,
                candidates=(*primed.candidates, *routed.candidates),
                is_incomplete=primed.is_incomplete or routed.is_incomplete,
                invalid_reason=primed.invalid_reason or routed.invalid_reason,
            )

        if fragment is None:
            thinking_text, visible_text = self._splitter.flush()
            sequence = state.next_sequence
            flush_native = True
        else:
            thinking_text, visible_text = self._splitter.feed(fragment.text)
            sequence = fragment.sequence
            flush_native = False

        return self._decode_split_parts(
            state,
            sequence,
            thinking_text,
            visible_text,
            flush_native,
        )

    def _decode_split_parts(
        self,
        state: ProtocolDecodeState,
        sequence: int,
        thinking_text: str,
        visible_text: str,
        flush_native: bool,
    ) -> ProtocolDecodeResult:
        """Decode splitter output through the native lanes without leaking tool JSON."""
        results: list[ProtocolDecodeResult] = []
        current_state = state
        if thinking_text:
            thinking_result = self._native_adapter.incremental_decode(
                ProtocolFragment(sequence=sequence, lane="reasoning", text=thinking_text),
                current_state,
            )
            results.append(thinking_result)
            current_state = thinking_result.state
        if visible_text:
            visible_result = self._native_adapter.incremental_decode(
                ProtocolFragment(sequence=sequence, lane="text", text=visible_text),
                current_state,
            )
            results.append(visible_result)
            current_state = visible_result.state
        if flush_native:
            flushed = self._native_adapter.incremental_decode(None, current_state)
            results.append(flushed)
            current_state = flushed.state
        if not results:
            return ProtocolDecodeResult(state=current_state)

        final = results[-1]
        return ProtocolDecodeResult(
            state=final.state,
            visible_text="".join(result.visible_text for result in results),
            thinking_text="".join(result.thinking_text for result in results),
            candidates=tuple(
                candidate for result in results for candidate in result.candidates
            ),
            is_incomplete=final.is_incomplete,
            invalid_reason=next(
                (result.invalid_reason for result in results if result.invalid_reason),
                None,
            ),
        )
