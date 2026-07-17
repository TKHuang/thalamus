"""Composer-2 marker-protocol adapter."""

from __future__ import annotations

import json

from claude_code.composer_tool_parser import ComposerEmit, ComposerStreamProcessor
from config.system_prompt import COMPOSER_TOOL_PROMPT_HEADER
from claude_code.tool_protocols import (
    DecodedToolCandidate,
    ProtocolDecodeResult,
    ProtocolDecodeState,
    ProtocolFragment,
    ToolProtocol,
)


class ComposerMarkerV1Adapter:
    """Small adapter wrapper around the proven Composer marker stream parser."""

    protocol = ToolProtocol.COMPOSER_MARKER_V1

    def __init__(self) -> None:
        self._processor = ComposerStreamProcessor()
        self._next_call_id = 0

    def render_tool_manifest(self, tools: list[dict], execution_policy: str) -> str:
        """Render deterministic schemas with Composer's marker grammar."""
        serialized_tools = json.dumps(
            tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        return (
            f"{execution_policy}\n\n{COMPOSER_TOOL_PROMPT_HEADER}\n"
            "Available client tools are the following JSON schemas:\n"
            f"{serialized_tools}"
        )

    def render_repair(self, tools: list[dict], interrupted_state: str) -> str:
        """Request a new complete marker block after interrupted output."""
        del tools
        return (
            "Repair the interrupted tool call. The existing system instruction contains "
            "the client tool schemas and marker grammar.\n"
            f"Interrupted output: {interrupted_state}\n"
            "Start over with one complete <|tool_calls_begin|> block."
        )

    def incremental_decode(
        self,
        fragment: ProtocolFragment | None,
        state: ProtocolDecodeState,
    ) -> ProtocolDecodeResult:
        """Delegate marker splitting to the existing stateful Composer parser."""
        if fragment is None:
            if self._processor.pending_tool_block():
                return ProtocolDecodeResult(state=state, is_incomplete=True)
            emit = self._processor.flush()
            next_sequence = state.next_sequence
            source_lane = "reasoning"
        elif fragment.lane == "reasoning":
            emit = self._processor.feed_thinking(fragment.text)
            next_sequence = max(state.next_sequence, fragment.sequence + 1)
            source_lane = "reasoning"
        elif fragment.lane == "text":
            emit = self._processor.feed_content(fragment.text)
            next_sequence = max(state.next_sequence, fragment.sequence + 1)
            source_lane = "text"
        else:
            return ProtocolDecodeResult(state=state, invalid_reason="invalid_lane")

        return ProtocolDecodeResult(
            state=ProtocolDecodeState(
                reasoning_buffer=state.reasoning_buffer + emit.thinking,
                next_sequence=next_sequence,
            ),
            visible_text=emit.text,
            thinking_text=emit.thinking,
            candidates=self._candidates_from_emit(emit, source_lane),
            is_incomplete=bool(self._processor.pending_tool_block()),
        )

    def _candidates_from_emit(
        self,
        emit: ComposerEmit,
        source_lane: str,
    ) -> tuple[DecodedToolCandidate, ...]:
        candidates: list[DecodedToolCandidate] = []
        for call in emit.tool_calls:
            raw_name = call.get("name")
            arguments = call.get("arguments")
            if not isinstance(raw_name, str):
                continue
            self._next_call_id += 1
            candidates.append(
                DecodedToolCandidate(
                    call_id=f"composer_{self._next_call_id}",
                    raw_name=raw_name,
                    arguments=arguments,
                    source_lane=source_lane,
                )
            )
        return tuple(candidates)
