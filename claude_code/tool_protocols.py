"""Immutable interfaces and protocol selection for Cursor tool-call decoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ToolProtocol(StrEnum):
    """Supported upstream tool-call wire grammars."""

    STANDARD_JSON_V1 = "standard_json_v1"
    COMPOSER_MARKER_V1 = "composer_marker_v1"
    LEGACY_THINK_TAG_V1 = "legacy_think_tag_v1"


@dataclass(frozen=True)
class ProtocolFragment:
    """A single ordered delta from either visible text or reasoning."""

    sequence: int
    lane: str
    text: str


@dataclass(frozen=True)
class DecodedToolCandidate:
    """A syntactically valid, but not yet authorized, tool-call candidate."""

    call_id: str
    raw_name: str
    arguments: object
    source_lane: str


@dataclass(frozen=True)
class ProtocolDecodeState:
    """Independent incremental buffers for visible text and reasoning."""

    text_buffer: str = ""
    reasoning_buffer: str = ""
    quarantined_text: str = ""
    quarantined_reasoning: str = ""
    next_sequence: int = 0


@dataclass(frozen=True)
class ProtocolDecodeResult:
    """The immutable result of decoding one protocol fragment."""

    state: ProtocolDecodeState
    visible_text: str = ""
    thinking_text: str = ""
    candidates: tuple[DecodedToolCandidate, ...] = ()
    is_incomplete: bool = False
    invalid_reason: str | None = None


class ProtocolAdapter(Protocol):
    """Renders and incrementally decodes a single tool-call protocol."""

    protocol: ToolProtocol

    def render_tool_manifest(self, tools: list[dict], execution_policy: str) -> str:
        raise NotImplementedError

    def render_repair(self, tools: list[dict], interrupted_state: str) -> str:
        raise NotImplementedError

    def incremental_decode(
        self,
        fragment: ProtocolFragment | None,
        state: ProtocolDecodeState,
    ) -> ProtocolDecodeResult:
        raise NotImplementedError


def classify_tool_protocol(effective_model: str) -> ToolProtocol:
    """Select the only observed model-specific tool grammars."""
    normalized = effective_model.lower()
    if normalized.startswith("composer-2"):
        return ToolProtocol.COMPOSER_MARKER_V1
    if normalized.startswith("composer-1.5"):
        return ToolProtocol.LEGACY_THINK_TAG_V1
    return ToolProtocol.STANDARD_JSON_V1


def create_protocol_adapter(protocol: ToolProtocol) -> ProtocolAdapter:
    """Create an adapter for every protocol selected by the classifier."""
    from claude_code.composer_tool_protocol import ComposerMarkerV1Adapter
    from claude_code.standard_tool_protocol import LegacyThinkTagV1Adapter, StandardJsonV1Adapter

    if protocol == ToolProtocol.STANDARD_JSON_V1:
        return StandardJsonV1Adapter()
    if protocol == ToolProtocol.COMPOSER_MARKER_V1:
        return ComposerMarkerV1Adapter()
    if protocol == ToolProtocol.LEGACY_THINK_TAG_V1:
        return LegacyThinkTagV1Adapter()
    raise ValueError(f"Unsupported tool protocol: {protocol}")
