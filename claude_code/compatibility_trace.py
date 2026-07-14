"""Privacy-safe structured compatibility telemetry for the Cursor pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("pipeline", "DEBUG")


@dataclass(frozen=True)
class CompatibilityTrace:
    """Stable compatibility metadata that intentionally excludes request content."""

    request_id: str
    attempt_id: str
    requested_model: str
    effective_model: str
    protocol_adapter: str
    client_format: str
    fallback_reason: str | None
    text_bytes: int
    reasoning_bytes: int
    tool_candidate_source: str | None
    candidate_count: int
    accepted_tool_names: tuple[str, ...]
    rejection_reason: str | None
    repair_attempted: bool
    terminal_result: str
    latency_ms: int


def _emit(event: str, trace: CompatibilityTrace) -> None:
    payload = {
        **asdict(trace),
        "accepted_tool_names": list(trace.accepted_tool_names),
    }
    logger.info(event, {"event": event, "trace": payload})


def emit_attempt_trace(trace: CompatibilityTrace) -> None:
    """Emit a completed model-attempt compatibility record."""
    _emit("compatibility_attempt", trace)


def emit_terminal_trace(trace: CompatibilityTrace) -> None:
    """Emit the compatibility record for the terminal pipeline disposition."""
    _emit("compatibility_terminal", trace)
