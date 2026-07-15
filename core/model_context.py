from __future__ import annotations

"""Shared model-id syntax and routing for explicit Cursor context modes."""

import re


MODEL_CONTEXT_MARKER_RE = re.compile(
    r"\[(?P<amount>\d+(?:\.\d+)?)(?P<unit>[kmg])\]",
    re.IGNORECASE,
)

# Replaced atomically whenever AvailableModels refreshes successfully.  Models
# not present here retain the legacy request builder's automatic behavior.
_dual_context_model_ids: frozenset[str] = frozenset()
_model_context_catalog: dict[str, dict[str, int]] = {}


def context_marker_for_length(context_length: int) -> str | None:
    """Format an exact compact marker such as ``1m`` or ``300k``."""
    if context_length <= 0:
        return None
    for divisor, suffix in (
        (1_000_000_000, "g"),
        (1_000_000, "m"),
        (1_000, "k"),
    ):
        if context_length % divisor == 0:
            return f"{context_length // divisor}{suffix}"
    return None


def add_context_marker(model_id: str, context_length: int) -> str | None:
    marker = context_marker_for_length(context_length)
    if marker is None or MODEL_CONTEXT_MARKER_RE.search(model_id):
        return None
    return f"{model_id}[{marker}]"


def strip_context_marker(model_id: str) -> str:
    return MODEL_CONTEXT_MARKER_RE.sub("", model_id)


def get_model_context_length(model_id: str) -> int | None:
    """Return the live context limit advertised for a model, when available."""
    values = _model_context_catalog.get(model_id)
    if values is None:
        values = _model_context_catalog.get(strip_context_marker(model_id))
    if values is None:
        return None

    if MODEL_CONTEXT_MARKER_RE.search(model_id):
        maximum = values.get("max_context_length")
        if isinstance(maximum, int) and maximum > 0:
            return maximum

    context_length = values.get("context_length")
    return context_length if isinstance(context_length, int) and context_length > 0 else None


def replace_model_context_catalog(metadata: dict[str, dict[str, int]]) -> None:
    """Register models whose normal and Max context modes are distinct."""
    global _dual_context_model_ids, _model_context_catalog
    _model_context_catalog = {
        model_id: dict(values)
        for model_id, values in metadata.items()
        if isinstance(values, dict)
    }
    dual_context_ids: set[str] = set()
    for model_id, values in metadata.items():
        if MODEL_CONTEXT_MARKER_RE.search(model_id):
            continue
        normal = values.get("context_length")
        maximum = values.get("max_context_length")
        if (
            isinstance(normal, int)
            and isinstance(maximum, int)
            and normal > 0
            and maximum > 0
            and normal != maximum
        ):
            dual_context_ids.add(model_id)
    _dual_context_model_ids = frozenset(dual_context_ids)


def resolve_model_context_mode(model_id: str) -> tuple[str, bool | None]:
    """Return upstream model id and an explicit large-context choice.

    ``True`` means a synthetic ``[1m]``-style id explicitly selected Max
    Context. ``False`` means a known dual-context model explicitly selected
    its normal context by omitting the marker. ``None`` preserves the legacy
    size-based behavior for models without two advertised context modes.
    """
    has_marker = MODEL_CONTEXT_MARKER_RE.search(model_id) is not None
    upstream_model_id = strip_context_marker(model_id)
    if has_marker:
        return upstream_model_id, True
    if upstream_model_id in _dual_context_model_ids:
        return upstream_model_id, False
    return upstream_model_id, None
