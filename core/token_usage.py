from __future__ import annotations

import json
import math
from typing import Any


def estimate_input_tokens(messages: list[dict], tools: list[dict]) -> int:
    """Estimate tokens from the complete outbound prompt and tool inventory."""
    payload: dict[str, Any] = {"messages": messages}
    if tools:
        payload["tools"] = tools
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized.encode("utf-8")) / 4))


def input_tokens_from_remaining_context(
    context_length: int | None,
    remaining_percent: float | None,
    fallback: int,
) -> int:
    """Convert Cursor's server-reported remaining-context percentage to used tokens."""
    if not context_length or context_length <= 0:
        return max(1, fallback)
    if remaining_percent is None or not 0.0 <= remaining_percent <= 100.0:
        return max(1, fallback)
    used_percent = 100.0 - remaining_percent
    return max(1, math.ceil(context_length * used_percent / 100.0))
