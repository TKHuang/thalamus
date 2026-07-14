"""Feature flag for opt-in raw request and response payload persistence."""

from __future__ import annotations

import os


def is_raw_payload_logging_enabled() -> bool:
    """Return whether raw payload files may be persisted for local diagnostics."""
    return os.getenv("THALAMUS_RAW_PAYLOAD_LOGGING", "").strip().lower() == "true"
