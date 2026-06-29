from __future__ import annotations
"""
Single source of truth for the Cursor desktop-client version Thalamus
impersonates toward api2.cursor.sh.

Cursor's backend rejects deprecated client versions with
"Your version of Cursor is no longer supported." When that happens, set
CURSOR_CLIENT_VERSION in .env to a currently-supported Cursor release
(match the installed app's CFBundleShortVersionString) — no code change needed.
"""

import os

# Fallback used only when the CURSOR_CLIENT_VERSION env var is unset.
DEFAULT_CURSOR_CLIENT_VERSION = "3.9.16"


def get_cursor_client_version() -> str:
    """Resolve the Cursor client version: env override, else the default."""
    return os.environ.get("CURSOR_CLIENT_VERSION", DEFAULT_CURSOR_CLIENT_VERSION)
