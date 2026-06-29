from __future__ import annotations
"""Platform-aware application paths for Thalamus.

Centralizes the writable locations for the persisted token store and the
log directory so the app works whether it runs from the repo (dev) or from
a packaged build installed in a read-only location (e.g. C:\\Program Files
on Windows, /Applications on macOS).

Resolved locations (no override set):
  data dir  -> %APPDATA%\\Thalamus            (Windows)
               ~/Library/Application Support/Thalamus   (macOS)
               ~/.local/share/Thalamus        (Linux)
  log dir   -> %LOCALAPPDATA%\\Thalamus\\Logs  (Windows)
               ~/Library/Logs/Thalamus         (macOS)
               ~/.local/state/Thalamus/log     (Linux)

Both can be overridden via THALAMUS_DATA_DIR / THALAMUS_LOG_DIR for tests
or for keeping everything inside the repo during development.

These functions only *compute* paths; callers are responsible for creating
directories before writing (token_manager and structured_logging both do).
"""

import os
from pathlib import Path

from platformdirs import user_data_dir, user_log_dir

_APP_NAME = "Thalamus"


def data_dir() -> Path:
    """Writable directory for the persisted token `.env` and app state."""
    override = os.environ.get("THALAMUS_DATA_DIR")
    if override:
        return Path(override)
    return Path(user_data_dir(_APP_NAME, appauthor=False))


def token_env_path() -> Path:
    """Path of the `.env` file that holds the auto-managed CURSOR_TOKEN."""
    return data_dir() / ".env"


def log_base_dir() -> Path:
    """Base directory under which per-session structured logs are written."""
    override = os.environ.get("THALAMUS_LOG_DIR")
    if override:
        return Path(override)
    return Path(user_log_dir(_APP_NAME, appauthor=False))
