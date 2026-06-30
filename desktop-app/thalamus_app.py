#!/usr/bin/env python3
"""Self-contained cross-platform desktop shell for Thalamus (pywebview).

The FastAPI backend (server.app) and the UI proxy (launcher_ui) run *in-process*
on background threads; nothing is spawned as an external process. That is what
makes a PyInstaller / Nuitka single-file build fully portable: it needs no repo
checkout, no .venv, and no thalamus_path.conf — the interpreter and every
dependency are baked into the one executable.

Dev usage (backend imported from the repo root, the parent of this directory):

    python desktop-app/thalamus_app.py

Note: the macOS Swift shell (ThalamusApp.swift) still spawns server.py and
launcher_ui.py as separate scripts; this module is the Windows/dev path.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

THALAMUS_PORT = int(os.environ.get("THALAMUS_PORT", "3013"))
UI_PORT = int(os.environ.get("UI_PORT", "3014"))
WIN_W, WIN_H = 440, 600


def _resource_bases() -> list[Path]:
    """Directories to search for bundled resources, most specific first.

    Covers PyInstaller (sys.frozen + sys._MEIPASS), Nuitka onefile and dev
    (the module sits next to its resources). The always-present __file__ base
    is what Nuitka onefile resolves to, so no Nuitka-specific branch is needed.
    """
    bases: list[Path] = []
    if getattr(sys, "frozen", False):
        bases.append(Path(sys.executable).resolve().parent)  # next to the .exe
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bases.append(Path(meipass))                       # PyInstaller onefile
    bases.append(Path(__file__).resolve().parent)             # dev + Nuitka onefile
    return bases


def find_resource(name: str) -> Path:
    for base in _resource_bases():
        candidate = base / name
        if candidate.exists():
            return candidate
    raise SystemExit(f"Bundled resource not found: {name}")


def _ensure_backend_importable() -> None:
    """In a dev checkout the backend lives at the repo root (the parent of this
    directory); put it on sys.path so ``import server`` resolves. In a frozen
    build the backend is bundled, so nothing to do."""
    if getattr(sys, "frozen", False):
        return
    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "server.py").exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.25)
    return False


def acquire_single_instance():
    """Best-effort single-instance lock on Windows; returns a handle to hold."""
    if sys.platform != "win32":
        return None
    import ctypes

    error_already_exists = 183
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, "Global\\ThalamusApp")
    if ctypes.windll.kernel32.GetLastError() == error_already_exists:
        raise SystemExit("Thalamus is already running.")
    return handle  # caller keeps this alive for the lock to persist


def start_backend() -> None:
    """Run the FastAPI app (server.app) with uvicorn on a daemon thread."""
    import uvicorn
    import server  # exposes `app`; its __main__ block is guarded, so import is safe

    config = uvicorn.Config(
        server.app, host="127.0.0.1", port=THALAMUS_PORT, log_level="info"
    )
    uv = uvicorn.Server(config)
    # uvicorn.Server.run() skips installing signal handlers off the main thread,
    # so running it on a worker thread is supported.
    threading.Thread(target=uv.run, name="thalamus-backend", daemon=True).start()


def start_ui() -> None:
    """Run the UI proxy HTTP server on a daemon thread."""
    import launcher_ui

    html = find_resource("index.html")
    threading.Thread(
        target=launcher_ui.serve,
        kwargs={"ui_port": UI_PORT, "thalamus_port": THALAMUS_PORT, "html_path": html},
        name="thalamus-ui",
        daemon=True,
    ).start()


def main() -> None:
    _lock = acquire_single_instance()  # noqa: F841 — held to keep the lock alive

    os.environ.setdefault("THALAMUS_HOST", "127.0.0.1")
    os.environ["PORT"] = str(THALAMUS_PORT)
    os.environ["THALAMUS_PORT"] = str(THALAMUS_PORT)
    os.environ["UI_PORT"] = str(UI_PORT)

    _ensure_backend_importable()
    start_backend()
    start_ui()

    if not wait_for_port("127.0.0.1", UI_PORT):
        raise SystemExit(f"UI server did not start on port {UI_PORT}.")

    # Imported lazily so the helpers above can be exercised without a display.
    import webview

    webview.create_window(
        "Thalamus", f"http://127.0.0.1:{UI_PORT}", width=WIN_W, height=WIN_H
    )
    # Daemon threads exit with the process when the window closes.
    webview.start()


if __name__ == "__main__":
    main()
