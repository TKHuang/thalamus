#!/usr/bin/env python3
"""Cross-platform desktop shell for Thalamus (pywebview).

This is the Windows equivalent of ThalamusApp.swift: it spawns the FastAPI
backend (server.py) and the UI proxy (launcher_ui.py) as child processes,
waits for the UI server to come up, then opens a native window pointed at it.
On Windows pywebview renders with WebView2/Edge-Chromium; on macOS it uses
WebKit (the same engine the Swift shell wraps), so it also runs there for dev.

Run from a checkout with a populated .venv:

    python desktop-app/thalamus_app.py

Resource resolution works both in dev (files sit in desktop-app/) and when
frozen by PyInstaller (files are added via --add-data; the repo path is read
from thalamus_path.conf written next to the executable at build time).
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

THALAMUS_PORT = int(os.environ.get("THALAMUS_PORT", "3013"))
UI_PORT = int(os.environ.get("UI_PORT", "3014"))
WIN_W, WIN_H = 440, 600

# CREATE_NO_WINDOW: stop child python processes flashing a console on Windows.
_CREATE_NO_WINDOW = 0x08000000


def _resource_bases() -> list[Path]:
    """Directories to search for bundled resources / config, most specific first."""
    bases: list[Path] = []
    if getattr(sys, "frozen", False):
        bases.append(Path(sys.executable).resolve().parent)  # onedir: next to .exe
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bases.append(Path(meipass))                       # onefile / _internal
    bases.append(Path(__file__).resolve().parent)             # dev: desktop-app/
    return bases


def find_resource(name: str) -> Path:
    for base in _resource_bases():
        candidate = base / name
        if candidate.exists():
            return candidate
    raise SystemExit(f"Bundled resource not found: {name}")


def resolve_repo_dir() -> Path:
    """Locate the repo root that holds server.py (it runs from there, with .venv)."""
    env_dir = os.environ.get("THALAMUS_DIR")
    if env_dir and (Path(env_dir) / "server.py").exists():
        return Path(env_dir).resolve()

    for base in _resource_bases():
        conf = base / "thalamus_path.conf"
        if conf.exists():
            repo = Path(conf.read_text(encoding="utf-8").strip())
            if (repo / "server.py").exists():
                return repo.resolve()

    # Dev fallback: repo root is the parent of desktop-app/.
    parent = Path(__file__).resolve().parent.parent
    if (parent / "server.py").exists():
        return parent

    raise SystemExit(
        "Could not locate the Thalamus repo (server.py). "
        "Set THALAMUS_DIR or provide desktop-app/thalamus_path.conf."
    )


def find_python(repo_dir: Path) -> str:
    """Find the interpreter that runs the backend — prefer the repo's venv."""
    candidates = [
        repo_dir / ".venv" / "Scripts" / "python.exe",  # Windows venv
        repo_dir / ".venv" / "bin" / "python3",          # POSIX venv
        repo_dir / "venv" / "Scripts" / "python.exe",
        repo_dir / "venv" / "bin" / "python3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # In dev (unfrozen) the current interpreter is usually the venv one already.
    if not getattr(sys, "frozen", False):
        return sys.executable

    for name in ("python3", "python", "py"):
        found = shutil.which(name)
        if found:
            return found

    raise SystemExit("No Python interpreter found. Create a .venv in the repo.")


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


def _popen(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.Popen:
    kwargs: dict = {"cwd": str(cwd), "env": env}
    if sys.platform == "win32":
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    return subprocess.Popen(args, **kwargs)


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


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


def start_servers(repo_dir: Path, python: str) -> list[subprocess.Popen]:
    launcher = find_resource("launcher_ui.py")

    env = dict(os.environ)
    env["PORT"] = str(THALAMUS_PORT)
    env["THALAMUS_HOST"] = "127.0.0.1"
    env["THALAMUS_PORT"] = str(THALAMUS_PORT)
    env["UI_PORT"] = str(UI_PORT)

    backend = _popen([python, "server.py"], cwd=repo_dir, env=env)
    ui = _popen([python, str(launcher)], cwd=repo_dir, env=env)
    return [backend, ui]


def main() -> None:
    _lock = acquire_single_instance()  # noqa: F841 — held to keep the lock alive

    repo_dir = resolve_repo_dir()
    python = find_python(repo_dir)
    procs = start_servers(repo_dir, python)

    if not wait_for_port("127.0.0.1", UI_PORT):
        for proc in procs:
            terminate(proc)
        raise SystemExit(f"UI server did not start on port {UI_PORT}.")

    # Imported lazily so the helpers above can be exercised without a display.
    import webview

    window = webview.create_window(
        "Thalamus", f"http://127.0.0.1:{UI_PORT}", width=WIN_W, height=WIN_H
    )

    def _cleanup() -> None:
        for proc in procs:
            terminate(proc)

    window.events.closed += _cleanup
    try:
        webview.start()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
