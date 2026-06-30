#!/usr/bin/env python3
"""Lightweight HTTP server that serves the Thalamus UI and proxies API calls.

Two entry points share the same handler:

  * ``python launcher_ui.py`` — standalone script; ports come from the
    THALAMUS_PORT / UI_PORT env vars. This is how the macOS Swift shell
    (ThalamusApp.swift) spawns it.
  * ``launcher_ui.serve(...)`` — imported in-process by the pywebview shell
    (thalamus_app.py) so a frozen single-file build needs no external Python.
"""

import http.server
import json
import os
import webbrowser
from pathlib import Path
from urllib.request import urlopen, Request

THALAMUS_PORT = int(os.environ.get("THALAMUS_PORT", "3013"))
UI_PORT = int(os.environ.get("UI_PORT", "3014"))
DEFAULT_HTML_PATH = Path(__file__).parent / "index.html"


def _proxy_get(api: str, path: str, timeout: float = 10) -> tuple[int, bytes]:
    try:
        r = urlopen(f"{api}{path}", timeout=timeout)
        return r.status, r.read()
    except Exception as e:
        return 502, json.dumps({"error": str(e)}).encode()


def _proxy_post(
    api: str,
    path: str,
    body: bytes = b"",
    timeout: float = 60,
    extra_headers: dict | None = None,
) -> tuple[int, bytes]:
    try:
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        req = Request(f"{api}{path}", data=body, method="POST", headers=headers)
        r = urlopen(req, timeout=timeout)
        return r.status, r.read()
    except Exception as e:
        return 502, json.dumps({"error": str(e)}).encode()


def build_handler(api_base: str, html_path: Path) -> type[http.server.BaseHTTPRequestHandler]:
    """Create a request handler bound to a backend base URL and an index.html path."""
    html_path = Path(html_path)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, body):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_path.read_bytes())

            elif self.path == "/api/health":
                self._json(*_proxy_get(api_base, "/health"))

            elif self.path == "/api/token_status":
                self._json(*_proxy_get(api_base, "/token/status"))

            elif self.path == "/api/login":
                c, b = _proxy_get(api_base, "/cursor/login")
                # Open the OAuth URL in the system browser and tell the UI whether it
                # worked, so a missing default browser (more common on Windows) shows
                # a "click to open" fallback instead of silently spinning the poll.
                try:
                    d = json.loads(b)
                    if d.get("url"):
                        d["browser_opened"] = bool(webbrowser.open(d["url"]))
                        b = json.dumps(d).encode()
                except Exception:
                    pass
                self._json(c, b)

            elif self.path.startswith("/api/poll"):
                qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                self._json(*_proxy_get(api_base, f"/cursor/poll?{qs}"))

            elif self.path == "/api/models":
                self._json(*_proxy_get(api_base, "/v1/models", timeout=15))

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else b""

            if self.path == "/api/clear":
                self._json(*_proxy_post(api_base, "/token/clear"))

            elif self.path == "/api/messages":
                self._json(*_proxy_post(
                    api_base, "/v1/messages", body=body, timeout=60,
                    extra_headers={"anthropic-version": "2023-06-01"},
                ))

            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def serve(
    ui_port: int | None = None,
    thalamus_port: int | None = None,
    html_path: Path | str | None = None,
) -> None:
    """Serve the UI forever. Blocks; run on a thread when embedding in-process."""
    ui_port = UI_PORT if ui_port is None else ui_port
    thalamus_port = THALAMUS_PORT if thalamus_port is None else thalamus_port
    html_path = DEFAULT_HTML_PATH if html_path is None else html_path

    api_base = f"http://127.0.0.1:{thalamus_port}"
    handler = build_handler(api_base, Path(html_path))
    srv = http.server.HTTPServer(("127.0.0.1", ui_port), handler)
    print(f"UI server on http://127.0.0.1:{ui_port}")
    srv.serve_forever()


if __name__ == "__main__":
    serve()
