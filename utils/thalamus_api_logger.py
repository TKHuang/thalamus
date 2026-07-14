from __future__ import annotations
"""
Thalamus external API logger.

When THALAMUS_RAW_PAYLOAD_LOGGING=true, logs Claude CLI → thalamus requests and responses:
  1. Request payload file:  logs/<session>/<date>/thalamus-api/payloads/<request_id>_request.json
  2. Response payload file: logs/<session>/<date>/thalamus-api/payloads/<request_id>_response.json
  3. One-line entry in:     logs/<session>/<date>/thalamus-api/api-calls.log

Covers:
  - POST /v1/messages (request body + SSE event sequence or JSON body)
  - POST /v1/messages/count_tokens (request + response)

All timestamps use Beijing time (UTC+8).
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta

from utils.raw_payload_logging import is_raw_payload_logging_enabled
from utils.structured_logging import ThalamusStructuredLogger

BEIJING_TZ = timezone(timedelta(hours=8))


def _beijing_now() -> datetime:
    return datetime.now(BEIJING_TZ)


def _beijing_timestamp() -> str:
    return _beijing_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _beijing_date_str() -> str:
    return _beijing_now().strftime("%Y-%m-%d")


def _payload_dir() -> str:
    session_dir = ThalamusStructuredLogger.session_log_dir()
    today = _beijing_date_str()
    d = os.path.join(session_dir, today, "thalamus-api", "payloads")
    os.makedirs(d, exist_ok=True)
    return d


def _api_log_path() -> str:
    session_dir = ThalamusStructuredLogger.session_log_dir()
    today = _beijing_date_str()
    d = os.path.join(session_dir, today, "thalamus-api")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "api-calls.log")


def _write_json(filepath: str, data) -> None:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass


def _append_line(filepath: str, line: str) -> None:
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def parse_sse_to_events(raw_sse: str) -> list[dict[str, str]]:
    """Parse raw SSE string into list of {event, data} dicts."""
    events: list[dict[str, str]] = []
    if not raw_sse or not raw_sse.strip():
        return events
    blocks = re.split(r"\n\n+", raw_sse.strip())
    for block in blocks:
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if event_name or data_lines:
            data_str = "\n".join(data_lines) if data_lines else ""
            events.append({"event": event_name or "(unknown)", "data": data_str})
    return events


def log_thalamus_request(
    request_id: str,
    endpoint: str,
    method: str,
    payload: dict,
    headers_summary: dict | None = None,
) -> str:
    """Write request payload to file when raw payload logging is enabled."""
    if not is_raw_payload_logging_enabled():
        return ""
    payload_dir = _payload_dir()
    filename = f"{request_id}_request.json"
    filepath = os.path.join(payload_dir, filename)

    data: dict = {
        "timestamp": _beijing_timestamp(),
        "request_id": request_id,
        "endpoint": endpoint,
        "method": method,
        "payload": payload,
    }
    if headers_summary:
        data["headers_summary"] = headers_summary

    _write_json(filepath, data)
    return os.path.abspath(filepath)


def log_thalamus_response(
    request_id: str,
    endpoint: str,
    status_code: int,
    sse_events_or_body: list[dict[str, str]] | dict,
    latency_ms: int | None = None,
) -> str:
    """Write response payload to file when raw payload logging is enabled.

    sse_events_or_body: For SSE: list of {"event": str, "data": str}. For JSON: response body dict.
    """
    if not is_raw_payload_logging_enabled():
        return ""
    payload_dir = _payload_dir()
    filename = f"{request_id}_response.json"
    filepath = os.path.join(payload_dir, filename)

    data: dict = {
        "timestamp": _beijing_timestamp(),
        "request_id": request_id,
        "endpoint": endpoint,
        "status_code": status_code,
        "latency_ms": latency_ms,
    }
    if isinstance(sse_events_or_body, list):
        data["sse_events"] = sse_events_or_body
        data["event_count"] = len(sse_events_or_body)
    else:
        data["body"] = sse_events_or_body

    _write_json(filepath, data)
    return os.path.abspath(filepath)


def log_thalamus_api_call(
    request_id: str,
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: int | None,
    req_path: str,
    res_path: str,
    error: str | None = None,
) -> None:
    """Append raw payload-file references when raw payload logging is enabled."""
    if not is_raw_payload_logging_enabled():
        return
    ts = _beijing_timestamp()
    err_part = f" error={error}" if error else ""
    line = (
        f"{ts} | {request_id} | {method} {endpoint} | status={status_code} | "
        f"latency={latency_ms}ms | "
        f"req={req_path} | "
        f"res={res_path}"
        f"{err_part}"
    )
    _append_line(_api_log_path(), line)
