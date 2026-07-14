from __future__ import annotations
"""
LLM API payload logger.

When THALAMUS_RAW_PAYLOAD_LOGGING=true, Cursor API calls get:
  1. A request payload file:  logs/<session>/<date>/pipeline/payloads/req_<id>_request.json
  2. A response payload file: logs/<session>/<date>/pipeline/payloads/req_<id>_response.json
  3. A one-line entry in:     logs/<session>/<date>/pipeline/llm-api-calls.log
     that references the absolute paths of both payload files.

All timestamps use Beijing time (UTC+8).
"""

import json
import os
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
    d = os.path.join(session_dir, today, "pipeline", "payloads")
    os.makedirs(d, exist_ok=True)
    return d


def _api_log_path() -> str:
    session_dir = ThalamusStructuredLogger.session_log_dir()
    today = _beijing_date_str()
    d = os.path.join(session_dir, today, "pipeline")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "llm-api-calls.log")


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


def log_llm_request(
    request_id: str,
    model: str,
    messages: list[dict],
    extra: dict | None = None,
) -> str:
    """Write request payload to file when raw payload logging is enabled."""
    if not is_raw_payload_logging_enabled():
        return ""
    payload_dir = _payload_dir()
    filename = f"{request_id}_request.json"
    filepath = os.path.join(payload_dir, filename)

    payload = {
        "timestamp": _beijing_timestamp(),
        "request_id": request_id,
        "model": model,
        "message_count": len(messages),
        "messages": messages,
    }
    if extra:
        payload["extra"] = extra

    _write_json(filepath, payload)
    return os.path.abspath(filepath)


def log_llm_response(
    request_id: str,
    model: str,
    response_text: str,
    tool_calls: list | None = None,
    error: str | None = None,
    latency_ms: int | None = None,
    extra: dict | None = None,
) -> str:
    """Write response payload to file when raw payload logging is enabled."""
    if not is_raw_payload_logging_enabled():
        return ""
    payload_dir = _payload_dir()
    filename = f"{request_id}_response.json"
    filepath = os.path.join(payload_dir, filename)

    payload = {
        "timestamp": _beijing_timestamp(),
        "request_id": request_id,
        "model": model,
        "latency_ms": latency_ms,
        "response_text_length": len(response_text) if response_text else 0,
        "response_text": response_text,
        "tool_calls": tool_calls,
        "error": error,
    }
    if extra:
        payload["extra"] = extra

    _write_json(filepath, payload)
    return os.path.abspath(filepath)


def log_llm_api_call(
    request_id: str,
    model: str,
    status: str,
    latency_ms: int | None,
    request_payload_path: str,
    response_payload_path: str,
    error: str | None = None,
):
    """Append raw payload-file references when raw payload logging is enabled."""
    if not is_raw_payload_logging_enabled():
        return
    ts = _beijing_timestamp()
    err_part = f" error={error}" if error else ""
    line = (
        f"{ts} | {request_id} | model={model} | status={status} | "
        f"latency={latency_ms}ms | "
        f"req={request_payload_path} | "
        f"res={response_payload_path}"
        f"{err_part}"
    )
    _append_line(_api_log_path(), line)
