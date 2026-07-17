from __future__ import annotations
"""
Multi-dimensional structured logging provider for thalamus-py.

Log files are partitioned along four axes:
  logs/<sessionTimestamp>/<YYYY-MM-DD>/<layer>/<topic>.log

- sessionTimestamp: created once when the process starts (isolates restarts).
- YYYY-MM-DD: rotates at midnight Beijing time.
- layer: logical subsystem (server, routes, core, auth, utils, pipeline).
- topic: module-specific channel.

All timestamps use Asia/Shanghai (UTC+8).

Port of: thalamus/src/utils/thalamus_structured_logging_provider.js
"""

import os
import sys
import atexit
from datetime import datetime, timezone, timedelta

from core.paths import log_base_dir

BEIJING_TZ = timezone(timedelta(hours=8))

LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}

ANSI = {
    "reset": "\x1b[0m",
    "bright": "\x1b[1m",
    "dim": "\x1b[2m",
    "gray": "\x1b[90m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}

LEVEL_COLORS = {
    "DEBUG": ANSI["gray"],
    "INFO": ANSI["cyan"],
    "WARN": ANSI["yellow"],
    "ERROR": ANSI["red"],
}

LOGGER_NAME_TO_LAYER_AND_TOPIC: dict[str, dict[str, str]] = {
    "server":             {"layer": "server",   "topic": "lifecycle"},
    "pipeline":           {"layer": "pipeline", "topic": "claude-code"},
    "anthropic-messages": {"layer": "routes",   "topic": "anthropic-messages"},
    "token-routes":       {"layer": "routes",   "topic": "token-routes"},
    "login-routes":       {"layer": "auth",     "topic": "browser-login"},
    "token-manager":      {"layer": "auth",     "topic": "token-manager"},
    "bearer-token":       {"layer": "auth",     "topic": "bearer-extractor"},
    "pkce-login":         {"layer": "auth",     "topic": "pkce-login"},
    "h2-client":          {"layer": "core",     "topic": "http2-client"},
    "agent-client":       {"layer": "core",     "topic": "agent-client"},
    "protobuf-parser":    {"layer": "core",     "topic": "protobuf-parser"},
    "protobuf-builder":   {"layer": "core",     "topic": "protobuf-builder"},
    "tool-parser":        {"layer": "pipeline", "topic": "tool-parser"},
    "tool-prompt":        {"layer": "pipeline", "topic": "tool-prompt-builder"},
    "sse-assembler":      {"layer": "pipeline", "topic": "sse-assembler"},
    "llm-payload":        {"layer": "pipeline", "topic": "llm-payload-logger"},
    "thalamus-api":       {"layer": "routes",   "topic": "thalamus-api"},
    "fallback":           {"layer": "fallback", "topic": "fallback-config"},
    "tool-registry":      {"layer": "utils",    "topic": "tool-registry"},
    "tool-mapping":       {"layer": "utils",    "topic": "tool-mapping"},
}


def _resolve_layer_and_topic(name: str) -> tuple[str, str]:
    if name in LOGGER_NAME_TO_LAYER_AND_TOPIC:
        entry = LOGGER_NAME_TO_LAYER_AND_TOPIC[name]
        return entry["layer"], entry["topic"]
    if "route" in name or "api" in name:
        return "routes", name
    if "fallback" in name or "cooldown" in name:
        return "fallback", name
    if "cursor" in name or "h2" in name or "proto" in name:
        return "core", name
    if "token" in name or "login" in name or "auth" in name:
        return "auth", name
    if "pipeline" in name or "tool" in name or "sse" in name:
        return "pipeline", name
    return "misc", name


def _beijing_now() -> datetime:
    return datetime.now(BEIJING_TZ)


def _beijing_timestamp() -> str:
    return _beijing_now().strftime("%Y-%m-%d %H:%M:%S")


def _beijing_date_str() -> str:
    return _beijing_now().strftime("%Y-%m-%d")


class StructuredLogWriter:
    """One writer per (layer, topic) channel. Rotates at midnight Beijing time."""

    def __init__(self, name: str, layer: str, topic: str, level: str = "INFO"):
        self.name = name
        self.layer = layer
        self.topic = topic
        self.level = LEVELS.get(level.upper(), LEVELS["INFO"])
        self._current_date: str | None = None
        self._file = None
        self._file_path: str | None = None

    def _ensure_file(self):
        today = _beijing_date_str()
        if self._current_date == today and self._file:
            return
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass

        session_dir = ThalamusStructuredLogger.session_log_dir()
        day_dir = os.path.join(session_dir, today, self.layer)
        os.makedirs(day_dir, exist_ok=True)

        self._file_path = os.path.join(day_dir, f"{self.topic}.log")
        self._file = open(self._file_path, "a", encoding="utf-8")
        self._current_date = today

    def _log(self, level: str, message: str, meta=None):
        if LEVELS.get(level, 0) < self.level:
            return

        ts = _beijing_timestamp()
        meta_str = ""
        if meta is not None:
            if isinstance(meta, Exception):
                import traceback
                meta_str = "\n" + "".join(traceback.format_exception(type(meta), meta, meta.__traceback__))
            elif isinstance(meta, dict):
                import json
                try:
                    meta_str = " " + json.dumps(meta, ensure_ascii=False, default=str)
                except Exception:
                    meta_str = f" {meta}"
            else:
                meta_str = f" {meta}"

        file_line = f"{ts} - {self.name} - {level} - {message}{meta_str}"

        color = LEVEL_COLORS.get(level, ANSI["reset"])
        short_time = ts[11:]
        console_line = (
            f"{ANSI['gray']}{short_time}{ANSI['reset']} "
            f"{ANSI['bright']}[{self.name}]{ANSI['reset']} "
            f"{color}{level}{ANSI['reset']}: {message}"
        )
        print(console_line, file=sys.stderr)

        try:
            self._ensure_file()
            self._file.write(file_line + "\n")
            self._file.flush()
        except Exception:
            pass

    def debug(self, message: str, meta=None):
        self._log("DEBUG", message, meta)

    def info(self, message: str, meta=None):
        self._log("INFO", message, meta)

    def warn(self, message: str, meta=None):
        self._log("WARN", message, meta)

    def error(self, message: str, meta=None):
        self._log("ERROR", message, meta)

    def close(self):
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


class ThalamusStructuredLogger:
    _loggers: dict[str, StructuredLogWriter] = {}
    _log_base_dir = str(log_base_dir())
    _session_start = _beijing_now()
    _session_dir: str | None = None

    @classmethod
    def session_log_dir(cls) -> str:
        if cls._session_dir:
            return cls._session_dir
        ts = cls._session_start.strftime("%Y-%m-%d_%H-%M-%S")
        cls._session_dir = os.path.join(cls._log_base_dir, ts)
        os.makedirs(cls._session_dir, exist_ok=True)
        return cls._session_dir

    @classmethod
    def get_logger(cls, name: str, level: str = "INFO") -> StructuredLogWriter:
        if name in cls._loggers:
            return cls._loggers[name]
        layer, topic = _resolve_layer_and_topic(name)
        writer = StructuredLogWriter(name, layer, topic, level)
        cls._loggers[name] = writer
        return writer

    @classmethod
    def close_all(cls):
        for writer in cls._loggers.values():
            writer.close()


atexit.register(ThalamusStructuredLogger.close_all)
