from __future__ import annotations
"""
Cursor Access Token Manager (Singleton).

Manages Cursor authentication tokens across the Thalamus process.

Token sources (checked in priority order):
  1. In-memory store (set via API or auto-captured from requests)
  2. CURSOR_TOKEN environment variable (loaded by dotenv at startup)

Auto-capture: Every incoming request with a valid Bearer token is
captured and stored so that internal subsystems can make loopback
calls without the original request.
"""

import os
import re
from datetime import datetime, timezone
from urllib.parse import unquote

from core.paths import token_env_path
from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("token-manager", "DEBUG")

# Persisted in a platform-writable data dir (see core/paths). Loaded back at
# startup by server.py via load_dotenv(override=True), so the read path and
# this write path always resolve to the same file across macOS/Windows.
ENV_FILE_PATH = token_env_path()

_store: dict = {
    'token': '',
    'source': 'none',
    'captured_at': None,
    'last_used_at': None,
    'requests_captured_from': 0,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_from_environment() -> None:
    env_token = os.environ.get('CURSOR_TOKEN', '')
    if env_token and not _store['token']:
        _store['token'] = env_token
        _store['source'] = 'env'
        _store['captured_at'] = _now_iso()
        logger.info(f'Token loaded from env | length={len(env_token)} | prefix={env_token[:20]}...')


def get_cursor_access_token() -> str:
    if _store['token']:
        _store['last_used_at'] = _now_iso()
    return _store['token']


def set_cursor_access_token(new_token: str | None, source: str = 'api') -> None:
    prev = _store['token']
    _store['token'] = new_token or ''
    _store['source'] = source if new_token else 'cleared'
    _store['captured_at'] = _now_iso() if new_token else None

    if new_token:
        logger.info(f'Token set | source={source} | length={len(new_token)} | prefix={new_token[:20]}...')
    else:
        logger.info('Token cleared')

    if new_token and new_token != prev:
        _persist_to_dot_env(new_token)


def has_cursor_access_token() -> bool:
    return len(_store['token']) > 0


def get_token_status() -> dict:
    token = _store['token']
    return {
        'has_token': len(token) > 0,
        'token_length': len(token),
        'token_preview': f'{token[:20]}...{token[-10:]}' if token else '',
        'source': _store['source'],
        'captured_at': _store['captured_at'],
        'last_used_at': _store['last_used_at'],
        'requests_captured_from': _store['requests_captured_from'],
    }


def capture_token_from_request(authorization_header: str | None) -> None:
    if not authorization_header:
        return

    m = re.match(r'^Bearer\s+(.+)$', authorization_header, re.IGNORECASE)
    if not m:
        return

    parts = [s.strip() for s in m.group(1).split(',') if s.strip()]
    if not parts:
        return

    try:
        token = unquote(parts[0])
    except Exception:
        token = parts[0]
    token = re.sub(r'%3A%3A', '::', token, flags=re.IGNORECASE)

    if len(token) < 100:
        return

    looks_like_cursor_token = '::' in token or token.startswith('eyJ')
    if not looks_like_cursor_token:
        return

    _store['requests_captured_from'] += 1

    if token != _store['token']:
        old_len = len(_store['token'])
        _store['token'] = token
        _store['source'] = 'auto-capture'
        _store['captured_at'] = _now_iso()

        if old_len == 0:
            logger.info(f'Token auto-captured from request | length={len(token)} | prefix={token[:20]}...')
        else:
            logger.info(f'Token auto-refreshed from request | length={len(token)} (was {old_len})')

        _persist_to_dot_env(token)


def _persist_to_dot_env(token: str) -> None:
    try:
        env_content = ''
        ENV_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ENV_FILE_PATH.exists():
            env_content = ENV_FILE_PATH.read_text(encoding='utf-8')

        token_regex = re.compile(r'^CURSOR_TOKEN=.*$', re.MULTILINE)
        new_token_line = f'CURSOR_TOKEN={token}'

        if token_regex.search(env_content):
            env_content = token_regex.sub(new_token_line, env_content)
        else:
            if env_content and not env_content.endswith('\n'):
                env_content += '\n'
            env_content += f'\n# Cursor authentication token (auto-managed by Thalamus)\n{new_token_line}\n'

        ENV_FILE_PATH.write_text(env_content, encoding='utf-8')
        logger.info(f'Token persisted to .env | length={len(token)}')
    except Exception:
        logger.exception('Failed to persist token to .env')


_init_from_environment()
