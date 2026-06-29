from __future__ import annotations
"""
Cursor PKCE login URL generator and auth-poll client.

Cursor uses a PKCE (Proof Key for Code Exchange) flow for browser-based login.
The server cannot obtain a session token directly — it generates a
challenge/verifier pair, directs the user to Cursor's login page, then polls
until the user completes login in the browser.

Port of: cursor_pkce_login_and_access_token_polling_client.js
"""

import hashlib
import secrets
import uuid
from base64 import urlsafe_b64encode

import httpx

from config.cursor_client import get_cursor_client_version
from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("pkce-login", "DEBUG")

CURSOR_LOGIN_BASE = "https://www.cursor.com/loginDeepControl"
CURSOR_AUTH_POLL_URL = "https://api2.cursor.sh/auth/poll"
CURSOR_CLIENT_VERSION = get_cursor_client_version()


def generate_pkce_verifier_and_challenge() -> tuple[str, str]:
    verifier = urlsafe_b64encode(secrets.token_bytes(43)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def build_cursor_login_url(login_uuid: str, challenge: str) -> str:
    return f"{CURSOR_LOGIN_BASE}?challenge={challenge}&uuid={login_uuid}&mode=login"


async def poll_cursor_auth_for_token(
    login_uuid: str, verifier: str, timeout_s: float = 5.0
) -> dict | None:
    """Poll Cursor's auth endpoint once. Returns parsed JSON on success, None if not yet logged in."""
    url = f"{CURSOR_AUTH_POLL_URL}?uuid={login_uuid}&verifier={verifier}"
    headers = {
        "User-Agent": (
            f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            f"(KHTML, like Gecko) Cursor/{CURSOR_CLIENT_VERSION} "
            f"Chrome/132.0.6834.210 Electron/34.3.4 Safari/537.36"
        ),
        "Accept": "*/*",
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("accessToken"):
                    return data
    except Exception:
        pass
    return None


def assemble_cursor_token(data: dict) -> str:
    """Convert poll response into the userId::accessToken format Cursor API expects."""
    access_token = data.get("accessToken", "")
    auth_id = data.get("authId", "")
    if "|" in auth_id:
        user_id = auth_id.split("|")[1]
        return f"{user_id}::{access_token}"
    return access_token
