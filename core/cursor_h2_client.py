from __future__ import annotations
"""
HTTP/2 client for Cursor API (api2.cursor.sh).

Connects to the Cloudflare IP while preserving correct TLS SNI (api2.cursor.sh).
"""

import os
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import httpcore

from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("h2-client", "DEBUG")

CURSOR_CLOUDFLARE_IP: str = os.environ.get("CURSOR_CLOUDFLARE_IP", "104.18.19.125")
CURSOR_API_HOST: str = "api2.cursor.sh"

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
# Cursor may announce a native tool call, then stay silent while generating the
# complete argument payload. Large single-file writes have exceeded 120 seconds
# before arriving as one frame, so streaming reads need a separate budget.
_DEFAULT_STREAM_READ_TIMEOUT_SECONDS = 600.0


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


_STREAMING_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=_positive_float_env(
        "CURSOR_STREAM_READ_TIMEOUT_SECONDS",
        _DEFAULT_STREAM_READ_TIMEOUT_SECONDS,
    ),
    write=120.0,
    pool=120.0,
)


def _build_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.set_alpn_protocols(["h2"])
    return ctx


class _CloudflareOverrideBackend(httpcore.AsyncNetworkBackend):
    """Redirects TCP connections for api2.cursor.sh to the Cloudflare IP
    while keeping the original hostname for TLS SNI."""

    def __init__(self) -> None:
        from httpcore._backends.auto import AutoBackend
        self._inner = AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object | None = None,
    ) -> httpcore.AsyncNetworkStream:
        target = CURSOR_CLOUDFLARE_IP if host == CURSOR_API_HOST else host
        if target != host:
            logger.debug(f"DNS override: {host} -> {target}:{port}")
        return await self._inner.connect_tcp(
            target, port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self, path: str, timeout: float | None = None, socket_options: object | None = None,
    ) -> httpcore.AsyncNetworkStream:
        return await self._inner.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _build_client(
    *,
    timeout: httpx.Timeout = _TIMEOUT,
    base_url: str | None = None,
) -> httpx.AsyncClient:
    """Build an httpx AsyncClient that routes api2.cursor.sh to the Cloudflare IP.

    The URL host stays as api2.cursor.sh so httpx/httpcore use it for TLS SNI.
    The custom network backend intercepts the TCP connect and redirects to the IP.
    """
    ssl_ctx = _build_ssl_context()
    backend = _CloudflareOverrideBackend()

    pool = httpcore.AsyncConnectionPool(
        ssl_context=ssl_ctx,
        http2=True,
        max_connections=10,
        max_keepalive_connections=5,
        network_backend=backend,
    )

    transport = httpx.AsyncHTTPTransport(http2=True, verify=ssl_ctx)
    transport._pool = pool  # noqa: SLF001

    return httpx.AsyncClient(
        transport=transport,
        base_url=base_url or f"https://{CURSOR_API_HOST}",
        timeout=timeout,
    )


@asynccontextmanager
async def open_streaming_h2_request(
    path: str,
    headers: dict[str, str],
    body: bytes,
    *,
    base_url: str | None = None,
) -> AsyncIterator[AsyncIterator[bytes]]:
    """Open a server-streaming HTTP/2 POST to api2.cursor.sh.

    The caller can break out of the iterator early (e.g. after receiving a tool
    call). The context manager will forcefully close the connection rather than
    waiting for the server to finish the stream.
    """
    client_kwargs = {"timeout": _STREAMING_TIMEOUT}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    client = _build_client(**client_kwargs)
    response_ctx = client.stream("POST", path, headers=headers, content=body)
    response = None
    stream_iterator = None
    exit_args = (None, None, None)
    try:
        response = await response_ctx.__aenter__()
        stream_iterator = response.aiter_bytes()
        logger.debug(
            f"Streaming response started: status={response.status_code} path={path}"
        )
        try:
            yield stream_iterator
        except BaseException as exc:
            exit_args = (type(exc), exc, exc.__traceback__)
            raise
    finally:
        if stream_iterator is not None:
            try:
                await stream_iterator.aclose()
            except Exception:
                pass
        if response is not None:
            try:
                await response.aclose()
            except Exception:
                pass
            try:
                await response_ctx.__aexit__(*exit_args)
            except Exception:
                pass
        try:
            await client.aclose()
        except Exception:
            pass


async def send_unary_h2_request(
    path: str,
    headers: dict[str, str],
    body: bytes,
) -> dict:
    """Send a unary (non-streaming) HTTP/2 POST and return the full response."""
    async with _build_client() as client:
        response = await client.post(path, headers=headers, content=body)
        logger.debug(
            f"Unary response: status={response.status_code} path={path} size={len(response.content)}"
        )
        return {"status": response.status_code, "buffer": response.content}
