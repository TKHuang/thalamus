from __future__ import annotations

import asyncio
import time
from uuid import NAMESPACE_DNS, uuid4, uuid5

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from google.protobuf.unknown_fields import UnknownFieldSet

from config.cursor_client import get_cursor_client_version
from core.bearer_token import extract_bearer_tokens, strip_cursor_user_prefix
from core.cursor_h2_client import send_unary_h2_request
from core.protobuf_builder import (
    compute_sha256_hex_digest,
    generate_obfuscated_machine_id_checksum,
)
from core.token_manager import get_cursor_access_token
from proto import cursor_api_pb2 as pb
from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("model-routes", "DEBUG")
router = APIRouter()

CURSOR_CLIENT_VERSION = get_cursor_client_version()

# AvailableModel carries a repeated string of usable reasoning-effort variant
# ids (e.g. "glm-5.2-high", "glm-5.2-max") in field 36. Our partial .proto only
# defines name/defaultOn/..., so field 36 lands in the unknown-field set.
_MODEL_VARIANT_FIELD = 36

# The model listing proxies a live Cursor RPC that intermittently returns 401
# (ERROR_NOT_LOGGED_IN) and pays a fresh TLS/H2 handshake on every call. Clients
# such as Hermes fetch /v1/models to populate their model picker and fall back
# to a single configured model when the fetch fails or is slow. Cache successful
# results, single-flight concurrent refreshes, and serve a stale list rather
# than erroring so a transient upstream blip never empties the picker.
_MODEL_CACHE_TTL_SECONDS = 300.0
_AVAILABLE_MODELS_MAX_ATTEMPTS = 3
_AVAILABLE_MODELS_RETRY_BACKOFF_SECONDS = 0.3

_model_cache: dict = {"ids": [], "fetched_at": 0.0}
_fetch_lock = asyncio.Lock()


def _extract_variant_model_ids(model) -> list[str]:
    """Read Cursor's effort-suffixed variant ids (field 36) for a model.

    Some models (e.g. glm-5.2) reject their bare name and are only usable via a
    variant, so these must be surfaced in the model listing.
    """
    ids: list[str] = []
    for field in UnknownFieldSet(model):
        if field.field_number == _MODEL_VARIANT_FIELD and field.wire_type == 2:
            try:
                ids.append(field.data.decode("utf-8"))
            except UnicodeDecodeError:
                continue
    return ids


def _collect_model_ids(resp) -> list[str]:
    """Bare model name plus every effort-suffixed variant, deduped, order-preserved.

    Bare names are kept (most work as-is); variants make suffix-only models like
    glm-5.2 selectable.
    """
    model_ids: list[str] = []
    seen: set[str] = set()
    for m in resp.models:
        for model_id in [m.name, *_extract_variant_model_ids(m)]:
            if model_id and model_id not in seen:
                seen.add(model_id)
                model_ids.append(model_id)
    return model_ids


async def _fetch_available_models(token: str) -> list[str]:
    chosen_auth = token.strip()
    checksum = generate_obfuscated_machine_id_checksum(chosen_auth)
    # Match the streaming chat path's header signature (x-client-key /
    # x-session-id / x-request-id). The leaner header set used previously is
    # more prone to Cursor's intermittent ERROR_NOT_LOGGED_IN 401.
    headers = {
        "authorization": f"Bearer {chosen_auth}",
        "connect-protocol-version": "1",
        "content-type": "application/proto",
        "user-agent": "connect-es/1.6.1",
        "x-client-key": compute_sha256_hex_digest(chosen_auth),
        "x-cursor-checksum": checksum,
        "x-cursor-client-version": CURSOR_CLIENT_VERSION,
        "x-cursor-config-version": str(uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-ghost-mode": "true",
        "x-request-id": str(uuid4()),
        "x-session-id": str(uuid5(NAMESPACE_DNS, chosen_auth)),
        "Host": "api2.cursor.sh",
    }

    last_status: int | None = None
    last_buffer = b""
    for attempt in range(1, _AVAILABLE_MODELS_MAX_ATTEMPTS + 1):
        result = await send_unary_h2_request(
            "/aiserver.v1.AiService/AvailableModels", headers, b""
        )
        if result["status"] == 200:
            resp = pb.AvailableModelsResponse()
            resp.ParseFromString(result["buffer"])
            return _collect_model_ids(resp)

        last_status, last_buffer = result["status"], result["buffer"]
        logger.warn(
            f"AvailableModels attempt {attempt}/{_AVAILABLE_MODELS_MAX_ATTEMPTS} "
            f"returned {last_status}"
        )
        if attempt < _AVAILABLE_MODELS_MAX_ATTEMPTS:
            await asyncio.sleep(_AVAILABLE_MODELS_RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError(
        f"Cursor AvailableModels returned {last_status}: {last_buffer[:500]}"
    )


def _models_response(model_ids: list[str]) -> JSONResponse:
    return JSONResponse(content={
        "object": "list",
        "data": [
            {"id": model_id, "object": "model", "created": 0, "owned_by": "cursor"}
            for model_id in model_ids
        ],
    })


async def _list_model_ids(token: str) -> list[str]:
    """Model ids via a TTL cache with single-flight refresh and stale-on-error
    fallback, so a transient upstream failure never empties a client's picker."""
    now = time.monotonic()
    if _model_cache["ids"] and now - _model_cache["fetched_at"] < _MODEL_CACHE_TTL_SECONDS:
        return _model_cache["ids"]

    async with _fetch_lock:
        # Another request may have refreshed the cache while we waited.
        now = time.monotonic()
        if _model_cache["ids"] and now - _model_cache["fetched_at"] < _MODEL_CACHE_TTL_SECONDS:
            return _model_cache["ids"]

        try:
            model_ids = await _fetch_available_models(token)
        except Exception as exc:
            if _model_cache["ids"]:
                logger.warn(
                    f"AvailableModels failed ({exc}); serving "
                    f"{len(_model_cache['ids'])} cached models"
                )
                return _model_cache["ids"]
            raise

        _model_cache["ids"] = model_ids
        _model_cache["fetched_at"] = time.monotonic()
        logger.info(f"AvailableModels returned {len(model_ids)} models")
        return model_ids


def _resolve_cursor_token(authorization_header: str | None) -> str:
    """Resolve the Cursor token for the upstream AvailableModels call.

    Mirror the chat pipeline (pipeline.py): only honor a client-supplied
    bearer when it looks like a real Cursor token (``user_…::`` prefixed or a
    bare ``eyJ`` JWT). Anything else — e.g. a placeholder ``api_key`` a client
    sends purely to satisfy its own "has credentials" gate (Hermes requires
    one to enable live /v1/models discovery) — is ignored in favor of the
    process env token, so a bogus bearer is never forwarded to Cursor (which
    would 401 with ERROR_NOT_LOGGED_IN).
    """
    tokens = extract_bearer_tokens(authorization_header)
    client_token = tokens[0] if tokens else ""
    if client_token and ("::" in client_token or client_token.startswith("eyJ")):
        return strip_cursor_user_prefix(client_token)
    return strip_cursor_user_prefix(get_cursor_access_token())


@router.get("/v1/models")
async def list_models(request: Request):
    """OpenAI-compatible model listing — proxies Cursor's AvailableModels."""
    token = _resolve_cursor_token(request.headers.get("authorization"))

    if not token:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "No auth token", "type": "authentication_error"}},
        )

    try:
        model_ids = await _list_model_ids(token)
    except Exception as exc:
        logger.error(f"AvailableModels failed: {exc}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": str(exc), "type": "upstream_error"}},
        )

    return _models_response(model_ids)


@router.get("/models")
async def list_models_alt(request: Request):
    """Alias without /v1 prefix (legacy compat)."""
    return await list_models(request)
