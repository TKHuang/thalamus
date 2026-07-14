from __future__ import annotations

import asyncio
import re
import time
from decimal import Decimal, InvalidOperation
from uuid import NAMESPACE_DNS, uuid4, uuid5

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from google.protobuf.empty_pb2 import Empty
from google.protobuf.message import DecodeError
from google.protobuf.unknown_fields import UnknownFieldSet

from config.cursor_client import get_cursor_client_version
from core.bearer_token import extract_bearer_tokens, strip_cursor_user_prefix
from core.cursor_h2_client import send_unary_h2_request
from core.model_context import (
    MODEL_CONTEXT_MARKER_RE,
    add_context_marker,
    replace_model_context_catalog,
    strip_context_marker,
)
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

# Cursor 3.11.x defines explicit numeric context limits in fields 15/16, but
# the live service currently leaves those fields unset and publishes the same
# information in TooltipData.markdown_content (field 7) instead.  Read both:
# numeric fields win once Cursor starts populating them; the live tooltip keeps
# current and future models dynamic without a locally-maintained model map.
_MODEL_TOOLTIP_FIELD = 8
_MODEL_TOOLTIP_FOR_MAX_MODE_FIELD = 20
_MODEL_CONTEXT_TOKEN_LIMIT_FIELD = 15
_MODEL_CONTEXT_TOKEN_LIMIT_FOR_MAX_MODE_FIELD = 16
_TOOLTIP_MARKDOWN_FIELD = 7
_CONTEXT_WINDOW_RE = re.compile(
    r"\b(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>[kmg])"
    r"(?:\s+token)?\s+context\s+window\b",
    re.IGNORECASE,
)
# The model listing proxies a live Cursor RPC that intermittently returns 401
# (ERROR_NOT_LOGGED_IN) and pays a fresh TLS/H2 handshake on every call. Clients
# such as Hermes fetch /v1/models to populate their model picker and fall back
# to a single configured model when the fetch fails or is slow. Cache successful
# results, single-flight concurrent refreshes, and serve a stale list rather
# than erroring so a transient upstream blip never empties the picker.
_MODEL_CACHE_TTL_SECONDS = 300.0
_AVAILABLE_MODELS_MAX_ATTEMPTS = 3
_AVAILABLE_MODELS_RETRY_BACKOFF_SECONDS = 0.3

_model_cache: dict = {"ids": [], "metadata": {}, "fetched_at": 0.0}
_fetch_lock = asyncio.Lock()


def _extract_variant_model_ids(model) -> list[str]:
    """Read Cursor's effort-suffixed variant ids (field 36) for a model.

    Some models (e.g. glm-5.2) reject their bare name and are only usable via a
    variant, so these must be surfaced in the model listing.
    """
    ids: list[str] = []
    # Prefer named fields if cursor_api.proto is updated later.
    for attr in ("legacySlugs", "legacy_slugs"):
        values = getattr(model, attr, None)
        if values:
            ids.extend(value for value in values if value)
            return ids

    for field in UnknownFieldSet(model):
        if field.field_number == _MODEL_VARIANT_FIELD and field.wire_type == 2:
            try:
                ids.append(field.data.decode("utf-8"))
            except UnicodeDecodeError:
                continue
    return ids


def _scaled_token_count(amount: str, unit: str) -> int | None:
    multipliers = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    try:
        value = int(Decimal(amount) * multipliers[unit.lower()])
    except (InvalidOperation, KeyError, ValueError):
        return None
    return value if value > 0 else None


def _context_length_from_text(text: str) -> int | None:
    match = _CONTEXT_WINDOW_RE.search(text)
    if match is None:
        return None
    return _scaled_token_count(match.group("amount"), match.group("unit"))


def _known_numeric_field(model, attrs: tuple[str, ...]) -> int | None:
    for attr in attrs:
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _unknown_varint_field(model, field_number: int) -> int | None:
    for field in UnknownFieldSet(model):
        if field.field_number == field_number and field.wire_type == 0 and field.data > 0:
            return int(field.data)
    return None


def _known_tooltip_context(model, attrs: tuple[str, ...]) -> int | None:
    for attr in attrs:
        tooltip = getattr(model, attr, None)
        if tooltip is None:
            continue
        markdown = getattr(tooltip, "markdownContent", None)
        if markdown is None:
            markdown = getattr(tooltip, "markdown_content", "")
        context_length = _context_length_from_text(markdown or "")
        if context_length is not None:
            return context_length
    return None


def _unknown_tooltip_context(model, field_number: int) -> int | None:
    """Decode TooltipData.markdown_content while our checked-in proto is partial."""
    for field in UnknownFieldSet(model):
        if field.field_number != field_number or field.wire_type != 2:
            continue
        tooltip = Empty()
        try:
            tooltip.ParseFromString(field.data)
        except DecodeError:
            continue
        for nested in UnknownFieldSet(tooltip):
            if nested.field_number != _TOOLTIP_MARKDOWN_FIELD or nested.wire_type != 2:
                continue
            try:
                markdown = nested.data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            context_length = _context_length_from_text(markdown)
            if context_length is not None:
                return context_length
    return None


def _extract_model_context_metadata(model) -> dict[str, int]:
    """Extract live context limits from Cursor's AvailableModel metadata."""
    context_length = (
        _known_numeric_field(model, ("contextTokenLimit", "context_token_limit"))
        or _unknown_varint_field(model, _MODEL_CONTEXT_TOKEN_LIMIT_FIELD)
        or _known_tooltip_context(model, ("tooltipData", "tooltip_data"))
        or _unknown_tooltip_context(model, _MODEL_TOOLTIP_FIELD)
    )
    max_context_length = (
        _known_numeric_field(
            model,
            ("contextTokenLimitForMaxMode", "context_token_limit_for_max_mode"),
        )
        or _unknown_varint_field(model, _MODEL_CONTEXT_TOKEN_LIMIT_FOR_MAX_MODE_FIELD)
        or _known_tooltip_context(
            model,
            ("tooltipDataForMaxMode", "tooltip_data_for_max_mode"),
        )
        or _unknown_tooltip_context(model, _MODEL_TOOLTIP_FOR_MAX_MODE_FIELD)
    )

    metadata: dict[str, int] = {}
    if context_length is not None:
        metadata["context_length"] = context_length
    if max_context_length is not None:
        metadata["max_context_length"] = max_context_length
    return metadata


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


def _collect_model_catalog(resp) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Collect model ids plus metadata supplied by the same Cursor response."""
    model_ids: list[str] = []
    metadata: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    for model in resp.models:
        model_metadata = _extract_model_context_metadata(model)
        for model_id in [model.name, *_extract_variant_model_ids(model)]:
            if not model_id:
                continue
            if model_id not in seen:
                seen.add(model_id)
                model_ids.append(model_id)
            if model_metadata:
                metadata[model_id] = dict(model_metadata)
            normal = model_metadata.get("context_length")
            maximum = model_metadata.get("max_context_length")
            if normal is None or maximum is None or normal == maximum:
                continue
            max_model_id = add_context_marker(model_id, maximum)
            if max_model_id is None:
                continue
            if max_model_id not in seen:
                seen.add(max_model_id)
                model_ids.append(max_model_id)
            metadata[max_model_id] = {
                "context_length": maximum,
                "max_context_length": maximum,
            }
    return model_ids, metadata


async def _fetch_available_models(
    token: str,
) -> tuple[list[str], dict[str, dict[str, int]]]:
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
            return _collect_model_catalog(resp)

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


def _model_listing_metadata(model_id: str) -> dict[str, int]:
    """Return live Cursor metadata, including a max-context marker if requested."""
    marker = MODEL_CONTEXT_MARKER_RE.search(model_id)
    base_model_id = strip_context_marker(model_id)
    cached = _model_cache["metadata"].get(model_id)
    if cached is None:
        cached = _model_cache["metadata"].get(base_model_id)
    if cached is None:
        return {}

    if marker is not None and "max_context_length" in cached:
        marker_length = _scaled_token_count(marker.group("amount"), marker.group("unit"))
        if marker_length == cached["max_context_length"]:
            return {"context_length": marker_length}
    if "context_length" in cached:
        return {"context_length": cached["context_length"]}
    return {}


def _model_listing_object(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "cursor",
        **_model_listing_metadata(model_id),
    }


def _models_response(model_ids: list[str]) -> JSONResponse:
    return JSONResponse(content={
        "object": "list",
        "data": [_model_listing_object(model_id) for model_id in model_ids],
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
            model_ids, model_metadata = await _fetch_available_models(token)
        except Exception as exc:
            if _model_cache["ids"]:
                logger.warn(
                    f"AvailableModels failed ({exc}); serving "
                    f"{len(_model_cache['ids'])} cached models"
                )
                return _model_cache["ids"]
            raise

        _model_cache["ids"] = model_ids
        _model_cache["metadata"] = model_metadata
        replace_model_context_catalog(model_metadata)
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


@router.get("/v1/models/{model_id}")
async def get_model_detail(model_id: str):
    """Return metadata only for models observed in the live Cursor catalog."""
    observed_ids = _model_cache["ids"]
    base_model_id = strip_context_marker(model_id)
    if model_id not in observed_ids and base_model_id not in observed_ids:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": "Model not found", "type": "not_found_error"}},
        )
    return JSONResponse(content=_model_listing_object(model_id))


@router.get("/models")
async def list_models_alt(request: Request):
    """Alias without /v1 prefix (legacy compat)."""
    return await list_models(request)
