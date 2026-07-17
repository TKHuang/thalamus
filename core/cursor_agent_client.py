"""Cursor Agent API transport for request-scoped client tools.

The legacy ChatService endpoint only exposes Cursor's fixed built-in tool
inventory.  AgentService accepts arbitrary MCP tool definitions as real
protobuf fields, which keeps tool availability independent of model vendor.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import gzip
import inspect
import json
import os
import re
import struct
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from config.cursor_client import get_cursor_client_version
from core.bearer_token import strip_cursor_user_prefix
from core.cursor_h2_client import open_streaming_h2_request, send_unary_h2_request
from core.protobuf_builder import (
    compute_sha256_hex_digest,
    generate_obfuscated_machine_id_checksum,
)
from core.protobuf_tool_call_parser import NativeToolCall
from proto import agent_api_pb2 as agent_pb
from proto import bidi_api_pb2 as bidi_pb
from utils.structured_logging import ThalamusStructuredLogger

logger = ThalamusStructuredLogger.get_logger("agent-client", "DEBUG")

DEFAULT_AGENT_BASE_URL = "https://agentn.us.api5.cursor.sh"
AGENT_RUN_PATH = "/agent.v1.AgentService/RunSSE"
BIDI_APPEND_PATH = "/aiserver.v1.BidiService/BidiAppend"
CLIENT_TOOL_PROVIDER = "thalamus-client"
_UPSTREAM_TOOL_NAME_PREFIX = "mcp_thalamus_client_"
_UPSTREAM_TOOL_NAME_MAX_CHARS = 64
_CLIENT_TOOL_TRANSPORT_CONTRACT = """<client_tool_transport_contract>
For executable work, use only the request-scoped MCP tool inventory attached
to this request. Cursor-native local tools are unavailable. Match the exact
model-facing MCP tool name and its input_schema; do not infer parameter aliases.
Do not narrate unavailable native-tool attempts to the user; silently select
the applicable request-scoped MCP tool. If no applicable MCP tool exists,
explain that limitation accurately.
</client_tool_transport_contract>"""
_NATIVE_EXEC_POLICY_REASON = (
    "Cursor-native local execution is disabled by this proxy; use one of "
    "the request-scoped MCP tools advertised by the client instead. Replan "
    "silently using the exact MCP tool name and input schema."
)
_ASK_QUESTION_REJECTION_REASON = (
    "The downstream client has no synchronous interaction-response channel. "
    "Continue with best judgment and complete the user's request without asking."
)
_INTERACTION_REJECTION_REASON = (
    "The downstream client has no synchronous approval or interaction channel "
    "for this Cursor operation. Continue with best judgment and use the "
    "request-scoped client tools instead."
)
_GLOBAL_HEARTBEAT_SECONDS = 5.0
_EXEC_HEARTBEAT_SECONDS = 3.0
_DEFAULT_SEMANTIC_STALL_SECONDS = 90.0
_DEFAULT_TOOL_ASSEMBLY_STALL_SECONDS = 600.0
_DEFAULT_PENDING_TOOL_TTL_SECONDS = 900.0
_WORKSPACE_EXCLUSION_CACHE_TTL_SECONDS = 3600.0
_WORKSPACE_EXCLUSION_CACHE_MAX_ENTRIES = 512
_WORKSPACE_EXCLUSION_REJECTION = (
    "Workspace context exclusion is not allowed for this user, team, or "
    "selected model"
)
_MAX_KV_BLOB_BYTES = 16 * 1024 * 1024
_MAX_KV_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_TOOL_RESULT_IMAGE_BYTES = 16 * 1024 * 1024
_MAX_TOOL_RESULT_IMAGE_TOTAL_BYTES = 32 * 1024 * 1024
_TOOL_RESULT_MEDIA_KEY = "_thalamus_tool_result_media"
_SYNTHETIC_TOOL_MEDIA_PROMPTS = frozenset(
    {
        "Attached media from tool result:",
        "Attached image(s) from tool result:",
    }
)
_SYNTHETIC_TOOL_MEDIA_UNAVAILABLE_ERRORS = frozenset(
    {
        "ERROR: Cannot read image (this model does not support image input). "
        "Inform the user.",
    }
)
_SUPPORTED_TOOL_RESULT_IMAGE_MIME_TYPES = frozenset(
    {"image/gif", "image/jpeg", "image/png", "image/webp"}
)

_WORKSPACE_EXCLUSION_UNSUPPORTED: OrderedDict[
    tuple[str, str, str], float
] = OrderedDict()

# Cursor Agent normally exposes its own workspace-local ToolCall inventory in
# addition to the request's MCP inventory.  Those tools run in Cursor's client
# process, not in the HTTP caller's workspace, so a transparent proxy must not
# advertise them.  The official Cursor CLI sends these proto oneof field names
# through x-cursor-agent-exclude-tools.  Keep mcp_tool_call deliberately absent:
# request-scoped harness tools are represented exclusively through that lane.
_CURSOR_NATIVE_TOOL_EXCLUSIONS = (
    "shell_tool_call",
    "delete_tool_call",
    "glob_tool_call",
    "grep_tool_call",
    "read_tool_call",
    "update_todos_tool_call",
    "read_todos_tool_call",
    "edit_tool_call",
    "ls_tool_call",
    "read_lints_tool_call",
    "sem_search_tool_call",
    "create_plan_tool_call",
    "web_search_tool_call",
    "task_tool_call",
    "list_mcp_resources_tool_call",
    "read_mcp_resource_tool_call",
    "apply_agent_diff_tool_call",
    "ask_question_tool_call",
    "fetch_tool_call",
    "switch_mode_tool_call",
    "generate_image_tool_call",
    "record_screen_tool_call",
    "computer_use_tool_call",
    "write_shell_stdin_tool_call",
    "reflect_tool_call",
    "setup_vm_environment_tool_call",
    "truncated_tool_call",
    "start_grind_execution_tool_call",
    "start_grind_planning_tool_call",
    "web_fetch_tool_call",
    "report_bugfix_results_tool_call",
    "ai_attribution_tool_call",
    "pr_management_tool_call",
    "mcp_auth_tool_call",
    "await_tool_call",
    "blame_by_file_path_tool_call",
    "get_mcp_tools_tool_call",
    "report_bug_tool_call",
    "set_active_branch_tool_call",
    "communicate_update_tool_call",
    "send_final_summary_tool_call",
    "update_pr_code_tour_tool_call",
    "replace_env_tool_call",
    "edit_pr_labels_tool_call",
    "record_ci_investigation_findings_tool_call",
    "send_message_tool_call",
    "fetch_cloud_agent_data_tool_call",
    "send_to_user_tool_call",
    "pi_read_tool_call",
    "pi_bash_tool_call",
    "pi_edit_tool_call",
    "pi_write_tool_call",
    "pi_grep_tool_call",
    "pi_find_tool_call",
    "pi_ls_tool_call",
    "connect_scm_tool_call",
    "search_conversations_tool_call",
)


def _positive_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _workspace_exclusion_cache_key(
    auth_key: str,
    model: str,
    base_url: str,
) -> tuple[str, str, str]:
    return auth_key, model, base_url.rstrip("/")


def _workspace_exclusion_is_unsupported(
    key: tuple[str, str, str],
) -> bool:
    expires_at = _WORKSPACE_EXCLUSION_UNSUPPORTED.get(key)
    if expires_at is None:
        return False
    if expires_at <= time.monotonic():
        _WORKSPACE_EXCLUSION_UNSUPPORTED.pop(key, None)
        return False
    _WORKSPACE_EXCLUSION_UNSUPPORTED.move_to_end(key)
    return True


def _remember_workspace_exclusion_unsupported(
    key: tuple[str, str, str],
) -> None:
    _WORKSPACE_EXCLUSION_UNSUPPORTED[key] = (
        time.monotonic() + _WORKSPACE_EXCLUSION_CACHE_TTL_SECONDS
    )
    _WORKSPACE_EXCLUSION_UNSUPPORTED.move_to_end(key)
    while (
        len(_WORKSPACE_EXCLUSION_UNSUPPORTED)
        > _WORKSPACE_EXCLUSION_CACHE_MAX_ENTRIES
    ):
        _WORKSPACE_EXCLUSION_UNSUPPORTED.popitem(last=False)


def _is_workspace_exclusion_rejection(errors: Any) -> bool:
    if not isinstance(errors, list):
        return False
    detail = "\n".join(str(error) for error in errors)
    return (
        "invalid_argument" in detail.lower()
        and _WORKSPACE_EXCLUSION_REJECTION in detail
    )


def _connect_frame(payload: bytes) -> bytes:
    return b"\0" + struct.pack(">I", len(payload)) + payload


def _request_headers(
    token: str,
    request_id: str,
    host: str,
    *,
    connect: bool,
) -> dict[str, str]:
    headers = {
        "authorization": f"Bearer {token}",
        "user-agent": "connect-es/1.6.1",
        "x-client-key": compute_sha256_hex_digest(token),
        "x-cursor-checksum": generate_obfuscated_machine_id_checksum(token),
        "x-cursor-client-version": get_cursor_client_version(),
        "x-cursor-config-version": str(uuid.uuid4()),
        "x-cursor-timezone": "Asia/Shanghai",
        "x-ghost-mode": "true",
        "x-request-id": request_id,
        "x-original-request-id": request_id,
        "x-session-id": str(uuid.uuid5(uuid.NAMESPACE_DNS, token)),
        "x-cursor-streaming": "true",
        "cookie": f"CursorCookie=Cookie-{token[:15]}",
        "Host": host,
    }
    headers["x-cursor-agent-exclude-tools"] = ",".join(
        _CURSOR_NATIVE_TOOL_EXCLUSIONS
    )
    # These headers are defense in depth only: current Agent builds may still
    # emit native Exec lanes after accepting the request.  _handle_exec's typed
    # policy results are the actual workspace isolation boundary.
    headers["x-cursor-agent-allowed-tools"] = "mcp_tool_call"
    if connect:
        headers.update(
            {
                "connect-accept-encoding": "gzip",
                "connect-protocol-version": "1",
                "content-type": "application/connect+proto",
            }
        )
    else:
        headers["content-type"] = "application/proto"
    return headers


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") in ("image", "image_url", "input_image"):
                    parts.append("[image]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        value = content.get("text") or content.get("content")
        return value if isinstance(value, str) else ""
    return str(content)


def _schema_value(schema: Any) -> struct_pb2.Value:
    normalized = schema if isinstance(schema, dict) else {"type": "object"}
    value = struct_pb2.Value()
    # ``Struct.update({})`` is a no-op and otherwise leaves Value.kind unset,
    # turning a deliberate empty JSON schema into protobuf null/absence.
    value.struct_value.SetInParent()
    value.struct_value.update(normalized)
    return value


def _upstream_tool_name(client_name: str) -> str:
    """Namespace a model-facing MCP name without changing the client contract.

    Cursor's Agent runtime also exposes native tools such as ``Write`` and
    ``Read``.  Advertising a request-owned MCP tool as bare ``write`` or
    ``read`` leaves some model vendors to conflate the two inventories.  The
    wire-facing name therefore gets a deterministic namespace, while
    ``tool_name`` and the downstream call keep the exact client-owned name.
    """
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", client_name).strip("_") or "tool"
    digest = compute_sha256_hex_digest(client_name)[:10]
    suffix = f"_{digest}"
    available = max(
        1,
        _UPSTREAM_TOOL_NAME_MAX_CHARS
        - len(_UPSTREAM_TOOL_NAME_PREFIX)
        - len(suffix),
    )
    return f"{_UPSTREAM_TOOL_NAME_PREFIX}{safe_name[:available]}{suffix}"


def _request_tool_schema(function: dict[str, Any]) -> dict[str, Any]:
    """Preserve the request-owned schema, including a deliberate empty object."""
    if "parameters" in function:
        schema = function["parameters"]
    elif "input_schema" in function:
        schema = function["input_schema"]
    else:
        schema = {"type": "object"}
    return schema if isinstance(schema, dict) else {"type": "object"}


def build_agent_tool_definitions(tools: list[dict]) -> list[agent_pb.McpToolDefinition]:
    definitions: list[agent_pb.McpToolDefinition] = []
    for tool in tools:
        function = tool.get("function") or tool
        name = function.get("name", "")
        if not isinstance(name, str) or not name:
            continue
        description = function.get("description", "")
        schema = _request_tool_schema(function)
        definitions.append(
            agent_pb.McpToolDefinition(
                name=_upstream_tool_name(name),
                description=description if isinstance(description, str) else "",
                input_schema=_schema_value(schema),
                provider_identifier=CLIENT_TOOL_PROVIDER,
                tool_name=name,
            )
        )
    return definitions


def _request_context(
    definitions: list[agent_pb.McpToolDefinition],
) -> agent_pb.RequestContext:
    """Build a complete, workspace-independent Agent request context."""
    return agent_pb.RequestContext(
        tools=definitions,
        git_repo_info_complete=True,
        mcp_info_complete=True,
        rules_info_complete=True,
        env_info_complete=True,
        repository_info_complete=True,
        custom_subagents_info_complete=True,
        agent_skills_info_complete=True,
        mcp_file_system_info_complete=True,
        git_status_info_complete=True,
    )


def _split_current_turn(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], str]:
    instructions = "\n\n".join(
        _content_text(message.get("content"))
        for message in messages
        if message.get("role") in ("system", "developer")
        and _content_text(message.get("content"))
    )
    conversation = [
        message
        for message in messages
        if message.get("role") not in ("system", "developer")
    ]
    current_index = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if conversation[index].get("role") == "user"
        ),
        None,
    )
    if current_index is None or current_index != len(conversation) - 1:
        current_text = "Continue from the latest tool result and complete the user's request."
        history_messages = conversation
    else:
        current_text = _content_text(conversation[current_index].get("content"))
        history_messages = conversation[:current_index]

    return current_text, history_messages, instructions


def _history_transcript(
    prior_messages: list[dict[str, Any]],
    instructions: str,
) -> str:
    parts: list[str] = []
    if instructions:
        parts.append(f"SYSTEM INSTRUCTIONS:\n{instructions}")
    for message in prior_messages:
        role = str(message.get("role") or "unknown").upper()
        text = _content_text(message.get("content"))
        if text:
            parts.append(f"{role}:\n{text}")
        for call in message.get("tool_calls") or []:
            function = call.get("function") or call
            parts.append(
                "ASSISTANT TOOL CALL:\n"
                f"id={call.get('id', '')}\n"
                f"name={function.get('name', '')}\n"
                f"arguments={function.get('arguments', '{}')}"
            )
        if message.get("role") == "tool":
            parts.append(f"TOOL CALL ID: {message.get('tool_call_id', '')}")
    if not parts:
        return ""
    return "<client_conversation_context>\n" + "\n\n".join(parts) + "\n</client_conversation_context>"


def build_agent_run_message(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict],
    *,
    exclude_workspace_context: bool = True,
) -> agent_pb.AgentClientMessage:
    prompt, prior_messages, instructions = _split_current_turn(messages)
    definitions = build_agent_tool_definitions(tools)
    conversation_id = str(uuid.uuid4())
    user = agent_pb.UserMessage(
        text=prompt,
        message_id=str(uuid.uuid4()),
        mode=agent_pb.AGENT_MODE_AGENT,
        rich_text=prompt,
    )
    # AgentRunRequest.custom_system_prompt is present in the descriptor, but
    # the deployed Cursor Agent runtime currently translates it to an
    # unsupported ``--system-prompt`` option and rejects the whole request.
    # Keep the compatible prepended-context transport until that runtime
    # capability can be negotiated explicitly.  Authorization does not depend
    # on this prose: namespaced identities, exact schemas, provider checks, and
    # typed native-tool rejection remain the enforcement boundary.
    context_instructions = instructions
    if definitions:
        context_instructions = "\n\n".join(
            part
            for part in (instructions, _CLIENT_TOOL_TRANSPORT_CONTRACT)
            if part
        )
    transcript = _history_transcript(prior_messages, context_instructions)
    prepended = []
    if transcript:
        prepended.append(
            agent_pb.UserMessage(
                text=transcript,
                message_id=str(uuid.uuid4()),
                mode=agent_pb.AGENT_MODE_AGENT,
                rich_text=transcript,
            )
        )
    action = agent_pb.ConversationAction(
        user_message_action=agent_pb.UserMessageAction(
            user_message=user,
            request_context=_request_context(definitions),
            send_to_interaction_listener=True,
            prepend_user_messages=prepended,
        )
    )
    run_request = agent_pb.AgentRunRequest(
        conversation_state=agent_pb.ConversationStateStructure(
            mode=agent_pb.AGENT_MODE_AGENT,
            agent_type="agent",
        ),
        action=action,
        mcp_tools=agent_pb.McpTools(mcp_tools=definitions),
        conversation_id=conversation_id,
        requested_model=agent_pb.RequestedModel(
            model_id=model,
            built_in_model=True,
        ),
        conversation_group_id=conversation_id,
    )
    if exclude_workspace_context:
        run_request.exclude_workspace_context = True
    return agent_pb.AgentClientMessage(run_request=run_request)


class _ConnectFrameDecoder:
    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed(self, chunk: bytes) -> list[agent_pb.AgentServerMessage]:
        self.buffer.extend(chunk)
        messages: list[agent_pb.AgentServerMessage] = []
        while len(self.buffer) >= 5:
            magic = self.buffer[0]
            size = int.from_bytes(self.buffer[1:5], "big")
            if len(self.buffer) < size + 5:
                break
            payload = bytes(self.buffer[5 : size + 5])
            del self.buffer[: size + 5]
            if magic in (1, 3):
                payload = gzip.decompress(payload)
            if magic in (2, 3):
                detail = payload.decode("utf-8", errors="replace")
                raise RuntimeError(f"Cursor Agent API error: {detail}")
            message = agent_pb.AgentServerMessage()
            message.ParseFromString(payload)
            messages.append(message)
        return messages


def _normalize_numbers(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_numbers(item) for item in value]
    return value


def _public_call_id(raw_call_id: str) -> str:
    """Return the stable, single-line call id exposed to harness clients."""
    return next(
        (part.strip() for part in raw_call_id.splitlines() if part.strip()),
        raw_call_id.strip(),
    )


def _native_mcp_call(
    args: agent_pb.McpArgs,
    fallback_call_id: str = "",
    *,
    allow_generated_call_id: bool = True,
    name_aliases: dict[str, str] | None = None,
    expected_provider: str | None = None,
) -> NativeToolCall | None:
    if (
        expected_provider
        and args.provider_identifier
        and args.provider_identifier != expected_provider
    ):
        return None
    wire_name = args.tool_name or args.name
    if not wire_name:
        return None
    name = (name_aliases or {}).get(wire_name, wire_name)
    arguments = {
        key: _normalize_numbers(MessageToDict(value))
        for key, value in args.args.items()
    }
    raw_arguments = json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    raw_call_id = args.tool_call_id or fallback_call_id
    call_id = _public_call_id(raw_call_id)
    if not call_id:
        if not allow_generated_call_id:
            return None
        call_id = f"call_{uuid.uuid4().hex}"
    return NativeToolCall(
        enum=49,
        call_id=call_id,
        name=name,
        raw_arguments=raw_arguments,
        arguments=arguments,
    )


def _native_tool_call(
    update: agent_pb.ToolCallStartedUpdate,
    name_aliases: dict[str, str] | None = None,
) -> NativeToolCall | None:
    tool_call = update.tool_call
    if not tool_call.HasField("mcp_tool_call"):
        return None
    return _native_mcp_call(
        tool_call.mcp_tool_call.args,
        update.call_id or tool_call.tool_call_id,
        allow_generated_call_id=False,
        name_aliases=name_aliases,
        expected_provider=CLIENT_TOOL_PROVIDER,
    )


def _internal_call_identifiers(*values: str) -> tuple[str, ...]:
    """Keep every non-empty wire ID component for internal correlation."""
    identifiers: list[str] = []
    for value in values:
        for part in str(value or "").splitlines():
            normalized = part.strip()
            if normalized and normalized not in identifiers:
                identifiers.append(normalized)
    return tuple(identifiers)


def _tool_update_identifiers(update: Any) -> tuple[tuple[str, ...], bool]:
    """Return internal aliases and whether an update has a public call ID."""
    public_candidates = [getattr(update, "call_id", "")]
    model_candidates = [getattr(update, "model_call_id", "")]
    tool_call = getattr(update, "tool_call", None)
    if tool_call is not None:
        public_candidates.append(getattr(tool_call, "tool_call_id", ""))
        if tool_call.HasField("mcp_tool_call"):
            public_candidates.append(tool_call.mcp_tool_call.args.tool_call_id)
    public_ids = _internal_call_identifiers(*public_candidates)
    model_ids = _internal_call_identifiers(*model_candidates)
    return tuple(dict.fromkeys((*public_ids, *model_ids))), bool(public_ids)


def _native_exec_tool_call(
    message: agent_pb.ExecServerMessage,
    name_aliases: dict[str, str] | None = None,
) -> NativeToolCall | None:
    if message.WhichOneof("message") != "mcp_args":
        return None
    return _native_mcp_call(
        message.mcp_args,
        message.exec_id or (str(message.id) if message.id else ""),
        name_aliases=name_aliases,
        expected_provider=CLIENT_TOOL_PROVIDER,
    )


def _native_local_exec_tool_call_id(
    message: agent_pb.ExecServerMessage,
    kind: str | None,
) -> str:
    if not kind:
        return ""
    args = getattr(message, kind, None)
    return str(getattr(args, "tool_call_id", "") or "")


def _native_local_exec_rejection(
    message: agent_pb.ExecServerMessage,
    kind: str | None,
) -> agent_pb.ExecClientMessage | None:
    """Return Cursor's domain-level policy result for a known local lane."""

    common = {"id": message.id, "exec_id": message.exec_id}
    reason = _NATIVE_EXEC_POLICY_REASON

    if kind in ("shell_args", "shell_stream_args"):
        args = getattr(message, kind)
        rejected = agent_pb.ShellRejected(
            command=args.command,
            working_directory=args.working_directory,
            reason=reason,
        )
        if kind == "shell_stream_args":
            return agent_pb.ExecClientMessage(
                **common,
                shell_stream=agent_pb.ShellStream(rejected=rejected),
            )
        return agent_pb.ExecClientMessage(
            **common,
            shell_result=agent_pb.ShellResult(rejected=rejected),
        )

    if kind == "write_args":
        return agent_pb.ExecClientMessage(
            **common,
            write_result=agent_pb.WriteResult(
                rejected=agent_pb.WriteRejected(
                    path=message.write_args.path,
                    reason=reason,
                )
            ),
        )

    if kind == "delete_args":
        return agent_pb.ExecClientMessage(
            **common,
            delete_result=agent_pb.DeleteResult(
                rejected=agent_pb.DeleteRejected(
                    path=message.delete_args.path,
                    reason=reason,
                )
            ),
        )

    if kind == "grep_args":
        # Cursor's GrepResult has no rejected/permission-denied variant.  A
        # typed policy error still avoids misclassifying this as an executor
        # transport failure.
        return agent_pb.ExecClientMessage(
            **common,
            grep_result=agent_pb.GrepResult(
                error=agent_pb.GrepError(error=reason)
            ),
        )

    if kind in ("read_args", "redacted_read_args"):
        args = getattr(message, kind)
        result_field = (
            "redacted_read_result"
            if kind == "redacted_read_args"
            else "read_result"
        )
        return agent_pb.ExecClientMessage(
            **common,
            **{
                result_field: agent_pb.ReadResult(
                    rejected=agent_pb.ReadRejected(
                        path=args.path,
                        reason=reason,
                    )
                )
            },
        )

    if kind == "ls_args":
        return agent_pb.ExecClientMessage(
            **common,
            ls_result=agent_pb.LsResult(
                rejected=agent_pb.LsRejected(
                    path=message.ls_args.path,
                    reason=reason,
                )
            ),
        )

    if kind == "diagnostics_args":
        return agent_pb.ExecClientMessage(
            **common,
            diagnostics_result=agent_pb.DiagnosticsResult(
                rejected=agent_pb.DiagnosticsRejected(
                    path=message.diagnostics_args.path,
                    reason=reason,
                )
            ),
        )

    return None


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("invalid protobuf varint")


def _wire_field_numbers(data: bytes) -> list[int]:
    """Return protobuf field numbers without logging field contents."""
    fields: list[int] = []
    offset = 0
    try:
        while offset < len(data):
            tag, offset = _read_varint(data, offset)
            field_number, wire_type = tag >> 3, tag & 7
            fields.append(field_number)
            if wire_type == 0:
                _, offset = _read_varint(data, offset)
            elif wire_type == 1:
                offset += 8
            elif wire_type == 2:
                size, offset = _read_varint(data, offset)
                offset += size
            elif wire_type == 5:
                offset += 4
            else:
                break
    except (IndexError, ValueError):
        pass
    return fields


async def _invoke_callback(callback: Callable[[str], Any] | None, value: str) -> None:
    if callback is None or not value:
        return
    result = callback(value)
    if inspect.isawaitable(result):
        await result


async def _invoke_tool_start_callback(
    callback: Callable[[str, str], Any] | None,
    call_id: str,
    name: str,
) -> None:
    if callback is None or not call_id or not name:
        return
    result = callback(call_id, name)
    if inspect.isawaitable(result):
        await result


@dataclass(frozen=True)
class _SessionEvent:
    kind: str
    value: Any = None


@dataclass
class _ToolAssembly:
    aliases: set[str]
    has_public_id: bool


class _CollectorAlreadyAttached(RuntimeError):
    pass


@dataclass
class _PendingExecution:
    id: int
    exec_id: str
    raw_call_id: str
    public_call_id: str
    call: NativeToolCall
    created_at: float = field(default_factory=time.monotonic)
    heartbeat_task: asyncio.Task[None] | None = None
    state: str = "announced"


_ACTIVE_SESSIONS: set["_AgentSession"] = set()
_SESSIONS_BY_TOOL_ID: dict[
    tuple[str, str], set["_AgentSession"]
] = {}


def _tool_result_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return normalized top-level tool results in request order."""
    results: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id"):
            results.append(message)

    # A repeated HTTP body or normalizer should not submit one execution twice.
    deduplicated: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for result in results:
        call_id = str(result.get("tool_call_id") or "")
        if call_id not in deduplicated:
            order.append(call_id)
        deduplicated[call_id] = result
    return [deduplicated[call_id] for call_id in order]


def _decode_synthetic_tool_media_carrier(
    message: dict[str, Any],
) -> list[tuple[str, bytes]] | None:
    """Decode OpenCode/AI-SDK's typed tool-result media companion.

    OpenAI-compatible tool results only carry text.  OpenCode therefore emits
    one synthetic user message immediately after an image-producing tool
    result.  Treat only its exact transport marker plus inline image data as a
    continuation envelope; ordinary user text, remote URLs, malformed data,
    and unsupported media remain hard instruction boundaries.
    """

    if message.get("role") != "user":
        return None
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return None

    marker_count = 0
    encoded_images: list[tuple[str, str]] = []
    for part in content:
        if not isinstance(part, dict):
            return None
        part_type = str(part.get("type") or "")
        if part_type in ("text", "input_text"):
            text = str(part.get("text") or "")
            if text not in _SYNTHETIC_TOOL_MEDIA_PROMPTS:
                return None
            marker_count += 1
            continue
        if part_type != "image_url":
            return None
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        if not isinstance(url, str) or not url.startswith("data:image/"):
            return None
        header, separator, payload = url.partition(",")
        if not separator or not header.endswith(";base64"):
            return None
        mime_type = header[5:-7].lower()
        if mime_type not in _SUPPORTED_TOOL_RESULT_IMAGE_MIME_TYPES:
            return None
        max_encoded = ((_MAX_TOOL_RESULT_IMAGE_BYTES + 2) // 3) * 4
        if not payload or len(payload) > max_encoded:
            return None
        encoded_images.append((mime_type, payload))

    if marker_count != 1 or not encoded_images:
        return None

    decoded: list[tuple[str, bytes]] = []
    total_bytes = 0
    for mime_type, payload in encoded_images:
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return None
        if not data or len(data) > _MAX_TOOL_RESULT_IMAGE_BYTES:
            return None
        total_bytes += len(data)
        if total_bytes > _MAX_TOOL_RESULT_IMAGE_TOTAL_BYTES:
            return None
        decoded.append((mime_type, data))
    return decoded


def _decode_synthetic_tool_media_unavailable_carrier(
    message: dict[str, Any],
) -> str | None:
    """Decode OpenCode/AI-SDK's exact unsupported-media companion.

    Some provider model descriptors reject image input before the request
    reaches this proxy.  In that case OpenCode replaces each attachment with
    a fixed transport error inside the same synthetic user carrier.  It is
    still part of the preceding tool result, not a fresh user instruction.
    Only exact, known transport text is accepted here.
    """

    if message.get("role") != "user":
        return None
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return None

    marker_count = 0
    errors: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            return None
        if str(part.get("type") or "") not in ("text", "input_text"):
            return None
        text = str(part.get("text") or "")
        if text in _SYNTHETIC_TOOL_MEDIA_PROMPTS:
            marker_count += 1
            continue
        if text not in _SYNTHETIC_TOOL_MEDIA_UNAVAILABLE_ERRORS:
            return None
        errors.append(text)

    if marker_count != 1 or not errors:
        return None
    return "\n".join(dict.fromkeys(errors))


def _trailing_tool_result_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only results that form the current tool-continuation suffix.

    Historical tool results remain in a harness transcript after an Agent turn
    has ended.  They must not be interpreted as a request to resume an already
    completed RunSSE session.  A genuine continuation is identified by one or
    more normalized tool messages at the literal end of the request.  The sole
    exception is a structurally exact synthetic media carrier emitted by
    OpenCode/AI-SDK for an image-producing tool result.  A later ordinary user,
    system, or developer message remains a new instruction boundary.
    """
    end = len(messages)
    media: list[tuple[str, bytes]] | None = None
    media_unavailable_error: str | None = None
    if end:
        media = _decode_synthetic_tool_media_carrier(messages[end - 1])
        if media is not None:
            end -= 1
        else:
            media_unavailable_error = (
                _decode_synthetic_tool_media_unavailable_carrier(
                    messages[end - 1]
                )
            )
            if media_unavailable_error is not None:
                end -= 1

    trailing: list[dict[str, Any]] = []
    for message in reversed(messages[:end]):
        if message.get("role") != "tool" or not message.get("tool_call_id"):
            break
        trailing.append(message)
    trailing.reverse()
    results = _tool_result_messages(trailing)
    if media is None and media_unavailable_error is None:
        return results
    # OpenCode pools all extracted media from a model turn into one carrier.
    # Without a per-image call id, only a single-result boundary is unambiguous.
    if len(results) != 1:
        return []
    enriched = dict(results[0])
    if media is not None:
        enriched[_TOOL_RESULT_MEDIA_KEY] = media
    if media_unavailable_error is not None:
        content = _content_text(enriched.get("content"))
        enriched["content"] = (
            f"{content}\n\n{media_unavailable_error}"
            if content
            else media_unavailable_error
        )
    return [enriched]


class _AgentSession:
    """Own one Cursor RunSSE stream across client-side tool round trips."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict],
        auth_token: str,
        *,
        session_id: str | None = None,
        client_request_id: str | None = None,
        origin_request_id: str | None = None,
        exclude_workspace_context: bool = True,
    ) -> None:
        self.token = strip_cursor_user_prefix(auth_token).strip()
        self.auth_key = compute_sha256_hex_digest(self.token)
        self.request_id = str(uuid.uuid4())
        initial_client_request_id = client_request_id or "unavailable"
        self.origin_request_id = origin_request_id or initial_client_request_id
        self.active_request_id = initial_client_request_id
        # Compatibility alias for tests and any downstream log parsing.  It now
        # follows the active continuation rather than remaining stuck on the
        # request that created the RunSSE session.
        self.client_request_id = self.active_request_id
        self.model = model
        self.session_id = session_id or str(uuid.uuid4())
        self.exclude_workspace_context = exclude_workspace_context
        self.base_url = os.environ.get(
            "CURSOR_AGENT_BASE_URL", DEFAULT_AGENT_BASE_URL
        ).rstrip("/")
        self.agent_host = urlparse(self.base_url).netloc
        self.run_message = build_agent_run_message(
            messages,
            model,
            tools,
            exclude_workspace_context=exclude_workspace_context,
        )
        self.tool_definitions = build_agent_tool_definitions(tools)
        self.tool_names = {
            definition.tool_name or definition.name
            for definition in self.tool_definitions
        }
        self.tool_name_aliases = {
            definition.name: definition.tool_name or definition.name
            for definition in self.tool_definitions
        }

        self.events: asyncio.Queue[_SessionEvent] = asyncio.Queue()
        self.deferred_events: deque[_SessionEvent] = deque()
        self.append_lock = asyncio.Lock()
        self.collect_lock = asyncio.Lock()
        self.collector_state_lock = asyncio.Lock()
        self.submit_lock = asyncio.Lock()
        self.collector_active = False
        self.next_append_sequence = 0
        self.pending: dict[str, _PendingExecution] = {}
        self.pending_by_exec_id: dict[str, _PendingExecution] = {}
        self.registered_tool_keys: set[tuple[str, str]] = set()
        self.blobs: dict[bytes, bytes] = {}

        self.started_at = time.monotonic()
        self.last_semantic_progress_at = self.started_at
        self.last_progress_log_at = self.started_at
        self.last_quiet_log_at = self.started_at
        self.chunk_count = 0
        self.partial_tool_argument_bytes = 0
        self.partial_tool_update_count = 0
        self.partial_tool_snapshot_bytes = 0
        self.active_request_started_at = self.started_at
        self.active_request_partial_argument_bytes_baseline = 0
        self.active_request_partial_update_count_baseline = 0
        self.active_request_partial_snapshot_bytes = 0
        self.failure_tool_assembly_group_count = 0
        self.partial_fingerprints: dict[str, bytes] = {}
        self.status_fingerprints: set[bytes] = set()
        self.previewed_tool_call_ids: set[str] = set()
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

        self.reader_task: asyncio.Task[None] | None = None
        self.global_heartbeat_task: asyncio.Task[None] | None = None
        self.watchdog_task: asyncio.Task[None] | None = None
        self.closed_event = asyncio.Event()
        self.closing = False
        self.terminal_seen = False
        self.error_enqueued = False
        self.failure_kind: str | None = None
        self.semantic_output_seen = False
        self.tool_candidate_seen = False
        self.execution_boundary_seen = False
        self.tool_result_submitted = False
        self.tool_assembly_groups: list[_ToolAssembly] = []

    def _log_context(self) -> str:
        return (
            f"request_id={self.active_request_id} | "
            f"origin_request_id={self.origin_request_id} | "
            f"active_request_id={self.active_request_id} | "
            f"agent_request_id={self.request_id} | model={self.model} | "
            f"session_id={self.session_id}"
        )

    def bind_active_request(self, client_request_id: str | None) -> None:
        """Attach one downstream continuation before returning its tool result."""

        next_request_id = client_request_id or "unavailable"
        previous_request_id = self.active_request_id
        self.active_request_id = next_request_id
        self.client_request_id = next_request_id
        self.active_request_started_at = time.monotonic()
        self.active_request_partial_argument_bytes_baseline = (
            self.partial_tool_argument_bytes
        )
        self.active_request_partial_update_count_baseline = (
            self.partial_tool_update_count
        )
        self.active_request_partial_snapshot_bytes = 0
        if next_request_id != previous_request_id:
            logger.debug(
                "Cursor Agent session resumed by a new downstream request | "
                f"previous_active_request_id={previous_request_id} | "
                f"{self._log_context()}"
            )

    def _active_request_partial_argument_bytes(self) -> int:
        return max(
            0,
            self.partial_tool_argument_bytes
            - self.active_request_partial_argument_bytes_baseline,
        )

    def _active_request_partial_update_count(self) -> int:
        return max(
            0,
            self.partial_tool_update_count
            - self.active_request_partial_update_count_baseline,
        )

    def can_retry_zero_content_stall(self, result: dict[str, Any]) -> bool:
        """Whether replaying only the transcript is provably side-effect safe."""
        return bool(
            self.failure_kind == "semantic_stall"
            and self._has_no_observed_work(result)
        )

    def can_retry_workspace_exclusion_rejection(
        self,
        result: dict[str, Any],
    ) -> bool:
        return bool(
            self.exclude_workspace_context
            and self.failure_kind == "protocol"
            and _is_workspace_exclusion_rejection(result.get("errors"))
            and self._has_no_observed_work(result)
        )

    def _has_no_observed_work(self, result: dict[str, Any]) -> bool:
        return bool(
            result.get("has_fatal_error")
            and not result.get("had_content")
            and not self.semantic_output_seen
            and not self.tool_candidate_seen
            and not self.execution_boundary_seen
            and not self.tool_result_submitted
            and not self.pending
            and not self.terminal_seen
        )

    def start(self) -> None:
        _ACTIVE_SESSIONS.add(self)
        self.reader_task = asyncio.create_task(
            self._run(), name=f"cursor-agent-{self.request_id}"
        )

    @property
    def is_closed(self) -> bool:
        return self.closed_event.is_set()

    def _mark_progress(self) -> None:
        self.last_semantic_progress_at = time.monotonic()

    def _mark_transition(self, kind: str, payload: bytes = b"") -> bool:
        fingerprint = kind.encode("utf-8") + b"\0" + payload
        if fingerprint in self.status_fingerprints:
            return False
        self.status_fingerprints.add(fingerprint)
        self._mark_progress()
        return True

    @property
    def tool_assembly_active(self) -> bool:
        return bool(self.tool_assembly_groups)

    def _start_tool_assembly(
        self,
        identifiers: tuple[str, ...],
        *,
        has_public_id: bool,
    ) -> None:
        self.tool_candidate_seen = True
        aliases = set(_internal_call_identifiers(*identifiers))
        if not aliases:
            return

        merged = set(aliases)
        merged_has_public_id = has_public_id
        remaining: list[_ToolAssembly] = []
        for group in self.tool_assembly_groups:
            if group.aliases & aliases:
                merged.update(group.aliases)
                merged_has_public_id = (
                    merged_has_public_id or group.has_public_id
                )
            else:
                remaining.append(group)
        remaining.append(
            _ToolAssembly(
                aliases=merged,
                has_public_id=merged_has_public_id,
            )
        )
        self.tool_assembly_groups = remaining

    def _finish_tool_assembly(
        self,
        *identifiers: str,
        allow_unresolved_single: bool = False,
    ) -> None:
        aliases = set(_internal_call_identifiers(*identifiers))
        matched = False
        remaining: list[_ToolAssembly] = []
        for group in self.tool_assembly_groups:
            if aliases and group.aliases & aliases:
                matched = True
            else:
                remaining.append(group)
        if (
            not matched
            and allow_unresolved_single
            and not aliases
            and len(remaining) == 1
            and not remaining[0].has_public_id
        ):
            remaining.clear()
        self.tool_assembly_groups = remaining

    def _clear_tool_assemblies(self) -> None:
        self.tool_assembly_groups.clear()

    async def _queue_tool_start(self, native: NativeToolCall | None) -> None:
        if native is None or native.call_id in self.previewed_tool_call_ids:
            return
        self.tool_candidate_seen = True
        self.previewed_tool_call_ids.add(native.call_id)
        logger.debug(
            "Agent tool identity available for downstream streaming | "
            f"tool={native.name} | call_id={native.call_id} | "
            f"{self._log_context()}"
        )
        await self.events.put(
            _SessionEvent("tool_start", (native.call_id, native.name))
        )

    async def _append_messages_locked(
        self,
        messages: list[agent_pb.AgentClientMessage],
    ) -> None:
        for message in messages:
            sequence = self.next_append_sequence
            request = bidi_pb.BidiAppendRequest(
                data=message.SerializeToString().hex(),
                request_id=bidi_pb.BidiRequestId(request_id=self.request_id),
                append_seqno=sequence,
            )
            response = await send_unary_h2_request(
                BIDI_APPEND_PATH,
                _request_headers(
                    self.token,
                    self.request_id,
                    "api2.cursor.sh",
                    connect=False,
                ),
                request.SerializeToString(),
            )
            if response["status"] != 200:
                raise RuntimeError(
                    f"Cursor BidiAppend failed with HTTP {response['status']}"
                )
            self.next_append_sequence += 1

    async def _append_messages(
        self,
        messages: list[agent_pb.AgentClientMessage],
    ) -> None:
        """Append an atomic logical batch without heartbeat interleaving."""
        async with self.append_lock:
            await self._append_messages_locked(messages)

    async def _append_message(self, message: agent_pb.AgentClientMessage) -> None:
        await self._append_messages([message])

    async def _send_exec_close(self, execution_id: int) -> None:
        await self._append_message(
            agent_pb.AgentClientMessage(
                exec_client_control_message=agent_pb.ExecClientControlMessage(
                    stream_close=agent_pb.ExecClientStreamClose(id=execution_id)
                )
            )
        )

    async def _send_exec_result_and_close(
        self,
        execution_id: int,
        result: agent_pb.ExecClientMessage,
    ) -> None:
        await self._append_messages(
            [
                agent_pb.AgentClientMessage(exec_client_message=result),
                agent_pb.AgentClientMessage(
                    exec_client_control_message=agent_pb.ExecClientControlMessage(
                        stream_close=agent_pb.ExecClientStreamClose(id=execution_id)
                    )
                ),
            ]
        )

    async def _send_exec_throw(
        self,
        execution_id: int,
        detail: str,
    ) -> None:
        await self._append_messages(
            [
                agent_pb.AgentClientMessage(
                    exec_client_control_message=agent_pb.ExecClientControlMessage(
                        throw=agent_pb.ExecClientThrow(
                            id=execution_id,
                            error=detail,
                            error_code="UNSUPPORTED_EXECUTION_LANE",
                        )
                    )
                ),
                agent_pb.AgentClientMessage(
                    exec_client_control_message=agent_pb.ExecClientControlMessage(
                        stream_close=agent_pb.ExecClientStreamClose(id=execution_id)
                    )
                ),
            ]
        )

    async def _global_heartbeats(self) -> None:
        try:
            while True:
                await asyncio.sleep(_GLOBAL_HEARTBEAT_SECONDS)
                await self._append_message(
                    agent_pb.AgentClientMessage(
                        client_heartbeat=agent_pb.ClientHeartbeat()
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(f"Cursor Agent global heartbeat failed: {exc}")

    async def _execution_heartbeats(self, pending: _PendingExecution) -> None:
        try:
            while pending.state == "announced":
                await asyncio.sleep(_EXEC_HEARTBEAT_SECONDS)
                async with self.append_lock:
                    if pending.state != "announced":
                        return
                    await self._append_messages_locked(
                        [
                            agent_pb.AgentClientMessage(
                                exec_client_control_message=(
                                    agent_pb.ExecClientControlMessage(
                                        heartbeat=agent_pb.ExecClientHeartbeat(
                                            id=pending.id
                                        )
                                    )
                                )
                            )
                        ]
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._fail(
                f"Cursor Agent execution heartbeat failed for {pending.id}: {exc}"
            )

    async def _watchdog(self) -> None:
        stall_seconds = _positive_float_env(
            "CURSOR_AGENT_SEMANTIC_STALL_SECONDS",
            _DEFAULT_SEMANTIC_STALL_SECONDS,
        )
        tool_assembly_stall_seconds = _positive_float_env(
            "CURSOR_AGENT_TOOL_ASSEMBLY_STALL_SECONDS",
            _DEFAULT_TOOL_ASSEMBLY_STALL_SECONDS,
        )
        pending_ttl = _positive_float_env(
            "CURSOR_AGENT_PENDING_TOOL_TTL_SECONDS",
            _DEFAULT_PENDING_TOOL_TTL_SECONDS,
        )
        while True:
            await asyncio.sleep(5)
            now = time.monotonic()
            if self.pending:
                oldest = min(item.created_at for item in self.pending.values())
                if now - oldest > pending_ttl:
                    await self._fail(
                        "Cursor Agent client-tool result timed out while the "
                        "upstream execution session remained open"
                    )
                    return
                continue
            quiet = now - self.last_semantic_progress_at
            allowed_quiet = (
                tool_assembly_stall_seconds
                if self.tool_assembly_active
                else stall_seconds
            )
            if quiet > allowed_quiet:
                failure_kind = (
                    "tool_assembly_stall"
                    if self.tool_assembly_active
                    else "semantic_stall"
                )
                await self._fail(
                    "Cursor Agent stream stalled without semantic progress for "
                    f"{quiet:.1f}s"
                    + (
                        " while assembling tool arguments"
                        if self.tool_assembly_active
                        else ""
                    ),
                    failure_kind=failure_kind,
                )
                return

    async def _cancel_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _fail(
        self,
        detail: str,
        *,
        failure_kind: str = "protocol",
    ) -> None:
        if not self.error_enqueued:
            self.failure_tool_assembly_group_count = len(
                self.tool_assembly_groups
            )
        self._clear_tool_assemblies()
        if not self.error_enqueued:
            self.error_enqueued = True
            self.failure_kind = failure_kind
            await self.events.put(_SessionEvent("error", detail))
            if (
                self.exclude_workspace_context
                and failure_kind == "protocol"
                and _is_workspace_exclusion_rejection([detail])
            ):
                # This is a feature-negotiation response, not a terminal
                # failure. call_cursor_agent will retry once with field 12
                # omitted and emit the single user-actionable recovery log.
                logger.debug(
                    "Agent optional workspace-exclusion probe rejected; "
                    "capability negotiation pending | "
                    f"{self._log_context()} | {detail}"
                )
            else:
                logger.error(
                    f"Agent session failed | {self._log_context()} | "
                    f"failure_kind={failure_kind} | {detail}"
                )
        if self.reader_task is not None and self.reader_task is not asyncio.current_task():
            self.reader_task.cancel()

    def _register_pending(self, pending: _PendingExecution) -> None:
        key = (self.auth_key, pending.public_call_id)
        sessions = _SESSIONS_BY_TOOL_ID.setdefault(key, set())
        if sessions and self not in sessions:
            logger.warn(
                "Agent public tool-call id collision detected; resume will "
                "require an unambiguous matching id set | "
                f"call_id={pending.public_call_id} | {self._log_context()}"
            )
        sessions.add(self)
        self.registered_tool_keys.add(key)
        self.pending[pending.public_call_id] = pending
        if pending.exec_id:
            self.pending_by_exec_id[pending.exec_id] = pending

    def _unregister(self) -> None:
        for key in self.registered_tool_keys:
            sessions = _SESSIONS_BY_TOOL_ID.get(key)
            if sessions is None:
                continue
            sessions.discard(self)
            if not sessions:
                _SESSIONS_BY_TOOL_ID.pop(key, None)
        self.registered_tool_keys.clear()
        _ACTIVE_SESSIONS.discard(self)

    def _mcp_state_result(
        self,
        execution: agent_pb.ExecServerMessage,
    ) -> agent_pb.ExecClientMessage:
        requested = set(execution.mcp_state_exec_args.server_identifiers)
        servers: list[agent_pb.McpStateServer] = []
        if not requested or CLIENT_TOOL_PROVIDER in requested:
            descriptors: list[agent_pb.McpToolDescriptor] = []
            for definition in self.tool_definitions:
                descriptor = agent_pb.McpToolDescriptor(
                    tool_name=definition.tool_name or definition.name,
                    description=definition.description,
                )
                descriptor.input_schema.CopyFrom(definition.input_schema)
                descriptors.append(descriptor)
            servers.append(
                agent_pb.McpStateServer(
                    server_name=CLIENT_TOOL_PROVIDER,
                    server_identifier=CLIENT_TOOL_PROVIDER,
                    tools=descriptors,
                    status="connected",
                )
            )
        return agent_pb.ExecClientMessage(
            id=execution.id,
            exec_id=execution.exec_id,
            mcp_state_exec_result=agent_pb.McpStateExecResult(
                success=agent_pb.McpStateSuccess(servers=servers)
            ),
        )

    async def _handle_exec(self, execution: agent_pb.ExecServerMessage) -> None:
        kind = execution.WhichOneof("message")
        if kind == "request_context_args":
            await self._send_exec_result_and_close(
                execution.id,
                agent_pb.ExecClientMessage(
                    id=execution.id,
                    exec_id=execution.exec_id,
                    request_context_result=agent_pb.RequestContextResult(
                        success=agent_pb.RequestContextSuccess(
                            request_context=_request_context(
                                self.tool_definitions
                            )
                        )
                    ),
                ),
            )
            self._mark_progress()
            logger.debug(
                "Agent request-context query answered without workspace data "
                "and closed | "
                f"exec_id={execution.exec_id} | {self._log_context()}"
            )
            return

        if kind in (
            "shell_allowlist_precheck_args",
            "web_fetch_allowlist_precheck_args",
        ):
            self.tool_candidate_seen = True
            precheck = getattr(execution, kind)
            self._finish_tool_assembly(
                str(getattr(precheck, "tool_call_id", "") or ""),
                execution.exec_id,
                str(execution.id) if execution.id else "",
                allow_unresolved_single=True,
            )
            if kind == "shell_allowlist_precheck_args":
                result = agent_pb.ExecClientMessage(
                    id=execution.id,
                    exec_id=execution.exec_id,
                    shell_allowlist_precheck_result=(
                        agent_pb.ShellAllowlistPrecheckResult(allowlisted=False)
                    ),
                )
            else:
                result = agent_pb.ExecClientMessage(
                    id=execution.id,
                    exec_id=execution.exec_id,
                    web_fetch_allowlist_precheck_result=(
                        agent_pb.WebFetchAllowlistPrecheckResult(allowlisted=False)
                    ),
                )
            await self._send_exec_result_and_close(execution.id, result)
            self._mark_progress()
            logger.debug(
                "Rejected Cursor native allowlist precheck without local "
                f"execution | lane={kind} | exec_id={execution.exec_id} | "
                f"{self._log_context()}"
            )
            return

        if kind == "mcp_allowlist_precheck_args":
            precheck = execution.mcp_allowlist_precheck_args
            wire_tool_name = precheck.tool_name
            client_tool_name = self.tool_name_aliases.get(
                wire_tool_name,
                wire_tool_name,
            )
            provider_allowed = (
                not precheck.provider_identifier
                or precheck.provider_identifier == CLIENT_TOOL_PROVIDER
            )
            allowlisted = provider_allowed and client_tool_name in self.tool_names
            if allowlisted:
                self.tool_candidate_seen = True
            await self._send_exec_result_and_close(
                execution.id,
                agent_pb.ExecClientMessage(
                    id=execution.id,
                    exec_id=execution.exec_id,
                    mcp_allowlist_precheck_result=agent_pb.McpAllowlistPrecheckResult(
                        allowlisted=allowlisted
                    ),
                ),
            )
            self._mark_progress()
            logger.debug(
                "Agent MCP allowlist precheck acknowledged and closed | "
                f"wire_tool={wire_tool_name} | tool={client_tool_name} | "
                f"allowlisted={allowlisted} | {self._log_context()}"
            )
            if allowlisted:
                self._start_tool_assembly(
                    _internal_call_identifiers(precheck.tool_call_id),
                    has_public_id=bool(precheck.tool_call_id),
                )
                await self._queue_tool_start(
                    _native_mcp_call(
                        agent_pb.McpArgs(
                            tool_name=client_tool_name,
                            tool_call_id=precheck.tool_call_id,
                        ),
                        allow_generated_call_id=False,
                        name_aliases=self.tool_name_aliases,
                        expected_provider=CLIENT_TOOL_PROVIDER,
                    )
                )
            else:
                precheck = execution.mcp_allowlist_precheck_args
                self._finish_tool_assembly(
                    precheck.tool_call_id,
                )
            return

        if kind == "mcp_state_exec_args":
            await self._send_exec_result_and_close(
                execution.id,
                self._mcp_state_result(execution),
            )
            self._mark_progress()
            logger.debug(
                "Agent MCP state query answered and closed | "
                f"{self._log_context()}"
            )
            return

        if kind == "mcp_args":
            self.tool_candidate_seen = True
            self.execution_boundary_seen = True
            self._finish_tool_assembly(
                execution.mcp_args.tool_call_id,
                execution.exec_id,
                str(execution.id) if execution.id else "",
                allow_unresolved_single=True,
            )
            provider = execution.mcp_args.provider_identifier
            if provider and provider != CLIENT_TOOL_PROVIDER:
                await self._send_exec_result_and_close(
                    execution.id,
                    agent_pb.ExecClientMessage(
                        id=execution.id,
                        exec_id=execution.exec_id,
                        mcp_result=agent_pb.McpResult(
                            server_not_found=agent_pb.McpServerNotFound(
                                name=provider,
                                available_servers=[CLIENT_TOOL_PROVIDER],
                            )
                        ),
                    ),
                )
                self._mark_progress()
                logger.warn(
                    "Agent requested an MCP provider outside the advertised "
                    f"inventory | provider={provider} | "
                    f"exec_id={execution.exec_id} | {self._log_context()}"
                )
                return
            native = _native_exec_tool_call(
                execution,
                self.tool_name_aliases,
            )
            if native is None:
                await self._send_exec_throw(
                    execution.id,
                    "MCP execution request did not contain a tool name",
                )
                self._mark_progress()
                return
            if native.name not in self.tool_names:
                await self._send_exec_result_and_close(
                    execution.id,
                    agent_pb.ExecClientMessage(
                        id=execution.id,
                        exec_id=execution.exec_id,
                        mcp_result=agent_pb.McpResult(
                            tool_not_found=agent_pb.McpToolNotFound(
                                name=native.name,
                                available_tools=sorted(self.tool_names),
                            )
                        ),
                    ),
                )
                self._mark_progress()
                logger.warn(
                    "Agent requested an MCP tool outside the advertised inventory | "
                    f"tool={native.name} | exec_id={execution.exec_id} | "
                    f"{self._log_context()}"
                )
                return
            raw_call_id = (
                execution.mcp_args.tool_call_id
                or execution.exec_id
                or str(execution.id)
            )
            pending = _PendingExecution(
                id=execution.id,
                exec_id=execution.exec_id,
                raw_call_id=raw_call_id,
                public_call_id=native.call_id,
                call=native,
            )
            self._register_pending(pending)
            pending.heartbeat_task = asyncio.create_task(
                self._execution_heartbeats(pending),
                name=f"cursor-exec-heartbeat-{execution.id}",
            )
            self._mark_progress()
            logger.debug(
                "Agent execution request received; preserving RunSSE session | "
                f"tool={native.name} | call_id={native.call_id} | "
                f"exec_id={execution.exec_id} | {self._log_context()}"
            )
            await self._queue_tool_start(native)
            await self.events.put(_SessionEvent("tool", native))
            return

        self.tool_candidate_seen = True
        self.execution_boundary_seen = True
        wire_fields = _wire_field_numbers(execution.SerializeToString())
        detail = (
            "Unsupported Cursor local execution lane "
            f"{kind or 'unknown'} (wire_fields={wire_fields})"
        )
        local_call_id = _native_local_exec_tool_call_id(execution, kind)
        rejection = _native_local_exec_rejection(execution, kind)
        self._finish_tool_assembly(
            local_call_id,
            execution.exec_id,
            str(execution.id) if execution.id else "",
            allow_unresolved_single=rejection is not None,
        )
        if rejection is not None:
            await self._send_exec_result_and_close(execution.id, rejection)
            logger.warn(
                "Rejected Cursor native local execution with a typed policy "
                f"result | lane={kind} | exec_id={execution.exec_id} | "
                f"wire_fields={wire_fields} | {self._log_context()}"
            )
        else:
            logger.warn(
                f"{detail} | exec_id={execution.exec_id} | "
                f"{self._log_context()}"
            )
            await self._send_exec_throw(execution.id, detail)
        self._mark_progress()

    async def _handle_kv(self, message: agent_pb.KvServerMessage) -> None:
        kind = message.WhichOneof("message")
        if kind == "get_blob_args":
            blob_id = bytes(message.get_blob_args.blob_id)
            result = agent_pb.GetBlobResult()
            if blob_id in self.blobs:
                result.blob_data = self.blobs[blob_id]
            response = agent_pb.KvClientMessage(id=message.id, get_blob_result=result)
        elif kind == "set_blob_args":
            blob_id = bytes(message.set_blob_args.blob_id)
            blob_data = bytes(message.set_blob_args.blob_data)
            existing_size = len(self.blobs.get(blob_id, b""))
            projected_total = (
                sum(len(value) for value in self.blobs.values())
                - existing_size
                + len(blob_data)
            )
            if (
                len(blob_data) > _MAX_KV_BLOB_BYTES
                or projected_total > _MAX_KV_TOTAL_BYTES
            ):
                set_result = agent_pb.SetBlobResult(
                    error=agent_pb.Error(
                        message="Cursor Agent KV blob exceeds proxy memory limits"
                    )
                )
            else:
                self.blobs[blob_id] = blob_data
                set_result = agent_pb.SetBlobResult()
            response = agent_pb.KvClientMessage(
                id=message.id,
                set_blob_result=set_result,
            )
        else:
            raise RuntimeError(
                "Cursor Agent sent an unsupported KV lane: "
                f"{kind or 'unknown'}"
            )
        await self._append_message(
            agent_pb.AgentClientMessage(kv_client_message=response)
        )
        self._mark_progress()

    async def _handle_query(self, query: agent_pb.InteractionQuery) -> None:
        # An approved interaction query can trigger upstream work.  Even if no
        # downstream-visible candidate has been emitted yet, retrying the whole
        # turn could duplicate that work.
        self.tool_candidate_seen = True
        kind = query.WhichOneof("query")
        if kind == "web_search_request_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                web_search_request_response=agent_pb.WebSearchRequestResponse(
                    approved=agent_pb.WebSearchRequestResponse.Approved()
                ),
            )
        elif kind == "ask_question_interaction_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                ask_question_interaction_response=(
                    agent_pb.AskQuestionInteractionResponse(
                        result=agent_pb.AskQuestionResult(
                            rejected=agent_pb.AskQuestionRejected(
                                reason=_ASK_QUESTION_REJECTION_REASON
                            )
                        )
                    )
                ),
            )
        elif kind == "switch_mode_request_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                switch_mode_request_response=(
                    agent_pb.SwitchModeRequestResponse(
                        rejected=agent_pb.SwitchModeRequestResponse.Rejected(
                            reason=_INTERACTION_REJECTION_REASON
                        )
                    )
                ),
            )
        elif kind == "create_plan_request_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                create_plan_request_response=(
                    agent_pb.CreatePlanRequestResponse(
                        result=agent_pb.CreatePlanResult(
                            error=agent_pb.CreatePlanError(
                                error=_INTERACTION_REJECTION_REASON
                            )
                        )
                    )
                ),
            )
        elif kind == "web_fetch_request_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                web_fetch_request_response=agent_pb.WebFetchRequestResponse(
                    approved=agent_pb.WebFetchRequestResponse.Approved()
                ),
            )
        elif kind == "pr_management_request_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                pr_management_result=agent_pb.PrManagementResult(
                    rejected=agent_pb.PrManagementRejected(
                        reason=_INTERACTION_REJECTION_REASON
                    )
                ),
            )
        elif kind == "mcp_auth_request_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                mcp_auth_request_response=agent_pb.McpAuthRequestResponse(
                    rejected=agent_pb.McpAuthRequestResponse.Rejected(
                        reason=_INTERACTION_REJECTION_REASON
                    )
                ),
            )
        elif kind == "generate_image_request_query":
            description = query.generate_image_request_query.args.description
            response = agent_pb.InteractionResponse(
                id=query.id,
                generate_image_request_response=agent_pb.GenerateImageRequestResponse(
                    approved=agent_pb.GenerateImageRequestResponse.Approved(
                        description=description
                    )
                ),
            )
        elif kind == "replace_env_args":
            response = agent_pb.InteractionResponse(
                id=query.id,
                replace_env_result=agent_pb.ReplaceEnvResult(
                    failure=agent_pb.ReplaceEnvFailure(
                        error_message=_INTERACTION_REJECTION_REASON
                    )
                ),
            )
        elif kind == "connect_scm_request_query":
            response = agent_pb.InteractionResponse(
                id=query.id,
                connect_scm_request_response=(
                    agent_pb.ConnectScmRequestResponse(
                        rejected=agent_pb.ConnectScmRequestResponse.Rejected(
                            reason=_INTERACTION_REJECTION_REASON
                        )
                    )
                ),
            )
        else:
            raise RuntimeError(
                "Cursor Agent sent an unsupported interaction query: "
                f"{kind or 'unknown'}"
            )
        await self._append_message(
            agent_pb.AgentClientMessage(interaction_response=response)
        )
        self._mark_progress()

    async def _handle_interaction(
        self,
        update: agent_pb.InteractionUpdate,
    ) -> bool:
        kind = update.WhichOneof("message")
        if kind == "text_delta":
            delta = update.text_delta.text
            if delta:
                self.semantic_output_seen = True
                self._mark_progress()
                await self.events.put(_SessionEvent("text", delta))
        elif kind == "thinking_delta":
            delta = update.thinking_delta.text
            if delta:
                self.semantic_output_seen = True
                self._mark_progress()
                await self.events.put(_SessionEvent("thinking", delta))
        elif kind == "partial_tool_call":
            self.tool_candidate_seen = True
            partial = update.partial_tool_call
            partial_ids, partial_has_public_id = _tool_update_identifiers(partial)
            self._start_tool_assembly(
                partial_ids,
                has_public_id=partial_has_public_id,
            )
            key = partial.call_id or partial.model_call_id or "unknown"
            fingerprint = partial.tool_call.SerializeToString()
            previous = self.partial_fingerprints.get(key)
            delta_bytes = len(partial.args_text_delta.encode("utf-8"))
            changed = bool(delta_bytes) or fingerprint != previous
            if changed:
                self.partial_fingerprints[key] = fingerprint
                self.partial_tool_argument_bytes += delta_bytes
                self.partial_tool_update_count += 1
                self.partial_tool_snapshot_bytes = max(
                    self.partial_tool_snapshot_bytes,
                    len(fingerprint),
                )
                self.active_request_partial_snapshot_bytes = max(
                    self.active_request_partial_snapshot_bytes,
                    len(fingerprint),
                )
                self._mark_progress()
                now = time.monotonic()
                if now - self.last_progress_log_at >= 10:
                    logger.debug(
                        "Agent tool arguments still streaming | "
                        f"agent_run_updates={self.partial_tool_update_count} | "
                        "active_request_updates="
                        f"{self._active_request_partial_update_count()} | "
                        "agent_run_delta_bytes="
                        f"{self.partial_tool_argument_bytes} | "
                        "active_request_delta_bytes="
                        f"{self._active_request_partial_argument_bytes()} | "
                        "agent_run_max_snapshot_bytes="
                        f"{self.partial_tool_snapshot_bytes} | "
                        "active_request_max_snapshot_bytes="
                        f"{self.active_request_partial_snapshot_bytes} | "
                        f"active_assemblies={len(self.tool_assembly_groups)} | "
                        "agent_run_elapsed_ms="
                        f"{int((now - self.started_at) * 1000)} | "
                        "active_request_elapsed_ms="
                        f"{int((now - self.active_request_started_at) * 1000)} | "
                        f"{self._log_context()}"
                    )
                    self.last_progress_log_at = now
            if partial.tool_call.HasField("mcp_tool_call"):
                partial_args = partial.tool_call.mcp_tool_call.args
                partial_call_id = _public_call_id(
                    partial.call_id or partial_args.tool_call_id
                )
                logger.debug(
                    "Agent partial MCP metadata | "
                    f"call_id={partial_call_id} | "
                    f"name={partial_args.name} | "
                    f"tool_name={partial_args.tool_name} | "
                    f"provider={partial_args.provider_identifier} | "
                    f"arg_keys={sorted(partial_args.args.keys())} | "
                    f"mcp_arg_wire_fields={_wire_field_numbers(partial_args.SerializeToString())} | "
                    f"mcp_call_wire_fields={_wire_field_numbers(partial.tool_call.mcp_tool_call.SerializeToString())} | "
                    f"{self._log_context()}"
                )
                await self._queue_tool_start(
                    _native_mcp_call(
                        partial.tool_call.mcp_tool_call.args,
                        partial.call_id or partial.tool_call.tool_call_id,
                        allow_generated_call_id=False,
                        name_aliases=self.tool_name_aliases,
                        expected_provider=CLIENT_TOOL_PROVIDER,
                    )
                )
        elif kind in ("tool_call_started", "tool_call_completed"):
            self.tool_candidate_seen = True
            # These are presentation/status updates. The synchronous execution
            # boundary is ExecServerMessage.mcp_args, not tool_call_started.
            status = getattr(update, kind)
            status_ids, status_has_public_id = _tool_update_identifiers(status)
            if kind == "tool_call_started":
                self._start_tool_assembly(
                    status_ids,
                    has_public_id=status_has_public_id,
                )
            else:
                self._finish_tool_assembly(
                    *status_ids,
                    allow_unresolved_single=True,
                )
            self._mark_transition(kind, status.SerializeToString())
            await self._queue_tool_start(
                _native_tool_call(status, self.tool_name_aliases)
            )
        elif kind == "tool_call_delta":
            self.tool_candidate_seen = True
            delta = update.tool_call_delta
            delta_ids, delta_has_public_id = _tool_update_identifiers(delta)
            self._start_tool_assembly(
                delta_ids,
                has_public_id=delta_has_public_id,
            )
            self._mark_transition(kind, delta.SerializeToString())
        elif kind == "thinking_completed":
            self._mark_transition(kind, update.thinking_completed.SerializeToString())
        elif kind == "heartbeat":
            now = time.monotonic()
            quiet = now - self.last_semantic_progress_at
            if quiet >= 30 and now - self.last_quiet_log_at >= 30:
                logger.warn(
                    "Agent stream alive without semantic progress | "
                    f"quiet_seconds={quiet:.1f} | "
                    f"agent_run_updates={self.partial_tool_update_count} | "
                    "active_request_updates="
                    f"{self._active_request_partial_update_count()} | "
                    "agent_run_max_snapshot_bytes="
                    f"{self.partial_tool_snapshot_bytes} | "
                    "active_request_max_snapshot_bytes="
                    f"{self.active_request_partial_snapshot_bytes} | "
                    f"active_assemblies={len(self.tool_assembly_groups)} | "
                    f"{self._log_context()}"
                )
                self.last_quiet_log_at = now
        elif kind == "turn_ended":
            self._clear_tool_assemblies()
            ended = update.turn_ended
            self.input_tokens = (
                ended.input_tokens if ended.HasField("input_tokens") else None
            )
            self.output_tokens = (
                ended.output_tokens if ended.HasField("output_tokens") else None
            )
            self.terminal_seen = True
            self._mark_progress()
            await self.events.put(_SessionEvent("end"))
            return True
        return False

    async def _handle_server_message(
        self,
        message: agent_pb.AgentServerMessage,
    ) -> bool:
        kind = message.WhichOneof("message")
        if kind == "interaction_update":
            return await self._handle_interaction(message.interaction_update)
        if kind == "exec_server_message":
            await self._handle_exec(message.exec_server_message)
            return False
        if kind == "kv_server_message":
            await self._handle_kv(message.kv_server_message)
            return False
        if kind == "interaction_query":
            await self._handle_query(message.interaction_query)
            return False
        if kind == "exec_server_control_message":
            control_kind = message.exec_server_control_message.WhichOneof("message")
            if control_kind == "abort":
                abort_id = message.exec_server_control_message.abort.id
                detail = f"Cursor Agent aborted local execution {abort_id}"
                await self._fail(detail)
                return True
            raise RuntimeError(
                "Cursor Agent sent an unsupported execution control: "
                f"{control_kind or 'unknown'}"
            )
        if kind == "conversation_checkpoint_update":
            self._mark_transition(
                kind,
                message.conversation_checkpoint_update.SerializeToString(),
            )
            return False
        raise RuntimeError(
            f"Cursor Agent sent an unsupported server lane: {kind or 'unknown'}"
        )

    async def _run(self) -> None:
        decoder = _ConnectFrameDecoder()
        connect_body = _connect_frame(
            bidi_pb.BidiRequestId(request_id=self.request_id).SerializeToString()
        )
        try:
            async with open_streaming_h2_request(
                AGENT_RUN_PATH,
                _request_headers(
                    self.token,
                    self.request_id,
                    self.agent_host,
                    connect=True,
                ),
                connect_body,
                base_url=self.base_url,
            ) as stream:
                await self._append_message(self.run_message)
                self.global_heartbeat_task = asyncio.create_task(
                    self._global_heartbeats(),
                    name=f"cursor-global-heartbeat-{self.request_id}",
                )
                self.watchdog_task = asyncio.create_task(
                    self._watchdog(),
                    name=f"cursor-watchdog-{self.request_id}",
                )
                async for chunk in stream:
                    self.chunk_count += 1
                    should_stop = False
                    for server_message in decoder.feed(chunk):
                        if await self._handle_server_message(server_message):
                            should_stop = True
                            break
                    if should_stop:
                        break
                if not self.terminal_seen and not self.closing and not self.error_enqueued:
                    await self._fail(
                        "Cursor Agent stream ended before a terminal interaction update"
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.closing:
                await self._fail(str(exc))
        finally:
            await self._cancel_task(self.global_heartbeat_task)
            await self._cancel_task(self.watchdog_task)
            for pending in list(self.pending.values()):
                await self._cancel_task(pending.heartbeat_task)
            self._unregister()
            self.closed_event.set()

    async def submit_tool_result(self, message: dict[str, Any]) -> bool:
        call_id = str(message.get("tool_call_id") or "")
        async with self.submit_lock:
            pending = self.pending.get(call_id)
            if pending is None or pending.state != "announced":
                return False
            content = _content_text(message.get("content"))
            result_content = [
                agent_pb.McpToolResultContentItem(
                    text=agent_pb.McpTextContent(text=content)
                )
            ]
            media = message.get(_TOOL_RESULT_MEDIA_KEY)
            if isinstance(media, list):
                for item in media:
                    if (
                        not isinstance(item, tuple)
                        or len(item) != 2
                        or item[0] not in _SUPPORTED_TOOL_RESULT_IMAGE_MIME_TYPES
                        or not isinstance(item[1], bytes)
                    ):
                        continue
                    result_content.append(
                        agent_pb.McpToolResultContentItem(
                            image=agent_pb.McpImageContent(
                                mime_type=item[0],
                                data=item[1],
                            )
                        )
                    )
            success = agent_pb.McpSuccess(
                content=result_content,
                is_error=bool(message.get("is_error")),
            )
            result = agent_pb.ExecClientMessage(
                id=pending.id,
                exec_id=pending.exec_id,
                mcp_result=agent_pb.McpResult(success=success),
            )
            async with self.append_lock:
                if pending.state != "announced":
                    return False
                pending.state = "result_sent"
                self.tool_result_submitted = True
                await self._append_messages_locked(
                    [
                        agent_pb.AgentClientMessage(exec_client_message=result),
                        agent_pb.AgentClientMessage(
                            exec_client_control_message=(
                                agent_pb.ExecClientControlMessage(
                                    stream_close=agent_pb.ExecClientStreamClose(
                                        id=pending.id
                                    )
                                )
                            )
                        ),
                    ]
                )
            await self._cancel_task(pending.heartbeat_task)
            self.pending.pop(call_id, None)
            if pending.exec_id:
                self.pending_by_exec_id.pop(pending.exec_id, None)
            self._mark_progress()
            logger.debug(
                "Agent client-tool result returned on original session | "
                f"call_id={call_id} | exec_id={pending.exec_id} | "
                f"is_error={bool(message.get('is_error'))} | "
                f"media_count={len(result_content) - 1} | "
                f"{self._log_context()}"
            )
            return True

    async def _collect_inner(
        self,
        *,
        on_text_delta: Callable[[str], Any] | None,
        on_thinking_delta: Callable[[str], Any] | None,
        on_tool_call_start: Callable[[str, str], Any] | None,
    ) -> dict[str, Any]:
        async with self.collect_lock:
            started_at = time.monotonic()
            first_event_latency_ms: float | None = None
            start_chunks = self.chunk_count
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            native_calls: list[NativeToolCall] = []
            errors: list[str] = []
            text_delta_count = 0
            thinking_delta_count = 0
            consumed_events = 0

            while True:
                event = (
                    self.deferred_events.popleft()
                    if self.deferred_events
                    else await self.events.get()
                )
                consumed_events += 1
                if first_event_latency_ms is None:
                    first_event_latency_ms = (time.monotonic() - started_at) * 1000
                if event.kind == "text":
                    value = str(event.value or "")
                    text_parts.append(value)
                    text_delta_count += 1
                    await _invoke_callback(on_text_delta, value)
                elif event.kind == "thinking":
                    value = str(event.value or "")
                    thinking_parts.append(value)
                    thinking_delta_count += 1
                    await _invoke_callback(on_thinking_delta, value)
                elif event.kind == "tool_start":
                    call_id, name = event.value
                    await _invoke_tool_start_callback(
                        on_tool_call_start,
                        str(call_id),
                        str(name),
                    )
                elif event.kind == "tool":
                    native_calls.append(event.value)
                    # Parallel MCP executions are normally delivered together.
                    # Let the reader finish the current upstream batch, then
                    # return every immediately available tool boundary.
                    await asyncio.sleep(0)
                    while True:
                        try:
                            following = self.events.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                        if following.kind == "tool_start":
                            call_id, name = following.value
                            await _invoke_tool_start_callback(
                                on_tool_call_start,
                                str(call_id),
                                str(name),
                            )
                            consumed_events += 1
                            continue
                        if following.kind != "tool":
                            self.deferred_events.append(following)
                            break
                        native_calls.append(following.value)
                        consumed_events += 1
                    break
                elif event.kind == "end":
                    break
                elif event.kind == "error":
                    errors.append(str(event.value or "Cursor Agent session failed"))
                    break

            text = "".join(text_parts)
            thinking = "".join(thinking_parts)
            fatal = bool(errors)
            result = {
                "text": text,
                "thinking": thinking,
                "composer_tool_calls": [],
                "native_tool_calls": native_calls,
                "interrupted_tool_state": "",
                "errors": errors,
                "context_remaining_percent": None,
                "had_content": bool(text or thinking or native_calls),
                "has_fatal_error": fatal,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                },
                "metrics": {
                    "stream_duration_ms": (time.monotonic() - started_at) * 1000,
                    "first_chunk_latency_ms": (
                        first_event_latency_ms
                        if first_event_latency_ms is not None
                        else -1
                    ),
                    "chunk_count": max(
                        consumed_events,
                        self.chunk_count - start_chunks,
                    ),
                    "text_delta_count": text_delta_count,
                    "thinking_delta_count": thinking_delta_count,
                    "partial_tool_argument_bytes": self.partial_tool_argument_bytes,
                    "partial_tool_update_count": self.partial_tool_update_count,
                    "agent_run_partial_tool_argument_bytes": (
                        self.partial_tool_argument_bytes
                    ),
                    "agent_run_partial_tool_update_count": (
                        self.partial_tool_update_count
                    ),
                    "agent_run_max_partial_tool_snapshot_bytes": (
                        self.partial_tool_snapshot_bytes
                    ),
                    "active_request_partial_tool_argument_bytes": (
                        self._active_request_partial_argument_bytes()
                    ),
                    "active_request_partial_tool_update_count": (
                        self._active_request_partial_update_count()
                    ),
                    "active_request_max_partial_tool_snapshot_bytes": (
                        self.active_request_partial_snapshot_bytes
                    ),
                    "active_tool_assembly_group_count": len(
                        self.tool_assembly_groups
                    ),
                    "failure_tool_assembly_group_count": (
                        self.failure_tool_assembly_group_count
                    ),
                    "protocol_error_count": len(errors),
                },
            }
            result["replay_safe"] = self._has_no_observed_work(result)
            return result

    async def collect(
        self,
        *,
        on_text_delta: Callable[[str], Any] | None,
        on_thinking_delta: Callable[[str], Any] | None,
        on_tool_call_start: Callable[[str, str], Any] | None,
    ) -> dict[str, Any]:
        async with self.collector_state_lock:
            if self.collector_active:
                raise _CollectorAlreadyAttached(
                    "Cursor Agent session already has an attached downstream collector"
                )
            self.collector_active = True
        try:
            return await self._collect_inner(
                on_text_delta=on_text_delta,
                on_thinking_delta=on_thinking_delta,
                on_tool_call_start=on_tool_call_start,
            )
        finally:
            async with self.collector_state_lock:
                self.collector_active = False

    async def close(self, reason: str = "client request cancelled") -> None:
        if self.closing:
            return
        self.closing = True
        self._clear_tool_assemblies()
        live_pending = [
            pending
            for pending in self.pending.values()
            if pending.state == "announced"
        ]
        if live_pending:
            with suppress(Exception):
                async with self.append_lock:
                    messages: list[agent_pb.AgentClientMessage] = []
                    for pending in live_pending:
                        pending.state = "aborted"
                        messages.extend(
                            [
                                agent_pb.AgentClientMessage(
                                    exec_client_control_message=(
                                        agent_pb.ExecClientControlMessage(
                                            throw=agent_pb.ExecClientThrow(
                                                id=pending.id,
                                                error=reason,
                                                error_code="CLIENT_SESSION_CLOSED",
                                            )
                                        )
                                    )
                                ),
                                agent_pb.AgentClientMessage(
                                    exec_client_control_message=(
                                        agent_pb.ExecClientControlMessage(
                                            stream_close=agent_pb.ExecClientStreamClose(
                                                id=pending.id
                                            )
                                        )
                                    )
                                ),
                            ]
                        )
                    await self._append_messages_locked(messages)
        for pending in live_pending:
            await self._cancel_task(pending.heartbeat_task)
        await self._cancel_task(self.reader_task)
        self._unregister()
        self.closed_event.set()


def _find_resumable_session(
    auth_key: str,
    tool_results: list[dict[str, Any]],
) -> _AgentSession | None:
    candidate_sets: list[set[_AgentSession]] = []
    for result in reversed(tool_results):
        call_id = str(result.get("tool_call_id") or "")
        candidates = {
            session
            for session in _SESSIONS_BY_TOOL_ID.get((auth_key, call_id), set())
            if not session.is_closed
            and not session.terminal_seen
            and not session.closing
            and not session.error_enqueued
            and (pending := session.pending.get(call_id)) is not None
            and pending.state == "announced"
        }
        # Every trailing result belongs to the same tool boundary.  Ignoring an
        # unmatched id here could attach a mixed/foreign continuation to the
        # wrong live session.
        if not candidates:
            return None
        candidate_sets.append(candidates)
    if not candidate_sets:
        return None
    candidates = set.intersection(*candidate_sets)
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        candidate_contexts = "; ".join(
            sorted(session._log_context() for session in candidates)
        )
        logger.error(
            "Ambiguous Cursor Agent resume: multiple live sessions share the "
            "same client tool-call id set | "
            f"candidates=[{candidate_contexts}]"
        )
    return None


def has_resumable_agent_session(
    messages: list[dict[str, Any]],
    auth_token: str,
) -> bool:
    """Whether this request can attach to an existing live Agent execution."""
    return bool(resumable_agent_tool_names(messages, auth_token))


def resumable_agent_tool_names(
    messages: list[dict[str, Any]],
    auth_token: str,
) -> tuple[str, ...]:
    """Return the original inventory for a matching live Agent execution."""
    token = strip_cursor_user_prefix(auth_token).strip()
    auth_key = compute_sha256_hex_digest(token)
    session = _find_resumable_session(
        auth_key,
        _trailing_tool_result_messages(messages),
    )
    return tuple(sorted(session.tool_names)) if session is not None else ()


async def _reset_agent_sessions_for_tests() -> None:
    """Close live sessions so event-loop-scoped tests cannot leak tasks."""
    sessions = list(_ACTIVE_SESSIONS)
    for session in sessions:
        await session.close("test cleanup")
    _SESSIONS_BY_TOOL_ID.clear()
    _ACTIVE_SESSIONS.clear()
    _WORKSPACE_EXCLUSION_UNSUPPORTED.clear()


async def call_cursor_agent(
    messages: list[dict[str, Any]],
    model: str,
    tools: list[dict],
    auth_token: str,
    *,
    on_text_delta: Callable[[str], Any] | None = None,
    on_thinking_delta: Callable[[str], Any] | None = None,
    on_tool_call_start: Callable[[str, str], Any] | None = None,
    client_request_id: str | None = None,
) -> dict[str, Any]:
    """Start or resume a Cursor Agent session at an external tool boundary."""
    token = strip_cursor_user_prefix(auth_token).strip()
    auth_key = compute_sha256_hex_digest(token)
    base_url = os.environ.get(
        "CURSOR_AGENT_BASE_URL", DEFAULT_AGENT_BASE_URL
    ).rstrip("/")
    capability_key = _workspace_exclusion_cache_key(auth_key, model, base_url)
    exclude_workspace_context = not _workspace_exclusion_is_unsupported(
        capability_key
    )
    tool_results = _trailing_tool_result_messages(messages)
    session = _find_resumable_session(auth_key, tool_results)
    try:
        if session is None:
            session = _AgentSession(
                messages,
                model,
                tools,
                auth_token,
                client_request_id=client_request_id,
                exclude_workspace_context=exclude_workspace_context,
            )
            if tool_results:
                unmatched_ids = [
                    str(result.get("tool_call_id") or "")
                    for result in tool_results
                ]
                logger.warn(
                    "No live Cursor Agent session matched trailing tool results; "
                    "falling back to a new transcript-backed run | "
                    f"unmatched_call_ids={unmatched_ids} | "
                    f"{session._log_context()}"
                )
            session.start()
        else:
            session.bind_active_request(client_request_id)
            for result in tool_results:
                call_id = str(result.get("tool_call_id") or "")
                if session in _SESSIONS_BY_TOOL_ID.get(
                    (auth_key, call_id), set()
                ):
                    await session.submit_tool_result(result)
        workspace_retry_used = False
        semantic_retry_used = False
        retry_metrics: dict[str, Any] = {}
        while True:
            result = await session.collect(
                on_text_delta=on_text_delta,
                on_thinking_delta=on_thinking_delta,
                on_tool_call_start=on_tool_call_start,
            )

            retry_kind = ""
            retry_exclude_workspace_context = session.exclude_workspace_context
            if (
                not workspace_retry_used
                and session.can_retry_workspace_exclusion_rejection(result)
            ):
                retry_kind = "workspace_exclusion_capability"
                retry_exclude_workspace_context = False
                workspace_retry_used = True
                session_capability_key = _workspace_exclusion_cache_key(
                    session.auth_key,
                    session.model,
                    session.base_url,
                )
                _remember_workspace_exclusion_unsupported(
                    session_capability_key
                )
                retry_metrics["workspace_exclusion_retry_count"] = 1
                retry_metrics["workspace_exclusion_initial_errors"] = list(
                    result.get("errors") or []
                )
            elif (
                not semantic_retry_used
                and session.can_retry_zero_content_stall(result)
            ):
                retry_kind = "zero_content_semantic_stall"
                semantic_retry_used = True
                retry_metrics["zero_content_retry_count"] = 1
                retry_metrics["zero_content_retry_initial_errors"] = list(
                    result.get("errors") or []
                )

            if not retry_kind:
                if retry_metrics:
                    metrics = dict(result.get("metrics") or {})
                    metrics.update(retry_metrics)
                    result["metrics"] = metrics
                return result

            failed_agent_request_id = session.request_id
            logical_session_id = session.session_id
            origin_request_id = getattr(
                session,
                "origin_request_id",
                getattr(session, "client_request_id", "unavailable"),
            )
            await session.close(f"retrying {retry_kind}")
            retry_session = _AgentSession(
                messages,
                model,
                tools,
                auth_token,
                session_id=logical_session_id,
                client_request_id=client_request_id,
                exclude_workspace_context=(
                    retry_exclude_workspace_context
                ),
            )
            if hasattr(retry_session, "origin_request_id"):
                retry_session.origin_request_id = origin_request_id
            if retry_kind == "workspace_exclusion_capability":
                logger.warn(
                    "Cursor rejected optional workspace-context exclusion; "
                    "retrying once with field 12 unset while retaining MCP-only "
                    "tool headers and typed request-context fallback | "
                    f"failed_agent_request_id={failed_agent_request_id} | "
                    f"{retry_session._log_context()}"
                )
            else:
                logger.warn(
                    "Retrying one zero-content semantic stall with a "
                    "transcript-backed Agent run; no text, reasoning, tool "
                    "candidate, execution boundary, or submitted tool result "
                    "was observed | "
                    f"failed_agent_request_id={failed_agent_request_id} | "
                    f"{retry_session._log_context()}"
                )
            session = retry_session
            session.start()
    except asyncio.CancelledError:
        # A downstream disconnect while the model is generating has no future
        # request on which to deliver the queued output. Tool-boundary returns
        # are normal returns and therefore keep the session alive.
        if session is not None:
            await session.close("downstream request cancelled")
        raise
    except _CollectorAlreadyAttached:
        raise
    except Exception:
        if session is not None:
            await session.close("agent collection failed")
        raise
