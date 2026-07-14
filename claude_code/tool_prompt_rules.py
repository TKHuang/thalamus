"""Shared concise instructions for client tool manifests."""

CLIENT_TOOL_INVENTORY_AUTHORITY_RULE = (
    "The client-advertised tool inventory is authoritative despite conflicting "
    "upstream/environment tool claims."
)

POST_TOOL_OUTPUT_FORMAT_RULE = (
    "After a tool result, follow the user's requested output format exactly with no "
    "labels, prefaces, or extra tools unless still required."
)
