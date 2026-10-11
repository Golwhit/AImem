"""A Model Context Protocol server that exposes the memory store over stdio.

Implemented with the standard library only: newline delimited JSON-RPC 2.0
messages on stdin/stdout, exactly as the MCP stdio transport specifies. The
``mcp`` Python SDK is deliberately *not* used (it cannot be installed on this
machine), so the protocol surface is implemented by hand and covered by tests.

Supported protocol revisions: 2025-06-18, 2025-03-26, 2024-11-05.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, IO, Mapping, Sequence

from . import __version__
from .store import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    ORDERINGS,
    TAG_MODES,
    MemoryNotFound,
    MemoryStore,
)

LATEST_PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")

SERVER_NAME = "aimem"
SERVER_TITLE = "AI Mem"

# JSON-RPC 2.0 / MCP error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

INSTRUCTIONS = (
    "Local long-term memory for the user. Call memory_search before asking the "
    "user to repeat context, and memory_add when you learn a durable fact, "
    "preference or decision. Keep memories short, self-contained sentences."
)


class MethodNotFound(Exception):
    """Raised for unknown JSON-RPC methods."""


class ToolError(Exception):
    """Raised inside a tool handler; becomes a result with ``isError: true``."""


# ----------------------------------------------------------- argument parsing


def _require_text(args: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    value = args.get(key, default)
    if value is None:
        raise ToolError(f"'{key}' is required and must be a non-empty string")
    if not isinstance(value, str):
        raise ToolError(f"'{key}' must be a string")
    if not value.strip():
        raise ToolError(f"'{key}' must not be empty")
    return value


def _optional_text(args: Mapping[str, Any], key: str, default: str | None = None) -> str | None:
    value = args.get(key, None)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ToolError(f"'{key}' must be a string")
    return value


def _bool_arg(args: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = args.get(key, None)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ToolError(f"'{key}' must be a boolean")


def _int_arg(
    args: Mapping[str, Any],
    key: str,
    default: int | None = None,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = args.get(key, None)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ToolError(f"'{key}' must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"'{key}' must be an integer") from exc
    if minimum is not None and number < minimum:
        number = minimum
    if maximum is not None and number > maximum:
        number = maximum
    return number


def _tags_arg(args: Mapping[str, Any], key: str = "tags") -> Any:
    value = args.get(key, None)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    raise ToolError(f"'{key}' must be a string or a list of strings")


def _enum_arg(args: Mapping[str, Any], key: str, choices: Sequence[str], default: str) -> str:
    value = args.get(key, None)
    if value is None:
        return default
    if value not in choices:
        raise ToolError(f"'{key}' must be one of {list(choices)}")
    return str(value)


def _annotations(*, read_only: bool, destructive: bool = False) -> dict[str, Any]:
    return {
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": True,
        "openWorldHint": False,
    }


# ------------------------------------------------------------------ registry

TOOLS: dict[str, dict[str, Any]] = {}


def _register(definition: dict[str, Any], handler: Callable[["McpServer", Mapping[str, Any]], dict[str, Any]]) -> None:
    TOOLS[definition["name"]] = {"definition": definition, "handler": handler}


def tool_definitions() -> list[dict[str, Any]]:
    """The ``tools/list`` payload."""
    return [entry["definition"] for entry in TOOLS.values()]


# --------------------------------------------------------------- memory_add


def _handle_add(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    record = server.store.add(
        _require_text(args, "text"),
        tags=_tags_arg(args),
        source=_optional_text(args, "source", "mcp") or "mcp",
        pinned=_bool_arg(args, "pinned"),
    )
    return {"memory": record}


_register(
    {
        "name": "memory_add",
        "title": "Add memory",
        "description": (
            "Store a durable fact, preference or decision in the user's local "
            "memory. Identical text (case-insensitive) is not duplicated: the "
            "existing memory's tags are merged instead and 'duplicated' is true."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The memory, one short self-contained sentence."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lower-case topic tags, e.g. ['preference', 'tooling'].",
                },
                "source": {"type": "string", "description": "Where the memory came from.", "default": "mcp"},
                "pinned": {"type": "boolean", "description": "Pin it so search ranks it higher.", "default": False},
            },
            "required": ["text"],
            "additionalProperties": False,
        },
        "annotations": _annotations(read_only=False),
    },
    _handle_add,
)


# ------------------------------------------------------------ memory_search


def _handle_search(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    if query is None:
        query = ""
    if not isinstance(query, str):
        raise ToolError("'query' must be a string")
    limit = _int_arg(args, "limit", DEFAULT_LIMIT, minimum=1, maximum=MAX_LIMIT) or DEFAULT_LIMIT
    records = server.store.search(
        query,
        limit=limit,
        tags=_tags_arg(args),
        tag_mode=_enum_arg(args, "tag_mode", TAG_MODES, "any"),
        source=_optional_text(args, "source"),
        pinned_only=_bool_arg(args, "pinned_only"),
    )
    return {"query": query, "count": len(records), "memories": records}


_register(
    {
        "name": "memory_search",
        "title": "Search memory",
        "description": (
            "Ranked full-text search over stored memories (BM25, prefix matching, "
            "accent-insensitive). Terms are ANDed first, then ORed if nothing "
            "matches. An empty query lists the most recent memories."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms; empty lists recent memories."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "default": DEFAULT_LIMIT,
                    "description": "Maximum number of results.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Only memories carrying these tags.",
                },
                "tag_mode": {
                    "type": "string",
                    "enum": list(TAG_MODES),
                    "default": "any",
                    "description": "'any' (default) or 'all' of the given tags.",
                },
                "source": {"type": "string", "description": "Only memories from this source."},
                "pinned_only": {"type": "boolean", "description": "Only pinned memories.", "default": False},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": _annotations(read_only=True),
    },
    _handle_search,
)


# ----------------------------------------------------------------- memory_get


def _handle_get(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    memory_id = _int_arg(args, "id", None, minimum=1)
    if memory_id is None:
        raise ToolError("'id' is required and must be a positive integer")
    return {"memory": server.store.get(memory_id)}


_register(
    {
        "name": "memory_get",
        "title": "Get memory",
        "description": "Fetch one memory by its id.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "Memory id, as returned by memory_add/search."}},
            "required": ["id"],
            "additionalProperties": False,
        },
        "annotations": _annotations(read_only=True),
    },
    _handle_get,
)


# -------------------------------------------------------------- memory_update


def _handle_update(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    memory_id = _int_arg(args, "id", None, minimum=1)
    if memory_id is None:
        raise ToolError("'id' is required and must be a positive integer")
    text = _optional_text(args, "text")
    tags = _tags_arg(args)
    source = _optional_text(args, "source")
    pinned = None if args.get("pinned", None) is None else _bool_arg(args, "pinned")
    if text is None and tags is None and source is None and pinned is None:
        raise ToolError("nothing to update: provide at least one of 'text', 'tags', 'source', 'pinned'")
    record = server.store.update(
        memory_id,
        text=text,
        tags=tags,
        source=source,
        pinned=pinned,
        tag_mode=_enum_arg(args, "tag_mode", ("replace", "add", "remove"), "replace"),
    )
    return {"memory": record}


_register(
    {
        "name": "memory_update",
        "title": "Update memory",
        "description": (
            "Patch an existing memory. Only the supplied fields change. 'tag_mode' "
            "controls tags: 'replace' (default), 'add' to merge, 'remove' to drop."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Memory id to patch."},
                "text": {"type": "string", "description": "Replacement text."},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to replace, add or remove.",
                },
                "tag_mode": {
                    "type": "string",
                    "enum": ["replace", "add", "remove"],
                    "default": "replace",
                    "description": "How to apply 'tags'.",
                },
                "source": {"type": "string", "description": "New source value."},
                "pinned": {"type": "boolean", "description": "New pin state."},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        "annotations": _annotations(read_only=False),
    },
    _handle_update,
)


# -------------------------------------------------------------- memory_delete


def _handle_delete(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    memory_id = _int_arg(args, "id", None, minimum=1)
    if memory_id is None:
        raise ToolError("'id' is required and must be a positive integer")
    return {"memory": server.store.delete(memory_id)}


_register(
    {
        "name": "memory_delete",
        "title": "Delete memory",
        "description": "Permanently delete one memory by id. This cannot be undone.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "integer", "description": "Memory id to delete."}},
            "required": ["id"],
            "additionalProperties": False,
        },
        "annotations": _annotations(read_only=False, destructive=True),
    },
    _handle_delete,
)


# ----------------------------------------------------------------- memory_list


def _handle_list(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    records = server.store.list_memories(
        limit=_int_arg(args, "limit", 20, minimum=1, maximum=MAX_LIMIT) or 20,
        offset=_int_arg(args, "offset", 0, minimum=0) or 0,
        tags=_tags_arg(args),
        tag_mode=_enum_arg(args, "tag_mode", TAG_MODES, "any"),
        source=_optional_text(args, "source"),
        pinned_only=_bool_arg(args, "pinned_only"),
        order=_enum_arg(args, "order", ORDERINGS, "recent"),
    )
    return {"count": len(records), "memories": records}


_register(
    {
        "name": "memory_list",
        "title": "List memories",
        "description": "List stored memories, newest first by default. Use memory_search for relevance ranking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "default": 20},
                "offset": {"type": "integer", "minimum": 0, "default": 0, "description": "Skip N results."},
                "order": {"type": "string", "enum": list(ORDERINGS), "default": "recent"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "tag_mode": {"type": "string", "enum": list(TAG_MODES), "default": "any"},
                "source": {"type": "string"},
                "pinned_only": {"type": "boolean", "default": False},
            },
            "additionalProperties": False,
        },
        "annotations": _annotations(read_only=True),
    },
    _handle_list,
)


# ----------------------------------------------------------------- memory_tags


def _handle_tags(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    items = server.store.tags(min_count=_int_arg(args, "min_count", 1, minimum=1) or 1)
    return {"count": len(items), "tags": items}


_register(
    {
        "name": "memory_tags",
        "title": "List memory tags",
        "description": "List every tag in use with the number of memories carrying it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_count": {"type": "integer", "minimum": 1, "default": 1},
            },
            "additionalProperties": False,
        },
        "annotations": _annotations(read_only=True),
    },
    _handle_tags,
)


# ---------------------------------------------------------------- memory_stats


def _handle_stats(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    return {"stats": server.store.stats()}


_register(
    {
        "name": "memory_stats",
        "title": "Memory store statistics",
        "description": "Totals, sources, top tags, database path and search index health.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": _annotations(read_only=True),
    },
    _handle_stats,
)


# -------------------------------------------------------------- memory_reindex


def _handle_reindex(server: "McpServer", args: Mapping[str, Any]) -> dict[str, Any]:
    return server.store.reindex()


_register(
    {
        "name": "memory_reindex",
        "title": "Rebuild search index",
        "description": (
            "Rebuild the full-text index from the stored memories. Use only if "
            "memory_stats reports index_ok false."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": _annotations(read_only=False),
    },
    _handle_reindex,
)


# ---------------------------------------------------------------- MCP server


def _error_text(exc: BaseException) -> str:
    """Unwrap KeyError-style messages so tool errors read cleanly."""
    if exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def tool_result(payload: dict[str, Any], *, is_error: bool = False) -> dict[str, Any]:
    """Build a ``tools/call`` result carrying both text and structured content."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}],
        "structuredContent": payload,
        "isError": is_error,
    }


class McpServer:
    """Hand-rolled MCP server: one JSON-RPC message in, one response out."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        name: str = SERVER_NAME,
        version: str = __version__,
    ) -> None:
        self.store = store
        self.name = name
        self.version = version
        self.protocol_version: str | None = None
        self.client_info: dict[str, Any] | None = None
        self.initialized = False

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Handle one decoded JSON-RPC message.

        Returns the response object, or ``None`` for notifications (which per
        JSON-RPC must never be answered).
        """
        if not isinstance(message, dict):
            return error_response(None, INVALID_REQUEST, "message must be a JSON object")
        if message.get("jsonrpc") != "2.0":
            return error_response(message.get("id"), INVALID_REQUEST, 'jsonrpc must be "2.0"')
        method = message.get("method")
        if not isinstance(method, str):
            return error_response(message.get("id"), INVALID_REQUEST, "method must be a string")
        if "id" not in message:
            self._handle_notification(method)
            return None

        request_id = message.get("id")
        params = message.get("params")
        try:
            result = self._dispatch(method, {} if params is None else params)
        except MethodNotFound as exc:
            return error_response(request_id, METHOD_NOT_FOUND, str(exc))
        except ValueError as exc:
            return error_response(request_id, INVALID_PARAMS, _error_text(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return error_response(request_id, INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _handle_notification(self, method: str) -> None:
        if method == "notifications/initialized":
            self.initialized = True
        # notifications/cancelled and anything else are ignored on purpose.

    def _dispatch(self, method: str, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ValueError("params must be a JSON object")
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": tool_definitions()}
        if method == "tools/call":
            return self._call_tool(params)
        if method == "resources/list":
            return {"resources": []}
        if method == "resources/templates/list":
            return {"resourceTemplates": []}
        if method == "prompts/list":
            return {"prompts": []}
        if method == "logging/setLevel":
            return {}
        raise MethodNotFound(f"Method not found: {method}")

    def _initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        requested = params.get("protocolVersion")
        negotiated = (
            requested
            if isinstance(requested, str) and requested in SUPPORTED_PROTOCOL_VERSIONS
            else LATEST_PROTOCOL_VERSION
        )
        self.protocol_version = negotiated
        client_info = params.get("clientInfo")
        self.client_info = client_info if isinstance(client_info, dict) else None
        return {
            "protocolVersion": negotiated,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "title": SERVER_TITLE, "version": self.version},
            "instructions": f"{INSTRUCTIONS} Database file: {self.store.path}",
        }

    def _call_tool(self, params: Mapping[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str):
            raise ValueError("tools/call requires a 'name' string")
        entry = TOOLS.get(name)
        if entry is None:
            # Protocol level error (-32602), per the MCP tools specification.
            raise ValueError(f"Unknown tool: {name}")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return tool_result({"error": "'arguments' must be a JSON object"}, is_error=True)
        try:
            payload = entry["handler"](self, arguments)
        except (ToolError, ValueError, MemoryNotFound) as exc:
            # Tool execution errors are reported in the result, not as -326xx.
            return tool_result({"error": _error_text(exc)}, is_error=True)
        except OSError as exc:  # e.g. the database file is not writable
            return tool_result({"error": f"{type(exc).__name__}: {exc}"}, is_error=True)
        return tool_result(payload)


def serve_stdio(
    store: MemoryStore | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
) -> int:
    """Serve MCP over newline-delimited JSON on stdin/stdout until EOF."""
    owns_store = store is None
    server = McpServer(store if store is not None else MemoryStore())
    in_stream = sys.stdin if stdin is None else stdin
    out_stream = sys.stdout if stdout is None else stdout
    try:
        for line in in_stream:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                response: dict[str, Any] | None = error_response(
                    None, PARSE_ERROR, f"Invalid JSON: {exc.msg}"
                )
            else:
                response = server.handle_message(message)
            if response is None:
                continue
            out_stream.write(json.dumps(response, ensure_ascii=False) + "\n")
            out_stream.flush()
    except (BrokenPipeError, KeyboardInterrupt):  # pragma: no cover - client vanished
        pass
    finally:
        if owns_store:
            server.store.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    sys.exit(serve_stdio())
