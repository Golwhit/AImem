"""Tests for the hand-rolled MCP server: protocol handling and tool behaviour."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from aimem import __version__
from aimem.mcp_server import (
    LATEST_PROTOCOL_VERSION,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    TOOLS,
    McpServer,
    error_response,
    serve_stdio,
)
from aimem.store import MemoryStore

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TOOLS = {
    "memory_add",
    "memory_search",
    "memory_get",
    "memory_update",
    "memory_delete",
    "memory_list",
    "memory_tags",
    "memory_stats",
    "memory_reindex",
}
READ_ONLY_TOOLS = {
    "memory_search",
    "memory_get",
    "memory_list",
    "memory_tags",
    "memory_stats",
}


class ServerTestCase(unittest.TestCase):
    """Base class wiring a server to a throw-away database."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")
        self.addCleanup(self.store.close)
        self.server = McpServer(self.store)

    def request(self, method: str, params: Any = None, request_id: Any = 1) -> dict[str, Any]:
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        response = self.server.handle_message(message)
        assert response is not None, "requests must always be answered"
        return response

    def call(self, name: str, arguments: dict[str, Any] | None = None, request_id: Any = 2) -> dict:
        return self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}, request_id
        )

    @staticmethod
    def structured(response: dict[str, Any]) -> dict[str, Any]:
        return response["result"]["structuredContent"]


class InitializeTests(ServerTestCase):
    def test_initialize_negotiates_protocol_and_reports_capabilities(self) -> None:
        response = self.request(
            "initialize",
            {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "cline", "version": "3.0.0"},
            },
        )
        result = response["result"]
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(result["protocolVersion"], LATEST_PROTOCOL_VERSION)
        self.assertIn("tools", result["capabilities"])
        self.assertEqual(result["serverInfo"]["name"], "aimem")
        self.assertEqual(result["serverInfo"]["version"], __version__)
        self.assertIn(str(self.store.path), result["instructions"])
        self.assertEqual(self.server.protocol_version, LATEST_PROTOCOL_VERSION)
        self.assertEqual(self.server.client_info, {"name": "cline", "version": "3.0.0"})

    def test_initialize_echoes_an_older_supported_version(self) -> None:
        result = self.request("initialize", {"protocolVersion": "2024-11-05"})["result"]
        self.assertEqual(result["protocolVersion"], "2024-11-05")

    def test_initialize_falls_back_for_unknown_or_missing_versions(self) -> None:
        for version in ("1.0.0", "9999-01-01", None):
            with self.subTest(version=version):
                params = {} if version is None else {"protocolVersion": version}
                result = self.request("initialize", params)["result"]
                self.assertEqual(result["protocolVersion"], LATEST_PROTOCOL_VERSION)


class ProtocolTests(ServerTestCase):
    def test_initialized_notification_is_never_answered(self) -> None:
        response = self.server.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
        self.assertIsNone(response)
        self.assertTrue(self.server.initialized)

    def test_unknown_notifications_are_ignored(self) -> None:
        self.assertIsNone(
            self.server.handle_message({"jsonrpc": "2.0", "method": "notifications/cancelled"})
        )

    def test_ping_and_empty_lists(self) -> None:
        self.assertEqual(self.request("ping")["result"], {})
        self.assertEqual(self.request("resources/list")["result"], {"resources": []})
        self.assertEqual(self.request("prompts/list")["result"], {"prompts": []})
        self.assertEqual(
            self.request("resources/templates/list")["result"], {"resourceTemplates": []}
        )
        self.assertEqual(self.request("logging/setLevel", {"level": "info"})["result"], {})

    def test_unknown_method_returns_method_not_found(self) -> None:
        response = self.request("does/not/exist")
        self.assertEqual(response["error"]["code"], METHOD_NOT_FOUND)
        self.assertNotIn("result", response)

    def test_invalid_envelopes_return_invalid_request(self) -> None:
        for message in (
            "not an object",
            [1, 2, 3],
            {"jsonrpc": "1.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 1, "method": 42},
        ):
            with self.subTest(message=message):
                response = self.server.handle_message(message)
                assert response is not None
                self.assertEqual(response["error"]["code"], INVALID_REQUEST)
                self.assertEqual(response["jsonrpc"], "2.0")

    def test_params_must_be_an_object(self) -> None:
        response = self.request("tools/list", params=[1, 2, 3])
        self.assertEqual(response["error"]["code"], INVALID_PARAMS)

    def test_error_response_helper_shape(self) -> None:
        self.assertEqual(
            error_response(7, METHOD_NOT_FOUND, "nope"),
            {"jsonrpc": "2.0", "id": 7, "error": {"code": METHOD_NOT_FOUND, "message": "nope"}},
        )


class ToolsListTests(ServerTestCase):
    def test_tools_list_exposes_the_expected_tools(self) -> None:
        tools = self.request("tools/list")["result"]["tools"]
        self.assertEqual({tool["name"] for tool in tools}, EXPECTED_TOOLS)
        self.assertEqual(len(tools), len(EXPECTED_TOOLS))
        for tool in tools:
            with self.subTest(tool=tool["name"]):
                self.assertTrue(tool["description"].strip())
                self.assertEqual(tool["inputSchema"]["type"], "object")
                self.assertIn("properties", tool["inputSchema"])
                self.assertIn(tool["name"], TOOLS)

    def test_annotations_mark_read_only_and_destructive_tools(self) -> None:
        by_name = {tool["name"]: tool for tool in self.request("tools/list")["result"]["tools"]}
        for name in EXPECTED_TOOLS:
            with self.subTest(tool=name):
                annotations = by_name[name]["annotations"]
                self.assertEqual(annotations["readOnlyHint"], name in READ_ONLY_TOOLS)
                self.assertFalse(annotations["openWorldHint"])
        self.assertTrue(by_name["memory_delete"]["annotations"]["destructiveHint"])
        self.assertFalse(by_name["memory_add"]["annotations"]["destructiveHint"])


class ToolCallTests(ServerTestCase):
    def test_add_then_search_round_trip(self) -> None:
        added = self.call("memory_add", {
            "text": "Prefers stdlib-only tooling",
            "tags": ["preference", "python"],
            "pinned": True,
        })
        self.assertFalse(added["result"]["isError"])
        memory = self.structured(added)["memory"]
        self.assertEqual(memory["tags"], ["preference", "python"])
        self.assertTrue(memory["pinned"])
        self.assertEqual(memory["source"], "mcp")
        # the text block mirrors the structured content
        self.assertEqual(json.loads(added["result"]["content"][0]["text"]), self.structured(added))

        found = self.call("memory_search", {"query": "stdlib"}, request_id=3)
        payload = self.structured(found)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["query"], "stdlib")
        self.assertEqual(payload["memories"][0]["id"], memory["id"])
        self.assertEqual(payload["memories"][0]["tags"], ["preference", "python"])

    def test_add_deduplicates_identical_text(self) -> None:
        first = self.structured(self.call("memory_add", {"text": "note"}))["memory"]
        second = self.structured(self.call("memory_add", {"text": "NOTE", "tags": ["x"]}))["memory"]
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["duplicated"])
        self.assertEqual(second["tags"], ["x"])
        self.assertEqual(self.store.count(), 1)

    def test_add_requires_text_as_a_tool_error(self) -> None:
        response = self.call("memory_add", {})
        self.assertTrue(response["result"]["isError"])
        self.assertIn("'text' is required", self.structured(response)["error"])
        self.assertNotIn("error", response)  # tool error, not a protocol error

    def test_invalid_argument_types_are_tool_errors(self) -> None:
        for arguments in ({"text": 42}, {"text": "x", "tags": [1]}, {"text": "x", "pinned": "yes"}):
            with self.subTest(arguments=arguments):
                response = self.call("memory_add", arguments)
                self.assertTrue(response["result"]["isError"])
                self.assertIn("error", self.structured(response))

    def test_search_defaults_and_validation(self) -> None:
        self.call("memory_add", {"text": "MCP config location", "tags": ["mcp"]})
        self.assertEqual(self.structured(self.call("memory_search", {"query": ""}))["count"], 1)
        self.assertEqual(
            self.structured(self.call("memory_search", {"query": "mcp", "tags": ["mcp"]}))["count"], 1
        )
        self.assertEqual(
            self.structured(self.call("memory_search", {"query": "mcp", "tags": ["other"]}))["count"], 0
        )
        self.assertTrue(
            self.call("memory_search", {"query": "x", "tag_mode": "sometimes"})["result"]["isError"]
        )
        self.assertTrue(self.call("memory_search", {"query": 5})["result"]["isError"])

    def test_get_update_and_delete_tools(self) -> None:
        memory = self.structured(self.call("memory_add", {"text": "patch me", "tags": ["a"]}))["memory"]
        memory_id = memory["id"]
        patched = self.structured(
            self.call(
                "memory_update",
                {"id": memory_id, "text": "patched", "tags": ["b"], "tag_mode": "add"},
            )
        )["memory"]
        self.assertEqual(patched["text"], "patched")
        self.assertEqual(patched["tags"], ["a", "b"])
        self.assertEqual(
            self.structured(self.call("memory_get", {"id": memory_id}))["memory"]["text"], "patched"
        )
        missing = self.call("memory_get", {"id": 999})
        self.assertTrue(missing["result"]["isError"])
        self.assertIn("no memory with id", self.structured(missing)["error"])
        self.assertTrue(self.call("memory_update", {"id": memory_id})["result"]["isError"])
        deleted = self.structured(self.call("memory_delete", {"id": memory_id}))["memory"]
        self.assertTrue(deleted["deleted"])
        self.assertEqual(self.store.count(), 0)

    def test_list_tags_stats_and_reindex_tools(self) -> None:
        self.call("memory_add", {"text": "one", "tags": ["a"]})
        self.call("memory_add", {"text": "two", "tags": ["a", "b"]})
        listing = self.structured(self.call("memory_list", {"limit": 10}))
        self.assertEqual([m["text"] for m in listing["memories"]], ["two", "one"])
        paged = self.structured(self.call("memory_list", {"limit": 1, "offset": 1}))
        self.assertEqual([m["text"] for m in paged["memories"]], ["one"])
        self.assertEqual(
            self.structured(self.call("memory_tags"))["tags"][0], {"tag": "a", "count": 2}
        )
        stats = self.structured(self.call("memory_stats"))["stats"]
        self.assertEqual(stats["total"], 2)
        self.assertTrue(stats["index_ok"])
        self.assertEqual(
            self.structured(self.call("memory_reindex")), {"reindexed": 2, "index_ok": True}
        )

    def test_unknown_tool_is_a_protocol_error(self) -> None:
        response = self.call("memory_nope", {})
        self.assertEqual(response["error"]["code"], INVALID_PARAMS)
        self.assertEqual(response["error"]["message"], "Unknown tool: memory_nope")
        self.assertNotIn("result", response)

    def test_missing_tool_name_and_bad_arguments(self) -> None:
        self.assertEqual(self.request("tools/call", {})["error"]["code"], INVALID_PARAMS)
        response = self.request("tools/call", {"name": "memory_add", "arguments": [1, 2]})
        self.assertTrue(response["result"]["isError"])


class StdioTransportTests(ServerTestCase):
    def _session(self) -> tuple[str, io.StringIO]:
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": LATEST_PROTOCOL_VERSION}}),
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "memory_add",
                                   "arguments": {"text": "stdio note", "tags": ["stdio"]}}}),
            "",
            "this is not json",
            json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "memory_search", "arguments": {"query": "stdio"}}}),
        ]
        out = io.StringIO()
        serve_stdio(self.store, stdin=io.StringIO("\n".join(lines) + "\n"), stdout=out)
        return "\n".join(lines), out

    def test_serve_stdio_answers_every_request_and_no_notification(self) -> None:
        _, out = self._session()
        responses = [json.loads(line) for line in out.getvalue().splitlines()]
        # 4 requests, one notification (unanswered), one blank line, one parse error
        self.assertEqual([item["id"] for item in responses], [1, 2, 3, None, 4])
        self.assertEqual(responses[3]["error"]["code"], PARSE_ERROR)
        self.assertEqual(
            responses[2]["result"]["structuredContent"]["memory"]["text"], "stdio note"
        )
        self.assertEqual(responses[4]["result"]["structuredContent"]["count"], 1)

    def test_serve_stdio_output_is_one_json_object_per_line(self) -> None:
        _, out = self._session()
        for line in out.getvalue().splitlines():
            with self.subTest(line=line[:40]):
                self.assertNotIn("\n", line)
                self.assertIsInstance(json.loads(line), dict)

    def test_serve_stdio_creates_its_own_store_when_none_is_given(self) -> None:
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"AIMEM_DB": str(Path(self._tmp.name) / "auto.db")}):
            self.assertEqual(serve_stdio(stdin=io.StringIO(payload), stdout=out), 0)
        self.assertEqual(json.loads(out.getvalue().strip())["result"], {})
        self.assertTrue((Path(self._tmp.name) / "auto.db").exists())


class SubprocessIntegrationTests(unittest.TestCase):
    """Drive the real stdio transport through ``python3 -m aimem mcp``."""

    def test_full_session_over_a_real_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.db"
            lines = [
                json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": LATEST_PROTOCOL_VERSION,
                               "clientInfo": {"name": "unittest", "version": "1"}},
                }),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "memory_add",
                               "arguments": {"text": "written by a subprocess",
                                             "tags": ["subprocess"]}},
                }),
                json.dumps({
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "memory_search", "arguments": {"query": "subprocess"}},
                }),
            ]
            completed = subprocess.run(
                [sys.executable, "-m", "aimem", "mcp", "--db", str(db_path)],
                input="\n".join(lines) + "\n",
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            responses = [
                json.loads(line) for line in completed.stdout.splitlines() if line.strip()
            ]
            self.assertEqual([item["id"] for item in responses], [1, 2, 3])
            self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "aimem")
            self.assertEqual(
                responses[2]["result"]["structuredContent"]["memories"][0]["text"],
                "written by a subprocess",
            )
            # the memory really landed on disk
            store = MemoryStore(db_path)
            self.addCleanup(store.close)
            self.assertEqual(store.count(), 1)

    def test_module_entry_point_respects_the_aimem_db_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "env.db"
            completed = subprocess.run(
                [sys.executable, "-m", "aimem.mcp_server"],
                input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n",
                capture_output=True,
                text=True,
                cwd=str(REPO_ROOT),
                env=dict(os.environ, AIMEM_DB=str(db_path)),
                timeout=120,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout.strip())["result"], {})
            self.assertTrue(db_path.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
