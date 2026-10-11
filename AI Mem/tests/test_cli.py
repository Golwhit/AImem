"""Tests for the ``aimem`` command line interface."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from aimem import __version__
from aimem.cli import EXIT_ERROR, EXIT_NOT_FOUND, EXIT_OK, build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_TEXT = "Prefers stdlib-only tooling on Linux"


class CliTestCase(unittest.TestCase):
    """Base class that runs ``main()`` in-process against a temp database."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "memory.db"

    def run_cli(self, *args: str, db: Path | None = None) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        argv = ["--db", str(db or self.path), *args]
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def json_cli(self, *args: str, db: Path | None = None) -> object:
        code, out, err = self.run_cli(*args, "--json", db=db)
        self.assertEqual(code, EXIT_OK, err)
        return json.loads(out)

    def seed(self) -> None:
        self.run_cli("add", SEARCH_TEXT, "--tag", "preference", "--tag", "python", "--pin")
        self.run_cli("add", "Cline MCP config location", "--tags", "mcp,config")


class AddAndSearchTests(CliTestCase):
    def test_add_then_search_human_output(self) -> None:
        code, out, _ = self.run_cli("add", SEARCH_TEXT, "--tag", "preference", "--pin")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("stored memory [1]", out)
        self.assertIn(SEARCH_TEXT, out)
        self.assertIn("*", out)  # pinned marker

        code, out, _ = self.run_cli("search", "stdlib")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("[1]", out)
        self.assertIn("match:", out)
        self.assertIn("1 memory(ies)", out)

    def test_add_joins_multiple_words_into_one_memory(self) -> None:
        code, _, _ = self.run_cli("add", "Prefers", "stdlib-only", "tooling")
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(self.json_cli("show", "1")["text"], "Prefers stdlib-only tooling")

    def test_duplicate_add_reports_a_merge(self) -> None:
        self.run_cli("add", "note about tooling", "--tags", "a")
        code, out, _ = self.run_cli("add", "NOTE about tooling", "--tags", "b")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("merged into memory [1]", out)
        self.assertIn("duplicate: merged", out)
        self.assertEqual(self.json_cli("show", "1")["tags"], ["a", "b"])

    def test_json_output_is_machine_readable(self) -> None:
        added = self.json_cli("add", SEARCH_TEXT, "--tags", "preference,python")
        self.assertEqual(added["id"], 1)
        self.assertEqual(added["tags"], ["preference", "python"])
        found = self.json_cli("search", "stdlib")
        self.assertEqual(found["count"], 1)
        self.assertEqual(found["memories"][0]["text"], SEARCH_TEXT)
        self.assertEqual(self.json_cli("list")["count"], 1)
        self.assertEqual(self.json_cli("search", "nosuchterm"), {"count": 0, "memories": []})

    def test_search_empty_query_lists_recent_memories(self) -> None:
        self.run_cli("add", "first")
        self.run_cli("add", "second")
        payload = self.json_cli("search")
        self.assertEqual([m["id"] for m in payload["memories"]], [2, 1])

    def test_search_filters(self) -> None:
        self.seed()
        self.assertEqual(self.json_cli("search", "mcp", "--tag", "mcp")["count"], 1)
        self.assertEqual(self.json_cli("search", "mcp", "--tag", "python")["count"], 0)
        self.assertEqual(self.json_cli("search", "--pinned")["count"], 1)
        self.assertEqual(self.json_cli("search", "--source", "nosuchsource")["count"], 0)

    def test_no_snippet_flag_hides_excerpts(self) -> None:
        self.seed()
        _, out, _ = self.run_cli("search", "stdlib", "--no-snippet")
        self.assertNotIn("match:", out)


class ErrorHandlingTests(CliTestCase):
    def test_missing_memory_exits_with_not_found(self) -> None:
        code, _, err = self.run_cli("show", "999")
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertEqual(err.strip(), "error: no memory with id 999")

    def test_invalid_id_exits_with_error(self) -> None:
        code, _, err = self.run_cli("show", "not-a-number")
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("invalid memory id", err)

    def test_rm_reports_missing_ids(self) -> None:
        self.seed()
        code, out, err = self.run_cli("rm", "1", "99")
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertIn("deleted 1 memory(ies): 1", out)
        self.assertIn("not found: 99", err)

    def test_rm_is_also_available_as_delete(self) -> None:
        self.seed()
        code, out, _ = self.run_cli("delete", "1")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("deleted 1 memory(ies)", out)

    def test_no_command_prints_help(self) -> None:
        code, out, _ = self.run_cli()
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("usage: aimem", out)

    def test_update_without_fields_is_rejected(self) -> None:
        self.seed()
        code, _, err = self.run_cli("update", "1")
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("nothing to update", err)

    def test_update_rejects_conflicting_tag_flags(self) -> None:
        self.seed()
        code, _, err = self.run_cli("update", "1", "--tag", "a", "--add-tag", "b")
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("not both", err)

    def test_import_of_invalid_json_exits_with_error(self) -> None:
        bad = Path(self._tmp.name) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        code, _, err = self.run_cli("import", str(bad))
        self.assertEqual(code, EXIT_ERROR)
        self.assertTrue(err.strip())

    def test_version_flag(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn(__version__, out.getvalue())

    def test_parser_accepts_db_before_or_after_the_subcommand(self) -> None:
        parser = build_parser()
        before = parser.parse_args(["--db", "/tmp/x.db", "list", "--limit", "3"])
        after = parser.parse_args(["list", "--limit", "3", "--db", "/tmp/x.db"])
        self.assertEqual((before.db, before.limit), ("/tmp/x.db", 3))
        self.assertEqual((after.db, after.limit), ("/tmp/x.db", 3))
        self.assertFalse(before.json)
        self.assertFalse(after.json)


class UpdateTests(CliTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.seed()

    def test_update_text_source_and_pin(self) -> None:
        code, out, _ = self.run_cli(
            "update", "1", "--text", "rewritten", "--source", "mcp", "--pin"
        )
        self.assertEqual(code, EXIT_OK)
        self.assertIn("updated memory [1]", out)
        record = self.json_cli("show", "1")
        self.assertEqual(record["text"], "rewritten")
        self.assertEqual(record["source"], "mcp")
        self.assertTrue(record["pinned"])
        self.assertEqual(self.json_cli("search", "rewritten")["count"], 1)

    def test_update_unpin_and_replace_tags(self) -> None:
        self.run_cli("update", "1", "--unpin", "--tag", "solo")
        record = self.json_cli("show", "1")
        self.assertFalse(record["pinned"])
        self.assertEqual(record["tags"], ["solo"])

    def test_update_add_and_remove_tags(self) -> None:
        self.run_cli("update", "1", "--add-tag", "extra")
        self.assertEqual(self.json_cli("show", "1")["tags"], ["preference", "python", "extra"])
        self.run_cli("update", "1", "--remove-tag", "python")
        self.assertEqual(self.json_cli("show", "1")["tags"], ["preference", "extra"])

    def test_update_unknown_id_exits_with_not_found(self) -> None:
        code, _, err = self.run_cli("update", "999", "--text", "x")
        self.assertEqual(code, EXIT_NOT_FOUND)
        self.assertIn("no memory with id 999", err)


class ListTagsStatsTests(CliTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.seed()

    def test_list_orders_and_pagination(self) -> None:
        self.assertEqual([m["id"] for m in self.json_cli("list")["memories"]], [2, 1])
        self.assertEqual(
            [m["id"] for m in self.json_cli("list", "--order", "oldest")["memories"]], [1, 2]
        )
        self.assertEqual([m["id"] for m in self.json_cli("list", "-n", "1")["memories"]], [2])
        self.assertEqual(
            [m["id"] for m in self.json_cli("list", "-n", "1", "--offset", "1")["memories"]],
            [1],
        )
        self.assertEqual(
            [m["id"] for m in self.json_cli("list", "--order", "pinned")["memories"]], [1, 2]
        )

    def test_list_tag_filter_any_and_all(self) -> None:
        self.run_cli("add", "needs both tags", "--tags", "preference,mcp")
        # any: memory 1 (preference), memory 2 (mcp), memory 3 (both)
        self.assertEqual(self.json_cli("list", "--tag", "preference", "--tag", "mcp")["count"], 3)
        self.assertEqual(
            self.json_cli("list", "--tag", "preference", "--tag", "mcp", "--tag-mode", "all")["count"],
            1,
        )

    def test_tags_output(self) -> None:
        code, out, _ = self.run_cli("tags")
        self.assertEqual(code, EXIT_OK)
        for tag in ("preference", "python", "mcp", "config"):
            self.assertIn(tag, out)
        payload = self.json_cli("tags")
        self.assertEqual(
            sorted(item["tag"] for item in payload),
            ["config", "mcp", "preference", "python"],
        )
        self.assertEqual(self.json_cli("tags", "--min-count", "2"), [])
        _, out, _ = self.run_cli("tags", "--min-count", "2")
        self.assertIn("no tags yet", out)

    def test_stats_output(self) -> None:
        code, out, _ = self.run_cli("stats")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("memories  : 2 (1 pinned)", out)
        self.assertIn("index     : ok", out)
        self.assertIn(str(self.path), out)
        payload = self.json_cli("stats")
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["pinned"], 1)
        self.assertEqual(payload["tag_count"], 4)
        self.assertTrue(payload["index_ok"])

    def test_reindex_output(self) -> None:
        code, out, _ = self.run_cli("reindex")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("reindexed 2 memories", out)
        self.assertEqual(self.json_cli("reindex"), {"reindexed": 2, "index_ok": True})


class ExportImportTests(CliTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.seed()

    def test_export_to_stdout_is_json(self) -> None:
        code, out, _ = self.run_cli("export")
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertEqual(payload["count"], 2)
        self.assertEqual(len(payload["memories"]), 2)

    def test_export_to_file_and_import_round_trip(self) -> None:
        target = Path(self._tmp.name) / "dump.json"
        code, out, _ = self.run_cli("export", "-o", str(target))
        self.assertEqual(code, EXIT_OK)
        self.assertIn("exported 2 memories", out)
        self.assertTrue(target.exists())

        second = Path(self._tmp.name) / "second.db"
        code, out, _ = self.run_cli("import", str(target), db=second)
        self.assertEqual(code, EXIT_OK)
        self.assertIn("imported 2", out)
        self.assertEqual(self.json_cli("list", db=second)["count"], 2)
        # re-importing merges instead of duplicating
        self.run_cli("import", str(target), db=second)
        self.assertEqual(self.json_cli("list", db=second)["count"], 2)

    def test_import_from_stdin_with_replace(self) -> None:
        payload = json.dumps({"memories": [{"text": "from stdin", "tags": ["stdin"]}]})
        with mock.patch("sys.stdin", io.StringIO(payload)):
            code, out, _ = self.run_cli("import", "-", "--replace")
        self.assertEqual(code, EXIT_OK)
        self.assertIn("<stdin>", out)
        self.assertEqual(self.json_cli("search", "stdin")["count"], 1)
        self.assertEqual(self.json_cli("list")["count"], 1)


class EnvironmentAndEntryPointTests(CliTestCase):
    def test_aimem_db_env_var_is_used_when_no_db_flag_is_given(self) -> None:
        env_path = Path(self._tmp.name) / "from-env.db"
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"AIMEM_DB": str(env_path)}):
            with redirect_stdout(out), redirect_stderr(err):
                code = main(["add", "env var note"])
        self.assertEqual(code, EXIT_OK, err.getvalue())
        self.assertTrue(env_path.exists())
        self.assertIn("stored memory [1]", out.getvalue())

    def test_db_flag_wins_over_the_env_var(self) -> None:
        env_path = Path(self._tmp.name) / "from-env.db"
        with mock.patch.dict(os.environ, {"AIMEM_DB": str(env_path)}):
            code, _, _ = self.run_cli("add", "flag note")
        self.assertEqual(code, EXIT_OK)
        self.assertTrue(self.path.exists())
        self.assertFalse(env_path.exists())

    def test_module_entry_point_end_to_end(self) -> None:
        db = Path(self._tmp.name) / "subprocess.db"
        add = subprocess.run(
            [sys.executable, "-m", "aimem", "--db", str(db), "add", "subprocess note",
             "--tag", "sub"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        self.assertEqual(add.returncode, 0, add.stderr)
        self.assertIn("stored memory [1]", add.stdout)
        search = subprocess.run(
            [sys.executable, "-m", "aimem", "--db", str(db), "search", "subprocess", "--json"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        self.assertEqual(search.returncode, 0, search.stderr)
        payload = json.loads(search.stdout)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["memories"][0]["tags"], ["sub"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
