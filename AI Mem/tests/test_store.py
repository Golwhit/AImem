"""Unit tests for :mod:`aimem.store`."""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aimem.store import (
    MAX_LIMIT,
    MemoryNotFound,
    MemoryStore,
    build_fts_query,
    format_tags,
    normalize_tags,
    parse_tag_column,
    resolve_db_path,
    utc_timestamp,
)

ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class StoreTestCase(unittest.TestCase):
    """Base class that gives every test a throw-away database."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "memory.db"
        self.store = MemoryStore(self.path)
        self.addCleanup(self.store.close)

    def add(self, text: str, **kwargs) -> dict:
        return self.store.add(text, **kwargs)


class HelperTests(unittest.TestCase):
    def test_normalize_tags_accepts_strings_lists_and_dedupes(self) -> None:
        self.assertEqual(normalize_tags("ai, ML;notes"), ["ai", "ml", "notes"])
        self.assertEqual(normalize_tags(["AI", "ai", " data-science "]), ["ai", "data-science"])
        self.assertEqual(normalize_tags(None), [])
        self.assertEqual(normalize_tags(["two words"]), ["two-words"])
        self.assertEqual(normalize_tags(["bad/tag#1", "ci_cd"]), ["badtag1", "ci_cd"])

    def test_format_and_parse_tags_roundtrip(self) -> None:
        self.assertEqual(format_tags(["a", "b"]), ",a,b,")
        self.assertEqual(format_tags([]), "")
        self.assertEqual(format_tags(None), "")
        self.assertEqual(parse_tag_column(",a,b,"), ["a", "b"])
        self.assertEqual(parse_tag_column(""), [])

    def test_build_fts_query_quotes_every_token_as_a_prefix(self) -> None:
        self.assertEqual(build_fts_query("Hello World"), '"hello"* AND "world"*')
        self.assertEqual(build_fts_query("a b", "or"), '"a"* OR "b"*')
        self.assertEqual(build_fts_query(""), "")
        self.assertEqual(build_fts_query("!!!"), "")
        self.assertEqual(build_fts_query(None), "")
        with self.assertRaises(ValueError):
            build_fts_query("x", "nope")

    def test_build_fts_query_neutralises_fts_syntax(self) -> None:
        # Nothing the caller types may be interpreted as FTS5 syntax.
        self.assertEqual(build_fts_query('"quoted" OR *'), '"quoted"* AND "or"*')

    def test_resolve_db_path_precedence(self) -> None:
        self.assertEqual(resolve_db_path("/tmp/explicit.db"), Path("/tmp/explicit.db"))
        with mock.patch.dict(os.environ, {"AIMEM_DB": "/tmp/from-env.db"}):
            self.assertEqual(resolve_db_path(None), Path("/tmp/from-env.db"))
            self.assertEqual(resolve_db_path("/tmp/explicit.db"), Path("/tmp/explicit.db"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(str(resolve_db_path(None)).endswith(".aimem/memory.db"))

    def test_utc_timestamp_format(self) -> None:
        self.assertRegex(utc_timestamp(), ISO_UTC_RE)


class AddTests(StoreTestCase):
    def test_add_returns_stored_record(self) -> None:
        record = self.add("Prefers stdlib-only tooling", tags=["Preference"], source="cli")
        self.assertEqual(record["id"], 1)
        self.assertEqual(record["text"], "Prefers stdlib-only tooling")
        self.assertEqual(record["tags"], ["preference"])
        self.assertEqual(record["source"], "cli")
        self.assertFalse(record["pinned"])
        self.assertFalse(record["duplicated"])
        self.assertRegex(record["created_at"], ISO_UTC_RE)
        self.assertEqual(record["created_at"], record["updated_at"])

    def test_add_trims_text_and_rejects_blanks(self) -> None:
        self.assertEqual(self.add("  spaced  ")["text"], "spaced")
        for blank in ("", "   ", None):
            with self.subTest(blank=blank):
                with self.assertRaises(ValueError):
                    self.store.add(blank)  # type: ignore[arg-type]

    def test_add_dedupes_case_insensitively_and_unions_tags(self) -> None:
        first = self.add("Prefers stdlib-only tooling", tags=["python"])
        second = self.add("prefers STDLIB-only tooling", tags=["linux"], pinned=True)
        self.assertEqual(second["id"], first["id"])
        self.assertTrue(second["duplicated"])
        self.assertEqual(second["tags"], ["python", "linux"])
        self.assertTrue(second["pinned"])
        self.assertEqual(self.store.count(), 1)


class SearchTests(StoreTestCase):
    def test_search_matches_prefixes(self) -> None:
        self.add("Stores memories in a sqlite database", tags=["storage"])
        ids = [record["id"] for record in self.store.search("memo")]
        self.assertEqual(ids, [1])
        self.assertEqual(self.store.search("nosuchterm"), [])

    def test_search_requires_all_terms_then_falls_back_to_any(self) -> None:
        self.add("Cline MCP config location", tags=["mcp"])
        self.add("Unrelated note", tags=["misc"])
        self.assertEqual(len(self.store.search("cline config")), 1)
        # 'cline' AND 'zzz' finds nothing, so the OR fallback returns the Cline row.
        fallback = self.store.search("cline zzz")
        self.assertEqual([record["id"] for record in fallback], [1])

    def test_search_ranks_more_relevant_first(self) -> None:
        self.add("sqlite notes", tags=["db"])
        self.add("sqlite sqlite sqlite tuning notes", tags=["db"])
        results = self.store.search("sqlite")
        self.assertEqual([record["id"] for record in results], [2, 1])
        self.assertLess(results[0]["score"], results[1]["score"])

    def test_search_is_accent_insensitive(self) -> None:
        self.add("Café notes about naïveté", tags=["ídeas"])
        self.assertEqual([r["id"] for r in self.store.search("cafe")], [1])
        self.assertEqual([r["id"] for r in self.store.search("naive")], [1])

    def test_search_survives_hostile_query_strings(self) -> None:
        self.add("Prefers stdlib-only tooling", tags=["preference"])
        for query in ['" OR *', "NEAR(", "a AND", "*", "-+~^", "((()))", "\u2603"]:
            with self.subTest(query=query):
                self.assertIsInstance(self.store.search(query), list)

    def test_search_empty_query_lists_recent_memories(self) -> None:
        self.add("first")
        self.add("second")
        self.assertEqual([r["id"] for r in self.store.search("")], [2, 1])
        self.assertEqual([r["id"] for r in self.store.search(None)], [2, 1])

    def test_search_snippet_highlights_the_match(self) -> None:
        self.add("Prefers stdlib-only tooling on Linux", tags=["preference"])
        record = self.store.search("stdlib")[0]
        self.assertIn("[stdlib]", record["snippet"])
        self.assertNotIn("snippet", self.store.search("stdlib", snippet=False)[0])

    def test_search_limit_is_clamped(self) -> None:
        for index in range(3):
            self.add(f"sqlite note number {index}")
        self.assertEqual(len(self.store.search("sqlite", limit=2)), 2)
        self.assertEqual(len(self.store.search("sqlite", limit=0)), 3)  # 0 means default
        self.assertEqual(len(self.store.search("sqlite", limit=10_000)), 3)
        self.assertEqual(MAX_LIMIT, 500)


class TagFilterTests(StoreTestCase):
    """Regression tests for tag filtering on the canonical ",tag,tag," column."""

    def setUp(self) -> None:
        super().setUp()
        self.mcp = self.add("Cline MCP config location", tags=["mcp", "config"])
        self.tooling = self.add("Prefers stdlib-only tooling", tags=["preference", "python"])
        self.both = self.add("Python and MCP together", tags=["python", "mcp"])

    def ids(self, **kwargs) -> list[int]:
        return [record["id"] for record in self.store.search("", **kwargs)]

    def test_filter_matches_tags_that_are_not_last(self) -> None:
        # The bug this guards: ",mcp," inside ",mcp,config," must match.
        self.assertEqual(self.ids(tags=["mcp"]), [3, 1])
        self.assertEqual(self.ids(tags=["config"]), [1])

    def test_filter_accepts_comma_separated_string(self) -> None:
        self.assertEqual(self.ids(tags="preference,python"), [3, 2])
        self.assertEqual(self.ids(tags="preference,python", tag_mode="all"), [2])

    def test_tag_mode_any_versus_all(self) -> None:
        self.assertEqual(self.ids(tags=["python", "mcp"], tag_mode="any"), [3, 2, 1])
        self.assertEqual(self.ids(tags=["python", "mcp"], tag_mode="all"), [3])

    def test_filter_does_not_cross_match_similar_tags(self) -> None:
        self.add("Dash tag", tags=["ci-cd"])
        self.add("Underscore tag", tags=["ci_cd"])
        self.add("Short tag", tags=["ci"])
        self.assertEqual([r["id"] for r in self.store.search("", tags=["ci"])], [6])
        self.assertEqual([r["id"] for r in self.store.search("", tags=["ci-cd"])], [4])
        self.assertEqual([r["id"] for r in self.store.search("", tags=["ci_cd"])], [5])

    def test_invalid_tag_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.store.search("", tag_mode="sometimes")

    def test_source_and_pinned_filters(self) -> None:
        self.add("Pinned cli memory", tags=["mcp"], source="cli", pinned=True)
        self.assertEqual(self.ids(source="cli"), [4])
        self.assertEqual(self.ids(pinned_only=True), [4])
        self.assertEqual([r["id"] for r in self.store.search("pinned", source="cli")], [4])

    def test_pinned_boost_reorders_matches(self) -> None:
        long_text = "sqlite ranking notes plus extra words to dilute the document score"
        short_text = "sqlite ranking notes"
        long_id = self.add(long_text, tags=["db"])["id"]
        short_id = self.add(short_text, tags=["db"])["id"]
        # Precondition: without pinning, the shorter document wins on BM25.
        unpinned = [r["id"] for r in self.store.search("sqlite")]
        self.assertEqual(unpinned, [short_id, long_id])
        self.store.update(long_id, pinned=True)
        boosted = [r["id"] for r in self.store.search("sqlite")]
        self.assertEqual(boosted[0], long_id, "pinning must lift a memory in the ranking")


class UpdateAndDeleteTests(StoreTestCase):
    def test_update_applies_tag_modes(self) -> None:
        record = self.add("note", tags=["a", "b", "c"])
        self.assertEqual(self.store.update(record["id"], tags=["x"])["tags"], ["x"])
        self.assertEqual(
            self.store.update(record["id"], tags=["a"], tag_mode="add")["tags"], ["x", "a"]
        )
        self.assertEqual(
            self.store.update(record["id"], tags=["x"], tag_mode="remove")["tags"], ["a"]
        )
        with self.assertRaises(ValueError):
            self.store.update(record["id"], tags=["y"], tag_mode="whatever")

    def test_update_text_source_and_pinned(self) -> None:
        record = self.add("old text", tags=["a"])
        updated = self.store.update(
            record["id"], text="  new text  ", source="mcp", pinned=True
        )
        self.assertEqual(updated["text"], "new text")
        self.assertEqual(updated["source"], "mcp")
        self.assertTrue(updated["pinned"])
        self.assertGreaterEqual(updated["updated_at"], updated["created_at"])
        # search must see the new text and no longer the old one
        self.assertEqual(len(self.store.search("new")), 1)
        self.assertEqual(self.store.search("old"), [])

    def test_update_with_no_fields_is_a_noop(self) -> None:
        record = self.add("unchanged", tags=["a"])
        current = self.store.update(record["id"])
        for key in ("id", "text", "tags", "source", "pinned", "created_at"):
            self.assertEqual(current[key], record[key])

    def test_update_and_delete_raise_for_unknown_ids(self) -> None:
        with self.assertRaises(MemoryNotFound):
            self.store.update(999, text="x")
        with self.assertRaises(MemoryNotFound):
            self.store.delete(999)
        with self.assertRaises(ValueError):
            self.store.get("not-an-id")

    def test_update_rejects_empty_text(self) -> None:
        record = self.add("keep me")
        with self.assertRaises(ValueError):
            self.store.update(record["id"], text="   ")
        self.assertEqual(self.store.get(record["id"])["text"], "keep me")

    def test_delete_removes_from_store_and_index(self) -> None:
        record = self.add("sqlite note")
        removed = self.store.delete(record["id"])
        self.assertTrue(removed["deleted"])
        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.store.search("sqlite"), [])
        self.assertTrue(self.store.stats()["index_ok"])


class PersistenceTests(StoreTestCase):
    def test_data_survives_reopening_the_file(self) -> None:
        self.store.add("durable note", tags=["storage"], source="cli")
        self.store.close()
        reopened = MemoryStore(self.path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.count(), 1)
        record = reopened.search("durable")[0]
        self.assertEqual(record["tags"], ["storage"])
        self.assertEqual(record["source"], "cli")

    def test_missing_parent_directories_are_created(self) -> None:
        nested = Path(self._tmp.name) / "deep" / "deeper" / "memory.db"
        store = MemoryStore(nested)
        self.addCleanup(store.close)
        store.add("nested note")
        self.assertTrue(nested.exists())


class ExportImportTests(StoreTestCase):
    def test_export_shape(self) -> None:
        self.add("first", tags=["a"], source="cli")
        payload = self.store.export_data()
        self.assertEqual(payload["count"], 1)
        self.assertEqual(len(payload["memories"]), 1)
        self.assertIn("exported_at", payload)
        self.assertEqual(payload["memories"][0]["tags"], ["a"])

    def test_import_round_trip_preserves_content_and_timestamps(self) -> None:
        original = self.add("round trip note", tags=["a", "b"], source="cli", pinned=True)
        payload = self.store.export_data()
        target = MemoryStore(Path(self._tmp.name) / "copy.db")
        self.addCleanup(target.close)
        result = target.import_data(payload)
        self.assertEqual(result["imported"], 1)
        self.assertEqual(result["total"], 1)
        imported = target.get(1)
        self.assertEqual(imported["text"], original["text"])
        self.assertEqual(imported["tags"], ["a", "b"])
        self.assertTrue(imported["pinned"])
        self.assertEqual(imported["created_at"], original["created_at"])

    def test_import_merges_duplicates_and_can_replace(self) -> None:
        self.add("keep", tags=["a"])
        payload = {
            "memories": [
                {"text": "KEEP", "tags": ["b"]},
                {"text": "brand new", "tags": ["c"]},
            ]
        }
        result = self.store.import_data(payload)
        self.assertEqual((result["imported"], result["updated"], result["total"]), (1, 1, 2))
        self.assertEqual(self.store.get(1)["tags"], ["a", "b"])
        result = self.store.import_data([{"text": "only one"}], replace=True)
        self.assertEqual(result["total"], 1)
        self.assertEqual(self.store.count(), 1)

    def test_import_skips_invalid_rows_and_rejects_bad_payloads(self) -> None:
        result = self.store.import_data([{"text": "ok"}, None, 42, {"text": "  "}, {}])
        self.assertEqual(result["imported"], 1)
        self.assertEqual(result["skipped"], 4)
        with self.assertRaises(ValueError):
            self.store.import_data("not a list")


class ListTagsStatsTests(StoreTestCase):
    def test_list_ordering_and_pagination(self) -> None:
        first = self.add("alpha")
        second = self.add("beta")
        first_id, second_id = first["id"], second["id"]
        self.store.connection.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z", first_id),
        )
        self.store.connection.execute(
            "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
            ("2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z", second_id),
        )
        self.store.connection.commit()
        self.assertEqual(
            [r["id"] for r in self.store.list_memories(order="recent")], [second_id, first_id]
        )
        self.assertEqual(
            [r["id"] for r in self.store.list_memories(order="oldest")], [first_id, second_id]
        )
        self.assertEqual(
            [r["id"] for r in self.store.list_memories(order="updated")], [first_id, second_id]
        )
        self.assertEqual([r["id"] for r in self.store.list_memories(limit=1, offset=1)], [first_id])
        self.store.update(first_id, pinned=True)
        self.assertEqual(
            [r["id"] for r in self.store.list_memories(order="pinned")], [first_id, second_id]
        )
        with self.assertRaises(ValueError):
            self.store.list_memories(order="sideways")

    def test_tags_reports_counts_and_honours_min_count(self) -> None:
        self.add("one", tags=["shared", "solo"])
        self.add("two", tags=["shared"])
        self.assertEqual(
            self.store.tags(), [{"tag": "shared", "count": 2}, {"tag": "solo", "count": 1}]
        )
        self.assertEqual(self.store.tags(min_count=2), [{"tag": "shared", "count": 2}])

    def test_stats_and_reindex(self) -> None:
        self.add("sqlite note", tags=["db"], source="cli", pinned=True)
        stats = self.store.stats()
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["pinned"], 1)
        self.assertEqual(stats["tag_count"], 1)
        self.assertEqual(stats["sources"], [{"source": "cli", "count": 1}])
        self.assertEqual(stats["db_path"], str(self.path))
        self.assertTrue(stats["index_ok"])

        # Simulate index drift, then verify reindex repairs it.
        self.store.connection.execute(
            "INSERT INTO memories_fts(memories_fts) VALUES('delete-all')"
        )
        self.store.connection.commit()
        self.assertFalse(self.store.stats()["index_ok"])
        self.assertEqual(self.store.search("sqlite"), [])
        self.assertEqual(self.store.reindex(), {"reindexed": 1, "index_ok": True})
        self.assertEqual(len(self.store.search("sqlite")), 1)

    def test_clear_returns_the_number_of_removed_rows(self) -> None:
        self.add("one")
        self.add("two")
        self.assertEqual(self.store.clear(), 2)
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
