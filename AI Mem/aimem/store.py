"""SQLite + FTS5 backed memory store.

Everything is kept in a single SQLite file. Full text search uses SQLite's
FTS5 extension with:

* BM25 ranking (``bm25(memories_fts, 10.0, 4.0)`` - text weighted over tags),
* prefix matching, so ``memo`` finds ``memory``,
* ``unicode61 remove_diacritics 2``, so ``cafe`` finds ``café``,
* an external content table, so the text is stored exactly once.

The module is pure standard library so ``aimem`` runs on a stock Python
install with no ``pip`` required.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "BM25_WEIGHTS",
    "DEFAULT_DB_ENV",
    "DEFAULT_DB_PATH",
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "SCHEMA_VERSION",
    "MemoryNotFound",
    "MemoryStore",
    "build_fts_query",
    "format_tags",
    "normalize_tags",
    "parse_tag_column",
    "resolve_db_path",
    "utc_timestamp",
]

DEFAULT_DB_ENV = "AIMEM_DB"
DEFAULT_DB_PATH = Path.home() / ".aimem" / "memory.db"
SCHEMA_VERSION = 1
DEFAULT_LIMIT = 10
MAX_LIMIT = 500
PINNED_BOOST = 1.0  # pinned scores are scaled by (1 + PINNED_BOOST); bm25 is
# negative, so a larger magnitude ranks first. Multiplicative so the boost
# stays meaningful whatever the absolute scale of bm25 happens to be.
BM25_WEIGHTS = (10.0, 4.0)  # (text, tags)
ORDERINGS = ("recent", "oldest", "updated", "pinned")
TAG_MODES = ("any", "all")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_ILLEGAL_TAG_CHARS_RE = re.compile(r"[^\w.\-]+", re.UNICODE)
_TAG_SPLIT_RE = re.compile(r"[,\n;]+")


def _coerce_id(memory_id: Any) -> int:
    """Accept ints and numeric strings, reject everything else clearly."""
    if isinstance(memory_id, bool):
        raise ValueError(f"invalid memory id: {memory_id!r}")
    try:
        return int(memory_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid memory id: {memory_id!r}") from exc


class MemoryNotFound(KeyError):
    """Raised when a memory id does not exist."""


def resolve_db_path(db: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the database path.

    Priority: explicit ``db`` argument, then the ``AIMEM_DB`` environment
    variable, then ``~/.aimem/memory.db``.
    """
    if db:
        return Path(db).expanduser()
    env = os.environ.get(DEFAULT_DB_ENV)
    if env:
        return Path(env).expanduser()
    return DEFAULT_DB_PATH


def utc_timestamp() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_tags(tags: Any) -> list[str]:
    """Normalise tags into a de-duplicated, lower-cased list.

    Accepts ``None``, a comma/semicolon separated string, or any iterable of
    strings. Whitespace becomes ``-`` and characters that cannot safely live
    inside the comma separated storage format are dropped.
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        parts = _TAG_SPLIT_RE.split(tags)
    else:
        parts = []
        for item in tags:
            if item is None:
                continue
            parts.extend(_TAG_SPLIT_RE.split(str(item)))

    cleaned: list[str] = []
    for part in parts:
        tag = _ILLEGAL_TAG_CHARS_RE.sub("", re.sub(r"\s+", "-", part.strip().lower()))
        tag = tag.strip("-.")
        if tag and tag not in cleaned:
            cleaned.append(tag)
    return cleaned


def format_tags(tags: Iterable[str] | None) -> str:
    """Render tags in the canonical ``",a,b,"`` storage form."""
    cleaned = normalize_tags(tags)
    return f",{','.join(cleaned)}," if cleaned else ""


def parse_tag_column(value: str | None) -> list[str]:
    """Inverse of :func:`format_tags`."""
    if not value:
        return []
    return [tag for tag in value.split(",") if tag]


def build_fts_query(raw: str | None, mode: str = "and") -> str:
    """Turn free text into a safe FTS5 query string.

    Every token becomes a quoted prefix term, which makes user input immune to
    FTS5 syntax errors while still allowing ``memo`` to match ``memory``.
    """
    if not raw:
        return ""
    tokens = [token.lower() for token in _TOKEN_RE.findall(raw)]
    if not tokens:
        return ""
    if mode not in ("and", "or"):
        raise ValueError(f"unsupported FTS mode: {mode!r}")
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(f'"{token}"*' for token in tokens)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT    NOT NULL,
    tags       TEXT    NOT NULL DEFAULT '',
    source     TEXT    NOT NULL DEFAULT '',
    pinned     INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_pinned  ON memories(pinned, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_source  ON memories(source);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    tags,
    content='memories',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, tags)
    VALUES ('delete', old.id, old.text, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, tags)
    VALUES ('delete', old.id, old.text, old.tags);
    INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
END;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class MemoryStore:
    """A local memory store backed by a single SQLite file."""

    def __init__(self, db_path: str | os.PathLike[str] | None = None, *, timeout: float = 10.0):
        self.path = resolve_db_path(db_path)
        self.timeout = timeout
        self._conn: sqlite3.Connection | None = None
        self.connect()

    # ------------------------------------------------------------- plumbing

    @property
    def connection(self) -> sqlite3.Connection:
        """The live SQLite connection, opened lazily on first use."""
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def connect(self) -> sqlite3.Connection:
        """Open the database, creating the file and schema when needed."""
        if self._conn is not None:
            return self._conn
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:  # pragma: no cover - filesystems without WAL
            pass
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        self._conn = conn
        return conn

    def close(self) -> None:
        """Commit and release the connection (idempotent)."""
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MemoryStore path={str(self.path)!r}>"

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _clamp_limit(limit: Any, default: int = DEFAULT_LIMIT) -> int:
        try:
            value = int(limit)
        except (TypeError, ValueError):
            value = default
        if value <= 0:
            value = default
        return min(value, MAX_LIMIT)

    @staticmethod
    def _filters(
        tags: Any = None,
        tag_mode: str = "any",
        source: str | None = None,
        pinned_only: bool = False,
    ) -> tuple[list[str], list[Any]]:
        """Build the WHERE fragments and params shared by search and list."""
        if tag_mode not in TAG_MODES:
            raise ValueError(f"tag_mode must be one of {TAG_MODES}, got {tag_mode!r}")
        where: list[str] = []
        params: list[Any] = []
        tag_list = normalize_tags(tags)
        if tag_list:
            joiner = " AND " if tag_mode == "all" else " OR "
            # instr() on the canonical ",tag,tag," column is an exact test:
            # unlike LIKE it has no wildcard semantics, so a tag such as
            # "ci_cd" can never match "ci-cd".
            where.append("(" + joiner.join(["instr(m.tags, ?) > 0"] * len(tag_list)) + ")")
            params.extend(f",{tag}," for tag in tag_list)
        if source:
            where.append("m.source = ?")
            params.append(str(source))
        if pinned_only:
            where.append("m.pinned = 1")
        return where, params

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": row["id"],
            "text": row["text"],
            "tags": parse_tag_column(row["tags"]),
            "source": row["source"],
            "pinned": bool(row["pinned"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        keys = set(row.keys())
        if "score" in keys:
            data["score"] = float(row["score"])
        if "snippet" in keys and row["snippet"] is not None:
            data["snippet"] = row["snippet"]
        return data

    def _select_columns(self) -> str:
        return "id, text, tags, source, pinned, created_at, updated_at"

    def _fetch(self, memory_id: Any) -> sqlite3.Row | None:
        return self.connection.execute(
            f"SELECT {self._select_columns()} FROM memories WHERE id = ?",
            (_coerce_id(memory_id),),
        ).fetchone()

    def _find_by_text(self, text: str) -> sqlite3.Row | None:
        return self.connection.execute(
            f"SELECT {self._select_columns()} FROM memories "
            "WHERE text = ? COLLATE NOCASE LIMIT 1",
            (text,),
        ).fetchone()

    # ------------------------------------------------------------- write API

    def add(
        self,
        text: str,
        tags: Any = None,
        source: str = "",
        pinned: bool = False,
    ) -> dict[str, Any]:
        """Store a memory, de-duplicating on identical text.

        When the same text already exists its tags are unioned, the source and
        pin flag are kept/raised, ``updated_at`` is refreshed, and the returned
        record carries ``duplicated: True``.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("memory text must not be empty")
        now = utc_timestamp()
        existing = self._find_by_text(cleaned)
        if existing is not None:
            merged = normalize_tags([*parse_tag_column(existing["tags"]), *normalize_tags(tags)])
            self.connection.execute(
                "UPDATE memories SET tags = ?, source = ?, pinned = ?, updated_at = ? WHERE id = ?",
                (
                    format_tags(merged),
                    str(source or existing["source"]),
                    int(bool(pinned) or bool(existing["pinned"])),
                    now,
                    existing["id"],
                ),
            )
            self.connection.commit()
            record = self.get(existing["id"])
            record["duplicated"] = True
            return record
        cursor = self.connection.execute(
            "INSERT INTO memories(text, tags, source, pinned, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cleaned, format_tags(tags), str(source or ""), 1 if pinned else 0, now, now),
        )
        self.connection.commit()
        record = self.get(cursor.lastrowid)
        record["duplicated"] = False
        return record

    def get(self, memory_id: Any) -> dict[str, Any]:
        """Fetch a single memory, raising :class:`MemoryNotFound` if missing."""
        memory_id = _coerce_id(memory_id)
        row = self._fetch(memory_id)
        if row is None:
            raise MemoryNotFound(f"no memory with id {memory_id}")
        return self._row_to_dict(row)

    def update(
        self,
        memory_id: Any,
        *,
        text: str | None = None,
        tags: Any = None,
        source: str | None = None,
        pinned: bool | None = None,
        tag_mode: str = "replace",
    ) -> dict[str, Any]:
        """Patch a memory. Only the supplied fields change.

        ``tag_mode`` controls how ``tags`` are applied: ``replace`` (default),
        ``add`` to merge, or ``remove`` to drop the given tags.
        """
        memory_id = _coerce_id(memory_id)
        row = self._fetch(memory_id)
        if row is None:
            raise MemoryNotFound(f"no memory with id {memory_id}")

        sets: list[str] = []
        params: list[Any] = []
        if text is not None:
            cleaned = text.strip()
            if not cleaned:
                raise ValueError("memory text must not be empty")
            sets.append("text = ?")
            params.append(cleaned)
        if tags is not None:
            new_tags = normalize_tags(tags)
            current = parse_tag_column(row["tags"])
            if tag_mode == "replace":
                merged = new_tags
            elif tag_mode == "add":
                merged = normalize_tags([*current, *new_tags])
            elif tag_mode == "remove":
                drop = set(new_tags)
                merged = [tag for tag in current if tag not in drop]
            else:
                raise ValueError("tag_mode must be one of ('replace', 'add', 'remove')")
            sets.append("tags = ?")
            params.append(format_tags(merged))
        if source is not None:
            sets.append("source = ?")
            params.append(str(source))
        if pinned is not None:
            sets.append("pinned = ?")
            params.append(1 if pinned else 0)

        if not sets:
            return self.get(memory_id)
        sets.append("updated_at = ?")
        params.extend([utc_timestamp(), row["id"]])
        self.connection.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params)
        self.connection.commit()
        return self.get(row["id"])

    def delete(self, memory_id: Any) -> dict[str, Any]:
        """Delete a memory and return the removed record."""
        memory_id = _coerce_id(memory_id)
        row = self._fetch(memory_id)
        if row is None:
            raise MemoryNotFound(f"no memory with id {memory_id}")
        self.connection.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
        self.connection.commit()
        record = self._row_to_dict(row)
        record["deleted"] = True
        return record

    # -------------------------------------------------------------- read API

    def search(
        self,
        query: str | None,
        limit: int = DEFAULT_LIMIT,
        *,
        tags: Any = None,
        tag_mode: str = "any",
        source: str | None = None,
        pinned_only: bool = False,
        snippet: bool = True,
    ) -> list[dict[str, Any]]:
        """Ranked full-text search over text and tags.

        Tokens are ANDed first for precision, then retried with OR for recall.
        An empty query degrades to :meth:`list_memories`.
        """
        limit = self._clamp_limit(limit)
        where, params = self._filters(tags, tag_mode, source, pinned_only)
        fts_query = build_fts_query(query, "and")
        if not fts_query:
            return self.list_memories(
                limit=limit,
                tags=tags,
                tag_mode=tag_mode,
                source=source,
                pinned_only=pinned_only,
            )
        rows = self._search_fts(fts_query, where, params, limit, snippet)
        if not rows:
            fallback = build_fts_query(query, "or")
            if fallback and fallback != fts_query:
                rows = self._search_fts(fallback, where, params, limit, snippet)
        return [self._row_to_dict(row) for row in rows]

    def _search_fts(
        self,
        fts_query: str,
        where: Sequence[str],
        params: Sequence[Any],
        limit: int,
        snippet: bool,
    ) -> list[sqlite3.Row]:
        select = (
            "SELECT m.id, m.text, m.tags, m.source, m.pinned, m.created_at, m.updated_at, "
            f"bm25(memories_fts, {BM25_WEIGHTS[0]}, {BM25_WEIGHTS[1]}) AS score"
        )
        if snippet:
            select += ", snippet(memories_fts, 0, '[', ']', ' ... ', 12) AS snippet"
        clauses = ["memories_fts MATCH ?", *where]
        sql = (
            f"{select} FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY score * (1.0 + {PINNED_BOOST} * m.pinned) ASC, m.updated_at DESC "
            f"LIMIT {limit}"
        )
        return self.connection.execute(sql, [fts_query, *params]).fetchall()

    def list_memories(
        self,
        limit: int = 20,
        offset: int = 0,
        *,
        tags: Any = None,
        tag_mode: str = "any",
        source: str | None = None,
        pinned_only: bool = False,
        order: str = "recent",
    ) -> list[dict[str, Any]]:
        """List memories without a text query.

        ``order`` is one of ``recent``, ``oldest``, ``updated``, ``pinned``.
        """
        if order not in ORDERINGS:
            raise ValueError(f"order must be one of {ORDERINGS}, got {order!r}")
        limit = self._clamp_limit(limit, default=20)
        try:
            offset = max(0, int(offset))
        except (TypeError, ValueError):
            offset = 0
        where, params = self._filters(tags, tag_mode, source, pinned_only)
        order_sql = {
            "recent": "m.created_at DESC, m.id DESC",
            "oldest": "m.created_at ASC, m.id ASC",
            "updated": "m.updated_at DESC, m.id DESC",
            "pinned": "m.pinned DESC, m.created_at DESC",
        }[order]
        sql = (
            "SELECT m.id, m.text, m.tags, m.source, m.pinned, m.created_at, m.updated_at "
            "FROM memories m"
        )
        if where:
            sql += f" WHERE {' AND '.join(where)}"
        sql += f" ORDER BY {order_sql} LIMIT {limit} OFFSET {offset}"
        return [self._row_to_dict(row) for row in self.connection.execute(sql, params).fetchall()]

    # ------------------------------------------------------------ aggregates

    def count(self) -> int:
        """Total number of stored memories."""
        return int(self.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def tags(self, *, min_count: int = 1) -> list[dict[str, Any]]:
        """All tags with usage counts, most used first."""
        counts: dict[str, int] = {}
        for row in self.connection.execute("SELECT tags FROM memories"):
            for tag in parse_tag_column(row["tags"]):
                counts[tag] = counts.get(tag, 0) + 1
        threshold = max(1, int(min_count or 1))
        items = [
            {"tag": tag, "count": count}
            for tag, count in counts.items()
            if count >= threshold
        ]
        items.sort(key=lambda item: (-item["count"], item["tag"]))
        return items

    def index_ok(self) -> bool:
        """Verify the FTS index against the ``memories`` content table.

        FTS5's plain ``integrity-check`` only inspects the index structure. For
        an external content table the ``(..., rank) VALUES('integrity-check', 1)``
        form also compares the index against the content table, which is what
        catches a stale or partially rebuilt index.
        """
        conn = self.connection
        owns_transaction = not conn.in_transaction
        try:
            conn.execute(
                "INSERT INTO memories_fts(memories_fts, rank) VALUES('integrity-check', 1)"
            )
        except sqlite3.DatabaseError:
            return False
        finally:
            # The command looks like an INSERT, so release the transaction it
            # opened - but never roll back work the caller already had pending.
            if owns_transaction and conn.in_transaction:
                conn.rollback()
        return True

    def stats(self) -> dict[str, Any]:
        """Store level diagnostics, including FTS index health."""
        conn = self.connection
        total = self.count()
        pinned = int(conn.execute("SELECT COUNT(*) FROM memories WHERE pinned = 1").fetchone()[0])
        sources = [
            {"source": row["source"], "count": row["n"]}
            for row in conn.execute(
                "SELECT source, COUNT(*) AS n FROM memories GROUP BY source ORDER BY n DESC"
            )
        ]
        tag_items = self.tags()
        version_row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        span = conn.execute("SELECT MIN(created_at), MAX(created_at) FROM memories").fetchone()
        return {
            "db_path": str(self.path),
            "db_size_bytes": self.path.stat().st_size if self.path.exists() else 0,
            "schema_version": int(version_row[0]) if version_row else SCHEMA_VERSION,
            "total": total,
            "pinned": pinned,
            "tag_count": len(tag_items),
            "tags": tag_items[:20],
            "sources": sources,
            "oldest": span[0],
            "newest": span[1],
            "index_ok": self.index_ok(),
        }

    # -------------------------------------------------------- import / export

    def _all_records(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            f"SELECT {self._select_columns()} FROM memories ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def export_data(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot of the whole store."""
        records = self._all_records()
        return {
            "version": SCHEMA_VERSION,
            "exported_at": utc_timestamp(),
            "count": len(records),
            "memories": records,
        }

    def import_data(self, payload: Any, *, replace: bool = False) -> dict[str, Any]:
        """Load memories from :meth:`export_data` output (or a bare list).

        Existing memories are matched by text: their tags are unioned instead of
        creating duplicates. ``created_at``/``updated_at`` from the payload are
        preserved for new rows.
        """
        if isinstance(payload, Mapping):
            records = payload.get("memories", [])
        elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            records = payload
        else:
            raise ValueError(
                "import payload must be a list of memories or an object with a 'memories' list"
            )

        if replace:
            self.clear()
        imported = updated = skipped = 0
        for item in records:
            if not isinstance(item, Mapping):
                skipped += 1
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                skipped += 1
                continue
            existing = self._find_by_text(text)
            if existing is not None:
                self.update(
                    existing["id"],
                    tags=[*parse_tag_column(existing["tags"]), *normalize_tags(item.get("tags"))],
                    source=str(item.get("source") or existing["source"]),
                    pinned=bool(item.get("pinned", False)) or bool(existing["pinned"]),
                )
                updated += 1
                continue
            record = self.add(
                text,
                tags=item.get("tags"),
                source=str(item.get("source") or ""),
                pinned=bool(item.get("pinned", False)),
            )
            if item.get("created_at"):
                created = str(item["created_at"])
                updated_at = str(item.get("updated_at") or created)
                self.connection.execute(
                    "UPDATE memories SET created_at = ?, updated_at = ? WHERE id = ?",
                    (created, updated_at, record["id"]),
                )
            imported += 1
        self.connection.commit()
        return {
            "imported": imported,
            "updated": updated,
            "skipped": skipped,
            "total": self.count(),
        }

    def clear(self) -> int:
        """Delete every memory. Returns the number of rows removed."""
        removed = self.count()
        self.connection.execute("DELETE FROM memories")
        self.connection.commit()
        return removed

    def reindex(self) -> dict[str, Any]:
        """Rebuild the FTS index from the content table."""
        self.connection.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
        self.connection.commit()
        return {"reindexed": self.count(), "index_ok": self.index_ok()}

