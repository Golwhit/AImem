# AI Mem - design notes

## Goal

Give AI agents (Cline, Claude, scripts, humans) one small, local, trustworthy
place to persist facts about the user's work: preferences, decisions,
snippets, gotchas. Reads and writes must be instant, offline, and inspectable
with `sqlite3` alone.

## Constraints discovered on this machine

| Constraint | Consequence |
| --- | --- |
| No `pip` module available, `venv` present but empty | Core must be pure standard library; no `requirements.txt` install step |
| SQLite **has FTS5** (`CREATE VIRTUAL TABLE ... USING fts5` works) | Use FTS5 + BM25 instead of hand-rolled TF-IDF |
| No `ollama`, no `OPENAI`/`ANTHROPIC` keys in the environment | Semantic/embedding search is a *later, optional* layer |
| Python 3.12.3, Node 18.19.1 | Python for the engine; Node available if a web UI is added later |

## Architecture

```
aimem/
  store.py       -> MemoryStore: all SQLite/FTS5 logic (no I/O beyond the DB)
  cli.py         -> argparse front end (add/search/list/show/update/rm/tags/stats/export/import/mcp)
  mcp_server.py  -> JSON-RPC 2.0 over stdio, exposes the store as MCP tools
  __main__.py    -> `python3 -m aimem ...` entry point
tests/           -> stdlib unittest, including a subprocess end-to-end MCP test
```

Layering rule: `cli.py` and `mcp_server.py` are thin adapters. Neither touches
SQL; all behaviour lives in `MemoryStore`, which is why both front ends can be
tested against the same store tests.

## Data model

```sql
memories(
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  text       TEXT NOT NULL,
  tags       TEXT NOT NULL DEFAULT '',   -- canonical ",tag-a,tag-b," form
  source     TEXT NOT NULL DEFAULT '',   -- e.g. "cline", "cli", "import"
  pinned     INTEGER NOT NULL DEFAULT 0, -- 0/1
  created_at TEXT NOT NULL,              -- UTC iso8601 Z
  updated_at TEXT NOT NULL
)
memories_fts USING fts5(text, tags, content='memories', content_rowid='id',
                        tokenize='unicode61 remove_diacritics 2')
```

Triggers (`memories_ai`, `memories_ad`, `memories_au`) keep the FTS index in
sync with the table, including the FTS5 "delete" rows required by an external
content table. `MemoryStore.reindex()` rebuilds the index if it ever drifts.

**Why `,tag,tag,`?** A leading and trailing comma makes tag filtering an exact
`instr(tags, ',tag,') > 0` test, so tag `ci` never matches tag `ci-cd`, no join
table or JSON parsing is needed, and the wildcard characters that `LIKE` would
interpret (`%`, `_`) can never cause a false positive.

## Search behaviour

1. Free text is tokenised to `\w+`, lower-cased, and each token becomes a
   quoted prefix term: `"hello"* AND "mem"*`. Quoting means no user input can
   produce an FTS5 syntax error.
2. First pass uses `AND` for precision. If it returns nothing, the same tokens
   are retried with `OR` for recall.
3. Ranking is BM25 with text weighted 10 and tags 4, then a *scale-free* pinned
   boost: `ORDER BY score * (1.0 + pinned) ASC, updated_at DESC`. bm25 is
   negative, so doubling a pinned memory's magnitude lifts it above
   non-pinned matches of comparable relevance. Absolute bm25 values on a small
   corpus can be as small as `1e-6`, which is why the boost is multiplicative
   (and why scores are never rounded before comparison).
4. Filters (tags, source, pinned) are applied as SQL predicates on the join,
   and the page size is clamped to `MAX_LIMIT` (500).
5. `snippet()` returns a highlighted excerpt (`[term]`) for CLI/JSON output.

## Write behaviour

* `add()` de-duplicates on the trimmed text (case-insensitive): re-adding a
  known memory unions its tags, refreshes `updated_at`, and reports
  `duplicated: true`. This keeps an agent looping over the same insight from
  filling the store with near-identical rows.
* `update(tag_mode=...)` supports `replace`, `add`, and `remove`, so an agent
  can annotate a memory without having to read-modify-write the tag list.

## MCP exposure

Protocol revision: **2025-06-18** (also accepts older clients - the server
echoes whichever supported version the client asks for). Implemented by hand
because the `mcp` Python SDK cannot be installed here. Supported methods:
`initialize`, `notifications/initialized`, `ping`, `tools/list`, `tools/call`,
plus empty `resources/list` / `prompts/list` responses for clients that probe
them. Notifications are never answered; unknown methods get JSON-RPC `-32601`.

Tools (snake_case, `memory_` prefixed):

| Tool | Kind | Notes |
| --- | --- | --- |
| `memory_add` | write | de-duplicating, tags/source/pinned |
| `memory_search` | read | ranked search with tag/source/pinned filters |
| `memory_get` | read | fetch one memory by id |
| `memory_update` | write | text/tags/source/pinned, `tag_mode` replace/add/remove |
| `memory_delete` | destructive | `destructiveHint: true` |
| `memory_list` | read | newest first, paginated |
| `memory_tags` | read | tag names + counts |
| `memory_stats` | read | totals, db path, index health |
| `memory_reindex` | write | rebuild the FTS index |

Each tool declares `annotations` (`readOnlyHint`, `destructiveHint`,
`idempotentHint`) so clients can decide what is safe to auto-approve, and each
result carries both a text block and `structuredContent`.

## Index health

`stats()` reports `index_ok`, which runs FTS5's *deep* integrity check
(`INSERT INTO memories_fts(memories_fts, rank) VALUES('integrity-check', 1)`).
The plain `integrity-check` form only validates the index structure and happily
reports OK when the external content table has drifted, so the `rank`-argument
form is used instead: it compares the index against the `memories` table and
raises on mismatch. `reindex()` (`'rebuild'`) always restores a consistent
state, and the check releases the transaction it opens without ever rolling
back a caller's pending work.

## Non-goals / next steps

* Embeddings & hybrid ranking - needs `pip` bootstrap or Ollama; the store
  interface is designed so this can be added behind `search()` without
  touching the CLI or the MCP tool schemas.
* Multi-user / sync - a single local file is the point.
