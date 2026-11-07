# AI Mem

Local, offline, **dependency-free** long-term memory for AI agents.

One SQLite file holds your memories. A CLI and an
[MCP](https://modelcontextprotocol.io) server expose the same store, so both
you (in a terminal) and agents such as Cline can search and curate it.

```
python3 -m aimem add "Prefers stdlib-only tooling on Linux" --tag preference --pin
python3 -m aimem search tooling
```

* **No install, no API keys, no network.** Pure Python standard library plus
  SQLite with FTS5 — verified working on a machine with no `pip` at all.
* **Real ranked search.** FTS5 + BM25, prefix matching (`memo` finds `memory`),
  accent-insensitive (`cafe` finds `café`), highlighted snippets, tag/source/
  pinned filters, `AND`-then-`OR` fallback for recall.
* **Agent ready.** 9 MCP tools with JSON schemas and `readOnlyHint` /
  `destructiveHint` annotations, plus machine-readable `--json` CLI output.
* **Yours alone.** A single file you can copy, diff, `sqlite3` into, export to
  JSON, or delete. Nothing leaves the machine.

Design notes and the reasoning behind each decision live in
[docs/DESIGN.md](docs/DESIGN.md).

---

## Requirements

| Need | Notes |
| --- | --- |
| Python **3.11+** | developed and tested on 3.12.3 |
| SQLite with **FTS5** | bundled with CPython's `sqlite3`; verified enabled here |
| Nothing else | no `pip install`, no `requirements.txt`, no virtualenv |

Run everything from the repository root so `python3 -m aimem` can import the
package. If you do have `pip`, `pip install -e .` also works and provides an
`aimem` console script (see `pyproject.toml`).

## Storage

| Setting | Value |
| --- | --- |
| Default database | `~/.aimem/memory.db` |
| Override per command | `--db /path/to/memory.db` |
| Override globally | `AIMEM_DB=/path/to/memory.db` |
| Backup | copy the file (plus `-wal`/`-shm` if the server is running) |
| Portable export | `python3 -m aimem export -o memories.json` |

The database is git-ignored: `data/`, `*.db`, `*.db-wal`, `*.db-shm`.

```console
$ python3 -m aimem stats
db        : /home/al/.aimem/memory.db
size      : 49152 bytes
memories  : 2 (1 pinned)
tags      : 5
sources   : cli=2
oldest    : 2026-09-24T19:11:50Z
newest    : 2026-09-24T19:11:50Z
index     : ok
schema    : v1
```

## CLI reference

`python3 -m aimem [--db PATH] [--json] COMMAND ...` — `--db`/`--json` may be
given before *or* after the command.

| Command | What it does |
| --- | --- |
| `add TEXT... [--tag T]... [--source S] [--pin]` | store a memory; identical text is merged, not duplicated |
| `search [TERMS...] [-n N] [--tag T]... [--tag-mode any\|all] [--source S] [--pinned] [--no-snippet]` | ranked search; no terms lists recent memories |
| `list [-n N] [--offset N] [--order recent\|oldest\|updated\|pinned] [...]` | browse the store |
| `show ID` | one memory by id |
| `update ID [--text T] [--tag T]... [--add-tag T]... [--remove-tag T]... [--source S] [--pin\|--unpin]` | patch a memory |
| `rm ID...` (alias `delete`) | delete one or more memories |
| `tags [--min-count N]` | tag inventory with counts |
| `stats` | totals, sources, index health, db path |
| `reindex` | rebuild the search index |
| `export [-o FILE]` | dump everything as JSON |
| `import FILE\|- [--replace]` | load JSON (`-` reads stdin); merges duplicates unless `--replace` |
| `mcp` | run the MCP server on stdio |

Everything supports `--json` for scripting and agents:

```console
$ python3 -m aimem search --tag preference --json
{
  "count": 1,
  "memories": [
    {
      "id": 1,
      "text": "Prefers stdlib-only tooling on Linux",
      "tags": ["preference", "python"],
      "source": "cli",
      "pinned": true,
      "created_at": "2026-09-24T19:11:50Z",
      "updated_at": "2026-09-24T19:11:50Z",
      "score": -1.999209798498617e-06,
      "snippet": "Prefers [stdlib]-only tooling on Linux"
    }
  ]
}
```

Exit codes: `0` success, `1` usage/runtime error, `2` memory id not found,
`130` interrupted.

## MCP server

```console
$ python3 -m aimem mcp                      # db from --db / $AIMEM_DB / ~/.aimem/memory.db
```

It speaks newline-delimited JSON-RPC 2.0 on stdin/stdout (the MCP **stdio**
transport), implements protocol revision **2025-06-18** and also accepts
`2025-03-26` / `2024-11-05` clients by echoing their version. No SDK, no
dependencies — the protocol is implemented in `aimem/mcp_server.py`.

### Tools

| Tool | Kind | Arguments |
| --- | --- | --- |
| `memory_add` | write | `text` (required), `tags`, `source`, `pinned` |
| `memory_search` | read | `query` (required), `limit`, `tags`, `tag_mode`, `source`, `pinned_only` |
| `memory_get` | read | `id` |
| `memory_update` | write | `id`, `text`, `tags`, `tag_mode` (replace/add/remove), `source`, `pinned` |
| `memory_delete` | **destructive** | `id` |
| `memory_list` | read | `limit`, `offset`, `order`, `tags`, `tag_mode`, `source`, `pinned_only` |
| `memory_tags` | read | `min_count` |
| `memory_stats` | read | — |
| `memory_reindex` | write | — |

Each result carries both a text block and `structuredContent`, and each tool
declares `readOnlyHint` / `destructiveHint` / `idempotentHint` so clients can
reason about what is safe to auto-approve. Argument problems come back as
`isError: true` results; only protocol problems (unknown tool/method, malformed
envelope) use JSON-RPC error codes.

### Wiring it into Cline

Cline stores MCP servers in `cline_mcp_settings.json` (open **MCP Servers →
Configure** in the Cline panel to be sure you have the right file; on this
machine it is `~/.cline/data/settings/cline_mcp_settings.json` and it is
currently empty). Paste the entry below — the full file is in
[examples/cline_mcp_settings.json](examples/cline_mcp_settings.json):

```json
{
  "mcpServers": {
    "aimem": {
      "command": "python3",
      "args": ["-m", "aimem", "mcp", "--db", "~/.aimem/memory.db"],
      "env": { "PYTHONPATH": "/home/al/Desktop/AI Mem" },
      "disabled": false,
      "autoApprove": ["memory_search", "memory_get", "memory_list", "memory_tags", "memory_stats"]
    }
  }
}
```

* `PYTHONPATH` must point at this repository: the client launches the process
  from an arbitrary working directory, so `python3 -m aimem` needs to find the
  package there.
* `~` in `--db` is expanded by AI Mem itself, not by the client.
* Reconnect/restart the server from the Cline MCP panel after editing the file,
  and check that the nine `memory_*` tools appear.
* `autoApprove` deliberately lists only read-only tools; writes and deletes stay
  behind an approval prompt.
* Then just ask: *"search my memory for stdlib tooling"* or *"remember that I
  prefer X"*. The server instructions tell the agent to search before asking you
  to repeat context, and to add durable facts as they come up.

### Manual smoke test

```console
$ printf '%s\n' \
    '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}' \
    '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
    '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"memory_add","arguments":{"text":"hello memory"}}}' \
    '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"memory_search","arguments":{"query":"hello"}}}' \
  | python3 -m aimem mcp --db /tmp/aimem-demo.db
```

## Project layout

```
aimem/
  __init__.py      public API (MemoryStore) and __version__
  store.py         schema, FTS5/BM25 search, tags, import/export, index health
  cli.py           argparse front end: every command, --json, exit codes
  mcp_server.py    MCP stdio server: 9 tools, hand-written JSON-RPC 2.0
  __main__.py      `python3 -m aimem`
tests/             stdlib unittest suite (store / cli / mcp)
docs/DESIGN.md     why each decision was made, plus the constraints found here
examples/          ready-to-paste MCP client configuration
```

## Tests

```console
$ python3 -m unittest discover -s tests -t . -v
...
Ran 98 tests in 1.1s

OK
```

What is covered:

* **store** — tag/text helpers, `add` de-duplication, prefix/accent/ranking
  search, hostile query strings, tag filters (including the regressions listed
  below), `AND`→`OR` fallback, update/delete, persistence across reopen,
  export/import round-trips, list orderings, `stats`, `reindex`, index drift.
* **CLI** — every command's output and exit code, `--json` shape, `--db` before
  and after the subcommand, `$AIMEM_DB`, stdin import, `--version`, and a real
  `python3 -m aimem` subprocess run.
* **MCP** — handshake and version negotiation, notifications never answered,
  `-32600` / `-32601` / `-32602` / `-32700` error paths, `tools/list` schema and
  annotations, every tool's happy path and validation errors, in-process stdio
  sessions, and a full session through a real `python3 -m aimem mcp`
  subprocess.

## Design decisions worth knowing

* **Tags are stored as `,a,b,` and matched with `instr()`, not `LIKE`.** A tag
  `ci` must not match `ci-cd`, and `LIKE` would treat `_`/`%` as wildcards. An
  early version of this code also only matched the *last* tag due to a missing
  trailing `%` — `tests/test_store.py::TagFilterTests` keeps that fixed.
* **The pinned boost is multiplicative.** bm25 scores here are as small as
  `1e-6`, so an additive boost would swamp relevance; pinned memories instead get
  `score * (1 + pinned)` and scores are never rounded before ranking.
* **`add()` de-duplicates** on trimmed, case-insensitive text and unions tags, so
  an agent that keeps re-noticing the same fact does not fill the store with
  near-duplicates.
* **Index health uses the deep FTS5 check** — the plain `integrity-check` cannot
  see a stale external-content index (see `docs/DESIGN.md`).
* **The MCP server is hand-written** because the `mcp` Python SDK is not
  installable here: no dependencies, and every protocol path is unit tested.

## Roadmap

* **Semantic search** — add embeddings (Ollama, or a `pip` bootstrap) behind
  `MemoryStore.search()` as a hybrid ranker; the CLI and MCP schemas would not
  change.
* **Web UI** — browse/search/edit memories in the same Express + Vite + React
  stack as the neighbouring `Desktop/Apps` project.
* **Bulk ingestion** — files/URLs → chunk → memories, with source attribution.
* **Memory hygiene** — confidence, expiry and "forget this" flows so the store
  ages gracefully.
