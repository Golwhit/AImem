"""Command line interface for AI Mem.

Examples::

    python3 -m aimem add "Prefers stdlib-only tooling" --tag preference --pin
    python3 -m aimem search stdlib -n 5
    python3 -m aimem list --order recent --json
    python3 -m aimem mcp           # run the MCP stdio server
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from . import __version__
from .store import (
    DEFAULT_DB_PATH,
    ORDERINGS,
    TAG_MODES,
    MemoryNotFound,
    MemoryStore,
)

PROG = "aimem"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NOT_FOUND = 2


# --------------------------------------------------------------------- output


def format_memory(record: dict[str, Any], *, snippet: bool = True) -> str:
    """Render one memory as a short human readable block."""
    marker = "*" if record.get("pinned") else " "
    tags = ",".join(record.get("tags") or []) or "-"
    head = f"[{record['id']}]{marker} {record.get('created_at', '')} tags={tags}"
    if record.get("source"):
        head += f" source={record['source']}"
    if "score" in record:
        head += f" score={record['score']:.3g}"
    if record.get("duplicated") is True:
        head += " (duplicate: merged)"
    lines = [head, f"    {record.get('text', '')}"]
    if snippet and record.get("snippet"):
        lines.append(f"    match: {record['snippet']}")
    return "\n".join(lines)


def print_records(records: Sequence[dict[str, Any]], *, as_json: bool, snippet: bool = True) -> None:
    """Print a list of memories either as JSON or as text blocks."""
    if as_json:
        print(json.dumps({"count": len(records), "memories": list(records)}, indent=2, ensure_ascii=False))
        return
    if not records:
        print("(no memories found)")
        return
    for record in records:
        print(format_memory(record, snippet=snippet))
    print(f"\n{len(records)} memory(ies)")


def print_payload(payload: Any, *, as_json: bool, human: Callable[[Any], str]) -> None:
    """Print a non-record payload (stats, tags, import results, ...)."""
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(human(payload))


# ---------------------------------------------------------------------- parser


def _common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add --db/--json, either as real flags or as pass-through ones.

    The subparser copies use ``SUPPRESS`` defaults so a flag given before the
    subcommand is not overwritten by the subparser default afterwards.
    """
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=argparse.SUPPRESS if suppress else None,
        help=f"memory database file (default: $AIMEM_DB or {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help="emit machine readable JSON",
    )


def _tag_filter_flags(parser: argparse.ArgumentParser) -> None:
    """Add the tag/source/pinned filter flags shared by search and list."""
    parser.add_argument("--tag", "--tags", dest="tags", action="append", default=None,
                        help="filter by tag (repeatable, or comma separated)")
    parser.add_argument("--tag-mode", choices=TAG_MODES, default="any",
                        help="match any/all of the given tags")
    parser.add_argument("--source", default=None, help="filter by source")
    parser.add_argument("--pinned", action="store_true", help="only pinned memories")


def build_parser() -> argparse.ArgumentParser:
    """Build the full argument parser (kept separate so tests can reuse it)."""
    sub_flags = argparse.ArgumentParser(add_help=False)
    _common_flags(sub_flags, suppress=True)

    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Local, dependency-free memory store for AI agents.",
        epilog="Database path: --db, else $AIMEM_DB, else ~/.aimem/memory.db",
    )
    _common_flags(parser, suppress=False)
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # add -------------------------------------------------------------------
    p_add = subparsers.add_parser("add", help="store a memory", parents=[sub_flags])
    p_add.add_argument("text", nargs="+", help="memory text")
    p_add.add_argument("--tag", "--tags", dest="tags", action="append", default=None,
                       help="tag (repeatable, or comma separated)")
    p_add.add_argument("--source", default="cli", help="where the memory came from")
    p_add.add_argument("--pin", action="store_true", help="pin the memory")
    p_add.set_defaults(func=cmd_add)

    # search ----------------------------------------------------------------
    p_search = subparsers.add_parser("search", help="ranked full-text search", parents=[sub_flags])
    p_search.add_argument("query", nargs="*", help="search terms (empty lists recent memories)")
    p_search.add_argument("-n", "--limit", type=int, default=10, help="max results (default 10)")
    _tag_filter_flags(p_search)
    p_search.add_argument("--no-snippet", action="store_true", help="hide match excerpts")
    p_search.set_defaults(func=cmd_search)

    # list ------------------------------------------------------------------
    p_list = subparsers.add_parser("list", help="list memories (newest first)", parents=[sub_flags])
    p_list.add_argument("-n", "--limit", type=int, default=20, help="max results (default 20)")
    p_list.add_argument("--offset", type=int, default=0, help="skip N results")
    p_list.add_argument("--order", choices=ORDERINGS, default="recent", help="sort order")
    _tag_filter_flags(p_list)
    p_list.set_defaults(func=cmd_list)

    # show ------------------------------------------------------------------
    p_show = subparsers.add_parser("show", help="show one memory by id", parents=[sub_flags])
    p_show.add_argument("id", help="memory id")
    p_show.set_defaults(func=cmd_show)

    # update ----------------------------------------------------------------
    p_update = subparsers.add_parser("update", help="patch a memory", parents=[sub_flags])
    p_update.add_argument("id", help="memory id")
    p_update.add_argument("--text", default=None, help="replace the memory text")
    p_update.add_argument("--tag", "--tags", dest="tags", action="append", default=None,
                          help="replace the tag list (repeatable)")
    p_update.add_argument("--add-tag", dest="add_tags", action="append", default=None,
                          help="add tags, keeping existing ones")
    p_update.add_argument("--remove-tag", dest="remove_tags", action="append", default=None,
                          help="drop tags")
    p_update.add_argument("--source", default=None, help="set the source")
    p_update.add_argument("--pin", dest="pinned", action="store_true", default=None, help="pin")
    p_update.add_argument("--unpin", dest="pinned", action="store_false", help="unpin")
    p_update.set_defaults(func=cmd_update)

    # rm --------------------------------------------------------------------
    p_rm = subparsers.add_parser("rm", aliases=["delete"], help="delete memories by id",
                                 parents=[sub_flags])
    p_rm.add_argument("ids", nargs="+", help="memory id(s) to delete")
    p_rm.set_defaults(func=cmd_rm)

    # tags / stats / reindex -------------------------------------------------
    p_tags = subparsers.add_parser("tags", help="list tags with counts", parents=[sub_flags])
    p_tags.add_argument("--min-count", type=int, default=1, help="hide tags used fewer times")
    p_tags.set_defaults(func=cmd_tags)

    p_stats = subparsers.add_parser("stats", help="store diagnostics", parents=[sub_flags])
    p_stats.set_defaults(func=cmd_stats)

    p_reindex = subparsers.add_parser("reindex", help="rebuild the search index",
                                      parents=[sub_flags])
    p_reindex.set_defaults(func=cmd_reindex)

    # export / import --------------------------------------------------------
    p_export = subparsers.add_parser("export", help="dump all memories as JSON",
                                     parents=[sub_flags])
    p_export.add_argument("-o", "--output", default=None, help="write to a file instead of stdout")
    p_export.set_defaults(func=cmd_export)

    p_import = subparsers.add_parser("import", help="load memories from JSON",
                                     parents=[sub_flags])
    p_import.add_argument("path", help="JSON file, or '-' for stdin")
    p_import.add_argument("--replace", action="store_true", help="clear the store first")
    p_import.set_defaults(func=cmd_import)

    # mcp --------------------------------------------------------------------
    p_mcp = subparsers.add_parser("mcp", help="run the MCP stdio server", parents=[sub_flags])
    p_mcp.set_defaults(func=cmd_mcp)

    return parser


# ------------------------------------------------------------------- commands


def _as_json(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "json", False))


def _error_message(exc: BaseException) -> str:
    """Unwrap KeyError-style messages so errors read cleanly."""
    if exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc)


def cmd_add(args: argparse.Namespace, store: MemoryStore) -> int:
    record = store.add(" ".join(args.text), tags=args.tags, source=args.source, pinned=args.pin)
    if _as_json(args):
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        verb = "merged into" if record["duplicated"] else "stored"
        print(f"{verb} memory [{record['id']}]")
        print(format_memory(record))
    return EXIT_OK


def cmd_search(args: argparse.Namespace, store: MemoryStore) -> int:
    records = store.search(
        " ".join(args.query),
        limit=args.limit,
        tags=args.tags,
        tag_mode=args.tag_mode,
        source=args.source,
        pinned_only=args.pinned,
    )
    print_records(records, as_json=_as_json(args), snippet=not args.no_snippet)
    return EXIT_OK


def cmd_list(args: argparse.Namespace, store: MemoryStore) -> int:
    records = store.list_memories(
        limit=args.limit,
        offset=args.offset,
        tags=args.tags,
        tag_mode=args.tag_mode,
        source=args.source,
        pinned_only=args.pinned,
        order=args.order,
    )
    print_records(records, as_json=_as_json(args))
    return EXIT_OK


def cmd_show(args: argparse.Namespace, store: MemoryStore) -> int:
    record = store.get(args.id)
    if _as_json(args):
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(format_memory(record, snippet=False))
    return EXIT_OK


def cmd_update(args: argparse.Namespace, store: MemoryStore) -> int:
    if args.tags and (args.add_tags or args.remove_tags):
        raise ValueError("use either --tag (replace) or --add-tag/--remove-tag, not both")
    if args.add_tags:
        tags, tag_mode = args.add_tags, "add"
    elif args.remove_tags:
        tags, tag_mode = args.remove_tags, "remove"
    else:
        tags, tag_mode = args.tags, "replace"
    if args.text is None and tags is None and args.source is None and args.pinned is None:
        print(
            "error: nothing to update (give --text, --tag/--add-tag/--remove-tag, "
            "--source or --pin/--unpin)",
            file=sys.stderr,
        )
        return EXIT_ERROR
    record = store.update(
        args.id,
        text=args.text,
        tags=tags,
        source=args.source,
        pinned=args.pinned,
        tag_mode=tag_mode,
    )
    if _as_json(args):
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print(f"updated memory [{record['id']}]")
        print(format_memory(record, snippet=False))
    return EXIT_OK


def cmd_rm(args: argparse.Namespace, store: MemoryStore) -> int:
    deleted: list[int] = []
    missing: list[str] = []
    for raw in args.ids:
        try:
            deleted.append(store.delete(raw)["id"])
        except MemoryNotFound:
            missing.append(str(raw))
    if _as_json(args):
        print(json.dumps({"deleted": deleted, "missing": missing}, indent=2))
    else:
        if deleted:
            print(f"deleted {len(deleted)} memory(ies): {', '.join(str(i) for i in deleted)}")
        if missing:
            print(f"not found: {', '.join(missing)}", file=sys.stderr)
    return EXIT_NOT_FOUND if missing else EXIT_OK


def _tags_human(items: list[dict[str, Any]]) -> str:
    if not items:
        return "no tags yet"
    width = max(len(str(item["count"])) for item in items)
    return "\n".join(f"{item['count']:>{width}}  {item['tag']}" for item in items)


def cmd_tags(args: argparse.Namespace, store: MemoryStore) -> int:
    print_payload(store.tags(min_count=args.min_count), as_json=_as_json(args), human=_tags_human)
    return EXIT_OK


def _stats_human(payload: dict[str, Any]) -> str:
    sources = ", ".join(
        f"{item['source'] or '(none)'}={item['count']}" for item in payload["sources"]
    ) or "-"
    index = (
        "ok" if payload["index_ok"] else "OUT OF SYNC - run 'reindex'"
    )
    lines = [
        f"db        : {payload['db_path']}",
        f"size      : {payload['db_size_bytes']} bytes",
        f"memories  : {payload['total']} ({payload['pinned']} pinned)",
        f"tags      : {payload['tag_count']}",
        f"sources   : {sources}",
        f"oldest    : {payload['oldest'] or '-'}",
        f"newest    : {payload['newest'] or '-'}",
        f"index     : {index}",
        f"schema    : v{payload['schema_version']}",
    ]
    return "\n".join(lines)


def cmd_stats(args: argparse.Namespace, store: MemoryStore) -> int:
    print_payload(store.stats(), as_json=_as_json(args), human=_stats_human)
    return EXIT_OK


def cmd_reindex(args: argparse.Namespace, store: MemoryStore) -> int:
    payload = store.reindex()
    human = lambda data: (  # noqa: E731 - tiny inline formatter
        f"reindexed {data['reindexed']} memories (index "
        f"{'ok' if data['index_ok'] else 'OUT OF SYNC'})"
    )
    print_payload(payload, as_json=_as_json(args), human=human)
    return EXIT_OK


# ------------------------------------------------------------ export / import


def cmd_export(args: argparse.Namespace, store: MemoryStore) -> int:
    payload = store.export_data()
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if not args.output:
        print(text)
        return EXIT_OK
    target = Path(args.output).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text + "\n", encoding="utf-8")
    summary = {"output": str(target), "count": payload["count"]}
    print(json.dumps(summary, indent=2) if _as_json(args)
          else f"exported {payload['count']} memories to {target}")
    return EXIT_OK


def cmd_import(args: argparse.Namespace, store: MemoryStore) -> int:
    if args.path == "-":
        raw, origin = sys.stdin.read(), "<stdin>"
    else:
        source = Path(args.path).expanduser()
        raw, origin = source.read_text(encoding="utf-8"), str(source)
    result = store.import_data(json.loads(raw), replace=args.replace)
    result["source"] = origin

    def human(data: dict[str, Any]) -> str:
        return (
            f"imported {data['imported']}, merged {data['updated']}, "
            f"skipped {data['skipped']} from {data['source']} (total {data['total']})"
        )

    print_payload(result, as_json=_as_json(args), human=human)
    return EXIT_OK


def cmd_mcp(args: argparse.Namespace, store: MemoryStore) -> int:
    from .mcp_server import serve_stdio

    return serve_stdio(store)


# ----------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv``, run the requested command, and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_ERROR

    store = MemoryStore(getattr(args, "db", None))
    try:
        return args.func(args, store)
    except MemoryNotFound as exc:
        print(f"error: {_error_message(exc)}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"error: {_error_message(exc)}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
