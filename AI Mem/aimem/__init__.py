"""AI Mem - a local, dependency-free memory store for AI agents.

Public API::

    from aimem import MemoryStore

    store = MemoryStore()
    store.add("Prefers stdlib-only tooling on Linux", tags=["preference"])
    store.search("stdlib")

The package has no third-party dependencies: it is pure Python standard
library plus SQLite (with FTS5) so it runs on a stock Python 3.11+ install.
"""

from .store import (
    DEFAULT_DB_ENV,
    DEFAULT_DB_PATH,
    MemoryNotFound,
    MemoryStore,
    resolve_db_path,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_DB_ENV",
    "DEFAULT_DB_PATH",
    "MemoryNotFound",
    "MemoryStore",
    "resolve_db_path",
    "__version__",
]
