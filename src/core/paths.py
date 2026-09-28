"""Where the project's data lives, decided in one place.

The database path is needed by three command line entry points and by the
Makefile. Defining it four times is how they drift apart, which is exactly what
happened: the Makefile was moved off the Desktop and the CLIs were not, so
running a module directly still looked for a database that no longer existed.

``XBRL_DB`` overrides the default, so a shell that exports it once needs no
flags at all.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DEFAULT_DB_ENV",
    "default_cache_path",
    "default_db_path",
    "default_filings_dir",
    "default_questions_dir",
]

DEFAULT_DB_ENV = "XBRL_DB"


def default_db_path() -> Path:
    """The ground truth database.

    Outside the repository by default. A multi-gigabyte SQLite write lasting
    several minutes is the workload iCloud, Dropbox and similar services
    interact with worst, and a checkout on the Desktop sits inside one.
    """
    override = os.environ.get(DEFAULT_DB_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "xbrl-data" / "xbrl.db"


def default_filings_dir() -> Path:
    """Cached filing text. Lives beside the database, not in the repository.

    These are fetched artifacts rather than source, they run to hundreds of
    megabytes, and gitignoring them inside a checkout has never stopped anyone
    accidentally syncing them.
    """
    return default_db_path().parent / "filings"


def default_questions_dir() -> Path:
    return default_db_path().parent / "questions"


def default_cache_path() -> Path:
    """The content-addressed model call cache.

    Beside the database, not under ``runs/`` in the checkout, and the reason is
    the same one that moved the database: a checkout on the Desktop sits inside
    a synced folder, and a SQLite file being written there can have the file
    swapped underneath it. The database failed that way with a bus error. The
    cache failed that way with "unable to open database file", 20 tasks into a
    50 task run.

    It does not belong under ``runs/`` regardless. Everything else there is a
    published artifact, committed as evidence for a claim. This is a cache: it
    is rebuildable, it is not evidence, and it is not committed.
    """
    return default_db_path().parent / "model-calls.db"
