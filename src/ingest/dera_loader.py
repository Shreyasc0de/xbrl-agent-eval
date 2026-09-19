"""Load SEC DERA Financial Statement Data Sets into SQLite.

This builds the answer key. Nothing in ``src/agents/`` may read the database it
produces, and nothing here may read ``data/filings/``. That separation is the
benchmark.

Four tab-delimited files per quarterly ZIP:

``sub.txt`` one row per submission, and the only safe source of fiscal year and
fiscal period labels. ``num.txt`` one row per numeric fact, which is the answer
key itself. ``pre.txt`` the presentation layer, needed for the ``negating`` flag
that decides sign. ``tag.txt`` tag metadata, needed to filter out company
specific extension tags.

Loading is strict. A file missing a required column raises rather than loading a
partial table, because a silently absent ``qtrs`` column produces an answer key
that is confidently wrong. Empty numeric values are stored as NULL and never as
zero: a zero would survive the question filters and be graded against.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "REQUIRED_COLUMNS",
    "SCHEMA_VERSION",
    "LoadReport",
    "SchemaVersionError",
    "create_indexes",
    "create_schema",
    "load_quarter",
    "open_database",
]

# Only the columns the benchmark uses. DERA files carry many more; ignoring the
# rest keeps the schema readable and the database small.
REQUIRED_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "sub": (
        "adsh",
        "cik",
        "name",
        "sic",
        "form",
        "period",
        "fy",
        "fp",
        "filed",
        "prevrpt",
    ),
    # segments is load-bearing, not optional. It carries the dimensional
    # breakdown of a fact: revenue by product line, by geography, by segment.
    # A row with an empty segments is the consolidated figure; a row with a
    # value is one slice of it. Both share every other identifying column, so
    # leaving segments out of the key collapses them and an arbitrary slice
    # wins.
    "num": (
        "adsh",
        "tag",
        "version",
        "coreg",
        "segments",
        "ddate",
        "qtrs",
        "uom",
        "value",
    ),
    "pre": (
        "adsh",
        "report",
        "line",
        "stmt",
        "inpth",
        "tag",
        "version",
        "plabel",
        "negating",
    ),
    "tag": (
        "tag",
        "version",
        "custom",
        "abstract",
        "datatype",
        "iord",
        "crdr",
        "tlabel",
    ),
}

_INTEGER_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "cik",
        "sic",
        "period",
        "fy",
        "filed",
        "prevrpt",
        "ddate",
        "qtrs",
        "report",
        "line",
        "inpth",
        "negating",
        "custom",
        "abstract",
    }
)
_REAL_COLUMNS: Final[frozenset[str]] = frozenset({"value"})

_PRIMARY_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "sub": ("adsh",),
    "num": ("adsh", "tag", "version", "coreg", "segments", "ddate", "qtrs"),
    "pre": ("adsh", "report", "line"),
    "tag": ("tag", "version"),
}

_BATCH_SIZE: Final[int] = 20_000

# Bumped whenever the schema or the primary keys change. A database written by
# an older loader is refused rather than appended to, because mixing rows keyed
# two different ways produces a silently wrong answer key.
SCHEMA_VERSION: Final[int] = 2


class SchemaVersionError(RuntimeError):
    """The database on disk was built by a loader with a different schema."""


@dataclass(frozen=True)
class LoadReport:
    """What a load actually did, per table.

    ``duplicates_ignored`` is reported rather than swallowed. A nonzero count
    across quarters is expected, since a submission can appear in more than one
    quarterly file, but a large count inside a single quarter means the primary
    key is wrong and the answer key cannot be trusted.
    """

    quarter: str
    rows_inserted: dict[str, int]
    duplicates_ignored: dict[str, int]
    null_values: int

    @property
    def total_rows(self) -> int:
        return sum(self.rows_inserted.values())


def _column_type(column: str) -> str:
    if column in _INTEGER_COLUMNS:
        return "INTEGER"
    if column in _REAL_COLUMNS:
        return "REAL"
    return "TEXT"


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    for table, columns in REQUIRED_COLUMNS.items():
        key_columns = _PRIMARY_KEYS[table]
        # Primary key columns are declared NOT NULL because SQLite permits
        # repeated NULLs inside a PRIMARY KEY. Without this, every fact whose
        # coreg is empty defeats the key and reloading a quarter silently
        # duplicates the answer key instead of being ignored.
        definitions = ", ".join(
            f"{column} {_column_type(column)}" + (" NOT NULL" if column in key_columns else "")
            for column in columns
        )
        primary_key = ", ".join(key_columns)
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ({definitions}, PRIMARY KEY ({primary_key}))"
        )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS load_log ("
        "quarter TEXT, tbl TEXT, rows_inserted INTEGER, duplicates_ignored INTEGER, "
        "loaded_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    connection.execute("CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)")

    row = connection.execute("SELECT version FROM schema_meta").fetchone()
    if row is None:
        connection.execute("INSERT INTO schema_meta (version) VALUES (?)", (SCHEMA_VERSION,))
    elif int(row[0]) != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"this database was built by loader schema v{row[0]}, but this is "
            f"v{SCHEMA_VERSION}. The fact primary key changed, so appending would "
            "mix rows keyed two different ways and produce a silently wrong answer "
            "key. Delete data/xbrl.db and reload; the downloaded archives are "
            "cached, so it costs no network."
        )
    connection.commit()


def create_indexes(connection: sqlite3.Connection) -> None:
    """Indexes chosen for the one query shape the benchmark runs constantly:

    look up a fact by submission, tag, period end date, and duration.
    """
    statements = (
        "CREATE INDEX IF NOT EXISTS idx_num_fact ON num (adsh, tag, ddate, qtrs)",
        "CREATE INDEX IF NOT EXISTS idx_num_tag ON num (tag, ddate, qtrs)",
        "CREATE INDEX IF NOT EXISTS idx_sub_cik ON sub (cik, fy, fp)",
        "CREATE INDEX IF NOT EXISTS idx_sub_form ON sub (form, fy, fp)",
        "CREATE INDEX IF NOT EXISTS idx_pre_fact ON pre (adsh, tag, version)",
        "CREATE INDEX IF NOT EXISTS idx_tag_custom ON tag (custom)",
    )
    for statement in statements:
        connection.execute(statement)
    connection.commit()


def _decode(raw: bytes) -> str:
    """Decode a DERA member.

    DERA files are usually UTF-8 but occasionally carry Latin-1 bytes in company
    names. Falling back on the encoding is safe because it cannot change a
    numeric value, only how a label is spelled. Any other fallback in this
    module would not be.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _read_rows(text: str, table: str) -> Iterator[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    header = reader.fieldnames
    if header is None:
        raise ValueError(f"{table}.txt has no header row")

    missing = [column for column in REQUIRED_COLUMNS[table] if column not in header]
    if missing:
        raise ValueError(
            f"{table}.txt is missing required column(s) {missing}. "
            "The DERA schema may have changed; loading a partial table would "
            "produce an answer key that is silently wrong."
        )
    yield from reader


def _coerce(column: str, raw: str | None, *, is_key: bool = False) -> object:
    """Convert a DERA cell to its stored type.

    An empty cell becomes NULL. It never becomes 0 or an empty string standing
    in for a number, because a zero-valued fact passes the question filters and
    would be graded against.

    The one exception is a text primary key column, where an empty cell is part
    of the key rather than a missing value. DERA leaves ``coreg`` empty for the
    parent registrant, which is the common case, and mapping that to NULL would
    defeat the primary key. This normalizes identity, never a value.
    """
    if raw is None:
        return "" if is_key and column not in _INTEGER_COLUMNS else None
    text = raw.strip()
    if not text:
        if is_key and column not in _INTEGER_COLUMNS and column not in _REAL_COLUMNS:
            return ""
        return None
    if column in _INTEGER_COLUMNS:
        return int(text)
    if column in _REAL_COLUMNS:
        return float(text)
    return text


def _insert_batch(
    connection: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    batch: list[tuple[object, ...]],
) -> tuple[int, int]:
    placeholders = ", ".join("?" for _ in columns)
    column_list = ", ".join(columns)
    before = connection.total_changes
    connection.executemany(
        f"INSERT OR IGNORE INTO {table} ({column_list}) VALUES ({placeholders})", batch
    )
    inserted = connection.total_changes - before
    return inserted, len(batch) - inserted


def load_quarter(connection: sqlite3.Connection, zip_path: Path, quarter: str) -> LoadReport:
    """Load one quarterly DERA ZIP. Returns what it did, per table."""
    if not zip_path.exists():
        raise FileNotFoundError(f"DERA archive not found: {zip_path}")

    inserted_counts: dict[str, int] = {}
    ignored_counts: dict[str, int] = {}
    null_values = 0

    with zipfile.ZipFile(zip_path) as archive:
        members = {name.lower() for name in archive.namelist()}
        for table in REQUIRED_COLUMNS:
            if f"{table}.txt" not in members:
                raise ValueError(f"{zip_path.name} does not contain {table}.txt")

        for table, columns in REQUIRED_COLUMNS.items():
            text = _decode(archive.read(f"{table}.txt"))
            inserted = ignored = 0
            batch: list[tuple[object, ...]] = []

            for row in _read_rows(text, table):
                key_columns = _PRIMARY_KEYS[table]
                values = tuple(
                    _coerce(column, row.get(column), is_key=column in key_columns)
                    for column in columns
                )
                if table == "num" and values[columns.index("value")] is None:
                    null_values += 1
                batch.append(values)
                if len(batch) >= _BATCH_SIZE:
                    added, skipped = _insert_batch(connection, table, columns, batch)
                    inserted += added
                    ignored += skipped
                    batch.clear()

            if batch:
                added, skipped = _insert_batch(connection, table, columns, batch)
                inserted += added
                ignored += skipped

            inserted_counts[table] = inserted
            ignored_counts[table] = ignored
            connection.execute(
                "INSERT INTO load_log (quarter, tbl, rows_inserted, duplicates_ignored) "
                "VALUES (?, ?, ?, ?)",
                (quarter, table, inserted, ignored),
            )

    connection.commit()
    return LoadReport(
        quarter=quarter,
        rows_inserted=inserted_counts,
        duplicates_ignored=ignored_counts,
        null_values=null_values,
    )


# ---------------------------------------------------------------- command line


def _cli() -> None:
    """Download and load quarterly DERA archives.

        python -m ingest.dera_loader --quarters 2024q1,2024q2 --db data/xbrl.db

    Runs on the host machine. SEC is not reachable from a sandbox.
    """
    import click

    from ingest.edgar import DERA_QUARTER_URL, client_from_env

    @click.command()
    @click.option(
        "--quarters",
        required=True,
        help="Comma-separated quarters, for example 2024q1,2024q2.",
    )
    @click.option(
        "--db",
        "db_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=Path("data/xbrl.db"),
        show_default=True,
    )
    @click.option(
        "--raw-dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("data/raw"),
        show_default=True,
    )
    def command(quarters: str, db_path: Path, raw_dir: Path) -> None:
        names = [quarter.strip().lower() for quarter in quarters.split(",") if quarter.strip()]
        if not names:
            raise click.UsageError("--quarters listed no quarters")

        client = client_from_env(raw_dir / ".http-cache")
        connection = open_database(db_path)
        create_schema(connection)

        total = 0
        for quarter in names:
            url = DERA_QUARTER_URL.format(quarter=quarter)
            destination = raw_dir / f"{quarter}.zip"
            # Say what is actually happening. Printing "downloading" on a cache
            # hit makes a reload look like a fresh fetch, which matters when the
            # whole point of the cache is that a rerun touches no network.
            cached = destination.exists() and destination.stat().st_size > 0
            click.echo(f"{quarter}: {'already downloaded' if cached else 'downloading'}")
            archive = client.download(url, destination)

            click.echo(f"{quarter}: loading {archive.stat().st_size / 1e6:.0f} MB")
            report = load_quarter(connection, archive, quarter)
            total += report.total_rows
            click.echo(
                f"{quarter}: {report.rows_inserted} inserted, "
                f"{report.duplicates_ignored} duplicates ignored, "
                f"{report.null_values} null values"
            )

        click.echo("building indexes")
        create_indexes(connection)
        click.echo(f"done: {total} rows across {len(names)} quarters into {db_path}")

    command()


if __name__ == "__main__":
    _cli()
