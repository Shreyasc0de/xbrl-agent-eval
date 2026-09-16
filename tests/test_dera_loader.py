"""DERA loader: schema strictness, type coercion, and the query the benchmark runs.

The fixtures here are synthetic, hand-built to exercise the loader's contract.
They are not real SEC data and no figure in them is used as a result.
"""

from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path

import pytest

from ingest.dera_loader import (
    REQUIRED_COLUMNS,
    create_indexes,
    create_schema,
    load_quarter,
    open_database,
)

SUB = (
    "adsh\tcik\tname\tsic\tform\tperiod\tfy\tfp\tfiled\tprevrpt\n"
    "0000000001-24-000001\t1750\tEXAMPLE INDUSTRIAL CORP\t3663\t10-K\t20241231\t2024\tFY\t20250214\t0\n"
    "0000000002-24-000001\t320193\tEXAMPLE TECH INC\t3571\t10-K\t20240928\t2024\tFY\t20241101\t0\n"
)

NUM = (
    "adsh\ttag\tversion\tcoreg\tddate\tqtrs\tuom\tvalue\n"
    "0000000001-24-000001\tRevenues\tus-gaap/2024\t\t20241231\t4\tUSD\t4000000000\n"
    "0000000001-24-000001\tCostOfGoodsAndServicesSold\tus-gaap/2024\t\t20241231\t4\tUSD\t2500000000\n"
    "0000000001-24-000001\tCashAndCashEquivalentsAtCarryingValue\tus-gaap/2024\t\t20241231\t0\tUSD\t812400000\n"
    "0000000001-24-000001\tSomeUntaggedThing\tus-gaap/2024\t\t20241231\t4\tUSD\t\n"
    "0000000002-24-000001\tRevenues\tus-gaap/2024\t\t20240928\t4\tUSD\t391035000000\n"
)

PRE = (
    "adsh\treport\tline\tstmt\tinpth\ttag\tversion\tplabel\tnegating\n"
    "0000000001-24-000001\t2\t1\tIS\t0\tRevenues\tus-gaap/2024\tTotal net sales\t0\n"
    "0000000001-24-000001\t2\t2\tIS\t0\tCostOfGoodsAndServicesSold\tus-gaap/2024\tCost of sales\t1\n"
)

TAG = (
    "tag\tversion\tcustom\tabstract\tdatatype\tiord\tcrdr\ttlabel\n"
    "Revenues\tus-gaap/2024\t0\t0\tmonetary\tD\tC\tRevenues\n"
    "CostOfGoodsAndServicesSold\tus-gaap/2024\t0\t0\tmonetary\tD\tD\tCost of goods sold\n"
    "CashAndCashEquivalentsAtCarryingValue\tus-gaap/2024\t0\t0\tmonetary\tI\tD\tCash\n"
    "SomeUntaggedThing\tus-gaap/2024\t0\t0\tmonetary\tD\tD\tThing\n"
    "ExmpAdjustedSegmentProfit\texmp/2024\t1\t0\tmonetary\tD\tC\tAdjusted segment profit\n"
)


def make_zip(path: Path, **overrides: str) -> Path:
    members = {"sub.txt": SUB, "num.txt": NUM, "pre.txt": PRE, "tag.txt": TAG}
    members.update({f"{k}.txt": v for k, v in overrides.items()})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    path.write_bytes(buffer.getvalue())
    return path


@pytest.fixture
def db(tmp_path: Path) -> sqlite3.Connection:
    connection = open_database(tmp_path / "xbrl.db")
    create_schema(connection)
    return connection


@pytest.fixture
def loaded(db: sqlite3.Connection, tmp_path: Path) -> sqlite3.Connection:
    load_quarter(db, make_zip(tmp_path / "2024q4.zip"), "2024q4")
    create_indexes(db)
    return db


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_all_four_tables_are_loaded(db: sqlite3.Connection, tmp_path: Path) -> None:
    report = load_quarter(db, make_zip(tmp_path / "2024q4.zip"), "2024q4")
    assert report.rows_inserted == {"sub": 2, "num": 5, "pre": 2, "tag": 5}
    assert report.total_rows == 14


def test_a_missing_archive_raises(db: sqlite3.Connection, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_quarter(db, tmp_path / "nope.zip", "2024q4")


def test_a_missing_member_file_raises(db: sqlite3.Connection, tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("sub.txt", SUB)
    path = tmp_path / "broken.zip"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="num.txt"):
        load_quarter(db, path, "2024q4")


def test_a_missing_required_column_raises_rather_than_loading_partially(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    """A silently absent qtrs column would make every duration fact ambiguous."""
    truncated = NUM.replace("\tqtrs", "").replace("\t4\tUSD", "\tUSD").replace("\t0\tUSD", "\tUSD")
    with pytest.raises(ValueError, match="qtrs"):
        load_quarter(db, make_zip(tmp_path / "bad.zip", num=truncated), "2024q4")


def test_loading_the_same_quarter_twice_is_idempotent(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    archive = make_zip(tmp_path / "2024q4.zip")
    load_quarter(db, archive, "2024q4")
    second = load_quarter(db, archive, "2024q4")
    assert second.rows_inserted["num"] == 0
    assert second.duplicates_ignored["num"] == 5
    assert db.execute("SELECT COUNT(*) FROM num").fetchone()[0] == 5


def test_the_load_is_logged(loaded: sqlite3.Connection) -> None:
    rows = loaded.execute(
        "SELECT tbl, rows_inserted FROM load_log WHERE quarter = '2024q4' ORDER BY tbl"
    ).fetchall()
    assert dict(rows) == {"sub": 2, "num": 5, "pre": 2, "tag": 5}


# --------------------------------------------------------------------------
# Type coercion
# --------------------------------------------------------------------------


def test_numeric_columns_are_stored_as_numbers(loaded: sqlite3.Connection) -> None:
    value, qtrs, ddate = loaded.execute(
        "SELECT value, qtrs, ddate FROM num WHERE tag = 'Revenues' AND adsh LIKE '%1-24%'"
    ).fetchone()
    assert value == 4_000_000_000.0
    assert qtrs == 4
    assert ddate == 20241231


def test_an_empty_value_becomes_null_and_never_zero(loaded: sqlite3.Connection) -> None:
    """A zero would pass the question filters and be graded against."""
    row = loaded.execute("SELECT value FROM num WHERE tag = 'SomeUntaggedThing'").fetchone()
    assert row[0] is None

    zeros = loaded.execute("SELECT COUNT(*) FROM num WHERE value = 0").fetchone()[0]
    assert zeros == 0


def test_null_values_are_counted_in_the_report(db: sqlite3.Connection, tmp_path: Path) -> None:
    report = load_quarter(db, make_zip(tmp_path / "2024q4.zip"), "2024q4")
    assert report.null_values == 1


def test_latin1_bytes_in_a_company_name_do_not_break_the_load(
    db: sqlite3.Connection, tmp_path: Path
) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "sub.txt", SUB.replace("EXAMPLE TECH INC", "CAF\xc9 CORP").encode("latin-1")
        )
        archive.writestr("num.txt", NUM)
        archive.writestr("pre.txt", PRE)
        archive.writestr("tag.txt", TAG)
    path = tmp_path / "latin.zip"
    path.write_bytes(buffer.getvalue())
    load_quarter(db, path, "2024q4")
    names = [r[0] for r in db.execute("SELECT name FROM sub").fetchall()]
    assert any("CAF" in n for n in names)


# --------------------------------------------------------------------------
# The query the benchmark actually runs
# --------------------------------------------------------------------------


def test_week_one_exit_criterion_one_query_per_fact(loaded: sqlite3.Connection) -> None:
    """Pull any tagged fact for any company in a single query.

    This is the Week 1 exit criterion from the spec, expressed as a test so it
    keeps holding.
    """
    row = loaded.execute(
        """
        SELECT n.value, n.uom, s.fy, s.fp, t.custom, p.negating
        FROM num n
        JOIN sub s ON s.adsh = n.adsh
        JOIN tag t ON t.tag = n.tag AND t.version = n.version
        LEFT JOIN pre p ON p.adsh = n.adsh AND p.tag = n.tag AND p.version = n.version
        WHERE s.cik = ? AND n.tag = ? AND n.ddate = ? AND n.qtrs = ?
        """,
        (1750, "CostOfGoodsAndServicesSold", 20241231, 4),
    ).fetchone()
    value, uom, fy, fp, custom, negating = row
    assert (value, uom, fy, fp, custom, negating) == (
        2_500_000_000.0,
        "USD",
        2024,
        "FY",
        0,
        1,
    )


def test_custom_extension_tags_are_identifiable_for_filtering(
    loaded: sqlite3.Connection,
) -> None:
    customs = loaded.execute("SELECT tag FROM tag WHERE custom = 1").fetchall()
    assert customs == [("ExmpAdjustedSegmentProfit",)]


def test_a_september_fiscal_year_end_is_still_fy2024(
    loaded: sqlite3.Connection,
) -> None:
    fy, fp, period = loaded.execute("SELECT fy, fp, period FROM sub WHERE cik = 320193").fetchone()
    assert (fy, fp) == (2024, "FY")
    assert period == 20240928


def test_instant_and_duration_facts_are_distinguishable(
    loaded: sqlite3.Connection,
) -> None:
    qtrs = loaded.execute(
        "SELECT qtrs FROM num WHERE tag = 'CashAndCashEquivalentsAtCarryingValue'"
    ).fetchone()[0]
    assert qtrs == 0


def test_required_columns_cover_what_the_spec_names(loaded: sqlite3.Connection) -> None:
    assert {"adsh", "tag", "ddate", "qtrs", "uom", "value"} <= set(REQUIRED_COLUMNS["num"])
    assert "negating" in REQUIRED_COLUMNS["pre"]
    assert "custom" in REQUIRED_COLUMNS["tag"]
