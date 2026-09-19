"""Shared synthetic ground-truth database for the question generation tests.

Every figure here is invented. Nothing in this file is real SEC data and no
number in it is ever reported as a result. The fixture exists to exercise the
generators' contracts: sign handling, ambiguity, restatement, peer logic.

The shape is four companies in one SIC major group over two fiscal years, which
is the smallest arrangement that supports a peer-based L5 question and a
three-company L4 comparison at the same time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ingest.dera_loader import create_indexes, create_schema, open_database
from questions.companies import Company
from questions.repository import FactRepository

SIC_SEMICONDUCTORS = 3674  # major group 36

ACME, BETA, GAMMA, DELTA = 1001, 1002, 1003, 1004

FIXTURE_COMPANIES = (
    Company("ACME", "Acme Devices Inc.", ACME, "ACME", "technology"),
    Company("BETA", "Beta Systems Inc.", BETA, "BETA", "technology"),
    Company("GAMA", "Gamma Semiconductor Corp.", GAMMA, "GAMMA", "technology"),
    Company("DLTA", "Delta Microelectronics Inc.", DELTA, "DELTA", "technology"),
)

_NAMES = {
    ACME: "ACME DEVICES INC",
    BETA: "BETA SYSTEMS INC",
    GAMMA: "GAMMA SEMICONDUCTOR CORP",
    DELTA: "DELTA MICROELECTRONICS INC",
}

US_GAAP = "us-gaap/2024"

# Standard tags used by the fixture, with the metadata the filters consult.
_STANDARD_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "GrossProfit",
    "CostOfGoodsAndServicesSold",
    "OperatingIncomeLoss",
    "NetIncomeLoss",
    "ResearchAndDevelopmentExpense",
    "Assets",
    "AssetsCurrent",
    "LiabilitiesCurrent",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
)


def _adsh(cik: int, fy: int) -> str:
    return f"{cik:010d}-{fy % 100:02d}-000001"


def _insert_submission(connection: sqlite3.Connection, cik: int, fy: int, *, filed: int) -> str:
    adsh = _adsh(cik, fy)
    connection.execute(
        "INSERT OR REPLACE INTO sub "
        "(adsh, cik, name, sic, form, period, fy, fp, filed, prevrpt) "
        "VALUES (?, ?, ?, ?, '10-K', ?, ?, 'FY', ?, 0)",
        (adsh, cik, _NAMES[cik], SIC_SEMICONDUCTORS, fy * 10000 + 1231, fy, filed),
    )
    return adsh


def _insert_fact(
    connection: sqlite3.Connection,
    adsh: str,
    tag: str,
    ddate: int,
    qtrs: int,
    value: float | None,
    uom: str = "USD",
    segments: str = "",
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO num "
        "(adsh, tag, version, coreg, segments, ddate, qtrs, uom, value) "
        "VALUES (?, ?, ?, '', ?, ?, ?, ?, ?)",
        (adsh, tag, US_GAAP, segments, ddate, qtrs, uom, value),
    )


def _insert_presentation(
    connection: sqlite3.Connection, adsh: str, tag: str, line: int, negating: int
) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO pre "
        "(adsh, report, line, stmt, inpth, tag, version, plabel, negating) "
        "VALUES (?, 2, ?, 'IS', 0, ?, ?, ?, ?)",
        (adsh, line, tag, US_GAAP, tag, negating),
    )


def _company_facts(connection: sqlite3.Connection, cik: int, fy: int, scale: float) -> None:
    """A complete, well-formed set of facts for one company-year."""
    adsh = _adsh(cik, fy)
    ddate = fy * 10000 + 1231
    duration = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": 1_000_000_000.0 * scale,
        "GrossProfit": 400_000_000.0 * scale,
        "CostOfGoodsAndServicesSold": 600_000_000.0 * scale,
        "OperatingIncomeLoss": 250_000_000.0 * scale,
        "NetIncomeLoss": 200_000_000.0 * scale,
        "ResearchAndDevelopmentExpense": 120_000_000.0 * scale,
    }
    instant = {
        "Assets": 5_000_000_000.0 * scale,
        "AssetsCurrent": 2_000_000_000.0 * scale,
        "LiabilitiesCurrent": 1_000_000_000.0 * scale,
        "Liabilities": 2_500_000_000.0 * scale,
        "StockholdersEquity": 2_500_000_000.0 * scale,
        "CashAndCashEquivalentsAtCarryingValue": 800_000_000.0 * scale,
    }
    for tag, value in duration.items():
        _insert_fact(connection, adsh, tag, ddate, 4, value)
    for tag, value in instant.items():
        _insert_fact(connection, adsh, tag, ddate, 0, value)

    # Cost of sales is presented in parentheses, so the presentation layer flips
    # its sign. Everything else is presented as stored.
    _insert_presentation(connection, adsh, "CostOfGoodsAndServicesSold", 2, negating=1)
    _insert_presentation(
        connection, adsh, "RevenueFromContractWithCustomerExcludingAssessedTax", 1, negating=0
    )


@pytest.fixture(scope="session")
def ground_truth_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("xbrl") / "xbrl.db"
    connection = open_database(path)
    create_schema(connection)

    for tag in _STANDARD_TAGS:
        connection.execute(
            "INSERT OR REPLACE INTO tag "
            "(tag, version, custom, abstract, datatype, iord, crdr, tlabel) "
            "VALUES (?, ?, 0, 0, 'monetary', 'D', 'C', ?)",
            (tag, US_GAAP, tag),
        )
    connection.execute(
        "INSERT OR REPLACE INTO tag "
        "(tag, version, custom, abstract, datatype, iord, crdr, tlabel) "
        "VALUES ('AcmeAdjustedRevenue', 'acme/2024', 1, 0, 'monetary', 'D', 'C', 'Adjusted')"
    )

    for cik, scale in ((ACME, 1.0), (BETA, 1.5), (GAMMA, 0.8), (DELTA, 2.0)):
        for fy, filed in ((2023, 20240215), (2024, 20250215)):
            _insert_submission(connection, cik, fy, filed=filed)
            _company_facts(connection, cik, fy, scale * (1.1 if fy == 2024 else 1.0))

    # ACME does not tag R&D in FY2024, while its three peers do. This is the
    # arrangement a valid L5 question is built from.
    connection.execute(
        "DELETE FROM num WHERE adsh = ? AND tag = 'ResearchAndDevelopmentExpense'",
        (_adsh(ACME, 2024),),
    )

    # DELTA reports revenue broken down by product line as well as consolidated.
    # The breakdown rows share every identifying column with the consolidated
    # row except segments, which is the collision that silently replaced
    # consolidated totals with single slices in the first real load.
    for fy in (2023, 2024):
        ddate = fy * 10000 + 1231
        for segment, share in (
            ("ProductOrService=Hardware;", 0.7),
            ("ProductOrService=Services;", 0.3),
        ):
            _insert_fact(
                connection,
                _adsh(DELTA, fy),
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                ddate,
                4,
                1_000_000_000.0 * 2.0 * (1.1 if fy == 2024 else 1.0) * share,
                segments=segment,
            )

    # BETA FY2023 tags two revenue concepts with different values, which makes
    # any revenue question about that filing ambiguous.
    _insert_fact(connection, _adsh(BETA, 2023), "Revenues", 20231231, 4, 1_400_000_000.0)

    # GAMMA restated FY2023 net income in its FY2024 filing.
    _insert_fact(connection, _adsh(GAMMA, 2024), "NetIncomeLoss", 20231231, 4, 111_000_000.0)

    connection.commit()
    create_indexes(connection)
    connection.close()
    return path


@pytest.fixture
def repository(ground_truth_db: Path) -> FactRepository:
    connection = sqlite3.connect(ground_truth_db)
    return FactRepository(connection)
