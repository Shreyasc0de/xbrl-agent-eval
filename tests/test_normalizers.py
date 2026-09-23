"""Unit, scale, sign, and fiscal period normalization.

Two sections, kept deliberately separate:

**Mechanical cases.** Arithmetic and enum semantics that follow from the DERA
field definitions alone. These need no filing and run everywhere.

**Hand-verified cases.** Figures read by hand from an actual filing, with the
accession number and page recorded. Per CLAUDE.md these are the real test of the
normalizers, because they are the only ones that catch a plausible-but-wrong
reading of the DERA schema. The parameter table is currently empty and the tests
are skipped. No invented figures are used as a stand-in.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from eval.normalizers import (
    FiscalPeriod,
    PeriodType,
    Scale,
    apply_presentation_sign,
    normalize_uom,
    parse_scale,
    period_type_from_qtrs,
    to_base_units,
)

# --------------------------------------------------------------------------
# Scale
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("units", Scale.UNITS),
        ("unit", Scale.UNITS),
        ("actuals", Scale.UNITS),
        ("thousands", Scale.THOUSANDS),
        ("in thousands", Scale.THOUSANDS),
        ("K", Scale.THOUSANDS),
        ("millions", Scale.MILLIONS),
        ("MM", Scale.MILLIONS),
        ("$ in millions", Scale.MILLIONS),
        ("billions", Scale.BILLIONS),
        ("B", Scale.BILLIONS),
    ],
)
def test_parse_scale_accepts_the_spellings_filings_actually_use(text: str, expected: Scale) -> None:
    assert parse_scale(text) is expected


def test_parse_scale_rejects_the_unknown_rather_than_defaulting() -> None:
    """No silent fallbacks. An unrecognized scale is an error, not 'units'."""
    with pytest.raises(ValueError):
        parse_scale("gazillions")


@pytest.mark.parametrize(
    ("value", "scale", "expected"),
    [
        (4000.0, Scale.MILLIONS, 4_000_000_000.0),
        (812_400.0, Scale.THOUSANDS, 812_400_000.0),
        (1.5, Scale.BILLIONS, 1_500_000_000.0),
        (391_035_000_000.0, Scale.UNITS, 391_035_000_000.0),
        (-2500.0, Scale.MILLIONS, -2_500_000_000.0),
    ],
)
def test_to_base_units(value: float, scale: Scale, expected: float) -> None:
    assert to_base_units(value, scale) == pytest.approx(expected, rel=1e-12)


def test_to_base_units_accepts_a_scale_string() -> None:
    assert to_base_units(4000.0, "millions") == pytest.approx(4_000_000_000.0)


def test_dera_values_are_already_in_base_units() -> None:
    """The DERA num.txt value field carries the actual value in its uom.

    iXBRL `scale` is a display artifact of the rendered filing, so ground truth
    needs no scaling and the agent's answer does. Encoding that asymmetry as a
    test keeps a future refactor from 'fixing' it symmetrically.
    """
    assert Scale.UNITS.multiplier == 1


# --------------------------------------------------------------------------
# Units of measure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("USD", "USD"),
        ("usd", "USD"),
        ("iso4217:USD", "USD"),
        ("shares", "shares"),
        ("USD/shares", "USD/shares"),
        ("pure", "pure"),
    ],
)
def test_normalize_uom(raw: str, expected: str) -> None:
    assert normalize_uom(raw) == expected


def test_normalize_uom_rejects_the_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_uom("doubloons")


# --------------------------------------------------------------------------
# Sign
# --------------------------------------------------------------------------
# Convention, decided and recorded here because the choice is not recoverable
# from the data: ground truth is stored AS PRESENTED. The pre.txt negating flag
# is applied, so the answer key matches what a human reads on the rendered
# statement. The agent only ever sees the rendered filing, so grading it against
# the raw XBRL sign would penalize it for a presentation artifact.


def test_negating_flag_flips_the_stored_value() -> None:
    assert apply_presentation_sign(2_500_000_000.0, negating=True) == -2_500_000_000.0


def test_without_the_negating_flag_the_stored_value_is_unchanged() -> None:
    assert apply_presentation_sign(2_500_000_000.0, negating=False) == 2_500_000_000.0


def test_negating_is_a_flip_not_an_absolute_value() -> None:
    """A value already stored negative flips to positive.

    This is the case that distinguishes a correct implementation from
    ``-abs(value)``, which happens to pass on every naturally-positive expense
    and fails on the rest.
    """
    assert apply_presentation_sign(-1_000.0, negating=True) == 1_000.0


def test_applying_the_flag_twice_returns_the_original() -> None:
    once = apply_presentation_sign(2_500.0, negating=True)
    assert apply_presentation_sign(once, negating=True) == 2_500.0


# --------------------------------------------------------------------------
# Fiscal periods
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qtrs", "expected"),
    [
        (0, PeriodType.INSTANT),
        (1, PeriodType.QUARTER),
        (2, PeriodType.DURATION),
        (3, PeriodType.DURATION),
        (4, PeriodType.YEAR),
    ],
)
def test_period_type_from_qtrs(qtrs: int, expected: PeriodType) -> None:
    """qtrs is a duration in quarters: 0 instantaneous, 1 a quarter, 4 a year."""
    assert period_type_from_qtrs(qtrs) is expected


def test_negative_qtrs_is_an_error() -> None:
    with pytest.raises(ValueError):
        period_type_from_qtrs(-1)


def test_instant_facts_have_no_duration() -> None:
    period = FiscalPeriod(fy=2024, fp="Q4", ddate=date(2024, 12, 31), qtrs=0)
    assert period.period_type is PeriodType.INSTANT
    assert period.duration_months == 0


def test_annual_facts_span_twelve_months() -> None:
    period = FiscalPeriod(fy=2024, fp="FY", ddate=date(2024, 12, 31), qtrs=4)
    assert period.duration_months == 12


def test_fiscal_year_is_not_calendar_year() -> None:
    """A September fiscal year end is FY2024, not FY2025 or a 2024 Q3 label.

    The alignment key is (fy, fp) from sub.txt, never the calendar month of
    ddate. This is the trap that silently produces a confidently wrong answer
    key for companies whose fiscal year does not end in December.
    """
    september_fy_end = FiscalPeriod(fy=2024, fp="FY", ddate=date(2024, 9, 28), qtrs=4)
    december_fy_end = FiscalPeriod(fy=2024, fp="FY", ddate=date(2024, 12, 31), qtrs=4)
    assert september_fy_end.alignment_key == december_fy_end.alignment_key
    assert september_fy_end.ddate != december_fy_end.ddate


def test_same_calendar_date_different_fiscal_year_does_not_align() -> None:
    a = FiscalPeriod(fy=2024, fp="Q1", ddate=date(2024, 1, 31), qtrs=1)
    b = FiscalPeriod(fy=2025, fp="Q1", ddate=date(2024, 1, 31), qtrs=1)
    assert a.alignment_key != b.alignment_key


def test_instant_and_duration_facts_never_align() -> None:
    """A balance sheet item and an income statement item can share a ddate.

    Comparing them would be a category error, so the alignment key includes the
    period type.
    """
    instant = FiscalPeriod(fy=2024, fp="FY", ddate=date(2024, 12, 31), qtrs=0)
    annual = FiscalPeriod(fy=2024, fp="FY", ddate=date(2024, 12, 31), qtrs=4)
    assert instant.alignment_key != annual.alignment_key


def test_invalid_fiscal_period_label_is_rejected() -> None:
    with pytest.raises(ValueError):
        FiscalPeriod(fy=2024, fp="Q5", ddate=date(2024, 12, 31), qtrs=1)


def test_prior_year_comparable_shifts_fiscal_year_only() -> None:
    """YoY alignment moves fy back by one and holds fp and period type fixed."""
    current = FiscalPeriod(fy=2024, fp="Q3", ddate=date(2024, 6, 29), qtrs=1)
    prior = current.prior_year()
    assert prior.fy == 2023
    assert prior.fp == "Q3"
    assert prior.qtrs == current.qtrs


# --------------------------------------------------------------------------
# Hand-verified figures from actual filings
# --------------------------------------------------------------------------
# CLAUDE.md requires these to be read by hand from a real filing, with the
# source recorded, and checked against what the answer key would serve. This is
# the only test in the suite that can catch a whole class of error the rest
# cannot: every other test compares the pipeline against itself or against a
# fixture the same code wrote, so a systematic misreading of DERA would pass
# all of them. A human reading a printed statement is the only independent
# check available.
#
# Two skip conditions, kept apart on purpose. An empty table is unfinished
# work. A missing database is an environment without ground truth loaded, such
# as CI. Collapsing them would let an empty table hide behind "no database
# here", which is the specific way this test could rot into decoration.

# (adsh, tag, uom, ddate, qtrs, expected_base_units, source_note)
#
# ddate and qtrs are not optional. A 10-K states the prior year alongside the
# current one, so adsh and tag alone identify two different figures, and the
# wrong one looks entirely plausible.
HAND_VERIFIED_FACTS: list[tuple[str, str, str, int, int, float, str]] = [
    # Example of the shape, kept commented out because it is not verified:
    # (
    #     "0000320193-24-000123",
    #     "CashAndCashEquivalentsAtCarryingValue",
    #     "USD",
    #     20240928,
    #     0,
    #     29_943_000_000.0,
    #     "Apple FY2024 10-K, consolidated balance sheet, in millions",
    # ),
]


def _ground_truth_available() -> bool:
    from core.paths import default_db_path

    return default_db_path().exists()


@pytest.mark.needs_ground_truth
# Order matters. pytest reports the condition closest to the function, and the
# empty table is the one that must be visible: "no database here" is a fact
# about the machine, while "no figures recorded" is unfinished work.
@pytest.mark.skipif(
    not _ground_truth_available(),
    reason="No ground truth database on this machine. Run on the host after make quarters.",
)
@pytest.mark.skipif(
    not HAND_VERIFIED_FACTS,
    reason="No hand-verified filing figures recorded yet. NOT a passing test.",
)
@pytest.mark.parametrize(
    ("adsh", "tag", "uom", "ddate", "qtrs", "expected", "source_note"),
    HAND_VERIFIED_FACTS,
)
def test_hand_verified_fact_matches_the_answer_key(
    adsh: str, tag: str, uom: str, ddate: int, qtrs: int, expected: float, source_note: str
) -> None:
    """A figure read by eye off the filing equals what the answer key serves.

    Consolidated rows only. The first real load stored dimensional breakdowns
    under the same key as consolidated totals, which turned one company's
    revenue into the revenue of a single hotel category. A hand-verified figure
    that did not filter on segments would reproduce that bug rather than catch
    it.
    """
    import sqlite3

    from core.paths import default_db_path
    from eval.grader import EXACT_ABSOLUTE_TOLERANCE

    connection = sqlite3.connect(default_db_path())
    try:
        rows = connection.execute(
            "SELECT value FROM num "
            "WHERE adsh = ? AND tag = ? AND uom = ? AND ddate = ? AND qtrs = ? "
            "AND coreg = '' AND segments = ''",
            (adsh, tag, uom, ddate, qtrs),
        ).fetchall()
    finally:
        connection.close()

    assert rows, (
        f"no consolidated fact for {tag} in {adsh} at ddate {ddate} qtrs {qtrs}. "
        f"Source: {source_note}"
    )
    assert len(rows) == 1, (
        f"{len(rows)} consolidated facts match {tag} in {adsh} at ddate {ddate} "
        f"qtrs {qtrs}; a hand-verified figure must identify exactly one row"
    )

    stored = rows[0][0]
    assert stored is not None, f"the answer key stores NULL for {tag} in {adsh}"
    assert math.isclose(
        float(stored), expected, rel_tol=0.0, abs_tol=EXACT_ABSOLUTE_TOLERANCE
    ), f"answer key says {stored!r}, the filing says {expected!r}. Source: {source_note}"
