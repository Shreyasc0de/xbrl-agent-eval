"""Normalization of units, scale, sign, and fiscal periods.

A large share of apparent agent errors are not reasoning failures. They are the
agent reading "4,000" off a statement headed "in millions" and reporting 4,000,
or reporting a cost as positive when the statement presents it in parentheses.
Separating those from genuinely wrong numbers is most of the useful analysis, so
normalization lives here, is tested on its own, and never guesses.

Two conventions are decided here rather than inferred at call sites, because
both affect the validity of the answer key:

**Scale.** DERA ``num.txt`` values are actual values in their unit of measure.
The iXBRL ``scale`` attribute is a rendering artifact of the human-readable
filing. So ground truth needs no scaling, and the agent's answer always does.

**Sign.** Ground truth is stored AS PRESENTED, meaning the ``negating`` flag
from ``pre.txt`` is applied. The agent only ever sees the rendered filing, so
grading against the raw XBRL sign would mark it wrong for a presentation
artifact it had no way to see through.

Nothing here falls back to a default on unrecognized input. An unknown scale or
unit raises, because a quietly defaulted scale produces an answer key that is
confidently wrong and nothing surfaces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Final

__all__ = [
    "FiscalPeriod",
    "PeriodType",
    "Scale",
    "apply_presentation_sign",
    "normalize_uom",
    "normalize_uom_with_factor",
    "parse_scale",
    "period_type_from_qtrs",
    "to_base_units",
]


class Scale(Enum):
    """Multiplier between a displayed figure and its value in base units."""

    UNITS = 1
    THOUSANDS = 10**3
    MILLIONS = 10**6
    BILLIONS = 10**9

    @property
    def multiplier(self) -> int:
        return int(self.value)


# Ordered longest-first within each scale so that "in thousands" is matched as a
# whole rather than partially. Matching is done on a normalized, lowercased
# string with punctuation and currency symbols removed.
_SCALE_WORDS: Final[dict[Scale, tuple[str, ...]]] = {
    Scale.UNITS: ("units", "unit", "actuals", "actual", "ones", "as reported", "1"),
    Scale.THOUSANDS: ("thousands", "thousand", "in thousands", "000s", "000", "k"),
    Scale.MILLIONS: ("millions", "million", "in millions", "mm", "mn", "m"),
    Scale.BILLIONS: ("billions", "billion", "in billions", "bn", "b"),
}

_UOM_ALIASES: Final[dict[str, str]] = {
    "usd": "USD",
    "iso4217:usd": "USD",
    "usd/shares": "USD/shares",
    "iso4217:usd/shares": "USD/shares",
    "usdpershare": "USD/shares",
    "shares": "shares",
    "xbrli:shares": "shares",
    "pure": "pure",
    "xbrli:pure": "pure",
}

# A ratio expressed as a percentage. XBRL stores these as ``pure`` decimals, so
# an agent answering "46.2 percent" against a truth of 0.462 has given the right
# figure in an equivalent unit. Refusing it, which is what the grader used to do,
# discarded correct answers as ungradeable and inflated the failure count.
#
# The conversion is deliberately one-directional and applied only to the agent's
# answer. Ground truth comes from the database, which never carries a percent
# unit, so converting the truth side would mean something had gone wrong
# upstream rather than that a unit needed translating.
#
# This cannot turn a wrong answer into a right one: dividing an incorrect figure
# by 100 leaves it incorrect. It can only rescue a correct one.
_PERCENT_ALIASES: Final[frozenset[str]] = frozenset(
    {"percent", "percentage", "%", "pct", "percents", "per cent"}
)

_VALID_FP: Final[frozenset[str]] = frozenset({"FY", "Q1", "Q2", "Q3", "Q4"})


def _normalize_scale_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s/]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_scale(text: str) -> Scale:
    """Parse a scale from the wording filings and agents actually use.

    Raises ``ValueError`` on anything unrecognized. Defaulting to units here
    would turn a parse failure into a silent factor-of-a-million error.
    """
    normalized = _normalize_scale_text(text)
    if not normalized:
        raise ValueError("empty scale string")

    tokens = set(normalized.split())
    for scale, words in _SCALE_WORDS.items():
        for word in words:
            if word == normalized or word in tokens:
                return scale
    raise ValueError(f"unrecognized scale: {text!r}")


def to_base_units(value: float, scale: Scale | str) -> float:
    """Convert a displayed figure to its value in base units."""
    resolved = scale if isinstance(scale, Scale) else parse_scale(scale)
    return value * resolved.multiplier


def normalize_uom(raw: str) -> str:
    """Canonicalize a unit of measure. Raises ``ValueError`` if unrecognized."""
    key = raw.strip().lower()
    if key not in _UOM_ALIASES:
        raise ValueError(f"unrecognized unit of measure: {raw!r}")
    return _UOM_ALIASES[key]


def normalize_uom_with_factor(raw: str) -> tuple[str, float]:
    """Canonical unit, plus the factor that converts a value into it.

    Everything except a percentage returns a factor of 1.0, so this is
    ``normalize_uom`` with one translation bolted on rather than a second unit
    system. ``normalize_uom`` itself still rejects "percent", because the only
    place a percent legitimately appears is an agent's answer, and a percent
    arriving from anywhere else is a bug worth raising on.
    """
    if raw.strip().lower() in _PERCENT_ALIASES:
        return "pure", 0.01
    return normalize_uom(raw), 1.0


def apply_presentation_sign(value: float, *, negating: bool) -> float:
    """Apply the ``pre.txt`` negating flag to a stored XBRL value.

    This is a flip, not a coercion to negative. A value already stored with a
    negative sign flips to positive, which is the behavior that distinguishes
    this from ``-abs(value)``.
    """
    return -value if negating else value


class PeriodType(Enum):
    """What a fact's ``qtrs`` duration means."""

    INSTANT = "instant"
    QUARTER = "quarter"
    DURATION = "duration"
    YEAR = "year"


def period_type_from_qtrs(qtrs: int) -> PeriodType:
    """Map the DERA ``qtrs`` field to a period type.

    ``qtrs`` is a duration measured in quarters: 0 for an instantaneous balance
    sheet item, 1 for a single quarter, 4 for a full year. Values of 2 and 3 are
    year-to-date durations. Getting this wrong produces an answer key that is
    confidently wrong rather than one that crashes.
    """
    if qtrs < 0:
        raise ValueError(f"qtrs must be non-negative, got {qtrs}")
    if qtrs == 0:
        return PeriodType.INSTANT
    if qtrs == 1:
        return PeriodType.QUARTER
    if qtrs == 4:
        return PeriodType.YEAR
    return PeriodType.DURATION


@dataclass(frozen=True)
class FiscalPeriod:
    """A fiscal period as the answer key identifies it.

    ``fy`` and ``fp`` come from ``sub.txt`` and are the only safe alignment key.
    ``ddate`` is the period end date and is deliberately excluded from the
    alignment key, because fiscal years do not line up with calendar years: a
    September fiscal year end and a December one are both FY2024.
    """

    fy: int
    fp: str
    ddate: date
    qtrs: int

    def __post_init__(self) -> None:
        if self.fp not in _VALID_FP:
            raise ValueError(f"invalid fiscal period label: {self.fp!r}")
        if self.qtrs < 0:
            raise ValueError(f"qtrs must be non-negative, got {self.qtrs}")

    @property
    def period_type(self) -> PeriodType:
        return period_type_from_qtrs(self.qtrs)

    @property
    def duration_months(self) -> int:
        return self.qtrs * 3

    @property
    def alignment_key(self) -> tuple[int, str, PeriodType]:
        """The key two facts must share to be comparable.

        Period type is part of the key because a balance sheet item and an
        income statement item can share a ``ddate`` while measuring different
        things, and comparing them is a category error.
        """
        return (self.fy, self.fp, self.period_type)

    def prior_year(self) -> FiscalPeriod:
        """The same fiscal period one year earlier, for year-over-year questions.

        ``ddate`` is shifted by the nominal 364 days rather than by calendar
        year, since a 52/53-week fiscal calendar does not repeat the same date.
        The shifted ``ddate`` is indicative only; alignment still goes through
        ``alignment_key``, which ignores it.
        """
        shifted = date.fromordinal(self.ddate.toordinal() - 364)
        return FiscalPeriod(fy=self.fy - 1, fp=self.fp, ddate=shifted, qtrs=self.qtrs)
