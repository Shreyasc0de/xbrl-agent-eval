"""The concepts questions are generated about, and the ratios derived from them.

Every concept here is a standard us-gaap tag. Company extension tags are
excluded by construction, not by filtering after the fact, because an extension
tag means something only to the company that defined it and a question about one
has no defensible single answer.

Some concepts have more than one standard tag in common use. Revenue is the
clearest case: a filing may carry ``Revenues``, or
``RevenueFromContractWithCustomerExcludingAssessedTax``, or both. Alternatives
are listed in preference order, but the filters treat a filing that tags two of
them with different values as ambiguous and drop the question rather than
picking one. Picking one silently is how a benchmark ends up grading against a
number the agent had no way to know was intended.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from eval.normalizers import PeriodType

__all__ = [
    "CONCEPTS",
    "RATIOS",
    "ConceptSpec",
    "RatioSpec",
    "concept_for_tag",
]


@dataclass(frozen=True)
class ConceptSpec:
    """A financial concept, its standard tags, and how it is reported.

    ``period_type`` is what makes a question well posed. Asking for cash "for
    fiscal 2024" is a category error, because cash is a balance at an instant,
    not a flow over a year. The generators phrase instant and duration concepts
    differently and the filters reject a fact whose ``qtrs`` disagrees.
    """

    key: str
    label: str
    tags: tuple[str, ...]
    period_type: PeriodType
    uom: str
    statement: str

    @property
    def primary_tag(self) -> str:
        return self.tags[0]


_CONCEPT_LIST: Final[tuple[ConceptSpec, ...]] = (
    ConceptSpec(
        key="revenue",
        label="total revenue",
        tags=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
        ),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="income statement",
    ),
    ConceptSpec(
        key="cost_of_revenue",
        label="cost of revenue",
        tags=("CostOfGoodsAndServicesSold", "CostOfRevenue"),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="income statement",
    ),
    ConceptSpec(
        key="gross_profit",
        label="gross profit",
        tags=("GrossProfit",),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="income statement",
    ),
    ConceptSpec(
        key="operating_income",
        label="operating income",
        tags=("OperatingIncomeLoss",),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="income statement",
    ),
    ConceptSpec(
        key="net_income",
        label="net income",
        tags=("NetIncomeLoss",),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="income statement",
    ),
    ConceptSpec(
        key="research_and_development",
        label="research and development expense",
        tags=("ResearchAndDevelopmentExpense",),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="income statement",
    ),
    ConceptSpec(
        key="selling_general_administrative",
        label="selling, general and administrative expense",
        tags=("SellingGeneralAndAdministrativeExpense",),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="income statement",
    ),
    ConceptSpec(
        key="operating_cash_flow",
        label="net cash provided by operating activities",
        tags=("NetCashProvidedByUsedInOperatingActivities",),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="cash flow statement",
    ),
    ConceptSpec(
        key="capital_expenditure",
        label="payments to acquire property, plant and equipment",
        tags=("PaymentsToAcquirePropertyPlantAndEquipment",),
        period_type=PeriodType.DURATION,
        uom="USD",
        statement="cash flow statement",
    ),
    ConceptSpec(
        key="assets",
        label="total assets",
        tags=("Assets",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="current_assets",
        label="total current assets",
        tags=("AssetsCurrent",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="liabilities",
        label="total liabilities",
        tags=("Liabilities",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="current_liabilities",
        label="total current liabilities",
        tags=("LiabilitiesCurrent",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="stockholders_equity",
        label="total stockholders' equity",
        tags=("StockholdersEquity",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="cash_and_equivalents",
        label="cash and cash equivalents",
        tags=("CashAndCashEquivalentsAtCarryingValue",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="inventory",
        label="inventory, net",
        tags=("InventoryNet",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="goodwill",
        label="goodwill",
        tags=("Goodwill",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="long_term_debt",
        label="long-term debt, noncurrent",
        tags=("LongTermDebtNoncurrent",),
        period_type=PeriodType.INSTANT,
        uom="USD",
        statement="balance sheet",
    ),
    ConceptSpec(
        key="diluted_eps",
        label="diluted earnings per share",
        tags=("EarningsPerShareDiluted",),
        period_type=PeriodType.DURATION,
        uom="USD/shares",
        statement="income statement",
    ),
)

CONCEPTS: Final[Mapping[str, ConceptSpec]] = {spec.key: spec for spec in _CONCEPT_LIST}

_TAG_TO_CONCEPT: Final[Mapping[str, ConceptSpec]] = {
    tag: spec for spec in _CONCEPT_LIST for tag in spec.tags
}


def concept_for_tag(tag: str) -> ConceptSpec | None:
    return _TAG_TO_CONCEPT.get(tag)


@dataclass(frozen=True)
class RatioSpec:
    """A ratio derived from exactly two concepts.

    Restricting ratios to two components is deliberate. A three-term formula
    multiplies the chance that one component is missing or ambiguous, and the
    resulting question tests the generator's luck more than the agent's ability.

    ``as_percentage`` only changes how the question is phrased and how the
    answer is scaled. The grading tolerance is relative either way.

    A ratio may divide a flow by a stock, as return on assets divides a year of
    net income by a balance sheet total. That mismatch is legitimate but it has
    to be deliberate, so ``is_flow_over_stock`` reports it and the generators
    take the instant component at the period end date. The reverse, a stock
    divided by a flow, is not a ratio anyone reports and is rejected by the
    registry validator.
    """

    key: str
    label: str
    numerator: str
    denominator: str
    as_percentage: bool
    uom: str = "pure"

    @property
    def is_flow_over_stock(self) -> bool:
        return (
            CONCEPTS[self.numerator].period_type is PeriodType.DURATION
            and CONCEPTS[self.denominator].period_type is PeriodType.INSTANT
        )

    def compute(self, numerator_value: float, denominator_value: float) -> float:
        if denominator_value == 0.0:
            raise ZeroDivisionError(f"{self.key}: denominator is zero")
        ratio = numerator_value / denominator_value
        return ratio * 100.0 if self.as_percentage else ratio


_RATIO_LIST: Final[tuple[RatioSpec, ...]] = (
    RatioSpec("gross_margin", "gross margin", "gross_profit", "revenue", True),
    RatioSpec("operating_margin", "operating margin", "operating_income", "revenue", True),
    RatioSpec("net_margin", "net profit margin", "net_income", "revenue", True),
    RatioSpec(
        "rnd_intensity",
        "research and development expense as a share of revenue",
        "research_and_development",
        "revenue",
        True,
    ),
    RatioSpec("current_ratio", "current ratio", "current_assets", "current_liabilities", False),
    RatioSpec(
        "debt_to_equity", "debt-to-equity ratio", "liabilities", "stockholders_equity", False
    ),
    RatioSpec("return_on_assets", "return on assets", "net_income", "assets", True),
    RatioSpec(
        "cash_conversion",
        "operating cash flow as a share of net income",
        "operating_cash_flow",
        "net_income",
        True,
    ),
)

RATIOS: Final[Mapping[str, RatioSpec]] = {spec.key: spec for spec in _RATIO_LIST}


def _validate_registry() -> None:
    """Catch a malformed registry at import time rather than mid-generation."""
    for ratio in _RATIO_LIST:
        for component in (ratio.numerator, ratio.denominator):
            if component not in CONCEPTS:
                raise ValueError(f"ratio {ratio.key} references unknown concept {component}")
        numerator = CONCEPTS[ratio.numerator]
        denominator = CONCEPTS[ratio.denominator]
        if numerator.period_type is denominator.period_type:
            continue
        # A flow over a stock is a real ratio: return on assets divides a year
        # of net income by a period-end balance. A stock over a flow is not,
        # and almost always means the components were transposed.
        if not (
            numerator.period_type is PeriodType.DURATION
            and denominator.period_type is PeriodType.INSTANT
        ):
            raise ValueError(
                f"ratio {ratio.key} divides a {numerator.period_type.value} concept by a "
                f"{denominator.period_type.value} one; the components look transposed"
            )

    seen: dict[str, str] = {}
    for spec in _CONCEPT_LIST:
        for tag in spec.tags:
            if tag in seen:
                raise ValueError(f"tag {tag} claimed by both {seen[tag]} and {spec.key}")
            seen[tag] = spec.key


_validate_registry()
