"""Resolving a concept to exactly one defensible fact, or dropping it with a reason.

Every question starts as a request: this filing, this concept, this period. Most
requests do not survive. The concept is not tagged, or it is tagged twice with
different values, or the figure was restated later, or it is zero and cannot be
graded on relative tolerance. This module is where a request either becomes a
single fact with a known sign and unit, or becomes a line in the drop log.

Resolution is deliberately conservative. Wherever the data admits two readings,
the question is dropped rather than resolved by preference. A benchmark that
silently picks one of two plausible answers is grading the agent against a coin
flip and reporting it as accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from eval.normalizers import (
    PeriodType,
    apply_presentation_sign,
    normalize_uom,
    period_type_from_qtrs,
)
from questions.concepts import ConceptSpec
from questions.models import DropReason, FactRef
from questions.repository import FactRepository, Submission

__all__ = ["Resolution", "resolve_concept"]

# A ratio denominator this small makes the result numerically unstable and the
# question meaningless, even though the value is not literally zero.
MIN_DENOMINATOR_MAGNITUDE: Final[float] = 1.0


@dataclass(frozen=True)
class Resolution:
    """Either a usable fact or the reason there is not one.

    Exactly one of ``fact`` and ``reason`` is set. Callers check ``ok`` rather
    than truthiness, because a legitimately negative fact value is falsy.
    """

    fact: FactRef | None = None
    reason: DropReason | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if (self.fact is None) == (self.reason is None):
            raise ValueError("a resolution carries exactly one of fact or reason")

    @property
    def ok(self) -> bool:
        return self.fact is not None

    @classmethod
    def failed(cls, reason: DropReason, detail: str = "") -> Resolution:
        return cls(fact=None, reason=reason, detail=detail)


def resolve_concept(
    repository: FactRepository,
    submission: Submission,
    concept: ConceptSpec,
    ddate: int,
    qtrs: int,
) -> Resolution:
    """Resolve one concept in one filing for one period to a single fact.

    The returned value is as presented, meaning the ``pre.txt`` negating flag
    has been applied, matching the convention the grader uses.
    """
    # The registry distinguishes only stocks from flows, while qtrs distinguishes
    # quarters from years from other year-to-date spans. The question that matters
    # here is the coarse one: a balance sheet item must be asked at an instant
    # (qtrs = 0) and an income or cash flow item over some span (qtrs > 0).
    # Which span is a separate decision, made by the generator.
    wants_instant = concept.period_type is PeriodType.INSTANT
    is_instant = period_type_from_qtrs(qtrs) is PeriodType.INSTANT
    if wants_instant != is_instant:
        expected = "an instant" if wants_instant else "a duration"
        return Resolution.failed(
            DropReason.WRONG_PERIOD_TYPE,
            f"{concept.key} must be asked over {expected}, asked with qtrs={qtrs}",
        )

    candidates = repository.facts_for_tags(submission.adsh, concept.tags, ddate, qtrs)
    if not candidates:
        return Resolution.failed(
            DropReason.MISSING_COMPONENT,
            f"no fact for {concept.key} in {submission.adsh} at ddate={ddate} qtrs={qtrs}",
        )

    standard = [fact for fact in candidates if not repository.is_custom_tag(fact.tag, fact.version)]
    if not standard:
        return Resolution.failed(
            DropReason.CUSTOM_EXTENSION_TAG,
            f"only extension tags present for {concept.key}: "
            f"{sorted({fact.tag for fact in candidates})}",
        )

    concrete = [fact for fact in standard if not repository.is_abstract_tag(fact.tag, fact.version)]
    if not concrete:
        return Resolution.failed(
            DropReason.ABSTRACT_TAG, f"only abstract tags present for {concept.key}"
        )

    valued = [fact for fact in concrete if fact.value is not None]
    if not valued:
        return Resolution.failed(
            DropReason.NULL_VALUE, f"{concept.key} is tagged but carries no value"
        )

    # Alternative standard tags for the same concept are acceptable only when
    # they agree. Disagreement means the filing distinguishes them and the
    # question does not, so there is no single right answer.
    distinct_values = {round(float(fact.value), 6) for fact in valued if fact.value is not None}
    if len(distinct_values) > 1:
        detail = ", ".join(f"{fact.tag}={fact.value}" for fact in valued)
        return Resolution.failed(
            DropReason.AMBIGUOUS_MULTIPLE_MATCHES,
            f"{concept.key} resolves to more than one value: {detail}",
        )

    # Preference order only decides which of several agreeing tags is cited.
    preference = {tag: index for index, tag in enumerate(concept.tags)}
    chosen = min(valued, key=lambda fact: preference.get(fact.tag, len(preference)))
    assert chosen.value is not None

    try:
        unit = normalize_uom(chosen.uom)
    except ValueError:
        return Resolution.failed(
            DropReason.UNKNOWN_UNIT, f"{concept.key} reported in unrecognized unit {chosen.uom!r}"
        )
    if unit != concept.uom:
        return Resolution.failed(
            DropReason.UNKNOWN_UNIT,
            f"{concept.key} expected in {concept.uom}, filing reports {unit}",
        )

    if chosen.value == 0.0:
        return Resolution.failed(
            DropReason.ZERO_VALUE,
            f"{concept.key} is zero; relative tolerance against zero is meaningless",
        )

    later = repository.later_values_for_fact(
        submission.cik, chosen.tag, ddate, qtrs, submission.filed
    )
    restated = [value for value in later if round(value, 6) not in distinct_values]
    if restated:
        return Resolution.failed(
            DropReason.RESTATED_FACT,
            f"{concept.key} was later reported as {sorted(set(restated))}, "
            f"original {chosen.value}",
        )

    negating = repository.negating(submission.adsh, chosen.tag, chosen.version)
    presented = apply_presentation_sign(float(chosen.value), negating=negating)

    return Resolution(
        fact=FactRef(
            adsh=chosen.adsh,
            tag=chosen.tag,
            version=chosen.version,
            ddate=chosen.ddate,
            qtrs=chosen.qtrs,
            uom=unit,
            value=presented,
        )
    )
