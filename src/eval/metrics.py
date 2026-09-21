"""Metrics, with intervals rather than bare point estimates.

Per-level sample sizes here are small. With roughly 700 questions across five
levels, a level holds on the order of a hundred, and the difference between 68%
and 73% on a hundred questions is not a difference. Every rate is therefore
reported with a 95% Wilson interval, and comparisons between systems are made on
overlapping intervals rather than on point estimates.

Wilson rather than the normal approximation because the normal interval is badly
behaved exactly where this benchmark lives: small samples and proportions near
0 or 1. A hallucination rate of 2 out of 140 has a normal interval that includes
negative numbers, which is not a defensible thing to publish.

Two rates are always reported together. An agent can drive its hallucination
rate to zero by abstaining on everything, so the false abstention rate on
answerable questions sits beside it and neither is quoted alone.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

__all__ = [
    "Interval",
    "Rate",
    "TaskOutcome",
    "percentile",
    "summarize_run",
    "wilson_interval",
]

Z_95: Final[float] = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def overlaps(self, other: Interval) -> bool:
        """Whether two intervals overlap, which is how systems are compared.

        Two systems whose intervals overlap have not been shown to differ. That
        is a weaker claim than "they are the same", and the README says so.
        """
        return self.low <= other.high and other.low <= self.high


@dataclass(frozen=True)
class Rate:
    """A proportion with its interval and the counts behind it.

    ``successes`` and ``total`` are kept so a reader can recompute the rate and
    the interval without trusting either number.
    """

    successes: int
    total: int
    point: float
    interval: Interval

    def to_json(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "total": self.total,
            "point": round(self.point, 4),
            "ci95_low": round(self.interval.low, 4),
            "ci95_high": round(self.interval.high, 4),
        }


def wilson_interval(successes: int, total: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion.

    An empty sample returns [0, 1] rather than raising or returning a point.
    That is the honest interval for no evidence, and it keeps a level with no
    questions from silently reporting 0% accuracy.
    """
    if successes < 0 or total < 0:
        raise ValueError("counts must be non-negative")
    if successes > total:
        raise ValueError(f"{successes} successes out of {total} is impossible")
    if total == 0:
        return Interval(0.0, 1.0)

    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    spread = (
        z
        / denominator
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    )
    return Interval(max(0.0, centre - spread), min(1.0, centre + spread))


def rate(successes: int, total: int) -> Rate:
    return Rate(
        successes=successes,
        total=total,
        point=successes / total if total else 0.0,
        interval=wilson_interval(successes, total),
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile. Empty input gives 0.0."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class TaskOutcome:
    """One graded task, reduced to what the metrics need.

    ``failed`` is distinct from ``correct=False``. A task that errored produced
    no answer, and counting it as wrong would credit the agent with an attempt
    it never made, while dropping it silently would inflate accuracy. It is
    excluded from accuracy and reported separately.
    """

    question_id: str
    level: int
    correct: bool
    failed: bool
    error_class: str
    cost_usd: float
    latency_ms: float
    tool_calls: int
    model_calls: int
    opened_correct_filing: bool
    opened_answer_passage: bool | None = None


def summarize_run(outcomes: Sequence[TaskOutcome]) -> dict[str, Any]:
    """Every published number, computed from task outcomes alone.

    This is the only place run-level numbers are produced. ``verify_results.py``
    calls it on the raw artifacts, so a claim in the README either matches what
    this function returns or the build fails.
    """
    graded = [outcome for outcome in outcomes if not outcome.failed]
    failed = [outcome for outcome in outcomes if outcome.failed]

    by_level: dict[str, Any] = {}
    for level in (1, 2, 3, 4, 5):
        at_level = [outcome for outcome in graded if outcome.level == level]
        correct = sum(1 for outcome in at_level if outcome.correct)
        by_level[str(level)] = rate(correct, len(at_level)).to_json()

    answerable = [outcome for outcome in graded if outcome.level != 5]
    unanswerable = [outcome for outcome in graded if outcome.level == 5]

    hallucinations = sum(1 for outcome in unanswerable if outcome.error_class == "hallucination")
    false_abstentions = sum(
        1 for outcome in answerable if outcome.error_class == "false_abstention"
    )

    correct_total = sum(1 for outcome in graded if outcome.correct)
    total_cost = sum(outcome.cost_usd for outcome in outcomes)
    latencies = [outcome.latency_ms for outcome in outcomes]

    # Filing-level, and named so. With a roster this small, finding the right
    # 10-K is a lookup rather than retrieval, so this number sits near 1.0 by
    # construction and says nothing about whether the agent read the right
    # passage inside it. Passage-level hits are measured separately.
    filing_eligible = [outcome for outcome in graded if outcome.level in (1, 2, 3)]
    filing_hits = sum(1 for outcome in filing_eligible if outcome.opened_correct_filing)

    # The metric that actually moves. Only annotated tasks are eligible: a task
    # whose answer passage was never located contributes nothing rather than a
    # zero, so the denominator says how much of the set this number covers.
    # A hit means the agent read a passage where the figure is printed, not that
    # it read the right one. See annotate.passages for why that distinction is
    # stated rather than smoothed over.
    passage_eligible = [
        outcome
        for outcome in graded
        if outcome.level != 5 and outcome.opened_answer_passage is not None
    ]
    passage_hits = sum(1 for outcome in passage_eligible if outcome.opened_answer_passage)

    taxonomy = Counter(outcome.error_class for outcome in graded if not outcome.correct)

    return {
        "tasks": len(outcomes),
        "graded": len(graded),
        "failed": len(failed),
        "accuracy_overall": rate(correct_total, len(graded)).to_json(),
        "accuracy_by_level": by_level,
        "hallucination_rate_l5": rate(hallucinations, len(unanswerable)).to_json(),
        "false_abstention_rate_l1_l4": rate(false_abstentions, len(answerable)).to_json(),
        "correct_filing_rate": rate(filing_hits, len(filing_eligible)).to_json(),
        "passage_hit_rate": (
            rate(passage_hits, len(passage_eligible)).to_json() if passage_eligible else None
        ),
        "passage_annotated": len(passage_eligible),
        "error_taxonomy": dict(sorted(taxonomy.items())),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_correct_answer_usd": (
            round(total_cost / correct_total, 6) if correct_total else None
        ),
        "latency_ms": {
            "median": round(percentile(latencies, 0.5), 1),
            "p95": round(percentile(latencies, 0.95), 1),
        },
        "tool_calls_per_task": {
            "median": round(percentile([o.tool_calls for o in outcomes], 0.5), 2),
            "p95": round(percentile([o.tool_calls for o in outcomes], 0.95), 2),
        },
        "model_calls": sum(outcome.model_calls for outcome in outcomes),
    }


def outcome_to_json(outcome: TaskOutcome) -> dict[str, Any]:
    return asdict(outcome)
