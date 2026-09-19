"""Metrics, checked against values computed independently of the implementation.

The Wilson interval cases come from the closed form worked by hand, not from
running this code and recording what it said. A test that asserts the code
agrees with itself proves nothing.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import Interval, TaskOutcome, percentile, summarize_run, wilson_interval


def outcome(
    level: int,
    correct: bool,
    error_class: str = "correct",
    *,
    failed: bool = False,
    cost: float = 0.0,
    latency: float = 100.0,
    opened: bool = True,
) -> TaskOutcome:
    return TaskOutcome(
        question_id=f"q{level}{error_class}{correct}{latency}",
        level=level,
        correct=correct,
        failed=failed,
        error_class=error_class,
        cost_usd=cost,
        latency_ms=latency,
        tool_calls=3,
        model_calls=4,
        opened_correct_filing=opened,
    )


# --------------------------------------------------------------------------
# Wilson intervals
# --------------------------------------------------------------------------


def test_wilson_matches_the_closed_form() -> None:
    """Worked by hand for 7 successes in 10 at z = 1.96."""
    z = 1.959963984540054
    n, k = 10, 7
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z / denominator * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))

    interval = wilson_interval(k, n)
    assert interval.low == pytest.approx(centre - spread)
    assert interval.high == pytest.approx(centre + spread)


def test_wilson_never_goes_below_zero_at_a_low_rate() -> None:
    """A hallucination rate of 2 in 140 has a normal interval that includes

    negative numbers. That is why this is Wilson and not the normal
    approximation.
    """
    interval = wilson_interval(2, 140)
    assert interval.low >= 0.0
    assert interval.high < 0.10


def test_wilson_never_exceeds_one_at_a_perfect_rate() -> None:
    interval = wilson_interval(40, 40)
    assert interval.high <= 1.0
    assert interval.low < 1.0


def test_a_perfect_small_sample_still_has_a_wide_interval() -> None:
    """Five out of five is not evidence of 100%, and the interval says so."""
    interval = wilson_interval(5, 5)
    assert interval.low < 0.6


def test_an_empty_sample_returns_the_whole_range() -> None:
    """No evidence is [0, 1], not 0%. A level with no questions must not read

    as a level the agent failed.
    """
    assert wilson_interval(0, 0) == Interval(0.0, 1.0)


def test_impossible_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_overlapping_intervals_are_detected() -> None:
    assert Interval(0.40, 0.60).overlaps(Interval(0.55, 0.75))
    assert not Interval(0.10, 0.20).overlaps(Interval(0.30, 0.40))


# --------------------------------------------------------------------------
# Percentiles
# --------------------------------------------------------------------------


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_percentile_of_empty_is_zero() -> None:
    assert percentile([], 0.5) == 0.0


def test_percentile_bounds_are_validated() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


# --------------------------------------------------------------------------
# Run summary
# --------------------------------------------------------------------------


def test_failed_tasks_are_excluded_from_accuracy_not_counted_wrong() -> None:
    """Counting a failure as incorrect credits the agent with an attempt it

    never made. Dropping it silently inflates accuracy. It gets its own count.
    """
    outcomes = [
        outcome(1, True),
        outcome(1, False, "wrong_value"),
        outcome(1, False, "task_failed", failed=True),
    ]
    summary = summarize_run(outcomes)
    assert summary["tasks"] == 3
    assert summary["graded"] == 2
    assert summary["failed"] == 1
    assert summary["accuracy_overall"]["total"] == 2
    assert summary["accuracy_overall"]["successes"] == 1


def test_hallucination_rate_counts_only_level_five() -> None:
    outcomes = [
        outcome(5, False, "hallucination"),
        outcome(5, True),
        outcome(1, False, "wrong_value"),
    ]
    summary = summarize_run(outcomes)
    assert summary["hallucination_rate_l5"]["successes"] == 1
    assert summary["hallucination_rate_l5"]["total"] == 2


def test_false_abstention_counts_only_answerable_questions() -> None:
    outcomes = [
        outcome(1, False, "false_abstention"),
        outcome(2, True),
        outcome(5, True),
    ]
    summary = summarize_run(outcomes)
    assert summary["false_abstention_rate_l1_l4"]["successes"] == 1
    assert summary["false_abstention_rate_l1_l4"]["total"] == 2


def test_both_abstention_rates_are_always_reported() -> None:
    """An agent can drive either to zero by sacrificing the other, so neither

    is ever published alone.
    """
    summary = summarize_run([outcome(1, True), outcome(5, True)])
    assert "hallucination_rate_l5" in summary
    assert "false_abstention_rate_l1_l4" in summary


def test_cost_per_correct_answer_is_none_when_nothing_is_correct() -> None:
    """Zero correct answers makes the figure undefined, not zero, and reporting

    zero would read as free rather than as worthless.
    """
    summary = summarize_run([outcome(1, False, "wrong_value", cost=0.5)])
    assert summary["cost_per_correct_answer_usd"] is None


def test_cost_per_correct_answer_divides_total_by_correct() -> None:
    outcomes = [outcome(1, True, cost=0.10), outcome(2, True, cost=0.30)]
    summary = summarize_run(outcomes)
    assert summary["cost_per_correct_answer_usd"] == pytest.approx(0.20)


def test_retrieval_precision_ignores_unanswerable_questions() -> None:
    """There is no correct filing to open for a concept that is not in one."""
    outcomes = [
        outcome(1, True, opened=True),
        outcome(2, False, "wrong_value", opened=False),
        outcome(5, True, opened=False),
    ]
    summary = summarize_run(outcomes)
    assert summary["retrieval_precision"]["total"] == 2
    assert summary["retrieval_precision"]["successes"] == 1


def test_every_level_appears_even_when_empty() -> None:
    summary = summarize_run([outcome(1, True)])
    assert set(summary["accuracy_by_level"]) == {"1", "2", "3", "4", "5"}
    assert summary["accuracy_by_level"]["3"]["total"] == 0


def test_error_taxonomy_counts_only_incorrect_graded_tasks() -> None:
    outcomes = [
        outcome(1, True),
        outcome(1, False, "scale_error"),
        outcome(1, False, "scale_error"),
        outcome(2, False, "sign_error"),
    ]
    summary = summarize_run(outcomes)
    assert summary["error_taxonomy"] == {"scale_error": 2, "sign_error": 1}
