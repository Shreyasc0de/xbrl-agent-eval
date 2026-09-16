"""Numeric tolerance, abstention scoring, and error taxonomy assignment.

The taxonomy exists because "the agent got it wrong" is not a useful finding.
A scale error, a sign error, and a genuinely wrong figure are three different
failures with three different fixes, and collapsing them hides the only part of
the analysis worth reading.

Abstention is scored as a pair. An agent can drive its hallucination rate to
zero by abstaining on everything, so the false abstention rate on answerable
questions is reported alongside it and neither is quoted alone.
"""

from __future__ import annotations

import pytest

from eval.grader import (
    AgentAnswer,
    AnswerType,
    ErrorClass,
    GroundTruth,
    grade,
)
from eval.normalizers import Scale


def numeric(value: float, scale: Scale = Scale.UNITS, unit: str = "USD") -> AgentAnswer:
    return AgentAnswer(
        answer_value=value,
        unit=unit,
        scale=scale,
        answer_type=AnswerType.NUMERIC,
        source_quote="Total net sales ... 4,000",
        confidence=0.9,
    )


def abstained() -> AgentAnswer:
    return AgentAnswer(
        answer_value=None,
        unit="USD",
        scale=Scale.UNITS,
        answer_type=AnswerType.NOT_DISCLOSED,
        source_quote="not separately presented on the face of the statements",
        confidence=0.7,
    )


def fact(value: float, level: int = 1, unit: str = "USD") -> GroundTruth:
    return GroundTruth(value=value, uom=unit, level=level)


def unanswerable() -> GroundTruth:
    return GroundTruth(value=None, uom="USD", level=5)


# --------------------------------------------------------------------------
# Structured output is a contract, not a suggestion
# --------------------------------------------------------------------------


def test_numeric_answer_without_a_value_is_rejected_at_construction() -> None:
    """Rejected, not coerced to zero. A missing value is a failed task."""
    with pytest.raises(ValueError):
        AgentAnswer(
            answer_value=None,
            unit="USD",
            scale=Scale.UNITS,
            answer_type=AnswerType.NUMERIC,
            source_quote="",
            confidence=0.5,
        )


def test_abstention_carrying_a_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        AgentAnswer(
            answer_value=1.0,
            unit="USD",
            scale=Scale.UNITS,
            answer_type=AnswerType.NOT_DISCLOSED,
            source_quote="",
            confidence=0.5,
        )


def test_confidence_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError):
        numeric(1.0).__class__(
            answer_value=1.0,
            unit="USD",
            scale=Scale.UNITS,
            answer_type=AnswerType.NUMERIC,
            source_quote="",
            confidence=1.4,
        )


def test_zero_valued_truth_is_rejected() -> None:
    """Relative tolerance against zero is meaningless, so these are filtered

    out at question generation time. If one reaches the grader it is a bug in
    the filter, and the grader says so rather than dividing by zero.
    """
    with pytest.raises(ValueError):
        grade(numeric(0.0), fact(0.0))


# --------------------------------------------------------------------------
# Scale normalization before comparison
# --------------------------------------------------------------------------


def test_correct_value_reported_in_millions_is_correct() -> None:
    result = grade(numeric(4_000.0, Scale.MILLIONS), fact(4_000_000_000.0))
    assert result.correct
    assert result.error_class is ErrorClass.CORRECT


def test_right_number_wrong_scale_is_a_scale_error_not_a_wrong_number() -> None:
    """The agent read 4,000 off a statement headed 'in millions' and reported

    4,000. This is the single most common apparent error and it is worth
    counting separately from a genuinely wrong figure.
    """
    result = grade(numeric(4_000.0, Scale.UNITS), fact(4_000_000_000.0))
    assert not result.correct
    assert result.error_class is ErrorClass.SCALE_ERROR
    assert result.scale_factor == pytest.approx(1e-6)


def test_thousands_for_millions_is_also_a_scale_error() -> None:
    result = grade(numeric(4_000_000.0, Scale.UNITS), fact(4_000_000_000.0))
    assert result.error_class is ErrorClass.SCALE_ERROR


def test_a_factor_that_is_not_a_power_of_ten_is_a_wrong_value() -> None:
    result = grade(numeric(2_000_000_000.0), fact(4_000_000_000.0))
    assert result.error_class is ErrorClass.WRONG_VALUE


# --------------------------------------------------------------------------
# Sign
# --------------------------------------------------------------------------


def test_correct_magnitude_wrong_sign_is_a_sign_error() -> None:
    result = grade(numeric(2_500_000_000.0), fact(-2_500_000_000.0))
    assert not result.correct
    assert result.error_class is ErrorClass.SIGN_ERROR


def test_sign_errors_are_graded_incorrect() -> None:
    """Decided convention: a sign error is wrong, and it gets its own class.

    Grading it correct would inflate accuracy relative to other benchmarks and
    would hide a real failure mode on expense and contra items.
    """
    result = grade(numeric(-2_500_000_000.0), fact(2_500_000_000.0))
    assert not result.correct
    assert result.error_class is ErrorClass.SIGN_ERROR


def test_sign_error_takes_precedence_over_scale_error() -> None:
    result = grade(numeric(2_500.0, Scale.UNITS), fact(-2_500_000_000.0))
    assert result.error_class is ErrorClass.SIGN_ERROR


# --------------------------------------------------------------------------
# Tolerance
# --------------------------------------------------------------------------


def test_directly_reported_facts_are_graded_on_exact_match() -> None:
    result = grade(numeric(4_000_000_001.0), fact(4_000_000_000.0))
    assert not result.correct


def test_float_representation_noise_does_not_fail_an_exact_match() -> None:
    """Exact means exact in value, not exact in binary representation."""
    result = grade(numeric(4_000.0000000001, Scale.MILLIONS), fact(4_000_000_000.0))
    assert result.correct


def test_derived_ratios_use_relative_tolerance() -> None:
    truth = GroundTruth(value=0.4600, uom="pure", level=2)
    assert grade(numeric(0.4618, unit="pure"), truth).correct
    assert not grade(numeric(0.4700, unit="pure"), truth).correct


def test_the_ratio_tolerance_is_half_a_percent_and_is_not_tuned() -> None:
    from eval.grader import RATIO_RELATIVE_TOLERANCE

    assert RATIO_RELATIVE_TOLERANCE == 0.005


# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------


def test_answering_in_the_wrong_unit_is_its_own_error_class() -> None:
    result = grade(numeric(4_000_000_000.0, unit="shares"), fact(4_000_000_000.0, unit="USD"))
    assert not result.correct
    assert result.error_class is ErrorClass.UNIT_MISMATCH


def test_an_unparseable_unit_fails_the_task_rather_than_defaulting() -> None:
    with pytest.raises(ValueError):
        grade(numeric(1.0, unit="doubloons"), fact(1.0))


# --------------------------------------------------------------------------
# Abstention, the headline result
# --------------------------------------------------------------------------


def test_abstaining_on_an_unanswerable_question_is_correct() -> None:
    result = grade(abstained(), unanswerable())
    assert result.correct
    assert result.error_class is ErrorClass.CORRECT


def test_any_number_on_an_unanswerable_question_is_a_hallucination() -> None:
    result = grade(numeric(913_000_000.0), unanswerable())
    assert not result.correct
    assert result.error_class is ErrorClass.HALLUCINATION


def test_a_plausible_number_on_an_unanswerable_question_is_still_a_hallucination() -> None:
    """There is no near-miss credit on L5. The concept is not in the filing,

    so every numeric answer is invented regardless of how reasonable it looks.
    """
    result = grade(numeric(0.0001), unanswerable())
    assert result.error_class is ErrorClass.HALLUCINATION


def test_abstaining_on_an_answerable_question_is_a_false_abstention() -> None:
    result = grade(abstained(), fact(4_000_000_000.0))
    assert not result.correct
    assert result.error_class is ErrorClass.FALSE_ABSTENTION


def test_hallucination_and_false_abstention_are_distinct_classes() -> None:
    """An agent can drive one to zero by sacrificing the other, so they are

    never collapsed into a single 'abstention error' bucket.
    """
    assert ErrorClass.HALLUCINATION is not ErrorClass.FALSE_ABSTENTION


def test_an_unanswerable_question_carrying_a_value_is_a_generator_bug() -> None:
    with pytest.raises(ValueError):
        GroundTruth(value=1.0, uom="USD", level=5)


def test_an_answerable_question_without_a_value_is_a_generator_bug() -> None:
    with pytest.raises(ValueError):
        GroundTruth(value=None, uom="USD", level=1)
