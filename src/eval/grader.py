"""Grading: tolerance, abstention scoring, and error taxonomy.

This is where most benchmarks quietly fail, so the rules are explicit and the
tolerances are stated rather than tuned until the results look good.

Tolerance. Directly reported facts are graded on exact match after scale
normalization. "Exact" is enforced as an absolute half-cent floor rather than a
relative tolerance, because a relative tolerance loose enough to absorb binary
floating point noise is also loose enough to accept an answer that is several
dollars wrong on a multi-billion-dollar figure. At these magnitudes a half cent
is far below the precision any filing reports, so the floor rejects every real
disagreement while accepting representation noise. Derived ratios use a relative
tolerance of 0.5%, which accommodates the rounding a human analyst would accept
when combining two figures read off a rendered statement.

Abstention. On an unanswerable question, correct means the agent said
``not_disclosed``, and any numeric answer is a hallucination. The inverse,
abstaining on a question that is answerable, is a false abstention. Both are
reported, always as a pair, because an agent can drive either to zero by
sacrificing the other.

Nothing here substitutes a default for input it cannot parse. An unrecognized
unit or a zero-valued truth raises, so a filter bug upstream surfaces as a
failed task rather than as a quietly wrong grade.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

from eval.normalizers import Scale, normalize_uom, normalize_uom_with_factor, to_base_units

__all__ = [
    "EXACT_ABSOLUTE_TOLERANCE",
    "EXACT_RELATIVE_TOLERANCE",
    "RATIO_RELATIVE_TOLERANCE",
    "AgentAnswer",
    "AnswerType",
    "ErrorClass",
    "Grade",
    "GroundTruth",
    "grade",
]

# Slack for binary floating point representation only. Not a grading tolerance.
# Half a cent absolute, which is below the reporting precision of any filing.
EXACT_ABSOLUTE_TOLERANCE: Final[float] = 0.005
# Companion relative term, for magnitudes where a double cannot represent cents
# at all. At 4e9 this is 4e-5, so the absolute floor is what binds in practice.
EXACT_RELATIVE_TOLERANCE: Final[float] = 1e-14

# Derived ratios. Justified in the README, not tuned against results.
RATIO_RELATIVE_TOLERANCE: Final[float] = 0.005

# A wrong answer counts as a scale error only if the ratio to the truth is a
# power of ten within this band. Wider than the exact tolerance because the
# agent's underlying figure may itself be slightly off.
_SCALE_RATIO_TOLERANCE: Final[float] = 0.01
_MAX_SCALE_EXPONENT: Final[int] = 12

# Levels whose answers are ratios or other derived quantities.
_DERIVED_LEVELS: Final[frozenset[int]] = frozenset({2, 3})


class AnswerType(Enum):
    NUMERIC = "numeric"
    NOT_DISCLOSED = "not_disclosed"


class ErrorClass(Enum):
    """The taxonomy reported in the results.

    ``WRONG_CONCEPT``, ``WRONG_PERIOD`` and ``RETRIEVAL_MISS`` cannot be
    determined from the numbers alone. They are assigned by the run analysis
    from the agent's retrieval trace and its cited source quote, and appear here
    so the taxonomy has a single definition.
    """

    CORRECT = "correct"
    SCALE_ERROR = "scale_error"
    SIGN_ERROR = "sign_error"
    UNIT_MISMATCH = "unit_mismatch"
    WRONG_VALUE = "wrong_value"
    HALLUCINATION = "hallucination"
    FALSE_ABSTENTION = "false_abstention"
    WRONG_CONCEPT = "wrong_concept"
    WRONG_PERIOD = "wrong_period"
    ARITHMETIC_ERROR = "arithmetic_error"
    RETRIEVAL_MISS = "retrieval_miss"


@dataclass(frozen=True)
class AgentAnswer:
    """The only answer shape the pipeline accepts.

    Free-text answers are ungradeable and are rejected at parse time rather
    than coerced into something that looks gradeable.
    """

    answer_value: float | None
    unit: str
    scale: Scale
    answer_type: AnswerType
    source_quote: str
    confidence: float

    def __post_init__(self) -> None:
        if self.answer_type is AnswerType.NUMERIC and self.answer_value is None:
            raise ValueError("a numeric answer must carry a value")
        if self.answer_type is AnswerType.NOT_DISCLOSED and self.answer_value is not None:
            raise ValueError("a not_disclosed answer must not carry a value")
        if self.answer_value is not None and not math.isfinite(self.answer_value):
            raise ValueError(f"answer_value must be finite, got {self.answer_value}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def value_in_base_units(self) -> float:
        if self.answer_value is None:
            raise ValueError("cannot convert an abstention to base units")
        return to_base_units(self.answer_value, self.scale)


@dataclass(frozen=True)
class GroundTruth:
    """The answer key for one question, in base units and as presented.

    ``value`` is ``None`` exactly when the question is unanswerable, which is
    level 5. Any other combination is a bug in the question generator and is
    rejected here rather than graded.
    """

    value: float | None
    uom: str
    level: int

    def __post_init__(self) -> None:
        if self.level not in (1, 2, 3, 4, 5):
            raise ValueError(f"level must be 1 through 5, got {self.level}")
        if self.level == 5 and self.value is not None:
            raise ValueError("an unanswerable question must not carry a value")
        if self.level != 5 and self.value is None:
            raise ValueError(f"an answerable question (level {self.level}) must carry a value")

    @property
    def is_unanswerable(self) -> bool:
        return self.level == 5

    @property
    def is_derived(self) -> bool:
        return self.level in _DERIVED_LEVELS


@dataclass(frozen=True)
class Grade:
    correct: bool
    error_class: ErrorClass
    predicted_base_units: float | None
    truth_base_units: float | None
    relative_error: float | None
    scale_factor: float | None


def _is_power_of_ten(ratio: float) -> bool:
    if ratio <= 0.0:
        return False
    exponent = round(math.log10(ratio))
    if exponent == 0 or abs(exponent) > _MAX_SCALE_EXPONENT:
        return False
    return math.isclose(ratio, 10.0**exponent, rel_tol=_SCALE_RATIO_TOLERANCE)


def grade(answer: AgentAnswer, truth: GroundTruth) -> Grade:
    """Grade one answer against the key.

    Raises ``ValueError`` on input the grader cannot honestly evaluate: a
    zero-valued truth, an unrecognized unit. Both indicate an upstream filter
    bug, and failing the task is the correct outcome.
    """
    if truth.is_unanswerable:
        if answer.answer_type is AnswerType.NOT_DISCLOSED:
            return Grade(True, ErrorClass.CORRECT, None, None, None, None)
        return Grade(
            False,
            ErrorClass.HALLUCINATION,
            answer.value_in_base_units,
            None,
            None,
            None,
        )

    assert truth.value is not None  # guaranteed by GroundTruth.__post_init__
    if truth.value == 0.0:
        raise ValueError(
            "zero-valued ground truth reached the grader; these are filtered at "
            "question generation because relative tolerance against zero is meaningless"
        )

    if answer.answer_type is AnswerType.NOT_DISCLOSED:
        return Grade(False, ErrorClass.FALSE_ABSTENTION, None, truth.value, None, None)

    # Unit comparison goes through normalization on both sides so that an
    # unrecognized unit raises instead of silently failing to match. Only the
    # agent's side carries a conversion factor, for percentages: a ratio
    # answered as "46.2 percent" against a truth of 0.462 is the right figure in
    # an equivalent unit. Truth comes from the database, which stores ratios as
    # pure decimals, so a factor on that side would mean something upstream was
    # wrong rather than that a unit needed translating.
    predicted_unit, unit_factor = normalize_uom_with_factor(answer.unit)
    truth_unit = normalize_uom(truth.uom)

    predicted = answer.value_in_base_units * unit_factor

    if predicted_unit != truth_unit:
        return Grade(False, ErrorClass.UNIT_MISMATCH, predicted, truth.value, None, None)

    relative_error = abs(predicted - truth.value) / abs(truth.value)
    scale_factor = predicted / truth.value

    if truth.is_derived:
        within = math.isclose(predicted, truth.value, rel_tol=RATIO_RELATIVE_TOLERANCE, abs_tol=0.0)
    else:
        within = math.isclose(
            predicted,
            truth.value,
            rel_tol=EXACT_RELATIVE_TOLERANCE,
            abs_tol=EXACT_ABSOLUTE_TOLERANCE,
        )

    if within:
        return Grade(
            True,
            ErrorClass.CORRECT,
            predicted,
            truth.value,
            relative_error,
            scale_factor,
        )

    # Sign is checked before scale. A value that is both sign-flipped and
    # mis-scaled is reported as a sign error, because the sign is the part a
    # reader of the rendered statement could have got right.
    if predicted != 0.0:
        magnitude_ratio = abs(predicted) / abs(truth.value)
        opposite_sign = (predicted < 0.0) != (truth.value < 0.0)
        if opposite_sign and (
            math.isclose(magnitude_ratio, 1.0, rel_tol=_SCALE_RATIO_TOLERANCE)
            or _is_power_of_ten(magnitude_ratio)
        ):
            return Grade(
                False,
                ErrorClass.SIGN_ERROR,
                predicted,
                truth.value,
                relative_error,
                scale_factor,
            )

        if _is_power_of_ten(magnitude_ratio):
            return Grade(
                False,
                ErrorClass.SCALE_ERROR,
                predicted,
                truth.value,
                relative_error,
                scale_factor,
            )

    return Grade(
        False,
        ErrorClass.WRONG_VALUE,
        predicted,
        truth.value,
        relative_error,
        scale_factor,
    )
