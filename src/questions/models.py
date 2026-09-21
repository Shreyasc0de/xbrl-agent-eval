"""Question records and the dropped-question log.

Two ideas carry the weight here.

**Provenance.** Every question records the exact ``num.txt`` rows its answer was
computed from. A question whose answer cannot be traced back to specific facts
is not a question, it is an assertion, and the whole point of generating the set
programmatically is that the answer key is derived rather than typed.

**The drop log.** Questions are discarded constantly: custom extension tags,
restated facts, null values, ambiguous matches. Every discard is recorded with
its reason. The log is not debugging output, it is evidence that the surviving
questions survived something, and it is published alongside the question set.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "DropLog",
    "DropReason",
    "DroppedQuestion",
    "FactRef",
    "Question",
    "QuestionSet",
]


class DropReason(Enum):
    """Why a candidate question did not make it into the set.

    Each value is a distinct, countable reason. A catch-all bucket would defeat
    the purpose of the log, so there is deliberately no ``OTHER``.
    """

    CUSTOM_EXTENSION_TAG = "custom_extension_tag"
    ABSTRACT_TAG = "abstract_tag"
    RESTATED_FACT = "restated_fact"
    NULL_VALUE = "null_value"
    ZERO_VALUE = "zero_value"
    AMBIGUOUS_MULTIPLE_MATCHES = "ambiguous_multiple_matches"
    MISSING_COMPONENT = "missing_component"
    UNKNOWN_UNIT = "unknown_unit"
    WRONG_PERIOD_TYPE = "wrong_period_type"
    DUPLICATE_QUESTION = "duplicate_question"
    INSUFFICIENT_PEERS = "insufficient_peers"
    NO_PRIOR_YEAR_FILING = "no_prior_year_filing"
    NO_ANNUAL_FILING = "no_annual_filing"
    CONCEPT_PRESENT_IN_FILING = "concept_present_in_filing"
    DENOMINATOR_TOO_SMALL = "denominator_too_small"


@dataclass(frozen=True)
class FactRef:
    """One ``num.txt`` row that an answer depends on.

    This is the audit trail. Given a question and its fact refs, the answer is
    recomputable from the database without rerunning the generator.
    """

    adsh: str
    tag: str
    version: str
    ddate: int
    qtrs: int
    uom: str
    value: float

    def as_key(self) -> tuple[str, str, str, int, int]:
        return (self.adsh, self.tag, self.version, self.ddate, self.qtrs)


@dataclass(frozen=True)
class Question:
    """One graded question.

    ``answer_value`` is ``None`` exactly when ``level`` is 5, matching the
    contract that ``eval.grader.GroundTruth`` enforces on the other side.
    """

    question_id: str
    level: int
    template_id: str
    text: str
    cik: int
    company_name: str
    primary_adsh: str
    fy: int
    fp: str
    answer_value: float | None
    answer_uom: str
    concept: str
    facts: tuple[FactRef, ...] = ()
    supporting_adsh: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.level not in (1, 2, 3, 4, 5):
            raise ValueError(f"level must be 1 through 5, got {self.level}")
        if self.level == 5:
            if self.answer_value is not None:
                raise ValueError("an unanswerable question must not carry a value")
            if self.facts:
                raise ValueError(
                    "an unanswerable question must not cite supporting facts; "
                    "the absence of facts is the answer"
                )
        else:
            if self.answer_value is None:
                raise ValueError(f"a level {self.level} question must carry a value")
            if not self.facts:
                raise ValueError(
                    f"a level {self.level} question must cite the facts its answer "
                    "was computed from; an answer with no provenance is not gradeable"
                )
        if not self.text.strip():
            raise ValueError("question text must not be empty")

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["facts"] = [asdict(fact) for fact in self.facts]
        return payload


def make_question_id(template_id: str, *parts: object) -> str:
    """Deterministic id, so regenerating the set reproduces the same ids.

    Two candidates that differ only in wording collide on purpose: that is how
    duplicates are detected.
    """
    material = "|".join([template_id, *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class DroppedQuestion:
    reason: DropReason
    level: int
    template_id: str
    cik: int | None
    adsh: str | None
    concept: str
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason"] = self.reason.value
        return payload


class DropLog:
    """Accumulates every discarded candidate, with counts by reason."""

    def __init__(self) -> None:
        self._entries: list[DroppedQuestion] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[DroppedQuestion]:
        return iter(self._entries)

    def record(
        self,
        reason: DropReason,
        *,
        level: int,
        template_id: str,
        concept: str,
        cik: int | None = None,
        adsh: str | None = None,
        detail: str = "",
    ) -> None:
        self._entries.append(
            DroppedQuestion(
                reason=reason,
                level=level,
                template_id=template_id,
                cik=cik,
                adsh=adsh,
                concept=concept,
                detail=detail,
            )
        )

    def counts_by_reason(self) -> dict[str, int]:
        counter = Counter(entry.reason.value for entry in self._entries)
        return dict(sorted(counter.items()))

    def counts_by_level(self) -> dict[int, int]:
        counter = Counter(entry.level for entry in self._entries)
        return dict(sorted(counter.items()))

    def write_jsonl(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self._entries:
                handle.write(json.dumps(entry.to_json(), sort_keys=True))
                handle.write("\n")
        return len(self._entries)


@dataclass
class QuestionSet:
    """A generated set, plus the log of everything that did not make it."""

    questions: list[Question] = field(default_factory=list)
    drop_log: DropLog = field(default_factory=DropLog)

    def add(self, question: Question) -> bool:
        """Add a question, rejecting an exact duplicate id.

        Returns whether it was added. A rejected duplicate is logged, because a
        generator quietly producing the same question twice would inflate the
        set size without adding coverage.
        """
        if any(existing.question_id == question.question_id for existing in self.questions):
            self.drop_log.record(
                DropReason.DUPLICATE_QUESTION,
                level=question.level,
                template_id=question.template_id,
                concept=question.concept,
                cik=question.cik,
                adsh=question.primary_adsh,
                detail=f"question_id {question.question_id} already present",
            )
            return False
        self.questions.append(question)
        return True

    def extend(self, questions: Iterable[Question]) -> int:
        return sum(1 for question in questions if self.add(question))

    def counts_by_level(self) -> dict[int, int]:
        counter = Counter(question.level for question in self.questions)
        return dict(sorted(counter.items()))

    def level_share(self, level: int) -> float:
        if not self.questions:
            return 0.0
        return self.counts_by_level().get(level, 0) / len(self.questions)

    def write_jsonl(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for question in sorted(self.questions, key=lambda q: q.question_id):
                handle.write(json.dumps(question.to_json(), sort_keys=True))
                handle.write("\n")
        return len(self.questions)

    def content_hash(self) -> str:
        """Hash of the question ids, for the run manifest.

        Order independent, so a regenerated set with the same content hashes the
        same regardless of generation order.
        """
        digest = hashlib.sha256()
        for question_id in sorted(q.question_id for q in self.questions):
            digest.update(question_id.encode("utf-8"))
        return digest.hexdigest()


def summarize(questions: Sequence[Question]) -> str:
    counter = Counter(question.level for question in questions)
    parts = [f"L{level}={counter.get(level, 0)}" for level in (1, 2, 3, 4, 5)]
    return f"{len(questions)} questions ({', '.join(parts)})"
