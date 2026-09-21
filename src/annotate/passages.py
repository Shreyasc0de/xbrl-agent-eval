"""Locate the passages that state a question's answer.

Why this exists
---------------

``correct_filing_rate`` asks whether the agent opened the right 10-K. With a
twenty-company roster that is a lookup, not retrieval, and the number sits near
1.0 whatever the agent does. The failure that actually matters looks like this:
the agent opens Apple's 10-K, searches "cash and cash equivalents", lands on the
Note 4 investment-category table, and reads the ``Cash`` row (27,199) instead of
the balance-sheet line (29,943). Right filing, wrong passage, wrong answer, and
a filing-level metric scores it a hit.

So the annotation records which chunks of a filing contain the answer as a
reader would see it printed, and the grader compares that set against the chunks
the agent actually opened.

What a hit means, precisely
---------------------------

A chunk is annotated when the truth value appears in it at some standard scale.
That is narrower than "this is the passage the answer comes from" and the
difference is not cosmetic:

* A figure printed in more than one place (a note and the statement it rolls up
  into) annotates both. The agent gets credit for either.
* A coincidental match is possible. Nothing here understands the table it is
  reading.

The metric this feeds is therefore named for what it measures: whether the agent
read a passage where the figure appears, not whether it understood it. Apple
above scores zero on it, which is the point. A question whose figure appears in
both the note and the statement scores one either way, which is the limit, and
it belongs in the README next to the number.

Short mantissas
---------------

A truth of 4,000,000,000 renders at billions as "4". Searching a filing for "4"
finds everything and means nothing. Rather than let such a question contribute a
free hit, it is refused outright with a recorded reason, in the same spirit as
the question generator's drop log. An annotation set therefore covers fewer
questions than the question set, and says by how much and why.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.tools import Chunk

__all__ = [
    "SCALES",
    "AnnotationSet",
    "PassageAnnotation",
    "SkipReason",
    "SkippedQuestion",
    "annotate_question",
    "rendered_forms",
]

# The scales an issuer actually prints statements in. Ordered smallest first so
# that a value matching several is reported in a stable order.
SCALES: dict[str, float] = {
    "units": 1.0,
    "thousands": 1e3,
    "millions": 1e6,
    "billions": 1e9,
}

# Below this many digits, a rendered figure is too common to carry information.
# Three is the smallest width at which a match is worth anything: "199" already
# appears freely in a 10-K, but requiring four would refuse a great many honest
# figures printed in millions.
MIN_MANTISSA_DIGITS: int = 3

# Integrality tolerance. Statements print whole units at their stated scale, so a
# value that does not divide cleanly was not printed at that scale.
_INTEGRAL_TOLERANCE: float = 1e-6


class SkipReason:
    """Why a question could not be annotated. No catch-all, by design."""

    MANTISSA_TOO_SHORT = "mantissa_too_short"
    NO_PRINTABLE_SCALE = "no_printable_scale"
    FILING_UNAVAILABLE = "filing_unavailable"
    NO_MATCHING_CHUNK = "no_matching_chunk"


@dataclass(frozen=True)
class PassageAnnotation:
    """The chunks of one filing that state one question's answer."""

    question_id: str
    adsh: str
    chunk_ids: tuple[str, ...]
    scales: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkippedQuestion:
    question_id: str
    adsh: str
    reason: str
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _group(digits: str) -> str:
    """Insert thousands separators, as a filing prints them."""
    return f"{int(digits):,}"


def rendered_forms(value: float, scale_name: str) -> tuple[str, ...]:
    """How ``value`` would be printed in a statement headed by ``scale_name``.

    Returns an empty tuple when the value does not divide cleanly into that
    scale, or when what is left is too short to search for. An empty tuple is
    the honest answer to "how would this be printed in millions" for a figure
    that is not printed in millions, and callers must treat it as a refusal
    rather than as zero matches.
    """
    divisor = SCALES[scale_name]
    scaled = value / divisor
    magnitude = abs(scaled)

    # Anything that rounds away to nothing, or does not land on a whole unit,
    # was not printed at this scale.
    if magnitude < 1:
        return ()
    rounded = round(magnitude)
    if abs(magnitude - rounded) > _INTEGRAL_TOLERANCE * max(1.0, magnitude):
        return ()

    digits = str(rounded)
    if len(digits) < MIN_MANTISSA_DIGITS:
        return ()

    plain = digits
    grouped = _group(digits)
    forms = {plain, grouped}
    if value < 0:
        # Filings print negatives in parentheses far more often than with a
        # minus sign, and both occur.
        forms |= {f"({plain})", f"({grouped})", f"-{plain}", f"-{grouped}"}
    return tuple(sorted(forms))


def _pattern(form: str) -> re.Pattern[str]:
    """Match a printed figure without matching a longer number containing it.

    ``27,199`` must not match inside ``127,199``, and ``27199`` must not match
    inside ``271990``. Digits, commas and the decimal point all count as part of
    a number for this purpose, so the guard looks for any of them on either side.
    """
    body = re.escape(form)
    return re.compile(rf"(?<![\d.,]){body}(?![\d.,])")


def annotate_question(
    question_id: str,
    adsh: str,
    value: float,
    chunks: Sequence[Chunk],
) -> PassageAnnotation | SkippedQuestion:
    """Find every chunk that prints ``value`` at some standard scale.

    Returns a ``SkippedQuestion`` rather than an empty annotation when there is
    nothing worth searching for or nothing was found. An empty annotation would
    be indistinguishable from "the agent had no chance", and would drag the
    metric down for a reason that has nothing to do with the agent.
    """
    searchable: dict[str, tuple[str, ...]] = {}
    for scale_name in SCALES:
        forms = rendered_forms(value, scale_name)
        if forms:
            searchable[scale_name] = forms

    if not searchable:
        # Distinguish "too short to search for" from "does not divide into any
        # scale", because they call for different fixes.
        any_scale_divides = any(
            abs(abs(value / divisor) - round(abs(value / divisor)))
            <= _INTEGRAL_TOLERANCE * max(1.0, abs(value / divisor))
            and abs(value / divisor) >= 1
            for divisor in SCALES.values()
        )
        reason = (
            SkipReason.MANTISSA_TOO_SHORT if any_scale_divides else SkipReason.NO_PRINTABLE_SCALE
        )
        return SkippedQuestion(
            question_id=question_id,
            adsh=adsh,
            reason=reason,
            detail=f"value {value!r} has no searchable rendering",
        )

    hits: dict[str, None] = {}
    matched_scales: dict[str, None] = {}
    for scale_name, forms in searchable.items():
        patterns = [_pattern(form) for form in forms]
        for chunk in chunks:
            if any(pattern.search(chunk.text) for pattern in patterns):
                hits[chunk.chunk_id] = None
                matched_scales[scale_name] = None

    if not hits:
        return SkippedQuestion(
            question_id=question_id,
            adsh=adsh,
            reason=SkipReason.NO_MATCHING_CHUNK,
            detail=f"value {value!r} does not appear in {len(chunks)} chunks at any scale",
        )

    return PassageAnnotation(
        question_id=question_id,
        adsh=adsh,
        chunk_ids=tuple(sorted(hits)),
        scales=tuple(sorted(matched_scales)),
    )


@dataclass
class AnnotationSet:
    """Annotations plus the questions that could not be annotated."""

    annotations: list[PassageAnnotation]
    skipped: list[SkippedQuestion]

    def by_question(self) -> dict[str, tuple[str, ...]]:
        return {entry.question_id: entry.chunk_ids for entry in self.annotations}

    def counts_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.skipped:
            counts[entry.reason] = counts.get(entry.reason, 0) + 1
        return dict(sorted(counts.items()))

    def write_jsonl(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in sorted(self.annotations, key=lambda a: a.question_id):
                handle.write(json.dumps(entry.to_json(), sort_keys=True))
                handle.write("\n")
        return len(self.annotations)

    def write_skipped(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in sorted(self.skipped, key=lambda s: s.question_id):
                handle.write(json.dumps(entry.to_json(), sort_keys=True))
                handle.write("\n")
        return len(self.skipped)


def read_annotations(path: Path) -> dict[str, tuple[str, ...]]:
    """Load ``question_id -> chunk ids``.

    A missing file is not an error here: a run made before annotations existed
    simply has no passage metric, which is different from having one of zero.
    """
    if not path.exists():
        return {}
    mapping: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[str(row["question_id"])] = tuple(str(c) for c in row["chunk_ids"])
    return mapping


def iter_answerable(questions: Iterable[dict[str, Any]]) -> list[tuple[str, str, float]]:
    """``(question_id, adsh, value)`` for every question with a numeric answer.

    Level 5 questions are excluded because the absence of a figure is their
    answer: there is no passage to find, and a passage that did state one would
    mean the question set is wrong.
    """
    rows: list[tuple[str, str, float]] = []
    for question in questions:
        if question.get("answer_value") is None:
            continue
        rows.append(
            (
                str(question["question_id"]),
                str(question["primary_adsh"]),
                float(question["answer_value"]),
            )
        )
    return rows
