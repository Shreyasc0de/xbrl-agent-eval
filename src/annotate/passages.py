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

What gets searched for
----------------------

The fact values an answer was computed from, not the answer itself. The first
version searched for the answer value and skipped 473 of 995 questions with
"no printable scale", which sounded like a data problem and was not. A level 2
answer is a ratio and a level 3 answer is a year-over-year change. Neither is
printed anywhere in a filing. What is printed is the revenue and the cost of
sales they are derived from, and the question already records exactly those rows
as its provenance.

Annotating on the facts fixes level 1 too, where the single fact value is the
answer, so one rule covers every level. It also follows each fact into its own
filing: a year-over-year question draws one input from this year's 10-K and one
from last year's, so the annotation captures whether the agent opened both.

A question is annotated only when every one of its facts was located, and it
counts as a hit only when the agent read a passage for every one of them. An
agent that read revenue, never read cost of sales, and produced a plausible
margin did not retrieve what it needed. Scoring that as a retrieval success
would push the failure into the reasoning bucket, which is the exact
misattribution this metric exists to prevent.

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
from collections.abc import Iterable, Mapping, Sequence
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
    "FactLocation",
    "annotate_question",
    "locate_fact",
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
    NO_FACTS = "no_facts"


@dataclass(frozen=True)
class PassageAnnotation:
    """Where each of a question's input facts is printed.

    ``fact_chunk_ids`` holds one group per fact, and a hit requires the agent to
    have opened a chunk from every group. ``chunk_ids`` is their union, kept for
    reading and for anything that only needs "was this passage relevant at all".
    """

    question_id: str
    adsh: str
    fact_chunk_ids: tuple[tuple[str, ...], ...]
    scales: tuple[str, ...]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for group in self.fact_chunk_ids:
            for chunk_id in group:
                seen[chunk_id] = None
        return tuple(sorted(seen))

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fact_chunk_ids"] = [list(group) for group in self.fact_chunk_ids]
        payload["chunk_ids"] = list(self.chunk_ids)
        return payload


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


@dataclass(frozen=True)
class FactLocation:
    """One input fact and where it is printed."""

    adsh: str
    value: float
    chunk_ids: tuple[str, ...]
    scale: str


def locate_fact(
    value: float,
    chunks: Sequence[Chunk],
) -> tuple[tuple[str, ...], str] | str:
    """Chunks printing ``value`` at some standard scale, or a skip reason."""
    searchable: dict[str, tuple[str, ...]] = {}
    for scale_name in SCALES:
        forms = rendered_forms(value, scale_name)
        if forms:
            searchable[scale_name] = forms

    if not searchable:
        divides = any(
            abs(value / divisor) >= 1
            and abs(abs(value / divisor) - round(abs(value / divisor)))
            <= _INTEGRAL_TOLERANCE * max(1.0, abs(value / divisor))
            for divisor in SCALES.values()
        )
        return SkipReason.MANTISSA_TOO_SHORT if divides else SkipReason.NO_PRINTABLE_SCALE

    hits: dict[str, None] = {}
    matched = ""
    for scale_name, forms in searchable.items():
        patterns = [_pattern(form) for form in forms]
        for chunk in chunks:
            if any(pattern.search(chunk.text) for pattern in patterns):
                hits[chunk.chunk_id] = None
                matched = matched or scale_name

    if not hits:
        return SkipReason.NO_MATCHING_CHUNK
    return tuple(sorted(hits)), matched


def annotate_question(
    question_id: str,
    adsh: str,
    facts: Sequence[tuple[str, float]],
    chunks_by_adsh: Mapping[str, Sequence[Chunk]],
) -> PassageAnnotation | SkippedQuestion:
    """Locate every fact the answer was computed from.

    All or nothing. A question with one input located and another not cannot be
    scored fairly: counting it would let an agent pass by reading half of what
    the answer needs, and failing it would penalize the agent for a passage
    nobody could find. It is skipped, and the denominator says so.
    """
    if not facts:
        return SkippedQuestion(
            question_id=question_id,
            adsh=adsh,
            reason=SkipReason.NO_FACTS,
            detail="question cites no facts, so there is nothing to locate",
        )

    groups: list[tuple[str, ...]] = []
    scales: dict[str, None] = {}
    for fact_adsh, value in facts:
        chunks = chunks_by_adsh.get(fact_adsh)
        if not chunks:
            return SkippedQuestion(
                question_id=question_id,
                adsh=adsh,
                reason=SkipReason.FILING_UNAVAILABLE,
                detail=f"no cached text for {fact_adsh}, which states one of the inputs",
            )
        located = locate_fact(value, chunks)
        if isinstance(located, str):
            return SkippedQuestion(
                question_id=question_id,
                adsh=adsh,
                reason=located,
                detail=f"input {value!r} in {fact_adsh}",
            )
        chunk_ids, scale = located
        groups.append(chunk_ids)
        scales[scale] = None

    return PassageAnnotation(
        question_id=question_id,
        adsh=adsh,
        fact_chunk_ids=tuple(groups),
        scales=tuple(sorted(scales)),
    )


@dataclass
class AnnotationSet:
    """Annotations plus the questions that could not be annotated."""

    annotations: list[PassageAnnotation]
    skipped: list[SkippedQuestion]

    def by_question(self) -> dict[str, tuple[tuple[str, ...], ...]]:
        return {entry.question_id: entry.fact_chunk_ids for entry in self.annotations}

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


def read_annotations(path: Path) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Load ``question_id -> one group of chunk ids per input fact``.

    A missing file is not an error here: a run made before annotations existed
    simply has no passage metric, which is different from having one of zero.
    """
    if not path.exists():
        return {}
    mapping: dict[str, tuple[tuple[str, ...], ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            mapping[str(row["question_id"])] = tuple(
                tuple(str(c) for c in group) for group in row["fact_chunk_ids"]
            )
    return mapping


def iter_answerable(
    questions: Iterable[dict[str, Any]],
) -> list[tuple[str, str, list[tuple[str, float]]]]:
    """``(question_id, primary_adsh, [(fact adsh, value), ...])`` per question.

    The facts, not the answer. Each carries its own accession number, because a
    year-over-year question takes one input from this year's filing and one from
    last year's.

    Level 5 questions are excluded because the absence of a figure is their
    answer: there is no passage to find, and a passage that did state one would
    mean the question set is wrong.
    """
    rows: list[tuple[str, str, list[tuple[str, float]]]] = []
    for question in questions:
        if question.get("answer_value") is None:
            continue
        facts = [
            (str(fact["adsh"]), float(fact["value"]))
            for fact in question.get("facts", ())
            if fact.get("value") is not None
        ]
        rows.append((str(question["question_id"]), str(question["primary_adsh"]), facts))
    return rows
