"""The passage annotator.

These tests exist because the annotator is the one module that sees both the
answer key and the filing text. If it is wrong, the retrieval numbers are wrong
in a way that looks plausible on the page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.tools import Chunk
from annotate.passages import (
    MIN_MANTISSA_DIGITS,
    SCALES,
    AnnotationSet,
    PassageAnnotation,
    SkippedQuestion,
    SkipReason,
    annotate_question,
    iter_answerable,
    read_annotations,
    rendered_forms,
)


def chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        adsh="0000320193-24-000123",
        heading="Item 8. Financial Statements",
        text=text,
        start_line=0,
        end_line=40,
    )


# --------------------------------------------------------------- rendering


def test_a_figure_printed_in_millions_is_found_from_its_base_units_truth() -> None:
    """The case the whole module exists for.

    Ground truth is 243686000000. The filing prints 243,686 under a heading
    saying the statement is in millions. Nothing links the two except this.
    """
    forms = rendered_forms(243_686_000_000.0, "millions")
    assert "243,686" in forms
    assert "243686" in forms


def test_a_value_that_does_not_divide_cleanly_is_not_printable_at_that_scale() -> None:
    assert rendered_forms(243_686_500_000.0, "millions") == ()


def test_a_value_below_one_at_a_scale_is_not_printable_there() -> None:
    assert rendered_forms(1_000.0, "millions") == ()


def test_a_short_mantissa_is_refused_rather_than_searched_for() -> None:
    """4,000,000,000 renders at billions as "4", which matches everything."""
    assert rendered_forms(4_000_000_000.0, "billions") == ()
    assert rendered_forms(4_000_000_000.0, "millions") == ("4,000", "4000")


def test_the_minimum_width_is_the_one_the_module_documents() -> None:
    """Guard the constant, so loosening it has to be deliberate."""
    assert MIN_MANTISSA_DIGITS == 3
    assert rendered_forms(99_000_000.0, "millions") == ()
    assert "100" in rendered_forms(100_000_000.0, "millions")


def test_a_negative_value_renders_in_parentheses_as_well_as_with_a_sign() -> None:
    forms = rendered_forms(-1_234_000_000.0, "millions")
    assert "(1,234)" in forms
    assert "-1,234" in forms
    assert "1,234" in forms


# ---------------------------------------------------------------- matching
#
# Annotation is keyed on the facts the answer was computed from, not on the
# answer. A level 2 ratio and a level 3 change are never printed in a filing;
# the figures they are derived from are.


def chunks_for(*pairs: tuple[str, list[Chunk]]) -> dict[str, list[Chunk]]:
    return dict(pairs)


def test_a_matching_fact_is_annotated_with_the_scale_that_matched() -> None:
    result = annotate_question(
        "q1",
        "a",
        [("a", 243_686_000_000.0)],
        chunks_for(
            (
                "a",
                [
                    chunk("a#0001", "Total liabilities\t$\t243,686"),
                    chunk("a#0002", "Nothing relevant here."),
                ],
            )
        ),
    )
    assert isinstance(result, PassageAnnotation)
    assert result.fact_chunk_ids == (("a#0001",),)
    assert result.chunk_ids == ("a#0001",)
    assert result.scales == ("millions",)


def test_a_figure_printed_twice_annotates_both_chunks() -> None:
    """Deliberate. The agent gets credit for reading either one."""
    result = annotate_question(
        "q1",
        "a",
        [("a", 243_686_000_000.0)],
        chunks_for(
            (
                "a",
                [
                    chunk("a#0001", "Total liabilities 243,686"),
                    chunk("a#0002", "as discussed above, 243,686 of liabilities"),
                ],
            )
        ),
    )
    assert isinstance(result, PassageAnnotation)
    assert result.fact_chunk_ids == (("a#0001", "a#0002"),)


def test_a_longer_number_containing_the_figure_is_not_a_match() -> None:
    """Without the boundary guard, 27,199 matches inside 127,199."""
    result = annotate_question(
        "q1",
        "a",
        [("a", 27_199_000_000.0)],
        chunks_for(("a", [chunk("a#0001", "Something 127,199 and 271,990 and 27,199.4 here")])),
    )
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.NO_MATCHING_CHUNK


def test_a_figure_never_printed_is_skipped_rather_than_annotated_empty() -> None:
    """An empty annotation would score the agent zero on an impossible task."""
    result = annotate_question(
        "q1",
        "a",
        [("a", 243_686_000_000.0)],
        chunks_for(("a", [chunk("a#0001", "Total liabilities 111,111")])),
    )
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.NO_MATCHING_CHUNK


def test_a_value_with_no_searchable_rendering_reports_why() -> None:
    result = annotate_question(
        "q1", "a", [("a", 4.0)], chunks_for(("a", [chunk("a#0001", "4 4 4 4")]))
    )
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.MANTISSA_TOO_SHORT


def test_a_derived_answer_is_annotated_through_its_inputs() -> None:
    """The bug this rewrite fixes.

    A gross margin of 0.4623 is printed nowhere. Searching for it skipped 473
    of 995 questions under a reason that sounded like missing data. Revenue and
    cost of sales are printed, and the question already cites both.
    """
    revenue = chunk("a#0004", "Total net sales\t$\t391,035")
    cost = chunk("a#0005", "Total cost of sales\t$\t210,352")
    result = annotate_question(
        "q1",
        "a",
        [("a", 391_035_000_000.0), ("a", 210_352_000_000.0)],
        chunks_for(("a", [revenue, cost])),
    )
    assert isinstance(result, PassageAnnotation)
    assert result.fact_chunk_ids == (("a#0004",), ("a#0005",))


def test_inputs_are_located_in_their_own_filings() -> None:
    """A year-over-year question reads this year's 10-K and last year's."""
    result = annotate_question(
        "q1",
        "current",
        [("current", 391_035_000_000.0), ("prior", 383_285_000_000.0)],
        chunks_for(
            ("current", [chunk("current#0004", "Total net sales 391,035")]),
            ("prior", [chunk("prior#0004", "Total net sales 383,285")]),
        ),
    )
    assert isinstance(result, PassageAnnotation)
    assert result.fact_chunk_ids == (("current#0004",), ("prior#0004",))


def test_a_question_with_one_input_missing_is_skipped_entirely() -> None:
    """All or nothing.

    Counting it would let an agent pass by reading half of what the answer
    needs. Failing it would penalize the agent for a passage nobody could find.
    """
    result = annotate_question(
        "q1",
        "a",
        [("a", 391_035_000_000.0), ("a", 210_352_000_000.0)],
        chunks_for(("a", [chunk("a#0004", "Total net sales 391,035")])),
    )
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.NO_MATCHING_CHUNK


def test_a_question_citing_no_facts_is_skipped_with_its_own_reason() -> None:
    result = annotate_question("q1", "a", [], chunks_for(("a", [chunk("a#0001", "x")])))
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.NO_FACTS


def test_an_uncached_filing_is_reported_as_unavailable() -> None:
    result = annotate_question("q1", "a", [("missing", 391_035_000_000.0)], {})
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.FILING_UNAVAILABLE


def test_the_apple_case_scores_a_miss() -> None:
    """The failure that motivated the metric, reproduced end to end.

    Truth is 29,943 million. The agent read the Note 4 chunk, which prints the
    Cash row of 27,199 and not the balance-sheet line. Filing-level retrieval
    called this a hit. Passage-level must not.
    """
    note_four = chunk("a#0025", "Cash\t$\t27,199\t$\t—\t$\t—\t$\t27,199")
    balance_sheet = chunk("a#0012", "Cash and cash equivalents\t$\t29,943")
    result = annotate_question(
        "q1", "a", [("a", 29_943_000_000.0)], chunks_for(("a", [note_four, balance_sheet]))
    )
    assert isinstance(result, PassageAnnotation)
    assert result.fact_chunk_ids == (("a#0012",),)
    assert "a#0025" not in result.chunk_ids


# ----------------------------------------------------------------- the set


def test_unanswerable_questions_are_not_annotated() -> None:
    """Level 5 has no figure. A passage stating one would mean the set is wrong."""
    rows = iter_answerable(
        [
            {
                "question_id": "q1",
                "primary_adsh": "a",
                "answer_value": 100.0,
                "facts": [{"adsh": "a", "value": 100.0}],
            },
            {"question_id": "q5", "primary_adsh": "a", "answer_value": None, "facts": []},
        ]
    )
    assert rows == [("q1", "a", [("a", 100.0)])]


def test_facts_keep_their_own_accession_numbers() -> None:
    rows = iter_answerable(
        [
            {
                "question_id": "q1",
                "primary_adsh": "current",
                "answer_value": 0.02,
                "facts": [
                    {"adsh": "current", "value": 391_035_000_000.0},
                    {"adsh": "prior", "value": 383_285_000_000.0},
                ],
            }
        ]
    )
    assert rows == [
        ("q1", "current", [("current", 391_035_000_000.0), ("prior", 383_285_000_000.0)])
    ]


def test_the_set_reports_why_questions_were_skipped() -> None:
    annotation_set = AnnotationSet(
        annotations=[PassageAnnotation("q1", "a", (("a#0001",),), ("millions",))],
        skipped=[
            SkippedQuestion("q2", "a", SkipReason.NO_MATCHING_CHUNK),
            SkippedQuestion("q3", "a", SkipReason.NO_MATCHING_CHUNK),
            SkippedQuestion("q4", "a", SkipReason.MANTISSA_TOO_SHORT),
        ],
    )
    assert annotation_set.counts_by_reason() == {
        "mantissa_too_short": 1,
        "no_matching_chunk": 2,
    }
    assert annotation_set.by_question() == {"q1": (("a#0001",),)}


def test_written_annotations_round_trip(tmp_path: Path) -> None:
    annotation_set = AnnotationSet(
        annotations=[
            PassageAnnotation("q1", "a", (("a#0001", "a#0002"), ("a#0005",)), ("millions",))
        ],
        skipped=[],
    )
    path = tmp_path / "passages.jsonl"
    annotation_set.write_jsonl(path)
    assert read_annotations(path) == {"q1": (("a#0001", "a#0002"), ("a#0005",))}


def test_a_missing_annotation_file_is_no_annotations_not_an_error() -> None:
    assert read_annotations(Path("/nonexistent/passages.jsonl")) == {}


@pytest.mark.parametrize("scale", ["units", "thousands", "millions", "billions"])
def test_every_scale_round_trips_a_representative_figure(scale: str) -> None:
    value = 123_456.0 * SCALES[scale]
    assert "123,456" in rendered_forms(value, scale)
