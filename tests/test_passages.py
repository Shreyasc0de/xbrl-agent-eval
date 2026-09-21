"""The passage annotator.

These tests exist because the annotator is the one module that sees both the
answer key and the filing text. If it is wrong, the retrieval numbers are wrong
in a way that looks plausible on the page.
"""

from __future__ import annotations

import pytest

from agents.tools import Chunk
from annotate.passages import (
    MIN_MANTISSA_DIGITS,
    AnnotationSet,
    PassageAnnotation,
    SkippedQuestion,
    SkipReason,
    annotate_question,
    iter_answerable,
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


def test_a_matching_chunk_is_annotated_with_the_scale_that_matched() -> None:
    chunks = [
        chunk("a#0001", "Total liabilities\t$\t243,686"),
        chunk("a#0002", "Nothing relevant here."),
    ]
    result = annotate_question("q1", "a", 243_686_000_000.0, chunks)
    assert isinstance(result, PassageAnnotation)
    assert result.chunk_ids == ("a#0001",)
    assert result.scales == ("millions",)


def test_a_figure_printed_twice_annotates_both_chunks() -> None:
    """Deliberate. The agent gets credit for reading either one."""
    chunks = [
        chunk("a#0001", "Total liabilities 243,686"),
        chunk("a#0002", "as discussed above, 243,686 of liabilities"),
    ]
    result = annotate_question("q1", "a", 243_686_000_000.0, chunks)
    assert isinstance(result, PassageAnnotation)
    assert result.chunk_ids == ("a#0001", "a#0002")


def test_a_longer_number_containing_the_figure_is_not_a_match() -> None:
    """Without the boundary guard, 27,199 matches inside 127,199."""
    chunks = [chunk("a#0001", "Something 127,199 and 271,990 and 27,199.4 here")]
    result = annotate_question("q1", "a", 27_199_000_000.0, chunks)
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.NO_MATCHING_CHUNK


def test_a_figure_never_printed_is_skipped_rather_than_annotated_empty() -> None:
    """An empty annotation would score the agent zero on an impossible task."""
    chunks = [chunk("a#0001", "Total liabilities 111,111")]
    result = annotate_question("q1", "a", 243_686_000_000.0, chunks)
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.NO_MATCHING_CHUNK


def test_a_value_with_no_searchable_rendering_reports_why() -> None:
    result = annotate_question("q1", "a", 4.0, [chunk("a#0001", "4 4 4 4")])
    assert isinstance(result, SkippedQuestion)
    assert result.reason == SkipReason.MANTISSA_TOO_SHORT


def test_the_apple_case_scores_a_miss() -> None:
    """The failure that motivated the metric, reproduced end to end.

    Truth is 29,943 million. The agent read the Note 4 chunk, which prints the
    Cash row of 27,199 and not the balance-sheet line. Filing-level retrieval
    called this a hit. Passage-level must not.
    """
    note_four = chunk("a#0025", "Cash\t$\t27,199\t$\t—\t$\t—\t$\t27,199")
    balance_sheet = chunk("a#0012", "Cash and cash equivalents\t$\t29,943")
    result = annotate_question("q1", "a", 29_943_000_000.0, [note_four, balance_sheet])
    assert isinstance(result, PassageAnnotation)
    assert result.chunk_ids == ("a#0012",)
    assert "a#0025" not in result.chunk_ids


# ----------------------------------------------------------------- the set


def test_unanswerable_questions_are_not_annotated() -> None:
    """Level 5 has no figure. A passage stating one would mean the set is wrong."""
    rows = iter_answerable(
        [
            {"question_id": "q1", "primary_adsh": "a", "answer_value": 100.0},
            {"question_id": "q5", "primary_adsh": "a", "answer_value": None},
        ]
    )
    assert rows == [("q1", "a", 100.0)]


def test_the_set_reports_why_questions_were_skipped() -> None:
    annotation_set = AnnotationSet(
        annotations=[PassageAnnotation("q1", "a", ("a#0001",), ("millions",))],
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
    assert annotation_set.by_question() == {"q1": ("a#0001",)}


def test_written_annotations_round_trip(tmp_path: object) -> None:
    from pathlib import Path

    from annotate.passages import read_annotations

    assert isinstance(tmp_path, Path)
    annotation_set = AnnotationSet(
        annotations=[PassageAnnotation("q1", "a", ("a#0001", "a#0002"), ("millions",))],
        skipped=[],
    )
    path = tmp_path / "passages.jsonl"
    annotation_set.write_jsonl(path)
    assert read_annotations(path) == {"q1": ("a#0001", "a#0002")}


def test_a_missing_annotation_file_is_no_annotations_not_an_error() -> None:
    from pathlib import Path

    from annotate.passages import read_annotations

    assert read_annotations(Path("/nonexistent/passages.jsonl")) == {}


@pytest.mark.parametrize("scale", ["units", "thousands", "millions", "billions"])
def test_every_scale_round_trips_a_representative_figure(scale: str) -> None:
    from annotate.passages import SCALES

    value = 123_456.0 * SCALES[scale]
    assert "123,456" in rendered_forms(value, scale)
