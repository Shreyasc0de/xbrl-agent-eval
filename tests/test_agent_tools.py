"""Agent tools and answer parsing.

The filing store is built from a small synthetic cache written by the test, not
from real SEC data. What is tested is behaviour under the conditions that
actually bite: a company name that matches nothing, a truncated section, a
calculator handed something that is not arithmetic, and a model that wraps its
answer in prose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.answer import AnswerParseError, ParsedAnswer, parse_answer
from agents.tools import FilingStore, ToolError, calculate

ACME_CIK = 1001
ACME_ADSH = "0000001001-25-000001"
BETA_ADSH = "0000001002-25-000001"

ACME_TEXT = """\
FORM 10-K
Item 1. Business
Acme Devices designs and sells industrial sensors.
Item 7. Management's Discussion and Analysis
Revenue grew on stronger unit volumes.
Item 8. Financial Statements
Total net sales\t$ 4,000
Cost of sales\t(2,500)
Gross profit\t1,500
Research and development\t480
Item 9. Controls and Procedures
No changes in internal control over financial reporting.
"""

BETA_TEXT = """\
FORM 10-K
Item 8. Financial Statements
Total net sales\t$ 9,100
Gross profit\t3,000
"""


@pytest.fixture
def store(tmp_path: Path) -> FilingStore:
    filings = tmp_path / "filings"
    (filings / f"{ACME_CIK:010d}").mkdir(parents=True)
    (filings / f"{1002:010d}").mkdir(parents=True)
    (filings / f"{ACME_CIK:010d}" / f"{ACME_ADSH}.txt").write_text(ACME_TEXT, encoding="utf-8")
    (filings / f"{1002:010d}" / f"{BETA_ADSH}.txt").write_text(BETA_TEXT, encoding="utf-8")

    (filings / "companies.json").write_text(
        json.dumps(
            [
                {"cik": ACME_CIK, "ticker": "ACME", "display_name": "Acme Devices Inc."},
                {"cik": 1002, "ticker": "BETA", "display_name": "Beta Systems Inc."},
            ]
        ),
        encoding="utf-8",
    )
    with (filings / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for cik, adsh, date in (
            (ACME_CIK, ACME_ADSH, "2025-02-14"),
            (1002, BETA_ADSH, "2025-02-20"),
        ):
            handle.write(
                json.dumps({"cik": cik, "adsh": adsh, "form": "10-K", "filing_date": date})
            )
            handle.write("\n")
    return FilingStore(filings)


# --------------------------------------------------------------------------
# list_filings
# --------------------------------------------------------------------------


def test_listing_by_ticker_and_by_name_agree(store: FilingStore) -> None:
    assert [f.adsh for f in store.list_filings("ACME")] == [ACME_ADSH]
    assert [f.adsh for f in store.list_filings("Acme Devices Inc.")] == [ACME_ADSH]


def test_listing_is_newest_first(store: FilingStore) -> None:
    assert [f.adsh for f in store.list_filings()] == [BETA_ADSH, ACME_ADSH]


def test_an_unmatched_company_returns_nothing_not_everything(store: FilingStore) -> None:
    """Silently widening the search is how an agent confidently reads the wrong

    company's 10-K and reports a number that is real but about someone else.
    """
    assert store.list_filings("Umbrella Corporation") == []


def test_an_ambiguous_substring_returns_nothing(store: FilingStore) -> None:
    assert store.list_filings("Inc.") == []


# --------------------------------------------------------------------------
# search_filing and read_section
# --------------------------------------------------------------------------


def test_search_finds_the_passage_holding_the_figure(store: FilingStore) -> None:
    hits = store.search_filing(ACME_ADSH, "total net sales")
    assert hits
    assert any("4,000" in hit.snippet for hit in hits)


def test_search_returns_chunk_ids_so_retrieval_can_be_scored(
    store: FilingStore,
) -> None:
    """Without ids, a wrong answer cannot be attributed to retrieval or to

    reasoning, and that distinction is most of the interesting analysis.
    """
    hits = store.search_filing(ACME_ADSH, "gross profit")
    assert all(hit.chunk_id.startswith(f"{ACME_ADSH}#") for hit in hits)


def test_search_on_an_unknown_filing_is_an_error(store: FilingStore) -> None:
    with pytest.raises(ToolError, match="no cached filing"):
        store.search_filing("0000000000-00-000000", "revenue")


def test_an_empty_query_is_an_error(store: FilingStore) -> None:
    with pytest.raises(ToolError):
        store.search_filing(ACME_ADSH, "   ")


def test_a_query_matching_nothing_returns_no_hits(store: FilingStore) -> None:
    assert store.search_filing(ACME_ADSH, "zzzznonexistent") == []


def test_read_section_returns_the_passage(store: FilingStore) -> None:
    hits = store.search_filing(ACME_ADSH, "cost of sales")
    text = store.read_section(hits[0].chunk_id)
    assert "Cost of sales" in text


def test_reading_an_unknown_section_is_an_error(store: FilingStore) -> None:
    with pytest.raises(ToolError):
        store.read_section(f"{ACME_ADSH}#9999")


def test_a_malformed_chunk_id_is_an_error(store: FilingStore) -> None:
    with pytest.raises(ToolError, match="malformed"):
        store.read_section("not-a-chunk-id")


def test_a_statement_line_keeps_its_label_and_figure(store: FilingStore) -> None:
    hits = store.search_filing(ACME_ADSH, "total net sales")
    line = next(line for line in hits[0].snippet.split("\n") if "Total net sales" in line)
    assert "4,000" in line


def test_chunks_carry_the_heading_they_fall_under(store: FilingStore) -> None:
    chunks = store.chunks(ACME_ADSH)
    assert any("Item" in chunk.heading for chunk in chunks if chunk.heading)


# --------------------------------------------------------------------------
# calculate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1500 / 4000 * 100", 37.5),
        ("(4000 - 2500) / 4000 * 100", 37.5),
        ("-250 + 100", -150.0),
        ("2 ** 10", 1024.0),
    ],
)
def test_calculate_evaluates_arithmetic(expression: str, expected: float) -> None:
    assert calculate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "revenue * 2",
        "[1, 2, 3]",
        "lambda: 1",
    ],
)
def test_calculate_refuses_anything_that_is_not_arithmetic(expression: str) -> None:
    """The agent under test is untrusted input by construction, and a calculator

    is the most obvious place to try something that is not arithmetic.
    """
    with pytest.raises(ToolError):
        calculate(expression)


def test_division_by_zero_is_an_error_not_infinity() -> None:
    with pytest.raises(ToolError, match="division by zero"):
        calculate("1 / 0")


def test_an_empty_expression_is_an_error() -> None:
    with pytest.raises(ToolError):
        calculate("  ")


# --------------------------------------------------------------------------
# Answer parsing
# --------------------------------------------------------------------------


def test_a_clean_json_answer_parses() -> None:
    answer = parse_answer(
        '{"answer_type": "numeric", "answer_value": 4000, "unit": "USD", '
        '"scale": "millions", "source_quote": "Total net sales 4,000", "confidence": 0.9}'
    )
    assert answer.answer_value == 4000.0
    assert answer.scale == "millions"


def test_an_answer_wrapped_in_prose_and_fences_still_parses() -> None:
    """A model that explains itself before answering is doing something normal."""
    output = (
        "Let me work through this.\n\n```json\n"
        '{"answer_type": "numeric", "answer_value": 1500, "unit": "USD", '
        '"scale": "millions", "source_quote": "Gross profit 1,500", "confidence": 0.8}'
        "\n```\n"
    )
    assert parse_answer(output).answer_value == 1500.0


def test_the_last_object_wins_when_a_model_shows_an_example_first() -> None:
    output = (
        'The format is {"answer_type": "numeric", "answer_value": 0} and my answer is '
        '{"answer_type": "numeric", "answer_value": 42, "unit": "USD", '
        '"scale": "units", "source_quote": "x", "confidence": 0.5}'
    )
    assert parse_answer(output).answer_value == 42.0


def test_a_free_text_answer_is_rejected_not_coerced() -> None:
    """A coerced answer is a fabricated result: the number was chosen by a

    regular expression, not by the agent.
    """
    with pytest.raises(AnswerParseError):
        parse_answer("Total net sales were about 4 billion dollars.")


def test_a_numeric_answer_with_no_value_is_rejected() -> None:
    with pytest.raises(AnswerParseError):
        parse_answer('{"answer_type": "numeric", "answer_value": null, "confidence": 0.5}')


def test_an_abstention_carrying_a_value_is_rejected() -> None:
    with pytest.raises(AnswerParseError):
        parse_answer('{"answer_type": "not_disclosed", "answer_value": 1, "confidence": 0.5}')


def test_an_abstention_parses() -> None:
    answer = parse_answer(
        '{"answer_type": "not_disclosed", "answer_value": null, "unit": "USD", '
        '"source_quote": "not separately presented", "confidence": 0.7}'
    )
    assert answer.answer_type == "not_disclosed"
    assert answer.answer_value is None


def test_an_unrecognized_scale_is_rejected_rather_than_defaulted() -> None:
    """The difference between units and millions is the difference between right

    and wrong, so it is never guessed.
    """
    with pytest.raises(AnswerParseError, match="scale"):
        parse_answer(
            '{"answer_type": "numeric", "answer_value": 4000, "unit": "USD", '
            '"scale": "gazillions", "confidence": 0.5}'
        )


def test_parentheses_in_a_numeric_string_read_as_negative() -> None:
    """Filings present negatives in parentheses and models copy that convention."""
    answer = parse_answer(
        '{"answer_type": "numeric", "answer_value": "(2,500)", "unit": "USD", '
        '"scale": "millions", "confidence": 0.6}'
    )
    assert answer.answer_value == -2500.0


def test_a_missing_answer_type_is_rejected() -> None:
    with pytest.raises(AnswerParseError, match="answer_type"):
        parse_answer('{"answer_value": 1, "confidence": 0.5}')


def test_confidence_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(AnswerParseError):
        parse_answer(
            '{"answer_type": "numeric", "answer_value": 1, "unit": "USD", '
            '"scale": "units", "confidence": 5}'
        )


def test_empty_output_is_rejected() -> None:
    with pytest.raises(AnswerParseError):
        parse_answer("")


def test_parsed_answers_are_immutable() -> None:
    answer = ParsedAnswer("numeric", 1.0, "USD", "units", "q", 0.5)
    with pytest.raises(AttributeError):
        answer.answer_value = 2.0  # type: ignore[misc]
