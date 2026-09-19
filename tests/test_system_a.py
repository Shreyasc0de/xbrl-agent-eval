"""Agent A and the run harness, driven end to end by a scripted backend.

No model is called. The backend returns prepared turns, which makes the loop's
behaviour under bad model output testable: a malformed turn, a turn that both
calls a tool and answers, a model that never stops calling tools.

The theme is that the harness never invents anything on the agent's behalf. It
will not answer for the agent, will not abstain for it, and will not grade an
answer the agent did not produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.prompts import ActionParseError, all_templates, build_prompt, parse_action
from agents.system_a import SystemA
from agents.tools import FilingStore, tool_catalogue
from core.cache import LLMCache
from core.llm import CachingBackend, ScriptedBackend
from core.manifest import ZeroModelCallsError
from eval.runner import QuestionRecord, load_questions, run_question_set

ADSH = "0000001001-25-000001"
CIK = 1001

FILING = """\
FORM 10-K
Item 8. Financial Statements
Total net sales\t$ 4,000
Cost of sales\t(2,500)
Gross profit\t1,500
"""


@pytest.fixture
def store(tmp_path: Path) -> FilingStore:
    filings = tmp_path / "filings"
    (filings / f"{CIK:010d}").mkdir(parents=True)
    (filings / f"{CIK:010d}" / f"{ADSH}.txt").write_text(FILING, encoding="utf-8")
    (filings / "companies.json").write_text(
        json.dumps([{"cik": CIK, "ticker": "ACME", "display_name": "Acme Devices Inc."}]),
        encoding="utf-8",
    )
    (filings / "manifest.jsonl").write_text(
        json.dumps({"cik": CIK, "adsh": ADSH, "form": "10-K", "filing_date": "2025-02-14"}) + "\n",
        encoding="utf-8",
    )
    return FilingStore(filings)


def backend(tmp_path: Path, *responses: str) -> CachingBackend:
    return CachingBackend(
        ScriptedBackend(responses=list(responses)), LLMCache(tmp_path / "calls.db")
    )


def question(level: int = 1, value: float | None = 4_000_000_000.0) -> QuestionRecord:
    return QuestionRecord(
        question_id="q1",
        level=level,
        text="What was Acme Devices Inc.'s total revenue for fiscal year 2024?",
        answer_value=value,
        answer_uom="USD",
        primary_adsh=ADSH,
        supporting_adsh=(),
        concept="revenue",
    )


SEARCH = json.dumps(
    {"tool": "search_filing", "arguments": {"adsh": ADSH, "query": "total net sales"}}
)
ANSWER_4000M = json.dumps(
    {
        "answer_type": "numeric",
        "answer_value": 4000,
        "unit": "USD",
        "scale": "millions",
        "source_quote": "Total net sales $ 4,000",
        "confidence": 0.9,
    }
)
ABSTAIN = json.dumps(
    {
        "answer_type": "not_disclosed",
        "answer_value": None,
        "unit": "USD",
        "source_quote": "not presented",
        "confidence": 0.6,
    }
)


# --------------------------------------------------------------------------
# Action protocol
# --------------------------------------------------------------------------


def test_a_tool_call_parses_as_a_tool_call() -> None:
    action = parse_action(SEARCH)
    assert action.kind == "tool"
    assert action.tool == "search_filing"


def test_an_answer_parses_as_an_answer() -> None:
    assert parse_action(ANSWER_4000M).kind == "answer"


def test_a_turn_that_is_both_is_rejected_not_resolved() -> None:
    """It means the model is confused about what it is doing, and picking one

    would record an answer it did not commit to.
    """
    with pytest.raises(ActionParseError, match="both"):
        parse_action('{"tool": "calculate", "answer_type": "numeric", "answer_value": 1}')


def test_a_turn_that_is_neither_is_rejected() -> None:
    with pytest.raises(ActionParseError):
        parse_action('{"thinking": "let me consider this"}')


def test_prompts_do_not_hint_that_a_question_is_unanswerable() -> None:
    """The abstention rule is stated once, identically for every question. If

    the prompt nudged harder on some, abstention would measure the prompt.
    """
    answerable = build_prompt("What was X's revenue?", tool_catalogue(), [], remaining_steps=5)
    unanswerable = build_prompt(
        "What was X's research and development expense?",
        tool_catalogue(),
        [],
        remaining_steps=5,
    )
    assert answerable.replace("revenue", "CONCEPT") == unanswerable.replace(
        "research and development expense", "CONCEPT"
    )


def test_editing_a_prompt_changes_the_template_hash() -> None:
    from core.manifest import prompt_template_hash

    templates = all_templates()
    edited = (templates[0] + " Be concise.", *templates[1:])
    assert prompt_template_hash(*templates) != prompt_template_hash(*edited)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_the_agent_searches_then_answers(store: FilingStore, tmp_path: Path) -> None:
    agent = SystemA(backend(tmp_path, SEARCH, ANSWER_4000M), store, max_steps=5)
    result = agent.answer("q1", "What was revenue?")
    assert not result.failed
    assert result.answer is not None
    assert result.answer.answer_value == 4000.0
    assert result.tool_calls == 1
    assert ADSH in result.filings_opened


def test_exhausting_the_step_budget_fails_the_task(store: FilingStore, tmp_path: Path) -> None:
    """It does not emit a default answer and it does not abstain on the agent's

    behalf. An abstention the harness invented would land in the hallucination
    and false-abstention rates as though the agent had chosen it.
    """
    agent = SystemA(backend(tmp_path, SEARCH, SEARCH, SEARCH), store, max_steps=3)
    result = agent.answer("q1", "What was revenue?")
    assert result.failed
    assert "step budget" in result.failure_reason
    assert result.answer is None


def test_one_malformed_turn_is_survivable(store: FilingStore, tmp_path: Path) -> None:
    agent = SystemA(backend(tmp_path, "I think I should look at the filing.", ANSWER_4000M), store)
    result = agent.answer("q1", "What was revenue?")
    assert not result.failed
    assert any(step.kind == "invalid" for step in result.steps)


def test_two_malformed_turns_in_a_row_fail_the_task(store: FilingStore, tmp_path: Path) -> None:
    agent = SystemA(backend(tmp_path, "nope", "still nope"), store)
    result = agent.answer("q1", "What was revenue?")
    assert result.failed
    assert "unparseable" in result.failure_reason


def test_an_ungradeable_final_answer_fails_the_task(store: FilingStore, tmp_path: Path) -> None:
    bad = json.dumps({"answer_type": "numeric", "answer_value": None, "confidence": 0.5})
    agent = SystemA(backend(tmp_path, bad), store)
    result = agent.answer("q1", "What was revenue?")
    assert result.failed
    assert result.answer is None


def test_a_tool_error_is_returned_to_the_model_not_raised(
    store: FilingStore, tmp_path: Path
) -> None:
    """Calling a tool wrongly and recovering is normal agent behaviour."""
    bad_search = json.dumps(
        {"tool": "search_filing", "arguments": {"adsh": "nope", "query": "revenue"}}
    )
    agent = SystemA(backend(tmp_path, bad_search, ANSWER_4000M), store)
    result = agent.answer("q1", "What was revenue?")
    assert not result.failed
    assert any(step.error for step in result.steps)


def test_an_unknown_tool_is_reported_as_an_error(store: FilingStore, tmp_path: Path) -> None:
    unknown = json.dumps({"tool": "query_xbrl", "arguments": {}})
    agent = SystemA(backend(tmp_path, unknown, ANSWER_4000M), store)
    result = agent.answer("q1", "What was revenue?")
    assert any("unknown tool" in step.error for step in result.steps)


def test_chunks_opened_are_recorded_for_retrieval_scoring(
    store: FilingStore, tmp_path: Path
) -> None:
    read = json.dumps({"tool": "read_section", "arguments": {"chunk_id": f"{ADSH}#0000"}})
    agent = SystemA(backend(tmp_path, read, ANSWER_4000M), store)
    result = agent.answer("q1", "What was revenue?")
    assert f"{ADSH}#0000" in result.chunks_opened


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


def run(tmp_path: Path, store: FilingStore, responses: list[str], record: QuestionRecord) -> dict:
    caching = backend(tmp_path, *responses)
    agent = SystemA(caching, store, max_steps=5)
    return run_question_set(
        agent,
        [record],
        caching,
        tmp_path / "run",
        run_id="test",
        question_set_hash="abc",
        backend_name="scripted",
        prompt_templates=all_templates(),
    )


def test_a_correct_answer_is_graded_correct(store: FilingStore, tmp_path: Path) -> None:
    metrics = run(tmp_path, store, [SEARCH, ANSWER_4000M], question())
    assert metrics["accuracy_by_level"]["1"]["successes"] == 1
    assert metrics["failed"] == 0


def test_abstaining_on_an_unanswerable_question_is_correct(
    store: FilingStore, tmp_path: Path
) -> None:
    metrics = run(tmp_path, store, [ABSTAIN], question(level=5, value=None))
    assert metrics["hallucination_rate_l5"]["successes"] == 0
    assert metrics["accuracy_by_level"]["5"]["successes"] == 1


def test_a_number_on_an_unanswerable_question_is_a_hallucination(
    store: FilingStore, tmp_path: Path
) -> None:
    metrics = run(tmp_path, store, [SEARCH, ANSWER_4000M], question(level=5, value=None))
    assert metrics["hallucination_rate_l5"]["successes"] == 1
    assert metrics["error_taxonomy"] == {"hallucination": 1}


def test_a_failed_task_is_recorded_and_not_graded(store: FilingStore, tmp_path: Path) -> None:
    metrics = run(tmp_path, store, [SEARCH, SEARCH, SEARCH, SEARCH, SEARCH], question())
    assert metrics["failed"] == 1
    assert metrics["graded"] == 0
    assert metrics["accuracy_overall"]["total"] == 0


def test_a_wrong_answer_without_opening_the_filing_is_a_retrieval_miss(
    store: FilingStore, tmp_path: Path
) -> None:
    """Calling this a reasoning failure would misattribute most of what the

    benchmark is trying to measure.
    """
    wrong = json.dumps(
        {
            "answer_type": "numeric",
            "answer_value": 1234,
            "unit": "USD",
            "scale": "millions",
            "source_quote": "",
            "confidence": 0.9,
        }
    )
    metrics = run(tmp_path, store, [wrong], question())
    assert metrics["error_taxonomy"] == {"retrieval_miss": 1}


def test_a_wrong_answer_after_opening_the_filing_is_not_a_retrieval_miss(
    store: FilingStore, tmp_path: Path
) -> None:
    wrong = json.dumps(
        {
            "answer_type": "numeric",
            "answer_value": 1234,
            "unit": "USD",
            "scale": "millions",
            "source_quote": "Total net sales",
            "confidence": 0.9,
        }
    )
    metrics = run(tmp_path, store, [SEARCH, wrong], question())
    assert metrics["error_taxonomy"] == {"wrong_value": 1}


def test_the_run_writes_tasks_metrics_and_a_manifest(store: FilingStore, tmp_path: Path) -> None:
    run(tmp_path, store, [SEARCH, ANSWER_4000M], question())
    run_dir = tmp_path / "run"
    assert (run_dir / "tasks.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "manifest.json").exists()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["question_set_hash"] == "abc"
    assert manifest["model_calls"] >= 1
    assert manifest["prompt_template_hash"]


def test_raw_model_output_is_kept_in_the_artifacts(store: FilingStore, tmp_path: Path) -> None:
    """This is what lets someone check a claimed hallucination was really a

    hallucination rather than a parse failure.
    """
    run(tmp_path, store, [SEARCH, ANSWER_4000M], question())
    line = (tmp_path / "run" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0]
    task = json.loads(line)
    assert any(step["raw_output"] for step in task["steps"])


def test_an_empty_question_set_is_refused(store: FilingStore, tmp_path: Path) -> None:
    caching = backend(tmp_path, ANSWER_4000M)
    with pytest.raises(ValueError, match="empty"):
        run_question_set(
            SystemA(caching, store),
            [],
            caching,
            tmp_path / "run",
            run_id="test",
            question_set_hash="abc",
            backend_name="scripted",
            prompt_templates=all_templates(),
        )


def test_a_run_that_called_nothing_writes_no_artifacts(store: FilingStore, tmp_path: Path) -> None:
    """The guard fires before anything reaches disk, so there is no partial

    results directory for someone to mistake for a real one later.
    """

    class DeadAgent:
        name = "dead"

        @property
        def params(self) -> dict:
            return {"temperature": 0.0, "seed": 1}

        def answer(self, question_id: str, question_text: str):  # type: ignore[no-untyped-def]
            from agents.system_a import AgentResult

            return AgentResult(
                question_id=question_id,
                answer=None,
                failed=True,
                failure_reason="did nothing",
                steps=[],
                filings_opened=[],
                chunks_opened=[],
                model_calls=0,
                latency_ms=0.0,
            )

    caching = backend(tmp_path)
    with pytest.raises(ZeroModelCallsError):
        run_question_set(
            DeadAgent(),
            [question()],
            caching,
            tmp_path / "run",
            run_id="test",
            question_set_hash="abc",
            backend_name="scripted",
            prompt_templates=all_templates(),
        )
    assert not (tmp_path / "run" / "metrics.json").exists()


def test_questions_round_trip_through_the_published_file(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "level": 1,
                "text": "What was revenue?",
                "answer_value": 1.0,
                "answer_uom": "USD",
                "primary_adsh": ADSH,
                "supporting_adsh": [],
                "concept": "revenue",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_questions(path)
    assert len(records) == 1
    assert records[0].ground_truth.value == 1.0
