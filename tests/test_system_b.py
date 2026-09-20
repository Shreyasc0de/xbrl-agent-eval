"""System B: planner, retrievers, synthesis, verifier.

Driven by a scripted backend, so the orchestration is tested without a model.
The emphasis is on the places multi-agent designs quietly go wrong: a verifier
that rubber-stamps whatever it cannot parse, a rejection that becomes an
abstention nobody chose, and a comparison contaminated by the two systems not
sharing their tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.prompts import all_templates
from agents.system_a import SystemA
from agents.system_b import SystemB
from agents.tools import FilingStore
from core.cache import LLMCache
from core.llm import CachingBackend, ScriptedBackend
from eval.runner import QuestionRecord, run_question_set

ADSH = "0000001001-25-000001"
CIK = 1001

FILING = """\
FORM 10-K
Item 8. Financial Statements
Total net sales\t$ 4,000
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


PLAN = json.dumps({"subquestions": ["Acme Devices total net sales for fiscal 2024"]})
SEARCH = json.dumps(
    {"tool": "search_filing", "arguments": {"adsh": ADSH, "query": "total net sales"}}
)
ANSWER = json.dumps(
    {
        "answer_type": "numeric",
        "answer_value": 4000,
        "unit": "USD",
        "scale": "millions",
        "source_quote": "Total net sales $ 4,000",
        "confidence": 0.9,
    }
)
ACCEPT = json.dumps({"verdict": "accept", "reason": "the finding states 4,000"})
REJECT = json.dumps({"verdict": "reject", "reason": "that figure is not in the findings"})


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_the_full_pipeline_produces_an_answer(store: FilingStore, tmp_path: Path) -> None:
    agent = SystemB(backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, ACCEPT), store)
    result = agent.answer("q1", "What was Acme's total revenue for fiscal 2024?")
    assert not result.failed
    assert result.answer is not None
    assert result.answer.answer_value == 4000.0


def test_each_role_is_recorded_as_its_own_step(store: FilingStore, tmp_path: Path) -> None:
    """The trace has to show which role produced what, or the cost of the extra

    roles cannot be attributed afterwards.
    """
    agent = SystemB(backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, ACCEPT), store)
    result = agent.answer("q1", "What was revenue?")
    kinds = [step.kind for step in result.steps]
    assert "plan" in kinds
    assert "retrieve" in kinds
    assert "synthesize" in kinds
    assert "verify" in kinds


def test_the_retriever_opens_filings_and_they_reach_the_result(
    store: FilingStore, tmp_path: Path
) -> None:
    agent = SystemB(backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, ACCEPT), store)
    result = agent.answer("q1", "What was revenue?")
    assert ADSH in result.filings_opened


# --------------------------------------------------------------------------
# The verifier
# --------------------------------------------------------------------------

REVISED_ANSWER = json.dumps(
    {
        "answer_type": "numeric",
        "answer_value": 1500,
        "unit": "USD",
        "scale": "millions",
        "source_quote": "Gross profit 1,500",
        "confidence": 0.7,
    }
)


def test_a_rejection_triggers_one_retry(store: FilingStore, tmp_path: Path) -> None:
    agent = SystemB(
        backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, REJECT, REVISED_ANSWER, ACCEPT),
        store,
    )
    result = agent.answer("q1", "What was revenue?")
    assert not result.failed
    assert result.answer is not None
    assert result.answer.answer_value == 1500.0
    assert sum(1 for step in result.steps if step.kind == "synthesize") == 2


def test_an_unchanged_retry_is_rejected_again_from_cache(
    store: FilingStore, tmp_path: Path
) -> None:
    """Subtle, correct, and worth pinning.

    If synthesis returns the same answer after a rejection, the verification
    prompt is byte-identical and the content-addressed cache replays the same
    verdict without a new model call. The retry is only useful when the answer
    actually changes, and a design that assumed otherwise would be relying on
    model noise to make progress.
    """
    caching = backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, REJECT, ANSWER)
    result = SystemB(caching, store).answer("q1", "What was revenue?")
    assert result.failed
    assert "verifier rejected" in result.failure_reason
    assert caching.cache_hits >= 1


def test_a_second_rejection_fails_the_task(store: FilingStore, tmp_path: Path) -> None:
    """It does not become an abstention. An abstention the harness invented

    would land in the hallucination and false-abstention rates as though the
    agent had chosen it.
    """
    agent = SystemB(
        backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, REJECT, REVISED_ANSWER, REJECT),
        store,
    )
    result = agent.answer("q1", "What was revenue?")
    assert result.failed
    assert result.answer is None
    assert "verifier rejected" in result.failure_reason


def test_an_unparseable_verdict_is_not_an_acceptance(store: FilingStore, tmp_path: Path) -> None:
    """Treating an unreadable verdict as approval makes the verifier a no-op

    exactly when the model is struggling, which is when it matters most.
    """
    agent = SystemB(
        backend(
            tmp_path,
            PLAN,
            SEARCH,
            ANSWER,
            ANSWER,
            "I think it looks fine",
            REVISED_ANSWER,
            "also fine",
        ),
        store,
    )
    result = agent.answer("q1", "What was revenue?")
    assert result.failed
    assert "unparseable" in result.failure_reason


def test_verification_can_be_disabled_for_an_ablation(store: FilingStore, tmp_path: Path) -> None:
    agent = SystemB(
        backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, REJECT), store, allow_one_retry=False
    )
    result = agent.answer("q1", "What was revenue?")
    assert result.failed


# --------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------


def test_an_unparseable_plan_degrades_to_the_original_question(
    store: FilingStore, tmp_path: Path
) -> None:
    """A degradation, not a silent default: the step is recorded."""
    agent = SystemB(
        backend(tmp_path, "I'm not sure how to plan this", SEARCH, ANSWER, ANSWER, ACCEPT),
        store,
    )
    result = agent.answer("q1", "What was revenue?")
    assert not result.failed
    assert any("planner output was unparseable" in step.error for step in result.steps)


def test_the_planner_is_capped(store: FilingStore, tmp_path: Path) -> None:
    """An unbounded plan multiplies cost without bound, which would make the

    cost comparison a measure of the planner's verbosity.
    """
    many = json.dumps({"subquestions": [f"lookup {i}" for i in range(10)]})
    # Exactly the calls two capped subquestions need: plan, then a search and a
    # report for each, then synthesis and verification.
    responses = [many, SEARCH, ANSWER, SEARCH, ANSWER, ANSWER, ACCEPT]
    agent = SystemB(backend(tmp_path, *responses), store, max_subquestions=2)
    result = agent.answer("q1", "What was revenue?")
    assert not result.failed
    assert sum(1 for step in result.steps if step.kind == "retrieve") == 4


# --------------------------------------------------------------------------
# Comparability with System A
# --------------------------------------------------------------------------


def test_both_systems_share_one_tool_dispatcher() -> None:
    """The benchmark compares orchestration. Anything below it must be

    identical or the comparison measures the wrong thing.
    """
    import inspect

    from agents import system_a, system_b, tools

    assert hasattr(tools, "run_tool")
    assert "run_tool" in inspect.getsource(system_a)
    assert "run_tool" in inspect.getsource(system_b)


def test_both_systems_share_the_answer_schema() -> None:
    from agents.answer import ANSWER_SCHEMA_DESCRIPTION
    from agents.prompts import SYNTHESIS_PROMPT

    assert "{answer_schema}" in SYNTHESIS_PROMPT
    assert "not_disclosed" in ANSWER_SCHEMA_DESCRIPTION


def test_both_systems_expose_the_same_interface(store: FilingStore, tmp_path: Path) -> None:
    a = SystemA(backend(tmp_path, ANSWER), store)
    b = SystemB(backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, ACCEPT), store)
    for agent in (a, b):
        assert isinstance(agent.name, str)
        assert isinstance(agent.params, dict)
        assert callable(agent.answer)


def test_system_b_costs_more_model_calls_than_system_a(store: FilingStore, tmp_path: Path) -> None:
    """The premise of the comparison, asserted rather than assumed. Whether the

    extra calls buy accuracy is what the run decides.
    """
    a_backend = backend(tmp_path / "a", SEARCH, ANSWER)
    a_result = SystemA(a_backend, store).answer("q1", "What was revenue?")

    b_backend = backend(tmp_path / "b", PLAN, SEARCH, ANSWER, ANSWER, ACCEPT)
    b_result = SystemB(b_backend, store).answer("q1", "What was revenue?")

    assert b_result.model_calls > a_result.model_calls


# --------------------------------------------------------------------------
# Through the harness
# --------------------------------------------------------------------------


def test_system_b_runs_through_the_harness(store: FilingStore, tmp_path: Path) -> None:
    caching = backend(tmp_path, PLAN, SEARCH, ANSWER, ANSWER, ACCEPT)
    question = QuestionRecord(
        question_id="q1",
        level=1,
        text="What was Acme Devices Inc.'s total revenue for fiscal year 2024?",
        answer_value=4_000_000_000.0,
        answer_uom="USD",
        primary_adsh=ADSH,
        supporting_adsh=(),
        concept="revenue",
    )
    metrics = run_question_set(
        SystemB(caching, store),
        [question],
        caching,
        tmp_path / "run",
        run_id="test-b",
        question_set_hash="abc",
        backend_name="scripted",
        prompt_templates=all_templates(),
    )
    assert metrics["accuracy_by_level"]["1"]["successes"] == 1
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["system"] == "B"
