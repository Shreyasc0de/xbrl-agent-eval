"""The run harness: question set in, graded artifacts out.

This is the only place a run's artifacts are written, and it writes them in one
order for a reason. Tasks run, outcomes are graded, the manifest asserts the run
actually happened, and only then does anything reach ``runs/``. A run that made
no model calls raises before a single file is written, so there is no partial
results directory for someone to mistake for a real one later.

A failed task is recorded as failed. It is not graded, not counted as wrong, and
not given a substitute answer. Counting a failure as incorrect would credit the
agent with an attempt it never made; dropping it silently would inflate
accuracy. It appears in the artifacts with its reason and in the metrics as its
own count.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from agents.system_a import AgentResult
from core.llm import ModelCallError
from core.manifest import RunManifest, prompt_template_hash
from eval.grader import AgentAnswer, AnswerType, ErrorClass, GroundTruth, grade
from eval.metrics import TaskOutcome, summarize_run
from eval.normalizers import Scale, parse_scale

__all__ = ["QuestionRecord", "TaskRecord", "load_questions", "run_question_set"]


class Agent(Protocol):
    name: str

    @property
    def params(self) -> dict[str, Any]: ...

    def answer(self, question_id: str, question_text: str) -> AgentResult: ...


@dataclass(frozen=True)
class QuestionRecord:
    """One question as it comes off disk.

    Deliberately not the generator's ``Question`` type. The runner reads a
    published question set file, which is the same thing a third party would
    read, so a field the file does not carry is a field the runner cannot use.
    """

    question_id: str
    level: int
    text: str
    answer_value: float | None
    answer_uom: str
    primary_adsh: str
    supporting_adsh: tuple[str, ...]
    concept: str

    @property
    def ground_truth(self) -> GroundTruth:
        return GroundTruth(value=self.answer_value, uom=self.answer_uom, level=self.level)

    @property
    def answer_bearing_filings(self) -> set[str]:
        return {self.primary_adsh, *self.supporting_adsh}


def load_questions(path: Path, limit: int | None = None) -> list[QuestionRecord]:
    records: list[QuestionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            QuestionRecord(
                question_id=row["question_id"],
                level=int(row["level"]),
                text=row["text"],
                answer_value=row["answer_value"],
                answer_uom=row["answer_uom"],
                primary_adsh=row["primary_adsh"],
                supporting_adsh=tuple(row.get("supporting_adsh", ())),
                concept=row.get("concept", ""),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


@dataclass
class TaskRecord:
    """Everything one task produced, written to the run artifacts verbatim.

    The raw model output for every step is kept. It is bulky, and it is the
    thing that lets someone check a claimed hallucination was really a
    hallucination rather than a parse failure.
    """

    question_id: str
    level: int
    question: str
    failed: bool
    failure_reason: str
    correct: bool | None
    error_class: str
    predicted: float | None
    truth: float | None
    tool_calls: int
    model_calls: int
    latency_ms: float
    cost_usd: float
    filings_opened: list[str]
    chunks_opened: list[str]
    opened_correct_filing: bool
    steps: list[dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "level": self.level,
            "question": self.question,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "correct": self.correct,
            "error_class": self.error_class,
            "predicted": self.predicted,
            "truth": self.truth,
            "tool_calls": self.tool_calls,
            "model_calls": self.model_calls,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": self.cost_usd,
            "filings_opened": self.filings_opened,
            "chunks_opened": self.chunks_opened,
            "opened_correct_filing": self.opened_correct_filing,
            "steps": self.steps,
        }


def _to_agent_answer(result: AgentResult) -> AgentAnswer:
    parsed = result.answer
    assert parsed is not None
    if parsed.answer_type == "not_disclosed":
        return AgentAnswer(
            answer_value=None,
            unit=parsed.unit,
            scale=Scale.UNITS,
            answer_type=AnswerType.NOT_DISCLOSED,
            source_quote=parsed.source_quote,
            confidence=parsed.confidence,
        )
    return AgentAnswer(
        answer_value=parsed.answer_value,
        unit=parsed.unit,
        scale=parse_scale(parsed.scale),
        answer_type=AnswerType.NUMERIC,
        source_quote=parsed.source_quote,
        confidence=parsed.confidence,
    )


def _grade_one(question: QuestionRecord, result: AgentResult) -> TaskRecord:
    opened_correct = bool(set(result.filings_opened) & question.answer_bearing_filings)
    steps = [
        {
            "index": step.index,
            "kind": step.kind,
            "tool": step.tool,
            "arguments": step.arguments,
            "observation": step.observation,
            "error": step.error,
            "raw_output": step.raw_output,
        }
        for step in result.steps
    ]

    if result.failed or result.answer is None:
        return TaskRecord(
            question_id=question.question_id,
            level=question.level,
            question=question.text,
            failed=True,
            failure_reason=result.failure_reason,
            correct=None,
            error_class="task_failed",
            predicted=None,
            truth=question.answer_value,
            tool_calls=result.tool_calls,
            model_calls=result.model_calls,
            latency_ms=result.latency_ms,
            cost_usd=0.0,
            filings_opened=result.filings_opened,
            chunks_opened=result.chunks_opened,
            opened_correct_filing=opened_correct,
            steps=steps,
        )

    try:
        graded = grade(_to_agent_answer(result), question.ground_truth)
        error_class = graded.error_class.value
        correct = graded.correct
        predicted = graded.predicted_base_units
    except ValueError as error:
        # An answer the grader cannot honestly evaluate, for example one in an
        # unrecognized unit. Recorded as a failed task rather than as incorrect,
        # because the agent was not shown to be wrong about the figure.
        return TaskRecord(
            question_id=question.question_id,
            level=question.level,
            question=question.text,
            failed=True,
            failure_reason=f"ungradeable answer: {error}",
            correct=None,
            error_class="task_failed",
            predicted=None,
            truth=question.answer_value,
            tool_calls=result.tool_calls,
            model_calls=result.model_calls,
            latency_ms=result.latency_ms,
            cost_usd=0.0,
            filings_opened=result.filings_opened,
            chunks_opened=result.chunks_opened,
            opened_correct_filing=opened_correct,
            steps=steps,
        )

    # Retrieval is scored separately from reasoning. An answer that is wrong and
    # was produced without ever opening the filing containing the figure is a
    # retrieval failure, and calling it a reasoning failure would misattribute
    # most of what this benchmark is trying to measure.
    if (
        not correct
        and question.level != 5
        and not opened_correct
        and error_class in (ErrorClass.WRONG_VALUE.value, ErrorClass.FALSE_ABSTENTION.value)
    ):
        error_class = ErrorClass.RETRIEVAL_MISS.value

    return TaskRecord(
        question_id=question.question_id,
        level=question.level,
        question=question.text,
        failed=False,
        failure_reason="",
        correct=correct,
        error_class=error_class,
        predicted=predicted,
        truth=question.answer_value,
        tool_calls=result.tool_calls,
        model_calls=result.model_calls,
        latency_ms=result.latency_ms,
        cost_usd=0.0,
        filings_opened=result.filings_opened,
        chunks_opened=result.chunks_opened,
        opened_correct_filing=opened_correct,
        steps=steps,
    )


def _outcome(record: TaskRecord) -> TaskOutcome:
    """Reduce a record to what the metrics need.

    Cost comes off the record rather than being passed alongside it, so the
    number the metrics use is the same number written to disk. Any other
    arrangement lets the published figure and the artifact drift apart, which
    is exactly what verify_results exists to catch.
    """
    return TaskOutcome(
        question_id=record.question_id,
        level=record.level,
        correct=bool(record.correct),
        failed=record.failed,
        error_class=record.error_class,
        cost_usd=record.cost_usd,
        latency_ms=record.latency_ms,
        tool_calls=record.tool_calls,
        model_calls=record.model_calls,
        opened_correct_filing=record.opened_correct_filing,
    )


def run_question_set(
    agent: Agent,
    questions: Sequence[QuestionRecord],
    backend: Any,
    run_dir: Path,
    *,
    run_id: str,
    question_set_hash: str,
    backend_name: str,
    prompt_templates: Iterable[str],
    progress: Any = None,
) -> dict[str, Any]:
    """Run every question, grade it, and write the artifacts.

    Nothing is written until the manifest's work assertion passes.
    """
    if not questions:
        raise ValueError("refusing to run an empty question set")

    params = agent.params
    manifest = RunManifest(
        run_id=run_id,
        system=agent.name,
        model_id=backend.model_id,
        backend=backend_name,
        temperature=float(params.get("temperature", 0.0)),
        seed=params.get("seed"),
        prompt_template_hash=prompt_template_hash(*prompt_templates),
        question_set_hash=question_set_hash,
        question_count=len(questions),
    )

    started = time.perf_counter()
    records: list[TaskRecord] = []
    # Cost is attributed per task by differencing the backend's running total
    # around each call, and stored on the record so it reaches disk. Without it
    # in the artifacts, cost per correct answer cannot be recomputed from raw
    # logs and the README figure would have to be taken on trust.
    cost_before = getattr(backend, "total_cost_usd", 0.0)

    for index, question in enumerate(questions, start=1):
        before = getattr(backend, "total_cost_usd", 0.0)
        try:
            result = agent.answer(question.question_id, question.text)
        except ModelCallError as error:
            result = AgentResult(
                question_id=question.question_id,
                answer=None,
                failed=True,
                failure_reason=f"model call failed: {error}",
                steps=[],
                filings_opened=[],
                chunks_opened=[],
                model_calls=0,
                latency_ms=0.0,
            )

        record = _grade_one(question, result)
        record.cost_usd = round(getattr(backend, "total_cost_usd", 0.0) - before, 10)
        records.append(record)

        if progress is not None:
            progress(index, len(questions), record)

    manifest.model_calls = getattr(backend, "calls", 0)
    manifest.cache_hits = getattr(backend, "cache_hits", 0)
    manifest.cache_misses = getattr(backend, "cache_misses", 0)
    manifest.prompt_tokens = getattr(backend, "prompt_tokens", 0)
    manifest.completion_tokens = getattr(backend, "completion_tokens", 0)
    manifest.total_cost_usd = getattr(backend, "total_cost_usd", 0.0) - cost_before
    manifest.tasks_completed = sum(1 for record in records if not record.failed)
    manifest.tasks_failed = sum(1 for record in records if record.failed)
    manifest.complete(time.perf_counter() - started)

    # Before anything reaches disk. A results directory from a run that called
    # nothing is fabricated output no matter what it contains.
    manifest.assert_work_happened()

    outcomes = [_outcome(record) for record in records]
    metrics = summarize_run(outcomes)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True))
            handle.write("\n")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest.write(run_dir / "manifest.json")

    return metrics
