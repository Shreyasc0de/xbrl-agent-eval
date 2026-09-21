"""System A: one model, a ReAct loop, and four tools.

The baseline the multi-agent system has to beat. It is deliberately plain, because
the comparison is only informative if System A is a fair representative of the
simple approach rather than a strawman.

Three things it will not do.

It will not answer past its step budget. When the budget runs out without a
final answer, the task fails and is recorded as failed. It does not emit a
default answer, and it does not abstain on the agent's behalf: an abstention the
harness invented would land in the hallucination and false-abstention rates as
though the agent had chosen it, which would corrupt the headline result.

It will not recover from a malformed turn by guessing. A turn that parses as
neither a tool call nor an answer is fed back to the model once as an error, and
a model that cannot produce valid output after that fails the task.

It will not touch anything but its tools. The filing store is the whole world.

Every step is recorded: the raw model output, the tool called, its arguments,
and whether it succeeded. The trace is what makes a wrong answer attributable to
retrieval or to reasoning afterwards.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agents.answer import AnswerParseError, ParsedAnswer, parse_answer
from agents.prompts import ActionParseError, build_prompt, parse_action
from agents.tools import FilingStore, run_tool, tool_catalogue

__all__ = ["AgentResult", "Step", "SystemA"]


@dataclass
class Step:
    """One turn of the loop."""

    index: int
    raw_output: str
    kind: str
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    error: str = ""


@dataclass
class AgentResult:
    """What one task produced.

    ``answer`` is ``None`` exactly when the task failed. A failed task has no
    answer at all, rather than a placeholder that would be graded.
    """

    question_id: str
    answer: ParsedAnswer | None
    failed: bool
    failure_reason: str
    steps: list[Step]
    filings_opened: list[str]
    chunks_opened: list[str]
    model_calls: int
    latency_ms: float

    @property
    def tool_calls(self) -> int:
        return sum(1 for step in self.steps if step.kind == "tool")


class SystemA:
    """Single agent, ReAct loop, hard step cap."""

    name = "A"

    def __init__(
        self,
        backend: Any,
        store: FilingStore,
        *,
        max_steps: int = 10,
        temperature: float = 0.0,
        seed: int | None = 7,
        num_predict: int = 700,
    ) -> None:
        self._backend = backend
        self._store = store
        self._max_steps = max_steps
        self._params: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
        }

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def answer(self, question_id: str, question_text: str) -> AgentResult:
        transcript: list[str] = []
        steps: list[Step] = []
        filings_opened: list[str] = []
        chunks_opened: list[str] = []
        model_calls = 0
        started = time.perf_counter()
        consecutive_parse_failures = 0
        seen_calls: set[tuple[str, str]] = set()

        for index in range(self._max_steps):
            remaining = self._max_steps - index
            prompt = build_prompt(
                question_text, tool_catalogue(), transcript, remaining_steps=remaining
            )

            completion = self._backend.complete(prompt, self._params)
            model_calls += 1

            try:
                action = parse_action(completion.text)
            except ActionParseError as error:
                consecutive_parse_failures += 1
                steps.append(
                    Step(
                        index=index,
                        raw_output=completion.text,
                        kind="invalid",
                        error=str(error),
                    )
                )
                if consecutive_parse_failures >= 2:
                    return self._failed(
                        question_id,
                        f"model produced unparseable output twice: {error}",
                        steps,
                        filings_opened,
                        chunks_opened,
                        model_calls,
                        started,
                    )
                transcript.append(
                    f"[your last turn was rejected: {error}. Reply with one JSON object.]"
                )
                continue

            consecutive_parse_failures = 0

            if action.kind == "answer":
                try:
                    # The payload is used when the model wrapped its answer as
                    # a tool call: the raw text's outermost object is the
                    # wrapper, and parsing that would fail on a valid answer.
                    source = (
                        json.dumps(action.payload) if action.payload is not None else action.raw
                    )
                    parsed = parse_answer(source)
                except AnswerParseError as error:
                    steps.append(
                        Step(
                            index=index,
                            raw_output=completion.text,
                            kind="invalid_answer",
                            error=str(error),
                        )
                    )
                    return self._failed(
                        question_id,
                        f"final answer was not gradeable: {error}",
                        steps,
                        filings_opened,
                        chunks_opened,
                        model_calls,
                        started,
                    )

                steps.append(Step(index=index, raw_output=completion.text, kind="answer"))
                return AgentResult(
                    question_id=question_id,
                    answer=parsed,
                    failed=False,
                    failure_reason="",
                    steps=steps,
                    filings_opened=filings_opened,
                    chunks_opened=chunks_opened,
                    model_calls=model_calls,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )

            arguments = action.arguments or {}

            # A small model with no scratchpad will re-emit its first plausible
            # action forever. Left alone it burns the whole budget on one call
            # and the task fails looking like a retrieval problem, which it is
            # not. Naming the repetition is cheap and turns a silent death
            # spiral into something the model can act on.
            signature = (action.tool, _render_arguments(arguments))
            if signature in seen_calls:
                repeat_note = (
                    f"You already called {action.tool} with exactly these arguments "
                    "and the result is above. Repeating it cannot tell you anything "
                    "new. Choose a different tool or a different argument, or give "
                    "your final answer."
                )
                steps.append(
                    Step(
                        index=index,
                        raw_output=completion.text,
                        kind="tool",
                        tool=action.tool,
                        arguments=arguments,
                        error=repeat_note,
                    )
                )
                transcript.append(f"> {action.tool}({_render_arguments(arguments)})")
                transcript.append(repeat_note)
                continue
            seen_calls.add(signature)

            observation, tool_error = self._run_tool(
                action.tool, arguments, filings_opened, chunks_opened
            )
            steps.append(
                Step(
                    index=index,
                    raw_output=completion.text,
                    kind="tool",
                    tool=action.tool,
                    arguments=arguments,
                    observation=observation,
                    error=tool_error,
                )
            )
            transcript.append(f"> {action.tool}({_render_arguments(arguments)})")
            transcript.append(tool_error or observation)

        return self._failed(
            question_id,
            f"step budget of {self._max_steps} exhausted without a final answer",
            steps,
            filings_opened,
            chunks_opened,
            model_calls,
            started,
        )

    # ------------------------------------------------------------------ tools

    def _run_tool(
        self,
        tool: str,
        arguments: dict[str, Any],
        filings_opened: list[str],
        chunks_opened: list[str],
    ) -> tuple[str, str]:
        return run_tool(self._store, tool, arguments, filings_opened, chunks_opened)

    def _failed(
        self,
        question_id: str,
        reason: str,
        steps: list[Step],
        filings_opened: list[str],
        chunks_opened: list[str],
        model_calls: int,
        started: float,
    ) -> AgentResult:
        return AgentResult(
            question_id=question_id,
            answer=None,
            failed=True,
            failure_reason=reason,
            steps=steps,
            filings_opened=filings_opened,
            chunks_opened=chunks_opened,
            model_calls=model_calls,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _render_arguments(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{name}={value!r}" for name, value in sorted(arguments.items()))
