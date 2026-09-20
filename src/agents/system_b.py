"""System B: planner, retrievers, synthesis, verifier.

The multi-agent comparison. A planner splits the question into lookups, a
retriever runs a bounded tool loop for each one, a synthesis step writes the
answer from the findings, and a verifier checks the answer against those
findings before it is accepted.

The expectation, stated before any run: this costs several times more than
System A for an accuracy difference inside the confidence interval. Publishing
that is the point. A candidate who measured the multi-agent premium and found
it unjustified is more credible than one who assumed it was worth paying, and
the metrics report cost per correct answer precisely so the comparison is about
value rather than accuracy alone.

Everything that could bias the comparison is shared with System A rather than
reimplemented: the same tools, the same filing store, the same answer schema,
the same abstention wording. The difference between the systems is the
orchestration and nothing else.

The same three refusals as System A apply, for the same reasons. It will not
answer past its budget, it will not recover from malformed output by guessing,
and a verifier rejection does not become an abstention invented by the harness.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from agents.answer import (
    ANSWER_SCHEMA_DESCRIPTION,
    AnswerParseError,
    ParsedAnswer,
    parse_answer,
)
from agents.prompts import (
    FINDINGS_HEADER,
    PLANNER_PROMPT,
    QUESTION_HEADER,
    RETRIEVER_PROMPT,
    SYNTHESIS_PROMPT,
    VERIFIER_PROMPT,
    ActionParseError,
    build_prompt,
    parse_action,
)
from agents.system_a import AgentResult, Step, _render_arguments
from agents.tools import FilingStore, run_tool, tool_catalogue, truncate_tool_output

__all__ = ["Finding", "SystemB"]

MAX_SUBQUESTIONS = 3


@dataclass
class Finding:
    """What one retriever came back with.

    ``text`` is the retriever's own report rather than raw tool output, because
    the synthesis step is deliberately given a summary and not the filing: if it
    could read the filing directly it would be System A with extra steps.
    """

    subquestion: str
    text: str
    chunks_opened: list[str] = field(default_factory=list)
    filings_opened: list[str] = field(default_factory=list)


class SystemB:
    """Planner, retrievers, synthesis, verifier."""

    name = "B"

    def __init__(
        self,
        backend: Any,
        store: FilingStore,
        *,
        max_retrieval_steps: int = 6,
        max_subquestions: int = MAX_SUBQUESTIONS,
        allow_one_retry: bool = True,
        temperature: float = 0.0,
        seed: int | None = 7,
        num_predict: int = 700,
    ) -> None:
        self._backend = backend
        self._store = store
        self._max_retrieval_steps = max_retrieval_steps
        self._max_subquestions = max_subquestions
        self._allow_one_retry = allow_one_retry
        self._params: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
        }

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    # ------------------------------------------------------------------ run

    def answer(self, question_id: str, question_text: str) -> AgentResult:
        state = _RunState(started=time.perf_counter())

        subquestions = self._plan(question_text, state)
        if not subquestions:
            return state.failed(question_id, "planner produced no subquestions")

        findings: list[Finding] = []
        for subquestion in subquestions:
            finding = self._retrieve(subquestion, state)
            findings.append(finding)

        parsed = self._synthesize(question_text, findings, state)
        if parsed is None:
            return state.failed(question_id, state.last_error or "synthesis produced no answer")

        accepted, reason = self._verify(question_text, findings, parsed, state)
        if not accepted and self._allow_one_retry:
            # Worth knowing: if synthesis returns the same answer, the second
            # verification prompt is byte-identical to the first, so the
            # content-addressed cache replays the same rejection without a new
            # model call. The retry only helps when synthesis actually changes
            # its answer, which is the correct behaviour and not obvious.
            state.note(f"verifier rejected the first answer: {reason}")
            retry = self._synthesize(question_text, findings, state, rejection=reason)
            if retry is not None:
                parsed = retry
                accepted, reason = self._verify(question_text, findings, parsed, state)

        if not accepted:
            # The verifier's judgement is recorded and the task fails. Turning a
            # rejection into "not_disclosed" would put an abstention the agent
            # never chose into the hallucination and false-abstention rates.
            return state.failed(question_id, f"verifier rejected the answer: {reason}")

        return AgentResult(
            question_id=question_id,
            answer=parsed,
            failed=False,
            failure_reason="",
            steps=state.steps,
            filings_opened=state.filings_opened,
            chunks_opened=state.chunks_opened,
            model_calls=state.model_calls,
            latency_ms=(time.perf_counter() - state.started) * 1000.0,
        )

    # ------------------------------------------------------------------ roles

    def _call(self, prompt: str, state: _RunState, kind: str) -> str:
        completion = self._backend.complete(prompt, self._params)
        state.model_calls += 1
        state.steps.append(Step(index=len(state.steps), raw_output=completion.text, kind=kind))
        return str(completion.text)

    def _plan(self, question: str, state: _RunState) -> list[str]:
        prompt = f"{PLANNER_PROMPT}\n\n{QUESTION_HEADER} {question}\n\nYour JSON object:"
        output = self._call(prompt, state, "plan")
        try:
            payload = json.loads(_last_object(output))
            raw = payload.get("subquestions", [])
        except (ValueError, AttributeError):
            # A planner that cannot produce a plan falls back to the original
            # question as a single lookup. This is a degradation, not a silent
            # default: the step is recorded and the run is still one retrieval
            # short of what the architecture intends.
            state.note("planner output was unparseable; using the question as one lookup")
            return [question]

        subquestions = [str(item).strip() for item in raw if str(item).strip()]
        return subquestions[: self._max_subquestions] or [question]

    def _retrieve(self, subquestion: str, state: _RunState) -> Finding:
        transcript: list[str] = []
        chunks: list[str] = []
        filings: list[str] = []
        consecutive_failures = 0

        for index in range(self._max_retrieval_steps):
            remaining = self._max_retrieval_steps - index
            prompt = (
                RETRIEVER_PROMPT
                + "\n\n"
                + build_prompt(subquestion, tool_catalogue(), transcript, remaining_steps=remaining)
            )
            output = self._call(prompt, state, "retrieve")

            try:
                action = parse_action(output)
            except ActionParseError:
                consecutive_failures += 1
                if consecutive_failures >= 2:
                    break
                transcript.append("[your last turn was rejected. Reply with one JSON object.]")
                continue
            consecutive_failures = 0

            if action.kind == "answer":
                break

            arguments = action.arguments or {}
            observation, error = run_tool(self._store, action.tool, arguments, filings, chunks)
            transcript.append(f"> {action.tool}({_render_arguments(arguments)})")
            transcript.append(error or observation)

        for adsh in filings:
            if adsh not in state.filings_opened:
                state.filings_opened.append(adsh)
        for chunk in chunks:
            if chunk not in state.chunks_opened:
                state.chunks_opened.append(chunk)

        report = "\n".join(transcript) if transcript else "nothing found"
        return Finding(
            subquestion=subquestion,
            text=truncate_tool_output(report),
            chunks_opened=chunks,
            filings_opened=filings,
        )

    def _render_findings(self, findings: list[Finding]) -> str:
        blocks = [
            f"[{index + 1}] {finding.subquestion}\n{finding.text}"
            for index, finding in enumerate(findings)
        ]
        return f"{FINDINGS_HEADER}\n\n" + "\n\n".join(blocks)

    def _synthesize(
        self,
        question: str,
        findings: list[Finding],
        state: _RunState,
        rejection: str = "",
    ) -> ParsedAnswer | None:
        parts = [
            SYNTHESIS_PROMPT.format(answer_schema=ANSWER_SCHEMA_DESCRIPTION),
            "",
            f"{QUESTION_HEADER} {question}",
            "",
            self._render_findings(findings),
        ]
        if rejection:
            parts.extend(["", f"A previous answer was rejected by review: {rejection}", ""])
        parts.extend(["", "Your JSON answer object:"])

        output = self._call("\n".join(parts), state, "synthesize")
        try:
            return parse_answer(output)
        except AnswerParseError as error:
            state.last_error = f"synthesis answer was not gradeable: {error}"
            return None

    def _verify(
        self,
        question: str,
        findings: list[Finding],
        answer: ParsedAnswer,
        state: _RunState,
    ) -> tuple[bool, str]:
        prompt = "\n".join(
            [
                VERIFIER_PROMPT,
                "",
                f"{QUESTION_HEADER} {question}",
                "",
                self._render_findings(findings),
                "",
                "Proposed answer:",
                answer.raw.strip(),
                "",
                "Your JSON object:",
            ]
        )
        output = self._call(prompt, state, "verify")
        try:
            payload = json.loads(_last_object(output))
            verdict = str(payload.get("verdict", "")).strip().lower()
            reason = str(payload.get("reason", "")).strip()
        except (ValueError, AttributeError):
            # An unreadable verdict is not an acceptance. Treating it as one
            # would make the verifier a no-op exactly when the model is
            # struggling, which is when it matters most.
            return False, "verifier output was unparseable"

        if verdict == "accept":
            return True, reason
        return False, reason or "no reason given"


# ---------------------------------------------------------------- internals


@dataclass
class _RunState:
    started: float
    steps: list[Step] = field(default_factory=list)
    filings_opened: list[str] = field(default_factory=list)
    chunks_opened: list[str] = field(default_factory=list)
    model_calls: int = 0
    last_error: str = ""

    def note(self, message: str) -> None:
        self.steps.append(Step(index=len(self.steps), raw_output="", kind="note", error=message))

    def failed(self, question_id: str, reason: str) -> AgentResult:
        return AgentResult(
            question_id=question_id,
            answer=None,
            failed=True,
            failure_reason=reason,
            steps=self.steps,
            filings_opened=self.filings_opened,
            chunks_opened=self.chunks_opened,
            model_calls=self.model_calls,
            latency_ms=(time.perf_counter() - self.started) * 1000.0,
        )


def _last_object(text: str) -> str:
    depth = 0
    start = -1
    found: list[str] = []
    for index, character in enumerate(text):
        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                found.append(text[start : index + 1])
                start = -1
    if not found:
        raise ValueError("no JSON object in output")
    return found[-1]
