"""Prompt templates and the action protocol.

Every template lives here so that ``prompt_template_hash`` over this module's
constants pins a run to the exact wording that produced it. Editing a prompt
changes the hash, which changes every cache key derived from it, which means an
edited prompt cannot quietly inherit an old run's numbers.

The action protocol is one JSON object per turn. A model either calls a tool or
gives its final answer, and the two are told apart by which keys are present
rather than by any wording. Small models are unreliable at prose formats and
perfectly reliable at emitting a single object, so the protocol is chosen to
make the failure mode a parse error rather than a misread intention.

The prompt does not hint that a question may be unanswerable beyond stating the
rule once, and it states that rule identically for every question. If the prompt
nudged harder on some questions than others, abstention would measure the
prompt rather than the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Final

from agents.answer import ANSWER_SCHEMA_DESCRIPTION

__all__ = [
    "ACTION_PROTOCOL",
    "FINDINGS_HEADER",
    "PLANNER_PROMPT",
    "RETRIEVER_PROMPT",
    "SYNTHESIS_PROMPT",
    "SYSTEM_PROMPT",
    "VERIFIER_PROMPT",
    "Action",
    "ActionParseError",
    "all_templates",
    "build_prompt",
    "parse_action",
]

SYSTEM_PROMPT: Final[str] = """\
You are a financial analyst answering a question about a company's SEC filing.

You can only see the filing as a human reader sees it. You cannot query
structured data. Everything you report must come from text you actually read
using the tools.

Work in small steps. Find the right filing, search it for the figure, read the
passage that contains it, and only then answer. Do not answer from memory: a
figure you recall rather than read is a guess, however confident it feels.

Some questions ask about figures a filing does not disclose. When that is the
case, say so. Do not estimate, derive, or infer a figure the filing does not
state."""

ACTION_PROTOCOL: Final[str] = f"""\
Reply with exactly one JSON object per turn, and nothing else.

To use a tool:

{{"tool": "<tool name>", "arguments": {{...}}}}

To give your final answer, reply with the answer object described below.

{ANSWER_SCHEMA_DESCRIPTION}"""

TOOLS_HEADER: Final[str] = "Tools available to you:"

QUESTION_HEADER: Final[str] = "Question:"

TRANSCRIPT_HEADER: Final[str] = "What you have done so far:"

STEP_LIMIT_WARNING: Final[str] = """\
You have {remaining} tool call(s) left. Answer with the answer object now, using
what you have read. If you did not find the figure, answer "not_disclosed"."""


# ------------------------------------------------------------------ system B
#
# The multi-agent prompts. These share SYSTEM_PROMPT and the answer schema with
# System A on purpose: the comparison is between orchestration strategies, not
# between prompts. If B answered with a different schema or a differently worded
# abstention rule, any difference in hallucination rate would be attributable to
# the wording rather than to the architecture.

PLANNER_PROMPT: Final[str] = """\
You are planning how to answer a question about an SEC filing.

Break the question into one to three specific things that must be looked up in
the filings. Each one should name a company and a figure. Do not answer the
question. Do not guess at any number.

Reply with exactly one JSON object and nothing else:

{"subquestions": ["...", "..."]}"""

RETRIEVER_PROMPT: Final[str] = """\
You are looking up one specific figure in SEC filings, as part of answering a
larger question. Find it and report exactly what the filing says.

Report only what you actually read. If you cannot find the figure, say so
plainly rather than estimating it."""

SYNTHESIS_PROMPT: Final[str] = """\
You are answering a question about an SEC filing, using findings gathered by
other analysts who read the filings for you.

Use only those findings. If they do not contain what the question asks for,
answer "not_disclosed" rather than estimating, deriving, or recalling a figure.

{answer_schema}"""

VERIFIER_PROMPT: Final[str] = """\
You are checking a proposed answer against the findings it was drawn from.

Accept it only if the findings actually support it. Reject it if the number
does not appear in the findings, if the arithmetic does not follow from them,
or if the findings do not cover what the question asked.

Rejecting a correct answer costs a retry. Accepting an unsupported one puts an
invented figure into the results, which is worse.

Reply with exactly one JSON object and nothing else:

{"verdict": "accept" or "reject", "reason": "one sentence"}"""

FINDINGS_HEADER: Final[str] = "Findings from the analysts:"


def all_templates() -> tuple[str, ...]:
    """Everything that goes into the prompt template hash."""
    return (
        SYSTEM_PROMPT,
        ACTION_PROTOCOL,
        TOOLS_HEADER,
        QUESTION_HEADER,
        TRANSCRIPT_HEADER,
        STEP_LIMIT_WARNING,
        PLANNER_PROMPT,
        RETRIEVER_PROMPT,
        SYNTHESIS_PROMPT,
        VERIFIER_PROMPT,
        FINDINGS_HEADER,
    )


class ActionParseError(ValueError):
    """The model's turn is neither a tool call nor a final answer."""


@dataclass(frozen=True)
class Action:
    """One turn: either a tool call or a final answer, never both."""

    kind: str
    tool: str = ""
    arguments: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    raw: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("tool", "answer"):
            raise ActionParseError(f"unknown action kind {self.kind!r}")


def render_tools(catalogue: Any) -> str:
    lines = [TOOLS_HEADER]
    for entry in catalogue:
        arguments = ", ".join(
            f"{name} ({description})" for name, description in entry["arguments"].items()
        )
        lines.append(f"- {entry['name']}({arguments}): {entry['description']}")
    return "\n".join(lines)


def build_prompt(
    question: str,
    catalogue: Any,
    transcript: list[str],
    *,
    remaining_steps: int,
) -> str:
    """Assemble the whole prompt for one turn.

    The full transcript is rebuilt each turn rather than kept as a chat history,
    because the content-addressed cache keys on the prompt string. Rebuilding
    makes the key describe the agent's entire state, so two runs that reach the
    same state reuse the same call.
    """
    parts = [SYSTEM_PROMPT, "", render_tools(catalogue), "", ACTION_PROTOCOL, ""]
    parts.append(f"{QUESTION_HEADER} {question}")

    if transcript:
        parts.extend(["", TRANSCRIPT_HEADER, *transcript])

    if remaining_steps <= 1:
        parts.extend(["", STEP_LIMIT_WARNING.format(remaining=remaining_steps)])

    parts.extend(["", "Your next JSON object:"])
    return "\n".join(parts)


def _extract_object(text: str) -> str:
    """Last balanced JSON object in the text.

    Shared shape with the answer parser: a model that shows an example before
    its real output is common, and the last object is the one it meant.
    """
    candidates: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : index + 1])
                start = -1

    if not candidates:
        raise ActionParseError("no JSON object in the model's turn")
    return candidates[-1]


def parse_action(output: str) -> Action:
    """Decide whether a turn is a tool call or a final answer.

    A turn carrying both a tool name and an answer type is rejected rather than
    resolved by precedence. It means the model is confused about what it is
    doing, and picking one for it would record an answer it did not commit to.
    """
    if not output or not output.strip():
        raise ActionParseError("the model produced no output")

    body = _extract_object(output)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ActionParseError(f"turn is not valid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ActionParseError("turn must be a JSON object")

    has_tool = "tool" in payload and payload["tool"]
    has_answer = "answer_type" in payload

    if has_tool and has_answer:
        raise ActionParseError(
            "turn contains both a tool call and a final answer; refusing to guess "
            "which the model meant"
        )

    if has_tool:
        tool = payload["tool"]
        if not isinstance(tool, str):
            raise ActionParseError("tool must be a string")
        arguments = payload.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ActionParseError("arguments must be an object")
        return Action(kind="tool", tool=tool.strip(), arguments=arguments, raw=output)

    if has_answer:
        return Action(kind="answer", payload=payload, raw=output)

    raise ActionParseError(
        "turn is neither a tool call nor an answer: no 'tool' and no 'answer_type'"
    )
