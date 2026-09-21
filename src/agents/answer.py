"""Parsing an agent's answer into the one shape the grader accepts.

Free-text answers are ungradeable. They are rejected at parse time rather than
coerced into something that looks gradeable, because a coerced answer is a
fabricated result: the number in it was chosen by a regular expression, not by
the agent.

The parser is strict about structure and forgiving about packaging. A model that
wraps its JSON in a code fence or writes a sentence before it is doing something
normal and its answer is still recoverable. A model that omits ``answer_type``,
or gives a numeric answer with no value, is not, and that task fails.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final

__all__ = [
    "ANSWER_SCHEMA_DESCRIPTION",
    "unwrap_answer_tool",
    "ParsedAnswer",
    "AnswerParseError",
    "parse_answer",
]


class AnswerParseError(ValueError):
    """The agent's output is not a gradeable answer."""


_FENCE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)

_VALID_TYPES: Final[frozenset[str]] = frozenset({"numeric", "not_disclosed"})

# Names models reach for when they treat answering as another tool call.
# Observed on qwen3:4b, which emitted a complete and correct answer object
# wrapped as {"tool": "answer", "arguments": {...}} and was rejected for it.
_ANSWER_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {"answer", "final_answer", "submit_answer", "give_answer", "respond"}
)


def unwrap_answer_tool(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap an answer the model packaged as a tool call.

    The protocol asks for two different shapes, a tool call and an answer
    object, and a model that has been calling tools all turn reasonably reaches
    for the tool shape when it finally has an answer. The object inside is
    complete and structurally valid.

    Unwrapping it is normalization, not coercion. Nothing about the answer is
    guessed or repaired: every field still has to be present and valid, and a
    wrapper carrying anything other than a well-formed answer is left alone to
    fail as a tool call. Rejecting this would measure whether the model can
    follow a format, which is not what the benchmark is for.
    """
    if payload.get("answer_type") is not None:
        return payload
    tool = payload.get("tool")
    arguments = payload.get("arguments")
    if (
        isinstance(tool, str)
        and tool.strip().lower() in _ANSWER_TOOL_NAMES
        and isinstance(arguments, dict)
        and arguments.get("answer_type") is not None
    ):
        return arguments
    return payload


_SCALE_WORDS: Final[frozenset[str]] = frozenset(
    {
        "units",
        "unit",
        "actuals",
        "actual",
        "ones",
        "thousands",
        "thousand",
        "millions",
        "million",
        "billions",
        "billion",
    }
)

ANSWER_SCHEMA_DESCRIPTION: Final[str] = """\
Answer with a single JSON object and nothing else:

{
  "answer_type": "numeric" or "not_disclosed",
  "answer_value": number, or null when answer_type is "not_disclosed",
  "unit": "USD", "USD/shares", "shares", or "pure" for ratios and percentages,
  "scale": "units", "thousands", "millions", or "billions",
  "source_quote": "the exact line from the filing the answer came from",
  "confidence": a number between 0 and 1
}

If the filing does not disclose the figure, answer with "not_disclosed" and
leave answer_value null. Do not estimate, infer, or calculate a figure the
filing does not state."""


@dataclass(frozen=True)
class ParsedAnswer:
    """An answer that is structurally valid. Whether it is correct is the grader's

    business, not this module's.
    """

    answer_type: str
    answer_value: float | None
    unit: str
    scale: str
    source_quote: str
    confidence: float
    raw: str = ""

    def __post_init__(self) -> None:
        if self.answer_type not in _VALID_TYPES:
            raise AnswerParseError(
                f"answer_type must be one of {sorted(_VALID_TYPES)}, got {self.answer_type!r}"
            )
        if self.answer_type == "numeric" and self.answer_value is None:
            raise AnswerParseError("a numeric answer must carry a value")
        if self.answer_type == "not_disclosed" and self.answer_value is not None:
            raise AnswerParseError("a not_disclosed answer must not carry a value")
        if not 0.0 <= self.confidence <= 1.0:
            raise AnswerParseError(f"confidence must be in [0, 1], got {self.confidence}")


def _extract_object(text: str) -> str:
    """Find the JSON object in a model's output.

    Takes the last balanced object rather than the first, because a model that
    reasons before answering often writes an example object partway through and
    its real answer last.
    """
    fenced = _FENCE.findall(text)
    if fenced:
        return str(fenced[-1])

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
        raise AnswerParseError(
            "no JSON object found in the agent's output; free-text answers are "
            "not gradeable and are not coerced into one"
        )
    return candidates[-1]


def _coerce_number(value: Any, field: str) -> float:
    """Accept a number, or a numeric string with the punctuation models emit.

    This strips thousands separators, currency symbols and surrounding
    parentheses. Parentheses are read as negative, which is how filings present
    them. It does not try to pull a number out of a sentence: if the field is
    prose, the answer is not structured and the task fails.
    """
    if isinstance(value, bool):
        raise AnswerParseError(f"{field} must be a number, got a boolean")
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        raise AnswerParseError(f"{field} must be a number, got {type(value).__name__}")

    text = value.strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()

    try:
        number = float(text)
    except ValueError as error:
        raise AnswerParseError(f"{field} is not a number: {value!r}") from error
    return -number if negative else number


def parse_answer(output: str) -> ParsedAnswer:
    """Parse a model's final output into a gradeable answer, or raise."""
    if not output or not output.strip():
        raise AnswerParseError("the agent produced no output")

    body = _extract_object(output)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise AnswerParseError(f"answer object is not valid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise AnswerParseError("answer must be a JSON object")

    payload = unwrap_answer_tool(payload)

    answer_type = payload.get("answer_type")
    if not isinstance(answer_type, str):
        raise AnswerParseError("answer_type is missing")
    answer_type = answer_type.strip().lower()

    raw_value = payload.get("answer_value")
    value = None if raw_value is None else _coerce_number(raw_value, "answer_value")

    unit = payload.get("unit", "")
    if not isinstance(unit, str):
        raise AnswerParseError("unit must be a string")

    scale = payload.get("scale", "units")
    if not isinstance(scale, str):
        raise AnswerParseError("scale must be a string")
    scale = scale.strip().lower() or "units"
    if answer_type == "numeric" and scale not in _SCALE_WORDS:
        raise AnswerParseError(
            f"scale must be one of {sorted(_SCALE_WORDS)}, got {scale!r}. "
            "An unrecognized scale cannot be defaulted: the difference between "
            "units and millions is the difference between right and wrong."
        )

    quote = payload.get("source_quote", "")
    if not isinstance(quote, str):
        raise AnswerParseError("source_quote must be a string")

    confidence_raw = payload.get("confidence", 0.0)
    confidence = _coerce_number(confidence_raw, "confidence")

    return ParsedAnswer(
        answer_type=answer_type,
        answer_value=value,
        unit=unit.strip() or "USD",
        scale=scale,
        source_quote=quote.strip(),
        confidence=confidence,
        raw=output,
    )
