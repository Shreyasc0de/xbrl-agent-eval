"""The tools an agent under test is given.

Everything here reads ``data/filings/`` and nothing else. This module imports no
part of ``ingest`` and no part of ``eval``, and it never opens a database. That
is not tidiness, it is the benchmark: the agent sees the human-readable filing,
the grader sees the XBRL, and the import graph is where that separation is
enforced rather than hoped for.

Retrieval is deterministic lexical scoring, BM25 over fixed chunks. Embeddings
would very likely retrieve better, but they would also make a run irreproducible
unless the model, its version and its normalization were all pinned, and the
point of this benchmark is that published numbers regenerate. An embedding
retriever is a cleaner ablation to add later than a default to start with.

Retrieval quality is measured separately from answer accuracy, so every search
returns the chunk ids it matched. Without that, a wrong answer cannot be
attributed to retrieval or to reasoning, and that distinction is most of the
interesting analysis.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

__all__ = [
    "Chunk",
    "FilingHandle",
    "FilingStore",
    "SearchHit",
    "ToolError",
    "calculate",
]

CHUNK_LINES: Final[int] = 40
CHUNK_OVERLAP: Final[int] = 8
MAX_SECTION_CHARS: Final[int] = 8_000
MAX_SNIPPET_CHARS: Final[int] = 700

_ITEM_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:PART\s+[IVX]+|Item\s+\d+[A-Za-z]?\.?)\b", re.IGNORECASE
)
_TOKEN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9][a-z0-9.,']*")


class ToolError(RuntimeError):
    """A tool was called with arguments it cannot satisfy.

    Raised rather than returning an error string, so a malformed tool call
    fails the task instead of feeding the agent a message it might treat as
    content.
    """


@dataclass(frozen=True)
class FilingHandle:
    cik: int
    adsh: str
    form: str
    filing_date: str
    company_name: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    adsh: str
    heading: str
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    heading: str
    score: float
    snippet: str


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _chunk_filing(adsh: str, text: str) -> list[Chunk]:
    """Split filing text into overlapping windows, labelled by the last heading.

    Overlap exists because a statement line and the heading that gives it
    meaning are often a few lines apart, and a hard window boundary between them
    produces a chunk whose numbers cannot be interpreted.
    """
    lines = text.split("\n")
    chunks: list[Chunk] = []
    heading = ""
    headings_at: dict[int, str] = {}

    for index, line in enumerate(lines):
        if _ITEM_HEADING.match(line):
            heading = line.strip()[:120]
        headings_at[index] = heading

    step = max(1, CHUNK_LINES - CHUNK_OVERLAP)
    for start in range(0, max(1, len(lines)), step):
        window = lines[start : start + CHUNK_LINES]
        body = "\n".join(window).strip()
        if not body:
            continue

        # A chunk that contains a heading is labelled by the first heading in
        # it, which is what a reader would call that passage. A chunk that
        # contains none inherits the heading in effect where it starts, so a
        # passage continuing a long section is still attributed to it.
        inner = next((line.strip()[:120] for line in window if _ITEM_HEADING.match(line)), "")
        chunks.append(
            Chunk(
                chunk_id=f"{adsh}#{len(chunks):04d}",
                adsh=adsh,
                heading=inner or headings_at.get(start, ""),
                text=body,
                start_line=start,
                end_line=start + len(window),
            )
        )
    return chunks


class _BM25:
    """Standard BM25 over a fixed corpus. No dependencies, fully deterministic."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        self._documents = [Counter(document) for document in documents]
        self._lengths = [len(document) for document in documents]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        frequency: Counter[str] = Counter()
        for document in documents:
            frequency.update(set(document))
        total = len(documents)
        self._idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def scores(self, query: Sequence[str]) -> list[float]:
        results: list[float] = []
        for index, document in enumerate(self._documents):
            length = self._lengths[index]
            score = 0.0
            for term in query:
                if term not in document:
                    continue
                frequency = document[term]
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * length / (self._average_length or 1.0)
                )
                score += self._idf.get(term, 0.0) * frequency * (self._k1 + 1) / denominator
            results.append(score)
        return results


class FilingStore:
    """The agent's whole view of the world.

    Backed by the text files and the two index files the ingest pipeline writes.
    If a filing is not cached, it does not exist as far as the agent is
    concerned, which is the correct behaviour: the alternative is an agent that
    reaches past its cache to the live web mid-run and makes the run
    irreproducible.
    """

    def __init__(self, filings_dir: Path) -> None:
        self.filings_dir = filings_dir
        self._handles: list[FilingHandle] = []
        self._by_adsh: dict[str, FilingHandle] = {}
        self._names: dict[str, int] = {}
        self._load_indexes()

    def _load_indexes(self) -> None:
        companies_path = self.filings_dir / "companies.json"
        names_by_cik: dict[int, str] = {}
        if companies_path.exists():
            for entry in json.loads(companies_path.read_text(encoding="utf-8")):
                cik = int(entry["cik"])
                names_by_cik[cik] = entry["display_name"]
                self._names[entry["display_name"].lower()] = cik
                self._names[entry["ticker"].lower()] = cik

        manifest_path = self.filings_dir / "manifest.jsonl"
        if not manifest_path.exists():
            return
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            handle = FilingHandle(
                cik=int(row["cik"]),
                adsh=row["adsh"],
                form=row["form"],
                filing_date=row["filing_date"],
                company_name=names_by_cik.get(int(row["cik"]), ""),
            )
            self._handles.append(handle)
            self._by_adsh[handle.adsh] = handle

    # ------------------------------------------------------------------ tools

    def list_filings(self, company: str = "", form: str = "") -> list[FilingHandle]:
        """Filings available for a company, newest first.

        Matching on company is by exact ticker or display name first, then by
        substring. An unmatched company returns an empty list rather than every
        filing, because silently widening the search is how an agent ends up
        confidently reading the wrong company's 10-K.
        """
        handles = self._handles
        if company:
            key = company.strip().lower()
            cik = self._names.get(key)
            if cik is None:
                matches = [value for name, value in self._names.items() if key in name]
                cik = matches[0] if len(set(matches)) == 1 else None
            if cik is None:
                return []
            handles = [handle for handle in handles if handle.cik == cik]
        if form:
            wanted = form.strip().upper()
            handles = [handle for handle in handles if handle.form.upper() == wanted]
        return sorted(handles, key=lambda handle: handle.filing_date, reverse=True)

    def _text(self, adsh: str) -> str:
        if adsh not in self._by_adsh:
            raise ToolError(f"no cached filing with accession number {adsh}")
        handle = self._by_adsh[adsh]
        path = self.filings_dir / f"{handle.cik:010d}" / f"{adsh}.txt"
        if not path.exists():
            raise ToolError(f"filing {adsh} is in the manifest but its text is missing")
        return path.read_text(encoding="utf-8")

    def chunks(self, adsh: str) -> list[Chunk]:
        return _chunk_filing(adsh, self._text(adsh))

    def search_filing(self, adsh: str, query: str, top_k: int = 5) -> list[SearchHit]:
        """Rank chunks of one filing against a query.

        Returns chunk ids as well as text, so retrieval precision can be scored
        separately from answer accuracy.
        """
        if not query.strip():
            raise ToolError("search_filing called with an empty query")
        if top_k < 1:
            raise ToolError(f"top_k must be at least 1, got {top_k}")

        chunks = self.chunks(adsh)
        if not chunks:
            return []

        index = _BM25([_tokenize(chunk.text) for chunk in chunks])
        scores = index.scores(_tokenize(query))
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: -pair[1])

        hits: list[SearchHit] = []
        for chunk, score in ranked[:top_k]:
            if score <= 0.0:
                continue
            snippet = chunk.text
            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[:MAX_SNIPPET_CHARS] + "\n[snippet truncated]"
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    heading=chunk.heading,
                    score=round(score, 4),
                    snippet=snippet,
                )
            )
        return hits

    def read_section(self, chunk_id: str) -> str:
        """Return one chunk in full, truncated with an explicit marker if long.

        The marker matters: an agent that cannot tell a truncated section from a
        complete one will confidently report that a figure is absent.
        """
        if "#" not in chunk_id:
            raise ToolError(f"malformed chunk id {chunk_id!r}, expected 'ADSH#NNNN'")
        adsh = chunk_id.split("#", 1)[0]
        for chunk in self.chunks(adsh):
            if chunk.chunk_id == chunk_id:
                if len(chunk.text) > MAX_SECTION_CHARS:
                    return chunk.text[:MAX_SECTION_CHARS] + "\n[section truncated]"
                return chunk.text
        raise ToolError(f"no section {chunk_id} in filing {adsh}")


# ---------------------------------------------------------------- calculator

_BINARY: Final[dict[type[ast.operator], Callable[[float, float], float]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise ToolError(f"unsupported constant {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_evaluate(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _evaluate(node.operand)
        raise ToolError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        handler = _BINARY.get(type(node.op))
        if handler is None:
            raise ToolError(f"unsupported operator {type(node.op).__name__}")
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if handler in (operator.truediv, operator.mod) and right == 0.0:
            raise ToolError("division by zero")
        return handler(left, right)
    raise ToolError(f"unsupported expression element {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression.

    Parsed and walked rather than passed to eval. A calculator tool is the most
    obvious place for a model to inject something that is not arithmetic, and
    the agent under test is untrusted input by construction.
    """
    if not expression.strip():
        raise ToolError("calculate called with an empty expression")
    if len(expression) > 500:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ToolError(f"could not parse expression: {error}") from error

    result = _evaluate(tree)
    if not math.isfinite(result):
        raise ToolError(f"expression produced a non-finite result: {result}")
    return result


@lru_cache(maxsize=8)
def _cached_store(path_str: str) -> FilingStore:
    return FilingStore(Path(path_str))


def tool_catalogue() -> Iterable[dict[str, Any]]:
    """Machine-readable tool descriptions, for prompt construction."""
    return (
        {
            "name": "list_filings",
            "description": "List cached filings for a company, newest first.",
            "arguments": {"company": "company name or ticker", "form": "10-K or 10-Q"},
        },
        {
            "name": "search_filing",
            "description": "Search one filing and return ranked passages with their ids.",
            "arguments": {
                "adsh": "accession number",
                "query": "search text",
                "top_k": "how many passages",
            },
        },
        {
            "name": "read_section",
            "description": "Read one passage in full, by the id search_filing returned.",
            "arguments": {"chunk_id": "passage id"},
        },
        {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression.",
            "arguments": {"expression": "for example (400 - 250) / 400 * 100"},
        },
    )
