"""The boundary invariant, checked structurally rather than by intention.

The iXBRL tests prove a filing's text is clean. This proves the agent side of
the codebase cannot reach the answer key at all: no module under ``src/agents/``
imports the ingest pipeline, the evaluation package, or a database driver.

A comment saying modules must stay separate is not an invariant. An import graph
that is checked on every commit is.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
AGENT_MODULES = sorted((SRC / "agents").glob("*.py"))

FORBIDDEN_ROOTS = {"ingest", "eval", "questions", "sqlite3"}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_there_are_agent_modules_to_check() -> None:
    """Guard against this suite passing because it found nothing to inspect."""
    assert AGENT_MODULES, "no modules found under src/agents/"


@pytest.mark.parametrize("module", AGENT_MODULES, ids=lambda p: p.name)
def test_no_agent_module_can_reach_the_answer_key(module: Path) -> None:
    """If an agent can read XBRL directly, every result the benchmark has ever

    produced is invalid, and nothing about the failure would be visible in the
    numbers.
    """
    offending = imported_roots(module) & FORBIDDEN_ROOTS
    assert not offending, (
        f"{module.name} imports {sorted(offending)}, which crosses the benchmark's " "one boundary"
    )


def test_the_check_would_actually_fail_if_violated(tmp_path: Path) -> None:
    """A guard that cannot fail is not a guard."""
    violation = tmp_path / "bad_agent.py"
    violation.write_text("from questions.repository import FactRepository\n", encoding="utf-8")
    assert imported_roots(violation) & FORBIDDEN_ROOTS
