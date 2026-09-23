"""Recompute every published number from raw run artifacts, and fail on a mismatch.

This is the script that makes the rest of the repository's claims checkable. It
reads only ``runs/*/tasks.jsonl`` and ``runs/*/manifest.json``, recomputes the
metrics from the per-task records, and compares the result against three things:
the ``metrics.json`` that run wrote, the results table in the README, and any
number quoted in README prose behind a claim marker.

It deliberately does not import the runner or trust ``metrics.json``. The whole
point is to answer "does the summary still match the logs", and a checker that
reuses the summary cannot answer that.

What it catches. A metrics file edited by hand. A README number that drifted
from the run that produced it. A results table that was never regenerated after
a rerun. A run whose manifest is missing fields, or that claims results while
reporting zero model calls.

What it cannot catch. Fabricated ``tasks.jsonl`` records. Nothing downstream can
detect those, which is why the run harness writes the raw model output for every
step: a reviewer who doubts a number can read the transcript that produced it.

Exit code is 0 when everything agrees and 1 otherwise, so CI fails the build on
a mismatch.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from eval.metrics import TaskOutcome, summarize_run

__all__ = [
    "RunArtifacts",
    "VerificationError",
    "load_runs",
    "recompute_metrics",
    "render_results_table",
    "write_results_table",
    "verify",
]

RESULTS_BEGIN: Final[str] = "<!-- results:begin -->"
RESULTS_END: Final[str] = "<!-- results:end -->"

# A number quoted in prose carries a marker naming the run and the metric path
# it came from, for example:
#     <!-- claim: 20260101T000000Z-systemA-qwen3-4b hallucination_rate_l5.point 0.14 -->
_CLAIM: Final[re.Pattern[str]] = re.compile(
    r"<!--\s*claim:\s*(?P<run>\S+)\s+(?P<path>\S+)\s+(?P<value>\S+)\s*-->"
)

# Fields a manifest must carry for a run to count as valid.
REQUIRED_MANIFEST_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "run_id",
        "system",
        "model_id",
        "backend",
        "temperature",
        "seed",
        "prompt_template_hash",
        "question_set_hash",
        "question_count",
        "git_commit",
        "started_at",
        "finished_at",
        "wall_clock_seconds",
        "model_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_cost_usd",
    }
)

# Metrics are rounded when written, so comparison allows the rounding but
# nothing larger. This is not a tolerance for disagreement.
_ROUNDING_SLACK: Final[float] = 1e-6


class VerificationError(RuntimeError):
    """A published number does not match the artifacts behind it."""


@dataclass(frozen=True)
class RunArtifacts:
    run_id: str
    directory: Path
    manifest: dict[str, Any]
    written_metrics: dict[str, Any]
    outcomes: list[TaskOutcome]


def _passage_flag(row: dict[str, Any]) -> bool | None:
    """Read the tri-state passage flag, keeping absent and false distinct.

    A run recorded before passage annotations existed has no such key at all,
    and that is not the same as the agent having missed the passage. The value
    is validated rather than coerced: a row carrying something other than a
    boolean or null is a corrupt artifact, and defaulting it would let a
    published number rest on one.
    """
    if "opened_answer_passage" not in row:
        return None
    value = row["opened_answer_passage"]
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(
        f"opened_answer_passage must be true, false or null, got {value!r} "
        f"in task {row.get('question_id')!r}"
    )


def _outcome_from_record(row: dict[str, Any]) -> TaskOutcome:
    return TaskOutcome(
        question_id=row["question_id"],
        level=int(row["level"]),
        correct=bool(row.get("correct")),
        failed=bool(row["failed"]),
        error_class=row["error_class"],
        cost_usd=float(row.get("cost_usd", 0.0)),
        latency_ms=float(row["latency_ms"]),
        tool_calls=int(row["tool_calls"]),
        model_calls=int(row["model_calls"]),
        opened_correct_filing=bool(row["opened_correct_filing"]),
        opened_answer_passage=_passage_flag(row),
    )


def load_runs(runs_dir: Path) -> list[RunArtifacts]:
    """Load every run directory that carries a complete set of artifacts.

    A directory missing any of the three files is reported as a failure rather
    than skipped. A half-written run directory is the shape a crashed run
    leaves behind, and quietly ignoring it would hide exactly the case where a
    published number came from an incomplete run.
    """
    runs: list[RunArtifacts] = []
    if not runs_dir.exists():
        return runs

    for directory in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        tasks_path = directory / "tasks.jsonl"
        metrics_path = directory / "metrics.json"
        manifest_path = directory / "manifest.json"
        present = [p for p in (tasks_path, metrics_path, manifest_path) if p.exists()]
        if not present:
            continue
        if len(present) != 3:
            missing = [p.name for p in (tasks_path, metrics_path, manifest_path) if not p.exists()]
            raise VerificationError(
                f"{directory.name}: incomplete run artifacts, missing {missing}"
            )

        outcomes = [
            _outcome_from_record(json.loads(line))
            for line in tasks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        runs.append(
            RunArtifacts(
                run_id=directory.name,
                directory=directory,
                manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
                written_metrics=json.loads(metrics_path.read_text(encoding="utf-8")),
                outcomes=outcomes,
            )
        )
    return runs


def recompute_metrics(run: RunArtifacts) -> dict[str, Any]:
    """Metrics computed from the per-task records alone."""
    return summarize_run(run.outcomes)


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            flat.update(_flatten(value, f"{prefix}{key}."))
    else:
        flat[prefix.rstrip(".")] = payload
    return flat


def _values_agree(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left == right)
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=_ROUNDING_SLACK)
    return bool(left == right)


def _check_manifest(run: RunArtifacts, problems: list[str]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(run.manifest))
    if missing:
        problems.append(f"{run.run_id}: manifest is missing {missing}")
    if int(run.manifest.get("model_calls", 0)) <= 0:
        problems.append(
            f"{run.run_id}: manifest reports zero model calls but the run wrote results"
        )
    if not str(run.manifest.get("git_commit", "")):
        problems.append(f"{run.run_id}: manifest has no git_commit")


def _check_metrics(run: RunArtifacts, problems: list[str]) -> None:
    recomputed = _flatten(recompute_metrics(run))
    written = _flatten(run.written_metrics)

    for key, expected in recomputed.items():
        if key not in written:
            problems.append(f"{run.run_id}: metrics.json is missing {key}")
            continue
        if not _values_agree(expected, written[key]):
            problems.append(
                f"{run.run_id}: {key} is {written[key]!r} in metrics.json but "
                f"{expected!r} when recomputed from tasks.jsonl"
            )

    extra = sorted(set(written) - set(recomputed))
    if extra:
        problems.append(
            f"{run.run_id}: metrics.json carries {extra}, which nothing in " "tasks.jsonl produces"
        )


def render_results_table(runs: list[RunArtifacts]) -> str:
    """The README results table, generated from artifacts.

    With no runs this returns the placeholder, so the table is generated in
    every state rather than being hand-written until the first run exists.
    """
    header = (
        "| System | Backend | Model | Graded | Accuracy | "
        "L5 hallucination rate | Passage hit rate | Cost per correct |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    if not runs:
        return f"{header}\n| _(no runs yet)_ | | | | | | | |"

    rows = []
    for run in sorted(runs, key=lambda r: r.run_id):
        metrics = recompute_metrics(run)
        accuracy = metrics["accuracy_overall"]
        hallucination = metrics["hallucination_rate_l5"]
        cost = metrics["cost_per_correct_answer_usd"]
        passage = metrics["passage_hit_rate"]
        # A run made before passage annotations existed says so, rather than
        # showing a rate computed over nothing. The denominator is printed
        # because a hit rate over nine questions is not the same claim as one
        # over six hundred, and a bare percentage hides which it is.
        passage_cell = (
            "not annotated"
            if passage is None
            else (
                f"{passage['point']:.1%} "
                f"[{passage['ci95_low']:.1%}, {passage['ci95_high']:.1%}] "
                f"(n={passage['total']})"
            )
        )
        rows.append(
            f"| {run.manifest['system']} "
            f"| {run.manifest['backend']} "
            f"| {run.manifest['model_id']} "
            f"| {accuracy['total']} "
            f"| {accuracy['point']:.1%} "
            f"[{accuracy['ci95_low']:.1%}, {accuracy['ci95_high']:.1%}] "
            f"| {hallucination['point']:.1%} "
            f"[{hallucination['ci95_low']:.1%}, {hallucination['ci95_high']:.1%}] "
            f"| {passage_cell} "
            f"| {'n/a' if cost is None else f'${cost:.4f}'} |"
        )
    return "\n".join([header, *rows])


def _extract_block(text: str) -> str | None:
    start = text.find(RESULTS_BEGIN)
    end = text.find(RESULTS_END)
    if start < 0 or end < 0:
        return None
    return text[start + len(RESULTS_BEGIN) : end].strip()


def _check_readme(readme: Path, runs: list[RunArtifacts], problems: list[str]) -> None:
    if not readme.exists():
        problems.append(f"{readme} does not exist")
        return

    text = readme.read_text(encoding="utf-8")

    block = _extract_block(text)
    if block is None:
        problems.append(
            f"{readme.name} has no {RESULTS_BEGIN} / {RESULTS_END} markers, so its "
            "results table cannot be checked against the artifacts"
        )
    else:
        expected = render_results_table(runs).strip()
        if block != expected:
            problems.append(
                f"{readme.name}: the results table does not match the artifacts.\n"
                f"--- in README ---\n{block}\n--- regenerated from runs/ ---\n{expected}"
            )

    by_id = {run.run_id: run for run in runs}
    for match in _CLAIM.finditer(text):
        run_id = match.group("run")
        path = match.group("path")
        claimed = match.group("value")

        run = by_id.get(run_id)
        if run is None:
            problems.append(
                f"{readme.name}: claims a number from run {run_id!r}, which has no "
                "artifacts under runs/"
            )
            continue

        flat = _flatten(recompute_metrics(run))
        if path not in flat:
            problems.append(f"{readme.name}: claim path {path!r} is not a metric")
            continue

        actual = flat[path]
        try:
            claimed_value: Any = float(claimed)
        except ValueError:
            claimed_value = claimed
        if not _values_agree(claimed_value, actual):
            problems.append(
                f"{readme.name}: claims {path} = {claimed} for {run_id}, but the "
                f"artifacts give {actual!r}"
            )


def verify(root: Path) -> tuple[list[str], list[RunArtifacts]]:
    """Return the problems found and the runs examined.

    Both, rather than just the problems, so a caller can report how much was
    actually checked. "No problems" over zero runs is a different statement
    from "no problems" over four.
    """
    try:
        runs = load_runs(root / "runs")
    except VerificationError as error:
        return [str(error)], []

    problems: list[str] = []
    for run in runs:
        _check_manifest(run, problems)
        _check_metrics(run, problems)

    _check_readme(root / "README.md", runs, problems)
    return problems, runs


def write_results_table(readme: Path, runs: list[RunArtifacts]) -> bool:
    """Replace the README's results block with the table generated from runs/.

    Returns whether the file changed. This is the other half of the check: the
    README claims the table is generated rather than hand-written, and until
    something generated it that claim was prose. Both sides call
    ``render_results_table``, so the writer cannot produce a table the verifier
    would reject.
    """
    text = readme.read_text(encoding="utf-8")
    start = text.find(RESULTS_BEGIN)
    end = text.find(RESULTS_END)
    if start < 0 or end < 0:
        raise VerificationError(
            f"{readme.name} has no {RESULTS_BEGIN} / {RESULTS_END} markers to write between"
        )

    table = render_results_table(runs).strip()
    updated = f"{text[: start + len(RESULTS_BEGIN)]}\n{table}\n{text[end:]}"
    if updated == text:
        return False
    readme.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv or [])
    write = "--write" in arguments
    if write:
        arguments.remove("--write")
    root = Path(arguments[0]) if arguments else Path.cwd()

    problems, runs = verify(root)

    if write:
        # Only the README's own table may be stale in write mode. A metrics
        # file that disagrees with its task records is a corrupt artifact, and
        # regenerating the table from it would launder that into the README.
        artifact_problems = [
            problem for problem in problems if not problem.startswith("README.md:")
        ]
        if artifact_problems:
            print(
                "verify_results: refusing to write; the artifacts themselves " "do not check out\n",
                file=sys.stderr,
            )
            for problem in artifact_problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        changed = write_results_table(root / "README.md", runs)
        print(
            f"verify_results: README results table {'rewritten' if changed else 'already current'}"
            f" from {len(runs)} run(s)"
        )
        return 0

    if problems:
        print(f"verify_results: {len(problems)} problem(s)\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if not runs:
        print("verify_results: no runs yet; README carries no unverified numbers")
    else:
        print(f"verify_results: {len(runs)} run(s) check out against their artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
