"""Verification, tested by breaking things on purpose.

A checker that cannot fail is theatre. Almost every test here edits an artifact
or a README number and asserts the check rejects it, because the value of this
script is entirely in what it refuses.

The one test that asserts success exists to stop the others passing for the
wrong reason: a checker that rejects everything is equally useless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from report.verify_results import (
    RESULTS_BEGIN,
    RESULTS_END,
    VerificationError,
    load_runs,
    render_results_table,
    verify,
)

RUN_ID = "20260101T000000Z-systemA-qwen3-4b"


def task(
    question_id: str,
    level: int,
    correct: bool,
    error_class: str,
    *,
    failed: bool = False,
    cost: float = 0.01,
) -> dict:
    return {
        "question_id": question_id,
        "level": level,
        "question": "What was X's revenue?",
        "failed": failed,
        "failure_reason": "",
        "correct": correct,
        "error_class": error_class,
        "predicted": 1.0,
        "truth": 1.0,
        "tool_calls": 2,
        "model_calls": 3,
        "latency_ms": 100.0,
        "cost_usd": cost,
        "filings_opened": ["0000000001-25-000001"],
        "chunks_opened": [],
        "opened_correct_filing": True,
        "steps": [{"index": 0, "kind": "answer", "raw_output": "{}"}],
    }


TASKS = [
    task("q1", 1, True, "correct"),
    task("q2", 1, False, "wrong_value"),
    task("q3", 2, True, "correct"),
    task("q4", 5, True, "correct"),
    task("q5", 5, False, "hallucination"),
]


def manifest() -> dict:
    return {
        "run_id": RUN_ID,
        "system": "A",
        "model_id": "qwen3:4b",
        "backend": "ollama",
        "temperature": 0.0,
        "seed": 7,
        "prompt_template_hash": "abc123",
        "question_set_hash": "def456",
        "question_count": 5,
        "git_commit": "0" * 40,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:05:00+00:00",
        "wall_clock_seconds": 300.0,
        "model_calls": 15,
        "cache_hits": 0,
        "cache_misses": 15,
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "total_cost_usd": 0.05,
        "tasks_completed": 5,
        "tasks_failed": 0,
    }


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one run whose artifacts agree with each other."""
    from eval.metrics import TaskOutcome, summarize_run

    run_dir = tmp_path / "runs" / RUN_ID
    run_dir.mkdir(parents=True)

    with (run_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for row in TASKS:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    outcomes = [
        TaskOutcome(
            question_id=row["question_id"],
            level=row["level"],
            correct=row["correct"],
            failed=row["failed"],
            error_class=row["error_class"],
            cost_usd=row["cost_usd"],
            latency_ms=row["latency_ms"],
            tool_calls=row["tool_calls"],
            model_calls=row["model_calls"],
            opened_correct_filing=row["opened_correct_filing"],
        )
        for row in TASKS
    ]
    (run_dir / "metrics.json").write_text(
        json.dumps(summarize_run(outcomes), indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest(), indent=2, sort_keys=True), encoding="utf-8"
    )

    runs = load_runs(tmp_path / "runs")
    (tmp_path / "README.md").write_text(
        f"# x\n\n{RESULTS_BEGIN}\n{render_results_table(runs)}\n{RESULTS_END}\n",
        encoding="utf-8",
    )
    return tmp_path


def problems(repo: Path) -> list[str]:
    found, _ = verify(repo)
    return found


# --------------------------------------------------------------------------
# The baseline
# --------------------------------------------------------------------------


def test_consistent_artifacts_pass(repo: Path) -> None:
    """Without this, a checker that rejects everything would look correct."""
    assert problems(repo) == []


def test_an_empty_repository_passes(tmp_path: Path) -> None:
    """No runs and no claims is a valid state, not a failure."""
    (tmp_path / "README.md").write_text(
        f"{RESULTS_BEGIN}\n{render_results_table([])}\n{RESULTS_END}\n", encoding="utf-8"
    )
    assert problems(tmp_path) == []


# --------------------------------------------------------------------------
# Tampering with the summary
# --------------------------------------------------------------------------


def test_an_edited_metrics_file_is_caught(repo: Path) -> None:
    """The scenario this whole script exists for: a summary that no longer

    matches the run behind it.
    """
    path = repo / "runs" / RUN_ID / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["accuracy_overall"]["point"] = 0.99
    path.write_text(json.dumps(metrics), encoding="utf-8")

    found = problems(repo)
    assert any("accuracy_overall.point" in p for p in found)
    assert any("recomputed from tasks.jsonl" in p for p in found)


def test_an_inflated_hallucination_rate_is_caught(repo: Path) -> None:
    path = repo / "runs" / RUN_ID / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["hallucination_rate_l5"]["successes"] = 0
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert any("hallucination_rate_l5" in p for p in problems(repo))


def test_a_metric_invented_in_the_summary_is_caught(repo: Path) -> None:
    path = repo / "runs" / RUN_ID / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["multi_agent_premium"] = 0.42
    path.write_text(json.dumps(metrics), encoding="utf-8")
    assert any("nothing in tasks.jsonl produces" in p for p in problems(repo))


def test_deleting_a_task_record_changes_the_recomputed_metrics(repo: Path) -> None:
    """Dropping an inconvenient task is the cheapest way to improve a result,

    so the recomputation has to notice.
    """
    path = repo / "runs" / RUN_ID / "tasks.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if '"q5"' not in line]
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    assert problems(repo)


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------


def test_a_manifest_missing_fields_is_caught(repo: Path) -> None:
    path = repo / "runs" / RUN_ID / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["prompt_template_hash"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any("missing" in p and "prompt_template_hash" in p for p in problems(repo))


def test_a_run_claiming_results_with_no_model_calls_is_caught(repo: Path) -> None:
    path = repo / "runs" / RUN_ID / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_calls"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any("zero model calls" in p for p in problems(repo))


def test_an_incomplete_run_directory_is_reported_not_skipped(repo: Path) -> None:
    """A half-written directory is what a crashed run leaves behind. Skipping

    it silently would hide the case where a number came from a partial run.
    """
    (repo / "runs" / RUN_ID / "metrics.json").unlink()
    found = problems(repo)
    assert any("incomplete run artifacts" in p for p in found)


# --------------------------------------------------------------------------
# The README
# --------------------------------------------------------------------------


def test_a_readme_table_that_drifted_is_caught(repo: Path) -> None:
    readme = repo / "README.md"
    text = readme.read_text(encoding="utf-8").replace("60.0%", "95.0%")
    readme.write_text(text, encoding="utf-8")
    assert any("does not match the artifacts" in p for p in problems(repo))


def test_a_readme_with_no_markers_is_caught(repo: Path) -> None:
    (repo / "README.md").write_text("# x\n\nno table here\n", encoding="utf-8")
    assert any("markers" in p for p in problems(repo))


def test_a_prose_claim_that_matches_passes(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + f"\nWe measured a 50% hallucination rate. "
        f"<!-- claim: {RUN_ID} hallucination_rate_l5.point 0.5 -->\n",
        encoding="utf-8",
    )
    assert problems(repo) == []


def test_a_prose_claim_that_does_not_match_is_caught(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + f"\nWe measured a 14% hallucination rate. "
        f"<!-- claim: {RUN_ID} hallucination_rate_l5.point 0.14 -->\n",
        encoding="utf-8",
    )
    assert any("artifacts give" in p for p in problems(repo))


def test_a_claim_about_a_run_that_does_not_exist_is_caught(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8")
        + "\n<!-- claim: some-run-that-never-happened accuracy_overall.point 0.9 -->\n",
        encoding="utf-8",
    )
    assert any("no artifacts under runs/" in p for p in problems(repo))


def test_a_claim_naming_a_metric_that_does_not_exist_is_caught(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8") + f"\n<!-- claim: {RUN_ID} made_up_metric 0.9 -->\n",
        encoding="utf-8",
    )
    assert any("is not a metric" in p for p in problems(repo))


# --------------------------------------------------------------------------
# Table rendering
# --------------------------------------------------------------------------


def test_the_table_reports_the_graded_count_not_the_task_count(repo: Path) -> None:
    """Accuracy is over graded tasks, so the denominator shown must be that."""
    runs = load_runs(repo / "runs")
    table = render_results_table(runs)
    assert "| 5 |" in table


def test_the_empty_table_is_a_placeholder_not_a_zero(tmp_path: Path) -> None:
    """A level with no runs must not read as a run that scored nothing."""
    table = render_results_table([])
    assert "no runs yet" in table
    assert "0.0%" not in table


def test_load_runs_on_a_missing_directory_is_not_an_error(tmp_path: Path) -> None:
    assert load_runs(tmp_path / "nothing-here") == []


def test_verification_error_is_raised_for_broken_artifacts(repo: Path) -> None:
    (repo / "runs" / RUN_ID / "tasks.jsonl").unlink()
    with pytest.raises(VerificationError):
        load_runs(repo / "runs")
