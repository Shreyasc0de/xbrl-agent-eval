"""Command line entry point for a benchmark run.

    python -m eval.run --system A --backend ollama --model qwen3:4b --limit 100

Runs on the host machine, since Ollama is reachable only there. Every run gets
its own directory under ``runs/`` holding the task traces, the metrics and the
manifest, and those three files together are the evidence for anything the
README later claims.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click

from agents.prompts import all_templates
from agents.system_a import SystemA
from agents.tools import FilingStore
from core.cache import LLMCache
from core.llm import CachingBackend, OllamaBackend
from core.paths import default_filings_dir, default_questions_dir
from eval.runner import load_questions, run_question_set

__all__ = ["main"]


@click.command()
@click.option("--system", type=click.Choice(["A"]), default="A", show_default=True)
@click.option("--backend", type=click.Choice(["ollama"]), default="ollama", show_default=True)
@click.option("--model", default="qwen3:4b", show_default=True)
@click.option(
    "--questions",
    "questions_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=lambda: default_questions_dir() / "questions.jsonl",
    show_default="alongside the database",
)
@click.option(
    "--filings",
    "filings_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=default_filings_dir,
    show_default="alongside the database",
)
@click.option(
    "--runs",
    "runs_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("runs"),
    show_default=True,
)
@click.option("--limit", type=int, default=None, help="Run only the first N questions.")
@click.option("--max-steps", default=10, show_default=True)
@click.option("--temperature", default=0.0, show_default=True)
@click.option("--seed", default=7, show_default=True)
@click.option("--ollama-url", default="http://localhost:11434", show_default=True)
def main(
    system: str,
    backend: str,
    model: str,
    questions_path: Path,
    filings_dir: Path,
    runs_dir: Path,
    limit: int | None,
    max_steps: int,
    temperature: float,
    seed: int,
    ollama_url: str,
) -> None:
    questions = load_questions(questions_path, limit=limit)
    if not questions:
        raise click.ClickException(f"{questions_path} contains no questions")

    summary_path = questions_path.parent / "summary.json"
    if summary_path.exists():
        question_set_hash = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "question_set_hash", "unknown"
        )
    else:
        question_set_hash = "unknown"

    store = FilingStore(filings_dir)
    caching = CachingBackend(
        OllamaBackend(model, ollama_url), LLMCache(runs_dir / "model-calls.db")
    )
    agent = SystemA(caching, store, max_steps=max_steps, temperature=temperature, seed=seed)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-system{system}-{model.replace(':', '-')}"
    run_dir = runs_dir / run_id

    failures = 0

    def progress(index: int, total: int, record: object) -> None:
        nonlocal failures
        if getattr(record, "failed", False):
            failures += 1
        if index % 10 == 0 or index == total:
            click.echo(f"  {index}/{total} ({failures} failed so far)")

    click.echo(f"run {run_id}: {len(questions)} questions, {model} via {backend}")
    metrics = run_question_set(
        agent,
        questions,
        caching,
        run_dir,
        run_id=run_id,
        question_set_hash=question_set_hash,
        backend_name=backend,
        prompt_templates=all_templates(),
        progress=progress,
    )

    accuracy = metrics["accuracy_overall"]
    hallucination = metrics["hallucination_rate_l5"]
    click.echo("")
    click.echo(
        f"accuracy {accuracy['point']:.1%} "
        f"[{accuracy['ci95_low']:.1%}, {accuracy['ci95_high']:.1%}] "
        f"on {accuracy['total']} graded"
    )
    click.echo(
        f"hallucination rate (L5) {hallucination['point']:.1%} "
        f"[{hallucination['ci95_low']:.1%}, {hallucination['ci95_high']:.1%}] "
        f"on {hallucination['total']}"
    )
    click.echo(f"failed tasks {metrics['failed']}")
    click.echo(f"artifacts: {run_dir}")


if __name__ == "__main__":
    main()
