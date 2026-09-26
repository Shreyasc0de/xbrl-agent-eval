"""Annotate a question set with the passages that state its answers.

Run after ``questions.generate`` and after the filings are cached. Reads the
generated questions as JSON and the filings as text; touches no database.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from agents.tools import Chunk, FilingStore, ToolError
from annotate.passages import (
    AnnotationSet,
    PassageAnnotation,
    SkippedQuestion,
    annotate_question,
    iter_answerable,
)
from core.paths import default_filings_dir, default_questions_dir


def _load_questions(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise click.ClickException(f"no question set at {path}; run questions.generate first")
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@click.command()
@click.option(
    "--questions-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=default_questions_dir,
    help="Directory holding questions.jsonl.",
)
@click.option(
    "--filings-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=default_filings_dir,
    help="Directory of cached filing text.",
)
def main(questions_dir: Path, filings_dir: Path) -> None:
    questions = _load_questions(questions_dir / "questions.jsonl")
    store = FilingStore(filings_dir)
    rows = iter_answerable(questions)

    annotations: list[PassageAnnotation] = []
    skipped: list[SkippedQuestion] = []
    chunk_cache: dict[str, list[Chunk]] = {}

    def chunks_for(adsh: str) -> list[Chunk]:
        if adsh not in chunk_cache:
            try:
                chunk_cache[adsh] = list(store.chunks(adsh))
            except ToolError as error:
                chunk_cache[adsh] = []
                click.echo(f"  {adsh}: {error}", err=True)
        return chunk_cache[adsh]

    for question_id, adsh, facts in rows:
        # A year-over-year question's inputs live in two different filings, so
        # the chunks are gathered per fact rather than per question.
        available = {fact_adsh: chunks_for(fact_adsh) for fact_adsh, _ in facts}
        result = annotate_question(question_id, adsh, facts, available)
        if isinstance(result, PassageAnnotation):
            annotations.append(result)
        else:
            skipped.append(result)

    annotation_set = AnnotationSet(annotations=annotations, skipped=skipped)
    written = annotation_set.write_jsonl(questions_dir / "passages.jsonl")
    skipped_written = annotation_set.write_skipped(questions_dir / "passages_skipped.jsonl")

    summary = {
        "answerable_questions": len(rows),
        "annotated": written,
        "skipped": skipped_written,
        "skipped_by_reason": annotation_set.counts_by_reason(),
        "median_chunks_per_question": (
            sorted(len(a.chunk_ids) for a in annotations)[len(annotations) // 2]
            if annotations
            else 0
        ),
        "median_facts_per_question": (
            sorted(len(a.fact_chunk_ids) for a in annotations)[len(annotations) // 2]
            if annotations
            else 0
        ),
    }
    (questions_dir / "passages_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    click.echo(f"answerable questions: {len(rows)}")
    click.echo(f"annotated: {written}  skipped: {skipped_written}")
    click.echo(f"skipped by reason: {summary['skipped_by_reason']}")
    click.echo(f"median chunks per annotated question: {summary['median_chunks_per_question']}")
    click.echo(f"median input facts per question: {summary['median_facts_per_question']}")

    if written == 0 and rows:
        click.echo("no question could be annotated; this is a failure, not a result", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
