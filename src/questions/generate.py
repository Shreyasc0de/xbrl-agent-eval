"""Command line entry point for question generation.

Run on the host machine after the DERA quarters are loaded:

    python -m questions.generate --db data/xbrl.db --out data/questions

Writes the question set and the dropped-question log side by side, because
neither is interpretable without the other. A set of 700 questions means one
thing if 200 candidates were dropped and something quite different if 20,000
were, and the log is what lets a reader tell.

The roster is verified against the loaded data before anything is generated. A
CIK that does not match the registrant name stops the run, since a wrong CIK
produces a question set that is internally consistent and about the wrong
companies.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import click

from core.paths import default_db_path, default_questions_dir
from questions.companies import ROSTER, RosterMismatchError, verify_roster
from questions.generators import GeneratorConfig, generate_all
from questions.repository import FactRepository

__all__ = ["main"]


@click.command()
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=default_db_path,
    show_default="~/xbrl-data/xbrl.db, or $XBRL_DB",
    help="SQLite ground truth built by the DERA loader.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=default_questions_dir,
    show_default="alongside the database",
    help="Directory for questions.jsonl, dropped.jsonl and summary.json.",
)
@click.option("--min-peers", default=3, show_default=True, help="L5: peers that must tag it.")
@click.option(
    "--peer-share",
    default=0.6,
    show_default=True,
    help="L5: share of comparable peers that must tag it.",
)
def main(db_path: Path, out_dir: Path, min_peers: int, peer_share: float) -> None:
    connection = sqlite3.connect(db_path)
    repository = FactRepository(connection)

    try:
        resolved = verify_roster(repository, ROSTER)
    except RosterMismatchError as error:
        click.echo(str(error), err=True)
        sys.exit(1)

    click.echo(f"roster verified: {len(resolved)} companies")

    config = GeneratorConfig(min_peers_tagging=min_peers, peer_tagging_share=peer_share)
    question_set = generate_all(repository, ROSTER, config)

    out_dir.mkdir(parents=True, exist_ok=True)
    questions_written = question_set.write_jsonl(out_dir / "questions.jsonl")
    drops_written = question_set.drop_log.write_jsonl(out_dir / "dropped.jsonl")

    summary = {
        "questions": questions_written,
        "by_level": {str(k): v for k, v in question_set.counts_by_level().items()},
        "l5_share": round(question_set.level_share(5), 4),
        "question_set_hash": question_set.content_hash(),
        "dropped": drops_written,
        "dropped_by_reason": question_set.drop_log.counts_by_reason(),
        "dropped_by_level": {str(k): v for k, v in question_set.drop_log.counts_by_level().items()},
        "config": {
            "min_peers_tagging": config.min_peers_tagging,
            "peer_tagging_share": config.peer_tagging_share,
        },
        "companies": resolved,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    click.echo(f"questions: {questions_written}  dropped: {drops_written}")
    click.echo(f"by level: {summary['by_level']}")
    click.echo(f"L5 share: {summary['l5_share']:.1%}")
    click.echo(f"written to {out_dir}")


if __name__ == "__main__":
    main()
