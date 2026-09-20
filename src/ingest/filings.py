"""Fetch filings from EDGAR, strip the XBRL out of them, and cache the text.

This is the pipeline that produces what the agent reads. Everything it writes
lands in ``data/filings/``, and nothing under ``src/agents/`` reads anything
else. The database on the other side of the boundary is never touched here.

The order of operations matters and is not negotiable: fetch, strip, assert
clean, convert to text, assert clean again, then write. The assertion runs twice
because the conversion step is itself a place where machine-readable markup
could reappear as visible text, and a leak that reaches disk is a leak that
reaches every run afterwards.

Nothing is written on failure. A filing that could not be fetched or could not
be cleaned has no cache entry at all, so a later run retries it rather than
serving an empty document that looks like a filing with nothing in it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from ingest.edgar import EdgarClient, EdgarFetchError
from ingest.ixbrl import IXBRLLeakError, filing_to_text, strip_ixbrl

__all__ = [
    "FilingRef",
    "CachedFiling",
    "FilingCacheError",
    "cache_filing",
    "cache_filings_for_cik",
    "filing_text_path",
    "list_filings",
    "parse_submissions",
    "primary_document_url",
    "submissions_url",
]

SUBMISSIONS_URL: Final[str] = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL: Final[str] = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
)
DEFAULT_FORMS: Final[tuple[str, ...]] = ("10-K", "10-Q")


class FilingCacheError(RuntimeError):
    """A filing could not be fetched or could not be cleaned."""


@dataclass(frozen=True)
class FilingRef:
    """One filing, identified well enough to fetch it."""

    cik: int
    adsh: str
    form: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def accession_nodash(self) -> str:
        return self.adsh.replace("-", "")

    @property
    def url(self) -> str:
        return primary_document_url(self.cik, self.adsh, self.primary_document)


@dataclass(frozen=True)
class CachedFiling:
    """A filing that reached disk, with enough detail to audit it later."""

    cik: int
    adsh: str
    form: str
    filing_date: str
    source_url: str
    text_path: str
    raw_bytes: int
    text_bytes: int
    text_sha256: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def submissions_url(cik: int) -> str:
    return SUBMISSIONS_URL.format(cik=cik)


def primary_document_url(cik: int, adsh: str, document: str) -> str:
    return ARCHIVE_URL.format(cik=cik, accession_nodash=adsh.replace("-", ""), document=document)


def parse_submissions(
    payload: dict[str, Any], cik: int, forms: Sequence[str] = DEFAULT_FORMS
) -> list[FilingRef]:
    """Pull filing references out of one submissions JSON document.

    EDGAR splits a long filing history across a ``recent`` block and a list of
    older files. This parses whichever blocks it is given; fetching the older
    files is the caller's job, so this stays a pure function and stays testable
    without a network.

    A row missing a primary document is skipped. There is nothing to fetch for
    it, and inventing a filename would produce a 404 that looks like a fetch
    failure rather than a data gap.
    """
    wanted = {form.upper() for form in forms}
    blocks: list[dict[str, Any]] = []

    filings = payload.get("filings")
    if isinstance(filings, dict) and isinstance(filings.get("recent"), dict):
        blocks.append(filings["recent"])
    elif "accessionNumber" in payload:
        blocks.append(payload)

    references: list[FilingRef] = []
    for block in blocks:
        accessions = block.get("accessionNumber") or []
        form_types = block.get("form") or []
        filing_dates = block.get("filingDate") or []
        report_dates = block.get("reportDate") or []
        documents = block.get("primaryDocument") or []

        for index, accession in enumerate(accessions):
            form = form_types[index] if index < len(form_types) else ""
            if form.upper() not in wanted:
                continue
            document = documents[index] if index < len(documents) else ""
            if not document:
                continue
            references.append(
                FilingRef(
                    cik=cik,
                    adsh=accession,
                    form=form,
                    filing_date=filing_dates[index] if index < len(filing_dates) else "",
                    report_date=report_dates[index] if index < len(report_dates) else "",
                    primary_document=document,
                )
            )
    return references


def older_submission_urls(payload: dict[str, Any]) -> list[str]:
    filings = payload.get("filings")
    if not isinstance(filings, dict):
        return []
    files = filings.get("files")
    if not isinstance(files, list):
        return []
    return [
        f"https://data.sec.gov/submissions/{entry['name']}"
        for entry in files
        if isinstance(entry, dict) and entry.get("name")
    ]


def list_filings(
    client: EdgarClient,
    cik: int,
    forms: Sequence[str] = DEFAULT_FORMS,
    *,
    include_older: bool = True,
) -> list[FilingRef]:
    """Every filing of the requested forms for one company, newest block first."""
    payload = json.loads(client.get(submissions_url(cik)).decode("utf-8"))
    references = parse_submissions(payload, cik, forms)

    if include_older:
        for url in older_submission_urls(payload):
            older = json.loads(client.get(url).decode("utf-8"))
            references.extend(parse_submissions(older, cik, forms))

    seen: set[str] = set()
    unique: list[FilingRef] = []
    for reference in references:
        if reference.adsh in seen:
            continue
        seen.add(reference.adsh)
        unique.append(reference)
    return unique


def filing_text_path(filings_dir: Path, cik: int, adsh: str) -> Path:
    return filings_dir / f"{cik:010d}" / f"{adsh}.txt"


def cache_filing(
    client: EdgarClient, reference: FilingRef, filings_dir: Path, *, force: bool = False
) -> CachedFiling:
    """Fetch one filing, clean it, and write the text the agent will read.

    Raises ``FilingCacheError`` if the document cannot be fetched or if stripping
    leaves any XBRL behind. Nothing is written in either case.
    """
    destination = filing_text_path(filings_dir, reference.cik, reference.adsh)

    if destination.exists() and not force:
        text = destination.read_text(encoding="utf-8")
        return CachedFiling(
            cik=reference.cik,
            adsh=reference.adsh,
            form=reference.form,
            filing_date=reference.filing_date,
            source_url=reference.url,
            text_path=str(destination),
            raw_bytes=0,
            text_bytes=len(text.encode("utf-8")),
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )

    raw = client.get(reference.url)

    try:
        # Bytes, not a decoded string: the document carries its own encoding
        # declaration and lxml should be the thing that reads it.
        text = filing_to_text(strip_ixbrl(raw))
    except IXBRLLeakError as error:
        raise FilingCacheError(
            f"{reference.adsh}: XBRL survived preprocessing, refusing to cache it. "
            f"An agent reading this document could read the answer key. {error}"
        ) from error
    except ValueError as error:
        raise FilingCacheError(f"{reference.adsh}: could not parse filing. {error}") from error

    if not text.strip():
        raise FilingCacheError(
            f"{reference.adsh}: stripped to an empty document. Caching this would "
            "later be indistinguishable from a filing with no content."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)

    return CachedFiling(
        cik=reference.cik,
        adsh=reference.adsh,
        form=reference.form,
        filing_date=reference.filing_date,
        source_url=reference.url,
        text_path=str(destination),
        raw_bytes=len(raw),
        text_bytes=len(text.encode("utf-8")),
        text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def cache_filings_for_cik(
    client: EdgarClient,
    cik: int,
    filings_dir: Path,
    forms: Sequence[str] = DEFAULT_FORMS,
    *,
    limit: int | None = None,
    only_adsh: set[str] | None = None,
) -> Iterator[CachedFiling | FilingCacheError]:
    """Cache filings for a company, yielding a result or an error per filing.

    ``only_adsh`` restricts the fetch to accession numbers that exist in the
    loaded ground truth. Without it this fetches whatever EDGAR lists most
    recently, which is not the same set: the database holds the quarters that
    were loaded, and EDGAR's newest filings can be a year or more later. Caching
    filings the answer key knows nothing about produces a benchmark where the
    agent is asked about documents it has and graded against facts drawn from
    documents it does not.

    Errors are yielded rather than raised so that one unparseable filing does
    not abandon the other nineteen companies. The caller decides what an
    acceptable failure count is, and records it. A partial run is fine; a run
    that hides which filings failed is not.

    Only fetch and cleaning failures are turned into yielded errors. Anything
    else propagates, because a bug in this module should stop the run rather
    than be logged as if EDGAR had misbehaved.
    """
    references = list_filings(client, cik, forms)
    if only_adsh is not None:
        references = [r for r in references if r.adsh in only_adsh]
    if limit is not None:
        references = references[:limit]

    for reference in references:
        try:
            yield cache_filing(client, reference, filings_dir)
        except FilingCacheError as error:
            yield error
        except EdgarFetchError as error:
            yield FilingCacheError(f"{reference.adsh}: fetch failed. {error}")


# ---------------------------------------------------------------- command line


def _cli() -> None:
    """Fetch and clean the filings the agent will read.

        python -m ingest.filings --db data/xbrl.db --out data/filings

    Companies come from the verified roster, so this runs after the DERA load.
    """
    import sqlite3

    import click

    from core.paths import default_db_path, default_filings_dir
    from ingest.edgar import client_from_env
    from questions.companies import ROSTER, RosterMismatchError, verify_roster
    from questions.repository import FactRepository

    @click.command()
    @click.option(
        "--db",
        "db_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=default_db_path,
        show_default="~/xbrl-data/xbrl.db, or $XBRL_DB",
    )
    @click.option(
        "--out",
        "filings_dir",
        type=click.Path(file_okay=False, path_type=Path),
        default=default_filings_dir,
        show_default="alongside the database",
    )
    @click.option("--forms", default="10-K,10-Q", show_default=True)
    @click.option("--limit", type=int, default=None, help="Filings per company, newest first.")
    def command(db_path: Path, filings_dir: Path, forms: str, limit: int | None) -> None:
        repository = FactRepository(sqlite3.connect(db_path))
        try:
            verify_roster(repository)
        except RosterMismatchError as error:
            raise click.ClickException(str(error)) from error

        wanted = tuple(form.strip() for form in forms.split(",") if form.strip())

        # The filings the agent may read are exactly the ones the answer key
        # covers. Anything else would be a document with no facts behind it.
        covered: dict[int, set[str]] = {
            company.cik: {
                submission.adsh
                for submission in repository.submissions_for_cik(company.cik, wanted)
            }
            for company in ROSTER
        }

        total_covered = sum(len(v) for v in covered.values())
        click.echo(f"ground truth covers {total_covered} filings across the roster")
        client = client_from_env(filings_dir / ".http-cache")
        if total_covered == 0:
            raise click.ClickException(
                "no roster filings in the loaded quarters; load the quarters that "
                "contain the filings you want to benchmark"
            )

        cached = 0
        failures: list[str] = []
        manifest = filings_dir / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)

        # The agent side of the boundary cannot import the roster, because the
        # roster module reaches the XBRL database. So the company index it needs
        # to turn a name in a question into a CIK is written out here, on this
        # side, as plain data.
        (filings_dir / "companies.json").write_text(
            json.dumps(
                [
                    {
                        "cik": company.cik,
                        "ticker": company.ticker,
                        "display_name": company.display_name,
                    }
                    for company in ROSTER
                ],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        with manifest.open("w", encoding="utf-8") as handle:
            for company in ROSTER:
                click.echo(f"{company.ticker}: fetching")
                for result in cache_filings_for_cik(
                    client,
                    company.cik,
                    filings_dir,
                    wanted,
                    limit=limit,
                    only_adsh=covered[company.cik],
                ):
                    if isinstance(result, FilingCacheError):
                        failures.append(f"{company.ticker}: {result}")
                        continue
                    handle.write(json.dumps(result.to_json(), sort_keys=True))
                    handle.write("\n")
                    cached += 1

        click.echo(f"cached {cached} filings, {len(failures)} failures")
        for failure in failures:
            click.echo(f"  FAILED {failure}", err=True)
        click.echo(f"manifest: {manifest}")

    command()


if __name__ == "__main__":
    _cli()
