"""The pipeline from an accession number to the text an agent reads.

No test here touches the network. The transport is injected throughout, so this
runs in CI and in a sandbox where SEC is blocked.

The heart of it is that a cached filing is clean. The iXBRL leak test proves the
stripper works on a document handed to it; these tests prove the stripper is
actually in the path between EDGAR and disk, which is a separate claim and the
one that was missing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from ingest.edgar import EdgarClient, EdgarFetchError, HttpResponse, RateLimiter
from ingest.filings import (
    FilingCacheError,
    FilingRef,
    cache_filing,
    cache_filings_for_cik,
    filing_text_path,
    list_filings,
    parse_submissions,
    primary_document_url,
    submissions_url,
)
from ingest.ixbrl import assert_no_ixbrl_leak

UA = "xbrl-agent-eval someone@example.com"
FIXTURE = Path(__file__).parent / "fixtures" / "sample_ixbrl.html"

SUBMISSIONS = {
    "cik": "1750",
    "filings": {
        "recent": {
            "accessionNumber": ["0000001750-25-000001", "0000001750-24-000009"],
            "form": ["10-K", "8-K"],
            "filingDate": ["2025-02-14", "2024-11-01"],
            "reportDate": ["2024-12-31", "2024-10-30"],
            "primaryDocument": ["exmp-20241231.htm", "exmp-8k.htm"],
        },
        "files": [{"name": "CIK0000001750-submissions-001.json"}],
    },
}

OLDER = {
    "accessionNumber": ["0000001750-24-000001"],
    "form": ["10-K"],
    "filingDate": ["2024-02-15"],
    "reportDate": ["2023-12-31"],
    "primaryDocument": ["exmp-20231231.htm"],
}


class RoutingTransport:
    """Serves a canned response per URL, and records what was requested."""

    def __init__(self, routes: dict[str, HttpResponse]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        self.calls.append(url)
        if url in self.routes:
            return self.routes[url]
        return HttpResponse(404, b"")


def client(tmp_path: Path, transport: RoutingTransport) -> EdgarClient:
    return EdgarClient(
        UA,
        tmp_path / "http",
        transport=transport,
        rate_limiter=RateLimiter(1000.0, monotonic=lambda: 0.0, sleep=lambda _: None),
        sleep=lambda _: None,
    )


def ok_json(payload: object) -> HttpResponse:
    return HttpResponse(200, json.dumps(payload).encode("utf-8"))


def ref() -> FilingRef:
    return FilingRef(
        cik=1750,
        adsh="0000001750-25-000001",
        form="10-K",
        filing_date="2025-02-14",
        report_date="2024-12-31",
        primary_document="exmp-20241231.htm",
    )


# --------------------------------------------------------------------------
# URLs and parsing
# --------------------------------------------------------------------------


def test_submissions_url_zero_pads_the_cik() -> None:
    assert submissions_url(1750) == "https://data.sec.gov/submissions/CIK0000001750.json"


def test_archive_url_strips_dashes_from_the_accession_number() -> None:
    url = primary_document_url(1750, "0000001750-25-000001", "exmp.htm")
    assert "000000175025000001" in url
    assert "0000001750-25-000001" not in url


def test_parse_submissions_keeps_only_the_requested_forms() -> None:
    references = parse_submissions(SUBMISSIONS, 1750, ("10-K",))
    assert [r.adsh for r in references] == ["0000001750-25-000001"]


def test_parse_submissions_skips_rows_with_no_primary_document() -> None:
    """Inventing a filename would produce a 404 that reads as a fetch failure

    rather than as the data gap it actually is.
    """
    payload = {
        "filings": {
            "recent": {
                "accessionNumber": ["0000001750-25-000002"],
                "form": ["10-K"],
                "filingDate": ["2025-02-14"],
                "reportDate": ["2024-12-31"],
                "primaryDocument": [""],
            }
        }
    }
    assert parse_submissions(payload, 1750, ("10-K",)) == []


def test_list_filings_follows_the_older_submission_files(tmp_path: Path) -> None:
    transport = RoutingTransport(
        {
            submissions_url(1750): ok_json(SUBMISSIONS),
            "https://data.sec.gov/submissions/CIK0000001750-submissions-001.json": ok_json(OLDER),
        }
    )
    references = list_filings(client(tmp_path, transport), 1750, ("10-K",))
    assert [r.adsh for r in references] == [
        "0000001750-25-000001",
        "0000001750-24-000001",
    ]


def test_list_filings_deduplicates_by_accession_number(tmp_path: Path) -> None:
    transport = RoutingTransport(
        {
            submissions_url(1750): ok_json(SUBMISSIONS),
            "https://data.sec.gov/submissions/CIK0000001750-submissions-001.json": ok_json(
                {
                    "accessionNumber": ["0000001750-25-000001"],
                    "form": ["10-K"],
                    "filingDate": ["2025-02-14"],
                    "reportDate": ["2024-12-31"],
                    "primaryDocument": ["exmp-20241231.htm"],
                }
            ),
        }
    )
    references = list_filings(client(tmp_path, transport), 1750, ("10-K",))
    assert len(references) == 1


# --------------------------------------------------------------------------
# The boundary, enforced in the pipeline and not only in the stripper
# --------------------------------------------------------------------------


def test_a_cached_filing_contains_no_xbrl(tmp_path: Path) -> None:
    """The leak test proves the stripper works. This proves it is in the path.

    Those are different claims, and only the second one protects a run.
    """
    raw = FIXTURE.read_bytes()
    transport = RoutingTransport({ref().url: HttpResponse(200, raw)})
    result = cache_filing(client(tmp_path, transport), ref(), tmp_path / "filings")

    text = Path(result.text_path).read_text(encoding="utf-8")
    assert_no_ixbrl_leak(text)
    assert "contextRef" not in text
    assert "ix:" not in text


def test_a_cached_filing_drops_the_hidden_facts(tmp_path: Path) -> None:
    """R&D appears in the fixture only inside ix:hidden, and the fixture says it

    is not presented on the face of the statements. If it reaches disk, every
    agent reading this filing can read a figure no human reader could.
    """
    transport = RoutingTransport({ref().url: HttpResponse(200, FIXTURE.read_bytes())})
    result = cache_filing(client(tmp_path, transport), ref(), tmp_path / "filings")
    text = Path(result.text_path).read_text(encoding="utf-8")
    assert "913" not in text


def test_a_cached_filing_keeps_the_displayed_figures(tmp_path: Path) -> None:
    """The pipeline must not be trivially safe by caching an empty document."""
    transport = RoutingTransport({ref().url: HttpResponse(200, FIXTURE.read_bytes())})
    result = cache_filing(client(tmp_path, transport), ref(), tmp_path / "filings")
    text = Path(result.text_path).read_text(encoding="utf-8")
    assert "4,000" in text
    assert "Total net sales" in text


def test_a_statement_line_keeps_its_label_and_figure_together(tmp_path: Path) -> None:
    """A converter that flattens the table destroys the association between a

    line label and its number, which is most of what the agent needs to read.
    """
    transport = RoutingTransport({ref().url: HttpResponse(200, FIXTURE.read_bytes())})
    result = cache_filing(client(tmp_path, transport), ref(), tmp_path / "filings")
    text = Path(result.text_path).read_text(encoding="utf-8")
    line = next(line for line in text.split("\n") if "Total net sales" in line)
    assert "4,000" in line


# --------------------------------------------------------------------------
# Failure writes nothing
# --------------------------------------------------------------------------


def test_a_fetch_failure_caches_no_filing(tmp_path: Path) -> None:
    transport = RoutingTransport({})
    with pytest.raises(EdgarFetchError):
        cache_filing(client(tmp_path, transport), ref(), tmp_path / "filings")
    assert not filing_text_path(tmp_path / "filings", 1750, ref().adsh).exists()


def test_a_document_that_strips_to_nothing_is_refused(tmp_path: Path) -> None:
    """An empty cached file is later indistinguishable from a filing with no

    content, which is the silent-fallback failure mode in a different costume.
    """
    empty = b"<html><body><div style='display:none'>hidden</div></body></html>"
    transport = RoutingTransport({ref().url: HttpResponse(200, empty)})
    with pytest.raises(FilingCacheError, match="empty"):
        cache_filing(client(tmp_path, transport), ref(), tmp_path / "filings")
    assert not filing_text_path(tmp_path / "filings", 1750, ref().adsh).exists()


def test_no_partial_file_survives_a_successful_write(tmp_path: Path) -> None:
    transport = RoutingTransport({ref().url: HttpResponse(200, FIXTURE.read_bytes())})
    cache_filing(client(tmp_path, transport), ref(), tmp_path / "filings")
    assert list((tmp_path / "filings").rglob("*.partial")) == []


def test_one_bad_filing_does_not_abandon_the_rest(tmp_path: Path) -> None:
    """A partial run is fine. A run that hides which filings failed is not."""
    good = ref()
    transport = RoutingTransport(
        {
            submissions_url(1750): ok_json(SUBMISSIONS),
            "https://data.sec.gov/submissions/CIK0000001750-submissions-001.json": ok_json(OLDER),
            good.url: HttpResponse(200, FIXTURE.read_bytes()),
            # The older filing is deliberately absent from the routes, so it 404s.
        }
    )
    results = list(
        cache_filings_for_cik(client(tmp_path, transport), 1750, tmp_path / "filings", ("10-K",))
    )
    assert len(results) == 2
    assert sum(1 for r in results if isinstance(r, FilingCacheError)) == 1
    assert sum(1 for r in results if not isinstance(r, FilingCacheError)) == 1


# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------


def test_a_second_pass_does_not_refetch(tmp_path: Path) -> None:
    transport = RoutingTransport({ref().url: HttpResponse(200, FIXTURE.read_bytes())})
    edgar = client(tmp_path, transport)
    first = cache_filing(edgar, ref(), tmp_path / "filings")
    before = len(transport.calls)
    second = cache_filing(edgar, ref(), tmp_path / "filings")
    assert len(transport.calls) == before
    assert first.text_sha256 == second.text_sha256


def test_download_skips_a_file_already_on_disk(tmp_path: Path) -> None:
    """Re-running the data pull must not re-download gigabytes of archives."""
    url = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2024q4.zip"
    transport = RoutingTransport({url: HttpResponse(200, b"PK\x03\x04zipbytes")})
    edgar = client(tmp_path, transport)
    destination = tmp_path / "raw" / "2024q4.zip"

    edgar.download(url, destination)
    assert destination.read_bytes() == b"PK\x03\x04zipbytes"

    edgar.download(url, destination)
    assert len(transport.calls) == 1


def test_a_failed_download_leaves_nothing_behind(tmp_path: Path) -> None:
    url = "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2099q4.zip"
    edgar = client(tmp_path, RoutingTransport({}))
    raw_dir = tmp_path / "raw"
    destination = raw_dir / "2099q4.zip"
    with pytest.raises(EdgarFetchError):
        edgar.download(url, destination)
    assert not destination.exists()
    leftovers = list(raw_dir.glob("*.partial")) if raw_dir.exists() else []
    assert leftovers == []
