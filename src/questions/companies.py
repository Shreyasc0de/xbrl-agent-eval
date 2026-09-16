"""The benchmark roster, and the check that confirms it against the data.

Twenty companies across five industries. The mix is the point: a bank tags
almost nothing the way a retailer does, and a benchmark drawn from one industry
measures how well an agent reads one tagging convention rather than how well it
reads filings.

**The CIKs here are unverified candidates.** They were written from memory, not
read from SEC, so every one is checked against ``sub.txt`` by
``verify_roster`` before any question is generated. A mismatch raises. This
matters more than it looks: a wrong CIK does not crash, it silently benchmarks a
different company, and every number downstream would be real, reproducible, and
about the wrong filing.

SIC codes are deliberately absent. Peer groups are derived from the ``sic``
column in the loaded data rather than from anything asserted here, so the peer
logic behind L5 rests on SEC's classification and not on mine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from questions.repository import FactRepository

__all__ = ["ROSTER", "Company", "RosterMismatchError", "verify_roster"]


@dataclass(frozen=True)
class Company:
    """One benchmark company.

    ``name_contains`` is the match used to confirm ``cik``. It is a substring of
    the registrant name as it appears in ``sub.txt``, uppercased, chosen to be
    specific enough to be wrong loudly if the CIK is wrong.
    """

    ticker: str
    display_name: str
    cik: int
    name_contains: str
    industry: str
    cik_verified: bool = False


ROSTER: Final[tuple[Company, ...]] = (
    # Technology
    Company("AAPL", "Apple Inc.", 320193, "APPLE", "technology"),
    Company("MSFT", "Microsoft Corporation", 789019, "MICROSOFT", "technology"),
    Company("NVDA", "NVIDIA Corporation", 1045810, "NVIDIA", "technology"),
    Company("GOOGL", "Alphabet Inc.", 1652044, "ALPHABET", "technology"),
    Company("AMZN", "Amazon.com, Inc.", 1018724, "AMAZON", "technology"),
    Company("INTC", "Intel Corporation", 50863, "INTEL", "technology"),
    # Banking
    Company("JPM", "JPMorgan Chase & Co.", 19617, "JPMORGAN", "banking"),
    Company("BAC", "Bank of America Corporation", 70858, "BANK OF AMERICA", "banking"),
    Company("WFC", "Wells Fargo & Company", 72971, "WELLS FARGO", "banking"),
    # Retail
    Company("WMT", "Walmart Inc.", 104169, "WALMART", "retail"),
    Company("COST", "Costco Wholesale Corporation", 909832, "COSTCO", "retail"),
    Company("TGT", "Target Corporation", 27419, "TARGET", "retail"),
    # Pharmaceuticals and biotechnology
    Company("PFE", "Pfizer Inc.", 78003, "PFIZER", "pharmaceuticals"),
    Company("MRK", "Merck & Co., Inc.", 310158, "MERCK", "pharmaceuticals"),
    Company("AMGN", "Amgen Inc.", 318154, "AMGEN", "pharmaceuticals"),
    Company("GILD", "Gilead Sciences, Inc.", 882095, "GILEAD", "pharmaceuticals"),
    # Industrials
    Company("CAT", "Caterpillar Inc.", 18230, "CATERPILLAR", "industrials"),
    Company("DE", "Deere & Company", 315189, "DEERE", "industrials"),
    Company("HON", "Honeywell International Inc.", 773840, "HONEYWELL", "industrials"),
    Company("MMM", "3M Company", 66740, "3M ", "industrials"),
)


class RosterMismatchError(RuntimeError):
    """A roster CIK does not match the registrant name in the loaded data."""


def verify_roster(
    repository: FactRepository, roster: tuple[Company, ...] = ROSTER
) -> dict[str, str]:
    """Confirm every roster CIK against ``sub.txt``. Raises on any mismatch.

    Returns the registrant name found for each ticker, so the caller can record
    in the run manifest exactly which entities the questions were built from.

    A company absent from the loaded quarters is a mismatch, not a warning. It
    means the roster and the data disagree about what this benchmark covers, and
    generating questions anyway would produce a set whose composition does not
    match what the README claims.
    """
    resolved: dict[str, str] = {}
    problems: list[str] = []

    for company in roster:
        submissions = repository.submissions_for_cik(company.cik)
        if not submissions:
            problems.append(
                f"{company.ticker}: CIK {company.cik} has no submissions in the loaded data"
            )
            continue

        found_name = submissions[0].name
        if company.name_contains.strip() not in found_name.upper():
            problems.append(
                f"{company.ticker}: CIK {company.cik} resolves to {found_name!r}, "
                f"which does not contain {company.name_contains.strip()!r}"
            )
            continue

        resolved[company.ticker] = found_name

    if problems:
        raise RosterMismatchError(
            "roster does not match the loaded data:\n  " + "\n  ".join(problems)
        )
    return resolved


def _validate_roster() -> None:
    tickers = [company.ticker for company in ROSTER]
    if len(set(tickers)) != len(tickers):
        raise ValueError("duplicate ticker in roster")
    ciks = [company.cik for company in ROSTER]
    if len(set(ciks)) != len(ciks):
        raise ValueError("duplicate CIK in roster")

    # The L5 peer logic needs a peer group with more than one other member, so
    # an industry represented by one or two companies cannot support it.
    by_industry: dict[str, int] = {}
    for company in ROSTER:
        by_industry[company.industry] = by_industry.get(company.industry, 0) + 1
    thin = {name: count for name, count in by_industry.items() if count < 3}
    if thin:
        raise ValueError(
            f"industries with fewer than three companies cannot support peer-based "
            f"L5 generation: {thin}"
        )


_validate_roster()
