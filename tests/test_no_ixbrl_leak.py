"""The boundary invariant.

Ground truth comes from structured XBRL. The agent under test only ever sees the
human-readable filing. If a tagged value or its machine-readable scaffolding
survives preprocessing, the agent can read the answer key out of its own context
and every result the benchmark has ever produced is silently invalid.

This is the first test in the repo and the one that must never be relaxed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ingest.ixbrl import IXBRLLeakError, assert_no_ixbrl_leak, strip_ixbrl

FIXTURE = Path(__file__).parent / "fixtures" / "sample_ixbrl.html"


@pytest.fixture(scope="module")
def raw_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def stripped(raw_html: str) -> str:
    return strip_ixbrl(raw_html)


def test_fixture_actually_contains_what_we_claim_to_strip(raw_html: str) -> None:
    """Guard against the fixture drifting into something the stripper trivially passes."""
    assert "ix:nonFraction" in raw_html
    assert "contextRef" in raw_html
    assert "unitRef" in raw_html
    assert "ix:hidden" in raw_html


@pytest.mark.parametrize(
    "forbidden",
    [
        "ix:",
        "contextRef",
        "unitRef",
        "xbrli:",
        "us-gaap:",
        "xlink:",
        "schemaRef",
    ],
)
def test_no_machine_readable_token_survives(stripped: str, forbidden: str) -> None:
    assert forbidden not in stripped, f"{forbidden!r} leaked into agent-visible text"


def test_no_ix_namespace_declaration_survives(stripped: str) -> None:
    assert not re.search(r"xmlns:\s*ix", stripped, flags=re.IGNORECASE)
    assert "inlineXBRL" not in stripped


@pytest.mark.parametrize("attr", ["scale", "decimals", "format", "sign"])
def test_no_fact_attribute_survives(stripped: str, attr: str) -> None:
    assert not re.search(rf"\b{attr}\s*=", stripped), f"{attr!r} attribute leaked"


def test_hidden_facts_are_removed_entirely(stripped: str) -> None:
    """ix:hidden holds tagged values a human reader never sees on the page.

    R&D expense appears in the fixture only inside ix:hidden, and the fixture's
    prose states it is not presented on the face of the statements. If the value
    survives, the agent can read a figure no human reader of this filing could.
    """
    assert "913" not in stripped
    assert "ResearchAndDevelopmentExpense" not in stripped


def test_displayed_values_do_survive(stripped: str) -> None:
    """The stripper must not be trivially correct by deleting the filing."""
    assert "4,000" in stripped
    assert "2,500" in stripped
    assert "812,400" in stripped
    assert "Total net sales" in stripped
    assert "Example Industrial Corp" in stripped


def test_assert_no_ixbrl_leak_rejects_raw_filing(raw_html: str) -> None:
    with pytest.raises(IXBRLLeakError):
        assert_no_ixbrl_leak(raw_html)


def test_assert_no_ixbrl_leak_accepts_stripped_filing(stripped: str) -> None:
    assert_no_ixbrl_leak(stripped)


def test_stripper_is_idempotent(stripped: str) -> None:
    assert strip_ixbrl(stripped) == stripped
