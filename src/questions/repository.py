"""Read-only access to the XBRL ground truth database.

This module and everything above it in ``src/questions/`` read ``data/xbrl.db``.
Nothing under ``src/agents/`` may import from here. That is the benchmark's one
invariant, and the import direction is how it is enforced.

Every method is a query. No method caches across calls, because the database is
written once by the loader and read afterwards, so staleness is not a risk and
a cache would only hide which queries are actually expensive.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

__all__ = ["FactRepository", "NumFact", "Submission"]

# The consolidated figure for the parent registrant, and nothing else.
#
# A non-empty coreg marks a co-registrant's figures, which sit in the same
# filing under a different reporting entity. A non-empty segments marks a
# dimensional breakdown: revenue for one product line, one geography, one
# reportable segment. Both share every other identifying column with the
# consolidated fact, so a query that does not exclude them can return a slice
# of a figure as though it were the whole, which is a wrong answer key that
# looks entirely plausible.
_CONSOLIDATED_ONLY: Final[str] = "coreg = '' AND segments = ''"


def _consolidated_only(prefix: str = "") -> str:
    """The consolidated-only predicate, optionally table-qualified."""
    if not prefix:
        return _CONSOLIDATED_ONLY
    return f"{prefix}.coreg = '' AND {prefix}.segments = ''"


@dataclass(frozen=True)
class Submission:
    adsh: str
    cik: int
    name: str
    sic: int | None
    form: str
    period: int | None
    fy: int
    fp: str
    filed: int
    prevrpt: int

    @property
    def sic_major_group(self) -> int | None:
        """First two SIC digits, the usual peer grouping granularity.

        Full four-digit SIC is too narrow to produce peer groups of useful size
        across a twenty-company benchmark.
        """
        if self.sic is None:
            return None
        return self.sic // 100


@dataclass(frozen=True)
class NumFact:
    adsh: str
    tag: str
    version: str
    ddate: int
    qtrs: int
    uom: str
    value: float | None


class FactRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    # ---------------------------------------------------------------- submissions

    def submissions_for_cik(
        self, cik: int, forms: tuple[str, ...] = ("10-K", "10-Q")
    ) -> list[Submission]:
        placeholders = ", ".join("?" for _ in forms)
        rows = self._connection.execute(
            f"""
            SELECT adsh, cik, name, sic, form, period, fy, fp, filed, prevrpt
            FROM sub
            WHERE cik = ? AND form IN ({placeholders}) AND fy IS NOT NULL AND fp IS NOT NULL
            ORDER BY filed
            """,
            (cik, *forms),
        ).fetchall()
        return [Submission(*row) for row in rows]

    def submission(self, adsh: str) -> Submission | None:
        row = self._connection.execute(
            """
            SELECT adsh, cik, name, sic, form, period, fy, fp, filed, prevrpt
            FROM sub WHERE adsh = ?
            """,
            (adsh,),
        ).fetchone()
        return Submission(*row) if row else None

    def peer_ciks(self, sic_major_group: int, exclude_cik: int) -> list[int]:
        """Companies sharing a SIC major group, used to establish expectation.

        Drawn from the whole loaded database rather than only the benchmark
        roster, because a wider peer set makes "this concept is commonly tagged
        in this industry" a stronger claim.
        """
        rows = self._connection.execute(
            """
            SELECT DISTINCT cik FROM sub
            WHERE sic IS NOT NULL AND sic / 100 = ? AND cik != ?
            """,
            (sic_major_group, exclude_cik),
        ).fetchall()
        return [row[0] for row in rows]

    # ---------------------------------------------------------------- facts

    def facts_for_tags(
        self, adsh: str, tags: tuple[str, ...], ddate: int, qtrs: int
    ) -> list[NumFact]:
        placeholders = ", ".join("?" for _ in tags)
        rows = self._connection.execute(
            f"""
            SELECT adsh, tag, version, ddate, qtrs, uom, value
            FROM num
            WHERE adsh = ? AND tag IN ({placeholders}) AND ddate = ? AND qtrs = ?
              AND {_CONSOLIDATED_ONLY}
            ORDER BY tag
            """,
            (adsh, *tags, ddate, qtrs),
        ).fetchall()
        return [NumFact(*row) for row in rows]

    def tags_present(self, adsh: str) -> set[str]:
        rows = self._connection.execute(
            f"SELECT DISTINCT tag FROM num WHERE adsh = ? AND {_CONSOLIDATED_ONLY}",
            (adsh,),
        ).fetchall()
        return {row[0] for row in rows}

    def period_ends(self, adsh: str, qtrs: int) -> list[int]:
        rows = self._connection.execute(
            f"""
            SELECT DISTINCT ddate FROM num
            WHERE adsh = ? AND qtrs = ? AND {_CONSOLIDATED_ONLY}
            ORDER BY ddate DESC
            """,
            (adsh, qtrs),
        ).fetchall()
        return [row[0] for row in rows]

    # ---------------------------------------------------------------- metadata

    def is_custom_tag(self, tag: str, version: str) -> bool:
        """True for a company extension tag, and for a tag with no metadata row.

        An unknown tag is treated as custom on purpose. The alternative is to
        assume it is standard, which would let an unrecognized extension into
        the question set.
        """
        row = self._connection.execute(
            "SELECT custom FROM tag WHERE tag = ? AND version = ?", (tag, version)
        ).fetchone()
        if row is None:
            return True
        return bool(row[0])

    def is_abstract_tag(self, tag: str, version: str) -> bool:
        row = self._connection.execute(
            "SELECT abstract FROM tag WHERE tag = ? AND version = ?", (tag, version)
        ).fetchone()
        return bool(row[0]) if row is not None else False

    def negating(self, adsh: str, tag: str, version: str) -> bool:
        """Whether the presentation layer flips this fact's sign.

        A fact with no presentation row is not negated. That is a real case:
        facts can be tagged without appearing on the face of a statement.
        Several presentation rows disagreeing is treated as negating only if
        every row agrees, since a concept presented both ways in one filing has
        no single as-presented sign.
        """
        rows = self._connection.execute(
            "SELECT negating FROM pre WHERE adsh = ? AND tag = ? AND version = ?",
            (adsh, tag, version),
        ).fetchall()
        flags = {bool(row[0]) for row in rows if row[0] is not None}
        return flags == {True}

    def later_values_for_fact(
        self, cik: int, tag: str, ddate: int, qtrs: int, filed_after: int
    ) -> list[float]:
        """Values for the same fact reported in submissions filed later.

        Used to detect restatements. Identical values are not restatements: the
        same figure legitimately reappears in a later 10-Q or 10-K comparative
        column. Only a changed value means the company restated.
        """
        rows = self._connection.execute(
            f"""
            SELECT n.value
            FROM num n
            JOIN sub s ON s.adsh = n.adsh
            WHERE s.cik = ? AND n.tag = ? AND n.ddate = ? AND n.qtrs = ?
              AND s.filed > ? AND n.value IS NOT NULL AND {_consolidated_only("n")}
            """,
            (cik, tag, ddate, qtrs, filed_after),
        ).fetchall()
        return [row[0] for row in rows]
