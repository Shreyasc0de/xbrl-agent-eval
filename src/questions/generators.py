"""Question generators for levels 1 through 5.

Questions are derived from facts, never written by hand. A generator's job is to
find a combination of facts that supports exactly one defensible answer, phrase
it, and record which facts the answer came from. When the facts do not support a
single answer, the candidate goes to the drop log instead.

One phrasing rule runs through all of this and is worth stating plainly, because
violating it would quietly destroy the headline result: **an unanswerable
question is phrased exactly like an answerable one.** If L5 questions read
differently, an agent can learn to abstain on the wording rather than on the
absence of the fact, and the measured hallucination rate becomes a measure of
prompt style. ``_ask`` is the single place question text is built, and every
level goes through it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

from eval.normalizers import PeriodType
from questions.companies import Company
from questions.concepts import CONCEPTS, RATIOS, ConceptSpec, RatioSpec
from questions.filters import MIN_DENOMINATOR_MAGNITUDE, resolve_concept
from questions.models import (
    DropLog,
    DropReason,
    FactRef,
    Question,
    QuestionSet,
    make_question_id,
)
from questions.repository import FactRepository, Submission

__all__ = [
    "GeneratorConfig",
    "generate_all",
    "generate_level_1",
    "generate_level_2",
    "generate_level_3",
    "generate_level_4",
    "generate_level_5",
]

_ANNUAL_QTRS: Final[int] = 4
_INSTANT_QTRS: Final[int] = 0


@dataclass(frozen=True)
class GeneratorConfig:
    """Knobs that affect what gets generated, recorded in the run manifest.

    ``min_peers_tagging`` and ``peer_tagging_share`` together decide when a
    concept counts as expected for an industry. Both are reported rather than
    tuned against results: loosening them inflates the L5 set with concepts no
    reader would actually expect, which would make the hallucination rate look
    worse for reasons that have nothing to do with the agent.
    """

    min_peers_tagging: int = 3
    peer_tagging_share: float = 0.6
    max_per_template: int | None = None

    def __post_init__(self) -> None:
        if self.min_peers_tagging < 2:
            raise ValueError("a peer group of fewer than two is not evidence of expectation")
        if not 0.0 < self.peer_tagging_share <= 1.0:
            raise ValueError("peer_tagging_share must be in (0, 1]")


def _period_label(submission: Submission) -> str:
    """How a period is named in a question.

    Fiscal, always, and never a calendar year. A question that says "in 2024"
    is ambiguous for every company whose fiscal year does not end in December.
    """
    if submission.fp == "FY":
        return f"fiscal year {submission.fy}"
    return f"{submission.fp} of fiscal year {submission.fy}"


def _ask(concept_label: str, company: str, submission: Submission, instant: bool) -> str:
    """Build question text. The only place question wording is constructed.

    Answerable and unanswerable questions are built here identically, so nothing
    about the phrasing distinguishes them.
    """
    period = _period_label(submission)
    if instant:
        return f"What was {company}'s {concept_label} as of the end of {period}?"
    return f"What was {company}'s {concept_label} for {period}?"


def _ddate_for(submission: Submission) -> int | None:
    return submission.period


def _qtrs_for(concept: ConceptSpec, submission: Submission) -> int:
    if concept.period_type is PeriodType.INSTANT:
        return _INSTANT_QTRS
    return _ANNUAL_QTRS if submission.fp == "FY" else 1


def _annual_submissions(submissions: Sequence[Submission]) -> list[Submission]:
    """Annual filings only, deduplicated to one per fiscal year.

    A company can file more than one 10-K touching the same fiscal year, for
    instance an amendment. The earliest filed one is the original, which is the
    document the benchmark treats as truth.
    """
    by_year: dict[int, Submission] = {}
    for submission in submissions:
        if submission.form != "10-K" or submission.fp != "FY" or submission.period is None:
            continue
        existing = by_year.get(submission.fy)
        if existing is None or submission.filed < existing.filed:
            by_year[submission.fy] = submission
    return [by_year[year] for year in sorted(by_year)]


# ------------------------------------------------------------------ level 1


def generate_level_1(
    repository: FactRepository,
    company: Company,
    submissions: Sequence[Submission],
    drop_log: DropLog,
) -> Iterator[Question]:
    """Single fact retrieval. One ``num.txt`` row is the answer."""
    template = "L1_single_fact"

    annual = _annual_submissions(submissions)
    if not annual:
        drop_log.record(
            DropReason.NO_ANNUAL_FILING,
            level=1,
            template_id=template,
            concept="*",
            cik=company.cik,
            detail=(
                "no 10-K in the loaded quarters; this company contributes no "
                "questions at any level"
            ),
        )

    for submission in annual:
        ddate = _ddate_for(submission)
        if ddate is None:
            continue

        for key in sorted(CONCEPTS):
            concept = CONCEPTS[key]
            qtrs = _qtrs_for(concept, submission)
            resolution = resolve_concept(repository, submission, concept, ddate, qtrs)

            if not resolution.ok:
                assert resolution.reason is not None
                drop_log.record(
                    resolution.reason,
                    level=1,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=submission.adsh,
                    detail=resolution.detail,
                )
                continue

            fact = resolution.fact
            assert fact is not None
            yield Question(
                question_id=make_question_id(
                    template, company.cik, submission.adsh, key, ddate, qtrs
                ),
                level=1,
                template_id=template,
                text=_ask(
                    concept.label,
                    company.display_name,
                    submission,
                    concept.period_type is PeriodType.INSTANT,
                ),
                cik=company.cik,
                company_name=company.display_name,
                primary_adsh=submission.adsh,
                fy=submission.fy,
                fp=submission.fp,
                answer_value=fact.value,
                answer_uom=concept.uom,
                concept=key,
                facts=(fact,),
            )


# ------------------------------------------------------------------ level 2


def _resolve_ratio_components(
    repository: FactRepository,
    submission: Submission,
    ratio: RatioSpec,
    ddate: int,
) -> tuple[FactRef, FactRef] | tuple[DropReason, str]:
    numerator_concept = CONCEPTS[ratio.numerator]
    denominator_concept = CONCEPTS[ratio.denominator]

    numerator = resolve_concept(
        repository,
        submission,
        numerator_concept,
        ddate,
        _qtrs_for(numerator_concept, submission),
    )
    if not numerator.ok:
        assert numerator.reason is not None
        return numerator.reason, f"numerator {ratio.numerator}: {numerator.detail}"

    denominator = resolve_concept(
        repository,
        submission,
        denominator_concept,
        ddate,
        _qtrs_for(denominator_concept, submission),
    )
    if not denominator.ok:
        assert denominator.reason is not None
        return denominator.reason, f"denominator {ratio.denominator}: {denominator.detail}"

    assert numerator.fact is not None
    assert denominator.fact is not None
    return numerator.fact, denominator.fact


def generate_level_2(
    repository: FactRepository,
    company: Company,
    submissions: Sequence[Submission],
    drop_log: DropLog,
) -> Iterator[Question]:
    """Derived ratios. Two facts from one filing, combined by a defined formula."""
    template = "L2_derived_ratio"

    for submission in _annual_submissions(submissions):
        ddate = _ddate_for(submission)
        if ddate is None:
            continue

        for key in sorted(RATIOS):
            ratio = RATIOS[key]
            resolved = _resolve_ratio_components(repository, submission, ratio, ddate)

            if isinstance(resolved[0], DropReason):
                reason, detail = resolved
                assert isinstance(reason, DropReason)
                drop_log.record(
                    reason,
                    level=2,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=submission.adsh,
                    detail=str(detail),
                )
                continue

            numerator_fact, denominator_fact = resolved
            assert isinstance(numerator_fact, FactRef)
            assert isinstance(denominator_fact, FactRef)

            if abs(denominator_fact.value) < MIN_DENOMINATOR_MAGNITUDE:
                drop_log.record(
                    DropReason.DENOMINATOR_TOO_SMALL,
                    level=2,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=submission.adsh,
                    detail=f"{ratio.denominator} = {denominator_fact.value}",
                )
                continue

            value = ratio.compute(numerator_fact.value, denominator_fact.value)
            label = f"{ratio.label} as a percentage" if ratio.as_percentage else ratio.label
            yield Question(
                question_id=make_question_id(template, company.cik, submission.adsh, key),
                level=2,
                template_id=template,
                text=_ask(label, company.display_name, submission, instant=False),
                cik=company.cik,
                company_name=company.display_name,
                primary_adsh=submission.adsh,
                fy=submission.fy,
                fp=submission.fp,
                answer_value=value,
                answer_uom=ratio.uom,
                concept=key,
                facts=(numerator_fact, denominator_fact),
                notes=(
                    f"{ratio.numerator} / {ratio.denominator}"
                    + (" x 100" if ratio.as_percentage else "")
                ),
            )


# ------------------------------------------------------------------ level 3


def generate_level_3(
    repository: FactRepository,
    company: Company,
    submissions: Sequence[Submission],
    drop_log: DropLog,
) -> Iterator[Question]:
    """Temporal. Year-over-year change, which spans two filings.

    The two periods are aligned on fiscal year, never on calendar date, and both
    facts must come from their own original filing.
    """
    template = "L3_yoy_change"
    annual = _annual_submissions(submissions)
    by_year = {submission.fy: submission for submission in annual}

    if not annual:
        drop_log.record(
            DropReason.NO_ANNUAL_FILING,
            level=3,
            template_id=template,
            concept="*",
            cik=company.cik,
            detail="no 10-K in the loaded quarters, so no year-over-year pair exists",
        )
        return

    for submission in annual:
        prior = by_year.get(submission.fy - 1)
        if prior is None:
            # Recorded rather than skipped. A level that produces nothing and
            # logs nothing is indistinguishable from a level that was never
            # run, and that is exactly the question the drop log exists to
            # answer.
            drop_log.record(
                DropReason.NO_PRIOR_YEAR_FILING,
                level=3,
                template_id=template,
                concept="*",
                cik=company.cik,
                adsh=submission.adsh,
                detail=(
                    f"FY{submission.fy} has no FY{submission.fy - 1} 10-K in the " "loaded quarters"
                ),
            )
            continue
        current_ddate = _ddate_for(submission)
        prior_ddate = _ddate_for(prior)
        if current_ddate is None or prior_ddate is None:
            continue

        for key in sorted(CONCEPTS):
            concept = CONCEPTS[key]
            current = resolve_concept(
                repository, submission, concept, current_ddate, _qtrs_for(concept, submission)
            )
            if not current.ok:
                assert current.reason is not None
                drop_log.record(
                    current.reason,
                    level=3,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=submission.adsh,
                    detail=f"current year: {current.detail}",
                )
                continue

            previous = resolve_concept(
                repository, prior, concept, prior_ddate, _qtrs_for(concept, prior)
            )
            if not previous.ok:
                assert previous.reason is not None
                drop_log.record(
                    previous.reason,
                    level=3,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=prior.adsh,
                    detail=f"prior year: {previous.detail}",
                )
                continue

            current_fact = current.fact
            previous_fact = previous.fact
            assert current_fact is not None
            assert previous_fact is not None

            if abs(previous_fact.value) < MIN_DENOMINATOR_MAGNITUDE:
                drop_log.record(
                    DropReason.DENOMINATOR_TOO_SMALL,
                    level=3,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=prior.adsh,
                    detail=f"prior-year {key} = {previous_fact.value}",
                )
                continue

            change = (current_fact.value - previous_fact.value) / abs(previous_fact.value) * 100.0
            period = _period_label(submission)
            text = (
                f"By what percentage did {company.display_name}'s {concept.label} change "
                f"from fiscal year {prior.fy} to {period}?"
            )
            yield Question(
                question_id=make_question_id(template, company.cik, submission.adsh, key),
                level=3,
                template_id=template,
                text=text,
                cik=company.cik,
                company_name=company.display_name,
                primary_adsh=submission.adsh,
                fy=submission.fy,
                fp=submission.fp,
                answer_value=change,
                answer_uom="pure",
                concept=key,
                facts=(current_fact, previous_fact),
                supporting_adsh=(prior.adsh,),
                notes=f"percent change in {key} from FY{prior.fy} to FY{submission.fy}",
            )


# ------------------------------------------------------------------ level 4


def generate_level_4(
    repository: FactRepository,
    companies: Sequence[Company],
    submissions_by_cik: dict[int, Sequence[Submission]],
    drop_log: DropLog,
) -> Iterator[Question]:
    """Cross-company comparison, answered with the extremum value.

    Asking for the value rather than the company keeps the answer numeric, so
    the structured output contract and the whole error taxonomy apply unchanged.
    The retrieval work is the same either way: the agent still has to open every
    filing in the comparison set.
    """
    template = "L4_cross_company_max"
    minimum_companies = 3

    by_industry: dict[str, list[Company]] = {}
    for company in companies:
        by_industry.setdefault(company.industry, []).append(company)

    for industry in sorted(by_industry):
        peers = sorted(by_industry[industry], key=lambda c: c.cik)
        if len(peers) < minimum_companies:
            drop_log.record(
                DropReason.INSUFFICIENT_PEERS,
                level=4,
                template_id=template,
                concept="*",
                detail=(
                    f"{industry} has {len(peers)} roster companies, "
                    f"fewer than the {minimum_companies} a comparison needs"
                ),
            )
            continue

        years = sorted(
            {
                submission.fy
                for company in peers
                for submission in _annual_submissions(submissions_by_cik.get(company.cik, []))
            }
        )

        for fy in years:
            for key in sorted(RATIOS):
                ratio = RATIOS[key]
                values: list[tuple[Company, Submission, float, tuple[FactRef, FactRef]]] = []
                incomplete = False

                for company in peers:
                    annual = _annual_submissions(submissions_by_cik.get(company.cik, []))
                    submission = next((s for s in annual if s.fy == fy), None)
                    if submission is None or submission.period is None:
                        incomplete = True
                        break

                    resolved = _resolve_ratio_components(
                        repository, submission, ratio, submission.period
                    )
                    if isinstance(resolved[0], DropReason):
                        incomplete = True
                        break

                    numerator_fact, denominator_fact = resolved
                    assert isinstance(numerator_fact, FactRef)
                    assert isinstance(denominator_fact, FactRef)
                    if abs(denominator_fact.value) < MIN_DENOMINATOR_MAGNITUDE:
                        incomplete = True
                        break

                    values.append(
                        (
                            company,
                            submission,
                            ratio.compute(numerator_fact.value, denominator_fact.value),
                            (numerator_fact, denominator_fact),
                        )
                    )

                # Every company in the comparison must resolve. A comparison
                # over a subset silently changes the question the agent was
                # asked, and its answer would be graded against the wrong set.
                if incomplete or len(values) < minimum_companies:
                    drop_log.record(
                        DropReason.MISSING_COMPONENT,
                        level=4,
                        template_id=template,
                        concept=key,
                        detail=(
                            f"{industry} FY{fy}: only {len(values)} of {len(peers)} "
                            "companies resolved"
                        ),
                    )
                    continue

                best = max(values, key=lambda item: item[2])
                names = [company.display_name for company, _, _, _ in values]
                listed = ", ".join(names[:-1]) + f" and {names[-1]}"
                label = f"{ratio.label} as a percentage" if ratio.as_percentage else ratio.label
                text = f"Among {listed}, what was the highest {label} for fiscal year {fy}?"

                best_company, best_submission, best_value, _ = best
                all_facts = tuple(fact for _, _, _, pair in values for fact in pair)
                yield Question(
                    question_id=make_question_id(template, industry, fy, key),
                    level=4,
                    template_id=template,
                    text=text,
                    cik=best_company.cik,
                    company_name=best_company.display_name,
                    primary_adsh=best_submission.adsh,
                    fy=fy,
                    fp="FY",
                    answer_value=best_value,
                    answer_uom=ratio.uom,
                    concept=key,
                    facts=all_facts,
                    supporting_adsh=tuple(submission.adsh for _, submission, _, _ in values),
                    notes=(
                        f"highest {key} among {len(values)} {industry} companies in FY{fy}; "
                        f"held by {best_company.ticker}"
                    ),
                )


# ------------------------------------------------------------------ level 5


def _peer_tagging_share(
    repository: FactRepository,
    submission: Submission,
    concept: ConceptSpec,
    peer_ciks: Sequence[int],
) -> tuple[int, int]:
    """How many peers tagged this concept in the same fiscal period.

    Returns (peers tagging it, peers with a filing for that period). Peers with
    no filing for the period are excluded from the denominator rather than
    counted as not tagging it, since a missing filing is not evidence about
    disclosure practice.
    """
    tagging = 0
    comparable = 0
    tags = set(concept.tags)

    for peer_cik in peer_ciks:
        peer_submissions = [
            peer
            for peer in repository.submissions_for_cik(peer_cik)
            if peer.fy == submission.fy and peer.fp == submission.fp and peer.form == "10-K"
        ]
        if not peer_submissions:
            continue
        comparable += 1
        peer = min(peer_submissions, key=lambda s: s.filed)
        if repository.tags_present(peer.adsh) & tags:
            tagging += 1

    return tagging, comparable


def generate_level_5(
    repository: FactRepository,
    company: Company,
    submissions: Sequence[Submission],
    drop_log: DropLog,
    config: GeneratorConfig,
) -> Iterator[Question]:
    """Unanswerable questions: the concept is absent, and peers say it is expected.

    Absence alone is not enough. A concept no company in the industry tags is
    not something a reader would look for, and asking about it tests nothing.
    Expectation is established from the data, by counting how many same-industry
    peers tagged the concept in the same fiscal period.

    The question text is built by the same ``_ask`` used for L1, so nothing in
    the wording marks it as unanswerable.
    """
    template = "L5_unanswerable"

    for submission in _annual_submissions(submissions):
        sic_group = submission.sic_major_group
        if sic_group is None:
            drop_log.record(
                DropReason.INSUFFICIENT_PEERS,
                level=5,
                template_id=template,
                concept="*",
                cik=company.cik,
                adsh=submission.adsh,
                detail="submission has no SIC code, so no peer group can be formed",
            )
            continue

        peer_ciks = repository.peer_ciks(sic_group, company.cik)
        present = repository.tags_present(submission.adsh)

        for key in sorted(CONCEPTS):
            concept = CONCEPTS[key]
            if present & set(concept.tags):
                drop_log.record(
                    DropReason.CONCEPT_PRESENT_IN_FILING,
                    level=5,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=submission.adsh,
                    detail="concept is tagged in this filing, so it is answerable",
                )
                continue

            tagging, comparable = _peer_tagging_share(repository, submission, concept, peer_ciks)
            share = tagging / comparable if comparable else 0.0

            if tagging < config.min_peers_tagging or share < config.peer_tagging_share:
                drop_log.record(
                    DropReason.INSUFFICIENT_PEERS,
                    level=5,
                    template_id=template,
                    concept=key,
                    cik=company.cik,
                    adsh=submission.adsh,
                    detail=(
                        f"{tagging} of {comparable} peers tag {key} "
                        f"(share {share:.2f}); need {config.min_peers_tagging} "
                        f"and {config.peer_tagging_share:.2f}"
                    ),
                )
                continue

            yield Question(
                question_id=make_question_id(template, company.cik, submission.adsh, key),
                level=5,
                template_id=template,
                text=_ask(
                    concept.label,
                    company.display_name,
                    submission,
                    concept.period_type is PeriodType.INSTANT,
                ),
                cik=company.cik,
                company_name=company.display_name,
                primary_adsh=submission.adsh,
                fy=submission.fy,
                fp=submission.fp,
                answer_value=None,
                answer_uom=concept.uom,
                concept=key,
                notes=(
                    f"not tagged in this filing; {tagging} of {comparable} "
                    f"SIC-{sic_group} peers tag it in FY{submission.fy}"
                ),
            )


# ------------------------------------------------------------------ all


def generate_all(
    repository: FactRepository,
    companies: Sequence[Company],
    config: GeneratorConfig | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> QuestionSet:
    """Run every generator over the roster and collect the result.

    ``progress`` is called with a label and the questions gathered so far. The
    loader grew a progress callback because a multi-million row load was silent
    for minutes and a stalled run was indistinguishable from a slow one.
    Generation had exactly the same problem and none of the reporting, which is
    why a run that had gone quadratic looked identical to one nearly finished.
    """
    settings = config if config is not None else GeneratorConfig()
    question_set = QuestionSet()
    drop_log = question_set.drop_log

    submissions_by_cik: dict[int, Sequence[Submission]] = {
        company.cik: repository.submissions_for_cik(company.cik) for company in companies
    }

    def report(label: str) -> None:
        if progress is not None:
            progress(label, len(question_set.questions))

    for index, company in enumerate(companies, start=1):
        submissions = submissions_by_cik[company.cik]
        question_set.extend(generate_level_1(repository, company, submissions, drop_log))
        question_set.extend(generate_level_2(repository, company, submissions, drop_log))
        question_set.extend(generate_level_3(repository, company, submissions, drop_log))
        question_set.extend(generate_level_5(repository, company, submissions, drop_log, settings))
        report(f"{company.ticker} ({index}/{len(companies)})")

    question_set.extend(generate_level_4(repository, companies, submissions_by_cik, drop_log))
    report("cross-company (L4)")
    return question_set
