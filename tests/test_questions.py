"""Question generation: filters, generators, and the properties the set must hold.

The fixture is synthetic and described in ``conftest.py``. These tests check the
contracts, not the numbers: that ambiguity is dropped rather than resolved, that
a restated fact never reaches the set, that signs follow the presentation layer,
and above all that an unanswerable question is indistinguishable from an
answerable one by its wording.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from questions.companies import ROSTER, RosterMismatchError, verify_roster
from questions.concepts import CONCEPTS, RATIOS
from questions.filters import resolve_concept
from questions.generators import (
    GeneratorConfig,
    generate_all,
    generate_level_1,
    generate_level_2,
    generate_level_3,
    generate_level_4,
    generate_level_5,
)
from questions.models import DropLog, DropReason, FactRef, Question, QuestionSet
from questions.repository import FactRepository
from tests.conftest import (
    ACME,
    BETA,
    DELTA,
    FIXTURE_COMPANIES,
    GAMMA,
    SIC_SEMICONDUCTORS,
)

ACME_COMPANY, BETA_COMPANY, GAMMA_COMPANY, DELTA_COMPANY = FIXTURE_COMPANIES


def submissions(repository: FactRepository, cik: int) -> list:
    return repository.submissions_for_cik(cik)


# --------------------------------------------------------------------------
# The Question contract
# --------------------------------------------------------------------------


def test_an_answerable_question_must_cite_its_facts() -> None:
    """An answer with no provenance is an assertion, not a derived answer."""
    with pytest.raises(ValueError, match="provenance"):
        Question(
            question_id="x",
            level=1,
            template_id="t",
            text="What was X's revenue?",
            cik=1,
            company_name="X",
            primary_adsh="a",
            fy=2024,
            fp="FY",
            answer_value=1.0,
            answer_uom="USD",
            concept="revenue",
            facts=(),
        )


def test_an_unanswerable_question_must_not_cite_facts() -> None:
    fact = FactRef("a", "Revenues", "us-gaap/2024", 20241231, 4, "USD", 1.0)
    with pytest.raises(ValueError):
        Question(
            question_id="x",
            level=5,
            template_id="t",
            text="What was X's revenue?",
            cik=1,
            company_name="X",
            primary_adsh="a",
            fy=2024,
            fp="FY",
            answer_value=None,
            answer_uom="USD",
            concept="revenue",
            facts=(fact,),
        )


def test_question_ids_are_deterministic(repository: FactRepository) -> None:
    """Regenerating the set must reproduce the same ids, or the question set

    hash in the run manifest means nothing.
    """
    first = generate_all(repository, FIXTURE_COMPANIES)
    second = generate_all(repository, FIXTURE_COMPANIES)
    assert first.content_hash() == second.content_hash()
    assert [q.question_id for q in first.questions] == [q.question_id for q in second.questions]


def test_duplicate_questions_are_rejected_and_logged() -> None:
    fact = FactRef("a", "Revenues", "us-gaap/2024", 20241231, 4, "USD", 1.0)
    question = Question(
        question_id="dup",
        level=1,
        template_id="t",
        text="What was X's revenue?",
        cik=1,
        company_name="X",
        primary_adsh="a",
        fy=2024,
        fp="FY",
        answer_value=1.0,
        answer_uom="USD",
        concept="revenue",
        facts=(fact,),
    )
    question_set = QuestionSet()
    assert question_set.add(question)
    assert not question_set.add(question)
    assert question_set.drop_log.counts_by_reason() == {"duplicate_question": 1}


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


def test_the_presentation_negating_flag_is_applied(repository: FactRepository) -> None:
    """Cost of sales is presented in parentheses in the fixture, so the answer

    key carries it negative even though num.txt stores it positive.
    """
    submission = next(s for s in submissions(repository, ACME) if s.fy == 2024)
    resolution = resolve_concept(
        repository, submission, CONCEPTS["cost_of_revenue"], submission.period, 4
    )
    assert resolution.ok
    assert resolution.fact is not None
    assert resolution.fact.value < 0


def test_a_concept_tagged_two_ways_is_ambiguous_not_resolved(
    repository: FactRepository,
) -> None:
    """BETA FY2023 tags two revenue concepts with different values. Picking one

    would grade the agent against a coin flip.
    """
    submission = next(s for s in submissions(repository, BETA) if s.fy == 2023)
    resolution = resolve_concept(repository, submission, CONCEPTS["revenue"], submission.period, 4)
    assert not resolution.ok
    assert resolution.reason is DropReason.AMBIGUOUS_MULTIPLE_MATCHES


def test_a_restated_fact_is_dropped(repository: FactRepository) -> None:
    """GAMMA restated FY2023 net income in its FY2024 filing."""
    submission = next(s for s in submissions(repository, GAMMA) if s.fy == 2023)
    resolution = resolve_concept(
        repository, submission, CONCEPTS["net_income"], submission.period, 4
    )
    assert not resolution.ok
    assert resolution.reason is DropReason.RESTATED_FACT


def test_an_unchanged_figure_repeated_later_is_not_a_restatement(
    repository: FactRepository,
) -> None:
    """The same figure legitimately reappears in later comparative columns.

    Treating that as a restatement would empty the question set.
    """
    submission = next(s for s in submissions(repository, ACME) if s.fy == 2023)
    resolution = resolve_concept(
        repository, submission, CONCEPTS["net_income"], submission.period, 4
    )
    assert resolution.ok


def test_asking_an_instant_concept_over_a_duration_is_rejected(
    repository: FactRepository,
) -> None:
    submission = next(s for s in submissions(repository, ACME) if s.fy == 2024)
    resolution = resolve_concept(repository, submission, CONCEPTS["assets"], submission.period, 4)
    assert not resolution.ok
    assert resolution.reason is DropReason.WRONG_PERIOD_TYPE


def test_a_missing_concept_is_dropped_as_missing(repository: FactRepository) -> None:
    submission = next(s for s in submissions(repository, ACME) if s.fy == 2024)
    resolution = resolve_concept(
        repository, submission, CONCEPTS["research_and_development"], submission.period, 4
    )
    assert not resolution.ok
    assert resolution.reason is DropReason.MISSING_COMPONENT


def test_an_unknown_tag_is_treated_as_a_custom_extension(
    repository: FactRepository,
) -> None:
    """Assuming an unrecognized tag is standard would let extensions in."""
    assert repository.is_custom_tag("SomeTagWithNoMetadataRow", "us-gaap/2024")


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------


def test_level_1_produces_facts_with_provenance(repository: FactRepository) -> None:
    log = DropLog()
    questions = list(generate_level_1(repository, ACME_COMPANY, submissions(repository, ACME), log))
    assert questions
    for question in questions:
        assert question.level == 1
        assert len(question.facts) == 1
        assert question.facts[0].adsh == question.primary_adsh


def test_level_1_asks_instant_and_duration_concepts_differently(
    repository: FactRepository,
) -> None:
    log = DropLog()
    questions = {
        q.concept: q
        for q in generate_level_1(repository, ACME_COMPANY, submissions(repository, ACME), log)
    }
    assert "as of the end of" in questions["assets"].text
    assert "as of the end of" not in questions["revenue"].text


def test_level_1_never_says_a_calendar_year(repository: FactRepository) -> None:
    """Fiscal and calendar years diverge, so every question says fiscal."""
    log = DropLog()
    for question in generate_level_1(repository, ACME_COMPANY, submissions(repository, ACME), log):
        assert "fiscal year" in question.text


def test_level_2_computes_a_ratio_from_two_cited_facts(
    repository: FactRepository,
) -> None:
    log = DropLog()
    questions = {
        q.concept: q
        for q in generate_level_2(repository, ACME_COMPANY, submissions(repository, ACME), log)
    }
    gross_margin = questions["gross_margin"]
    assert len(gross_margin.facts) == 2
    numerator, denominator = gross_margin.facts
    expected = RATIOS["gross_margin"].compute(numerator.value, denominator.value)
    assert gross_margin.answer_value == pytest.approx(expected)


def test_level_2_answers_are_recomputable_from_their_facts(
    repository: FactRepository,
) -> None:
    """The provenance is not decoration: the answer must fall out of it."""
    log = DropLog()
    for question in generate_level_2(
        repository, DELTA_COMPANY, submissions(repository, DELTA), log
    ):
        ratio = RATIOS[question.concept]
        numerator, denominator = question.facts
        assert question.answer_value == pytest.approx(
            ratio.compute(numerator.value, denominator.value)
        )


def test_level_3_spans_two_filings_and_aligns_on_fiscal_year(
    repository: FactRepository,
) -> None:
    log = DropLog()
    questions = list(generate_level_3(repository, ACME_COMPANY, submissions(repository, ACME), log))
    assert questions
    for question in questions:
        assert question.supporting_adsh
        assert question.supporting_adsh[0] != question.primary_adsh
        assert len(question.facts) == 2
        assert "fiscal year" in question.text


def test_level_3_percent_change_matches_its_facts(repository: FactRepository) -> None:
    log = DropLog()
    question = next(
        q
        for q in generate_level_3(repository, ACME_COMPANY, submissions(repository, ACME), log)
        if q.concept == "revenue"
    )
    current, previous = question.facts
    expected = (current.value - previous.value) / abs(previous.value) * 100.0
    assert question.answer_value == pytest.approx(expected)


def test_level_4_compares_at_least_three_companies(repository: FactRepository) -> None:
    log = DropLog()
    by_cik = {c.cik: submissions(repository, c.cik) for c in FIXTURE_COMPANIES}
    questions = list(generate_level_4(repository, FIXTURE_COMPANIES, by_cik, log))
    assert questions
    for question in questions:
        assert len(question.supporting_adsh) >= 3


def test_level_4_answer_is_the_maximum_over_the_comparison_set(
    repository: FactRepository,
) -> None:
    log = DropLog()
    by_cik = {c.cik: submissions(repository, c.cik) for c in FIXTURE_COMPANIES}
    question = next(
        q
        for q in generate_level_4(repository, FIXTURE_COMPANIES, by_cik, log)
        if q.concept == "gross_margin" and q.fy == 2024
    )
    ratio = RATIOS["gross_margin"]
    pairs = [question.facts[i : i + 2] for i in range(0, len(question.facts), 2)]
    computed = [ratio.compute(pair[0].value, pair[1].value) for pair in pairs]
    assert question.answer_value == pytest.approx(max(computed))


def test_level_4_drops_a_comparison_when_any_company_is_missing(
    repository: FactRepository,
) -> None:
    """A comparison over a subset is a different question than the one asked."""
    log = DropLog()
    by_cik = {c.cik: submissions(repository, c.cik) for c in FIXTURE_COMPANIES}
    list(generate_level_4(repository, FIXTURE_COMPANIES, by_cik, log))
    assert DropReason.MISSING_COMPONENT.value in log.counts_by_reason()


# --------------------------------------------------------------------------
# Level 5, the headline
# --------------------------------------------------------------------------


def test_level_5_finds_the_concept_peers_tag_and_this_company_does_not(
    repository: FactRepository,
) -> None:
    log = DropLog()
    config = GeneratorConfig(min_peers_tagging=3, peer_tagging_share=0.6)
    questions = list(
        generate_level_5(repository, ACME_COMPANY, submissions(repository, ACME), log, config)
    )
    concepts = {q.concept for q in questions}
    assert "research_and_development" in concepts
    for question in questions:
        assert question.answer_value is None
        assert question.facts == ()


def test_level_5_is_indistinguishable_from_level_1_by_wording(
    repository: FactRepository,
) -> None:
    """The headline result depends on this.

    If unanswerable questions read differently, an agent can learn to abstain on
    the phrasing rather than on the absence of the fact, and the measured
    hallucination rate stops being about hallucination at all.
    """
    log = DropLog()
    config = GeneratorConfig(min_peers_tagging=3, peer_tagging_share=0.6)

    unanswerable = next(
        q
        for q in generate_level_5(
            repository, ACME_COMPANY, submissions(repository, ACME), log, config
        )
        if q.concept == "research_and_development" and q.fy == 2024
    )
    answerable = next(
        q
        for q in generate_level_1(repository, BETA_COMPANY, submissions(repository, BETA), log)
        if q.concept == "research_and_development" and q.fy == 2024
    )

    def shape(text: str, company: str) -> str:
        return text.replace(company, "<COMPANY>")

    assert shape(unanswerable.text, ACME_COMPANY.display_name) == shape(
        answerable.text, BETA_COMPANY.display_name
    )


def test_level_5_will_not_ask_about_a_concept_the_filing_tags(
    repository: FactRepository,
) -> None:
    log = DropLog()
    config = GeneratorConfig()
    questions = list(
        generate_level_5(repository, ACME_COMPANY, submissions(repository, ACME), log, config)
    )
    for question in questions:
        present = repository.tags_present(question.primary_adsh)
        assert not (present & set(CONCEPTS[question.concept].tags))


def test_level_5_requires_peer_evidence_not_just_absence(
    repository: FactRepository,
) -> None:
    """Raising the peer bar above what the fixture can supply must empty L5.

    Absence alone is not a question. A concept nobody in the industry tags is
    not something a reader would look for.
    """
    log = DropLog()
    strict = GeneratorConfig(min_peers_tagging=99, peer_tagging_share=1.0)
    questions = list(
        generate_level_5(repository, ACME_COMPANY, submissions(repository, ACME), log, strict)
    )
    assert questions == []
    assert DropReason.INSUFFICIENT_PEERS.value in log.counts_by_reason()


def test_a_generator_config_with_a_trivial_peer_group_is_rejected() -> None:
    with pytest.raises(ValueError):
        GeneratorConfig(min_peers_tagging=1)


# --------------------------------------------------------------------------
# The set as a whole
# --------------------------------------------------------------------------


def test_generate_all_produces_every_level(repository: FactRepository) -> None:
    question_set = generate_all(repository, FIXTURE_COMPANIES)
    levels = question_set.counts_by_level()
    for level in (1, 2, 3, 4, 5):
        assert levels.get(level, 0) > 0, f"no level {level} questions generated"


def test_every_answerable_question_has_recomputable_provenance(
    repository: FactRepository,
) -> None:
    question_set = generate_all(repository, FIXTURE_COMPANIES)
    for question in question_set.questions:
        if question.level == 5:
            assert question.answer_value is None
            assert question.facts == ()
        else:
            assert question.answer_value is not None
            assert question.facts


def test_a_company_with_no_annual_filing_is_logged_not_skipped(
    repository: FactRepository,
) -> None:
    """A level that produces nothing and logs nothing is indistinguishable from

    a level that was never run. On the first real question set, L3 and L4 came
    back empty with no entries at all, and the drop log could not say whether
    that was the data or a bug.
    """
    log = DropLog()
    list(generate_level_1(repository, ACME_COMPANY, [], log))
    assert DropReason.NO_ANNUAL_FILING.value in log.counts_by_reason()


def test_a_year_with_no_prior_year_filing_is_logged(
    repository: FactRepository,
) -> None:
    """The exact condition that made L3 silently empty: one fiscal year loaded,

    so no year-over-year pair exists.
    """
    log = DropLog()
    one_year = [s for s in submissions(repository, ACME) if s.fy == 2023]
    list(generate_level_3(repository, ACME_COMPANY, one_year, log))
    assert DropReason.NO_PRIOR_YEAR_FILING.value in log.counts_by_reason()


def test_an_industry_with_too_few_companies_is_logged(
    repository: FactRepository,
) -> None:
    log = DropLog()
    by_cik = {FIXTURE_COMPANIES[0].cik: submissions(repository, ACME)}
    list(generate_level_4(repository, FIXTURE_COMPANIES[:1], by_cik, log))
    assert DropReason.INSUFFICIENT_PEERS.value in log.counts_by_reason()


def test_the_drop_log_records_reasons_and_is_written_as_jsonl(
    repository: FactRepository, tmp_path: Path
) -> None:
    """The log is evidence, not debugging output, so it ships with the set."""
    question_set = generate_all(repository, FIXTURE_COMPANIES)
    assert len(question_set.drop_log) > 0

    path = tmp_path / "dropped.jsonl"
    written = question_set.drop_log.write_jsonl(path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert written == len(lines)

    first = json.loads(lines[0])
    assert first["reason"] in {reason.value for reason in DropReason}
    assert "detail" in first


def test_the_question_set_round_trips_through_jsonl(
    repository: FactRepository, tmp_path: Path
) -> None:
    question_set = generate_all(repository, FIXTURE_COMPANIES)
    path = tmp_path / "questions.jsonl"
    written = question_set.write_jsonl(path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert written == len(lines) == len(question_set.questions)
    payload = json.loads(lines[0])
    assert {"question_id", "level", "text", "answer_value", "facts"} <= set(payload)


def test_no_question_text_leaks_a_tag_name(repository: FactRepository) -> None:
    """A question that names the XBRL tag would hand the agent the answer key's

    own vocabulary, which is a softer version of the same leak the iXBRL
    stripper exists to prevent.
    """
    question_set = generate_all(repository, FIXTURE_COMPANIES)
    all_tags = {tag for concept in CONCEPTS.values() for tag in concept.tags}
    for question in question_set.questions:
        for tag in all_tags:
            assert tag not in question.text


# --------------------------------------------------------------------------
# Roster
# --------------------------------------------------------------------------


def test_the_roster_ciks_are_flagged_unverified() -> None:
    """They were written from memory. Nothing may treat them as confirmed until

    verify_roster has checked them against the loaded data.
    """
    assert all(not company.cik_verified for company in ROSTER)


def test_verify_roster_raises_when_a_company_is_absent(
    repository: FactRepository,
) -> None:
    """A roster that does not match the data is a stop, not a warning: it means

    the benchmark's composition differs from what the README would claim.
    """
    with pytest.raises(RosterMismatchError):
        verify_roster(repository, ROSTER)


def test_verify_roster_accepts_a_roster_that_matches(
    repository: FactRepository,
) -> None:
    resolved = verify_roster(repository, FIXTURE_COMPANIES)
    assert set(resolved) == {company.ticker for company in FIXTURE_COMPANIES}


def test_verify_roster_catches_a_cik_pointing_at_the_wrong_company(
    repository: FactRepository,
) -> None:
    """The failure this guards against does not crash on its own: a wrong CIK

    silently benchmarks a different company and every number stays plausible.
    """
    transposed = (
        FIXTURE_COMPANIES[0].__class__(
            ticker="ACME",
            display_name="Acme Devices Inc.",
            cik=BETA,
            name_contains="ACME",
            industry="technology",
        ),
    )
    with pytest.raises(RosterMismatchError, match="does not contain"):
        verify_roster(repository, transposed)


# --------------------------------------------------------------------------
# Repository caching
# --------------------------------------------------------------------------
# Added after question generation went from seconds to over half an hour when
# the database grew from two quarters to eight. The cost was one query repeated,
# not one query that was slow.


def test_repeated_tag_lookups_hit_the_cache(repository: FactRepository) -> None:
    adsh = repository.submissions_for_cik(ACME)[0].adsh
    repository.tags_present(adsh)
    before = repository.cache_stats()["hits"]
    repository.tags_present(adsh)
    repository.tags_present(adsh)
    assert repository.cache_stats()["hits"] == before + 2


def test_a_cached_tag_set_cannot_be_corrupted_by_its_caller(
    repository: FactRepository,
) -> None:
    """The cache hands out a copy.

    A caller that mutated the returned set would otherwise change what every
    later caller sees, and the peer logic does set arithmetic on this result.
    """
    adsh = repository.submissions_for_cik(ACME)[0].adsh
    first = repository.tags_present(adsh)
    original = set(first)
    first.clear()
    assert repository.tags_present(adsh) == original


def test_repeated_submission_lookups_hit_the_cache(repository: FactRepository) -> None:
    repository.submissions_for_cik(ACME)
    before = repository.cache_stats()["hits"]
    repository.submissions_for_cik(ACME)
    assert repository.cache_stats()["hits"] == before + 1


def test_different_form_filters_are_cached_separately(repository: FactRepository) -> None:
    """The form tuple is part of the key, or a 10-K query serves 10-Q results."""
    annual = repository.submissions_for_cik(ACME, forms=("10-K",))
    both = repository.submissions_for_cik(ACME, forms=("10-K", "10-Q"))
    assert all(submission.form == "10-K" for submission in annual)
    assert len(both) >= len(annual)


def test_peer_lookup_finds_companies_in_the_same_major_group(
    repository: FactRepository,
) -> None:
    """Guards the rewrite from sic / 100 = ? to a sargable range."""
    peers = repository.peer_ciks(SIC_SEMICONDUCTORS // 100, exclude_cik=ACME)
    assert ACME not in peers
    assert {BETA, GAMMA, DELTA} <= set(peers)


def test_generate_all_reports_progress(repository: FactRepository) -> None:
    """A silent generator is indistinguishable from a hung one."""
    seen: list[tuple[str, int]] = []
    generate_all(repository, FIXTURE_COMPANIES, progress=lambda a, b: seen.append((a, b)))
    assert len(seen) == len(FIXTURE_COMPANIES) + 1
    assert seen[-1][0] == "cross-company (L4)"
