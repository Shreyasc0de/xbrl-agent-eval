# xbrl-agent-eval

A benchmark that grades agentic financial analysts against SEC XBRL ground truth, with a published hallucination rate.

The benchmark rests on one asymmetry: **ground truth comes from structured XBRL, and the agent under test only ever sees the human-readable filing.** If the agent could query XBRL directly the task would be a lookup and the benchmark would measure nothing. That boundary is enforced in code, not in a prompt, and `tests/test_no_ixbrl_leak.py` is the test that holds it.

## Results

**No runs have been executed yet. This table is empty by design and will be populated only from artifacts under `runs/`.**

Per the evidence rules in `CLAUDE.md`, no number appears here until it is computed from a real run and checked by `src/report/verify_results.py` in CI. Placeholder and illustrative values are not written to this file.

| System | Backend | L1 | L2 | L3 | L4 | L5 hallucination rate | Cost per correct answer |
|---|---|---|---|---|---|---|---|
| _(no runs yet)_ | | | | | | | |

## Status

| Component | State |
|---|---|
| Repo scaffold, tooling, CI | in place |
| `test_no_ixbrl_leak.py` | written, passing |
| `test_normalizers.py` | mechanical cases passing; hand-verified-figure cases skipped pending real filing figures |
| `test_grader.py` | written, passing |
| iXBRL stripper | implemented |
| DERA loader | implemented, not yet run against real data |
| EDGAR fetcher | implemented, not yet run against real data |
| Question generation (L1 to L5) | not started |
| Agent A, Agent B | not started |
| Grader, metrics, error taxonomy | not started |

Everything marked "not yet run against real data" has been unit-tested against fixtures only. SEC is not reachable from the development sandbox, so the ingest path is exercised on the host machine.

## Architecture

```
src/
  ingest/      DERA loader, EDGAR fetcher, iXBRL stripper
  questions/   generators per difficulty level, filters, validation
  agents/      system_a.py (single agent), system_b.py (multi-agent), tools.py
  eval/        grader, normalizers, metrics, error taxonomy
  report/      figures, tables, verify_results.py
  core/        run manifests, LLM call cache, rate-limited HTTP client
```

`src/core/` is a deliberate addition to the layout in `docs/SPEC.md`. It holds the run infrastructure that both `agents/` and `report/` need, specifically the content-addressed model call cache and the run manifest writer. Keeping it separate is what lets `agents/` stay free of imports from `eval/` and `ingest/`.

The boundary invariant in code: agent tools read from `data/filings/`, the grader reads from `data/xbrl.db`, and no module imports from both.

## Setup

```
make install
cp .env.example .env     # fill in SEC_CONTACT_EMAIL, SEC requires a real contact
make check               # ruff, mypy, pytest
```

## Data

Run on the host machine, not in a sandbox. SEC enforces a rate limit of roughly 10 requests per second and requires a descriptive User-Agent containing a contact email. Every filing is fetched once and cached to disk, which is also what makes runs reproducible.

```
make data QUARTERS=2023q1,2023q2,2023q3,2023q4,2024q1,2024q2,2024q3,2024q4
```

Nothing under `data/` is committed. Run artifacts under `runs/` are committed, since they are the evidence for every published claim.

## Evidence rules

Summarized from `CLAUDE.md`, which is the authoritative version:

no fabricated results ever; every run asserts that its model call count is greater than zero before writing output; no silent fallbacks, a failure is recorded as a failure; every number in this README is regenerable by `src/report/verify_results.py` from raw artifacts, checked in CI; notebooks never produce a published number; every run writes a complete manifest.

## License

Not yet chosen.
