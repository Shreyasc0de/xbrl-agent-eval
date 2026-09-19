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
| Runnable data pull (`make data`) | in place, not yet run against SEC |
| `test_no_ixbrl_leak.py` | written, passing |
| `test_normalizers.py` | mechanical cases passing; hand-verified-figure cases skipped pending real filing figures |
| `test_grader.py` | written, passing |
| iXBRL stripper and text conversion | implemented |
| DERA loader and its CLI | implemented, not yet run against real data |
| EDGAR fetcher | implemented, not yet run against real data |
| Filing fetch, strip and cache pipeline | implemented, not yet run against real data |
| Question generation (L1 to L5) | implemented, not yet run against real data |
| Drop log and question set serialization | implemented |
| Company roster | proposed, CIKs unverified until checked against loaded data |
| Model call cache, run manifest, backend interface | implemented |
| Agent tools over the filing cache | implemented |
| Structured answer parsing | implemented |
| Agent A, the single-agent ReAct loop | implemented, not yet run against a model |
| Run harness and run artifacts | implemented, not yet run against a model |
| Metrics with Wilson intervals | implemented |
| Agent B, the multi-agent system | not started |
| verify_results.py | not started |

Everything marked "not yet run against real data" has been unit-tested against fixtures only. SEC is not reachable from the development sandbox, so the ingest path and question generation are exercised on the host machine.

No question set has been generated. The spec's Week 2 exit criterion, 600 or more validated questions with a dropped-question log, is not met and will not be until the DERA quarters are loaded and `python -m questions.generate` is run.

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

The boundary is checked structurally. `tests/test_agent_boundary.py` parses every module under `src/agents/` and fails the build if any of them imports `ingest`, `eval`, `questions` or `sqlite3`. A comment saying the modules must stay separate is not an invariant; an import graph checked on every commit is.

Every rate is reported with a 95% Wilson interval rather than as a point estimate. Per-level sample sizes are small, and the difference between 68% and 73% on a hundred questions is not a difference. Wilson rather than the normal approximation because the normal interval misbehaves exactly where this benchmark lives: small samples and proportions near zero, where it can include negative numbers.

A failed task is not a wrong answer. A task that errored produced no answer at all, so counting it as incorrect would credit the agent with an attempt it never made, and dropping it silently would inflate accuracy. Failures are excluded from accuracy and reported as their own count.

Retrieval is deterministic lexical scoring, BM25 over fixed overlapping chunks, with no embeddings. Embeddings would likely retrieve better, but a run would stop being reproducible unless the embedding model, its version and its normalization were all pinned, and reproducibility is the property this benchmark is built around. An embedding retriever is a cleaner ablation to add later than a default to start with.

`src/core/` is a deliberate addition to the layout in `docs/SPEC.md`. It holds the run infrastructure that both `agents/` and `report/` need, specifically the content-addressed model call cache and the run manifest writer. Keeping it separate is what lets `agents/` stay free of imports from `eval/` and `ingest/`.

The boundary invariant in code: agent tools read from `data/filings/`, the grader reads from `data/xbrl.db`, and no module imports from both.

## Setup

Needs Python 3.10, 3.11 or 3.12. The upper bound is deliberate: the pinned scientific dependencies publish wheels for those versions, and on a newer interpreter pip falls back to building them from source and the install fails in a way that is easy to scroll past.

```
make install             # picks a supported interpreter, or says why it cannot
cp .env.example .env     # fill in SEC_CONTACT_EMAIL, SEC requires a real contact
make check               # ruff, mypy, pytest
```

`make install` ends by importing every package, so a partial install fails loudly rather than leaving a venv that looks fine until the first run. If the interpreter it picks is wrong, override it:

```
make install PYTHON=/opt/homebrew/bin/python3.12
```

`make doctor` reports what is set up and what is not, and needs nothing installed to run:

```
$ make doctor
python on PATH:   /usr/bin/python3 Python 3.12.4
chosen for venv:  python3.12
venv:             .venv, python3.12
packages:         importable
env file:         .env present
ground truth:     missing, run make quarters
filings:          missing, run make filings
questions:        missing, run make questions
ollama:           2 model(s)
```

## Data

Run on the host machine, not in a sandbox. SEC enforces a rate limit of roughly 10 requests per second and requires a descriptive User-Agent containing a contact email. Every filing is fetched once and cached to disk, which is also what makes runs reproducible.

```
make data QUARTERS=2023q1,2023q2,2023q3,2023q4,2024q1,2024q2,2024q3,2024q4
```

That runs three stages in order, and each can also be run on its own:

```
python -m ingest.dera_loader --quarters 2024q3,2024q4   # ground truth into SQLite
python -m ingest.filings                                # filings the agent reads
python -m questions.generate                            # the question set
```

Then run the benchmark:

```
python -m eval.run --system A --model qwen3:4b --limit 100
```

Each run writes `tasks.jsonl`, `metrics.json` and `manifest.json` into its own directory under `runs/`. The task traces keep the raw model output for every step, which is what lets a reader check that a claimed hallucination was a hallucination and not a parse failure.

The second stage is where the benchmark's boundary is enforced in practice. Each filing is fetched, stripped of inline XBRL, converted to text, checked for leaks twice, and only then written to `data/filings/`. A filing that cannot be cleaned is never cached, so a leak cannot reach a run through the back door of a stale cache file.

This writes `questions.jsonl`, `dropped.jsonl` and `summary.json` together. The drop log is not debugging output: a set of 700 questions means one thing if 200 candidates were dropped and something quite different if 20,000 were, and only the log distinguishes them.

Two conventions the generators follow, both chosen deliberately. Unanswerable questions are phrased identically to answerable ones, so an agent cannot learn to abstain on wording rather than on the absence of a fact. And a concept counts as unanswerable only when the filing does not tag it and a majority of same-industry peers do, so absence alone is never mistaken for a question worth asking.

Nothing under `data/` is committed. Run artifacts under `runs/` are committed, since they are the evidence for every published claim.

## Evidence rules

Summarized from `CLAUDE.md`, which is the authoritative version:

no fabricated results ever; every run asserts that its model call count is greater than zero before writing output; no silent fallbacks, a failure is recorded as a failure; every number in this README is regenerable by `src/report/verify_results.py` from raw artifacts, checked in CI; notebooks never produce a published number; every run writes a complete manifest.

## License

Not yet chosen.
