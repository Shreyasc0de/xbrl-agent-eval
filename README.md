# xbrl-agent-eval

A benchmark that grades agentic financial analysts against SEC XBRL ground truth, with a published hallucination rate.

The benchmark rests on one asymmetry: **ground truth comes from structured XBRL, and the agent under test only ever sees the human-readable filing.** If the agent could query XBRL directly the task would be a lookup and the benchmark would measure nothing. That boundary is enforced in code, not in a prompt, and `tests/test_no_ixbrl_leak.py` is the test that holds it.

## Results

**This table is empty. Runs have been executed against a local model, but none has produced a result worth publishing: the question set is 94 questions covering three of five levels, and committing a headline number computed from it would overstate what has been measured.**

Per the evidence rules in `CLAUDE.md`, no number appears here until it is computed from a real run and checked by `src/report/verify_results.py` in CI. Placeholder and illustrative values are not written to this file.

<!-- results:begin -->
| System | Backend | Model | Graded | Accuracy | L5 hallucination rate | Passage hit rate | Cost per correct |
|---|---|---|---|---|---|---|---|
| _(no runs yet)_ | | | | | | | |
<!-- results:end -->

This table is generated from `runs/`, not written by hand. `src/report/verify_results.py` regenerates it from the raw task records and fails the build if it differs, so it cannot drift away from the runs behind it. A number quoted in prose carries a claim marker naming the run and metric it came from, and is checked the same way.

## Status

Machinery is complete and has been run against real SEC data. No benchmark run has produced a publishable result yet, because the question set is still too small to be worth publishing.

| Component | State |
|---|---|
| Repo scaffold, tooling, CI | in place |
| iXBRL stripper and text conversion | implemented, run against real filings |
| DERA loader and its CLI | implemented, run against real quarters |
| EDGAR fetcher | implemented, run against SEC |
| Filing fetch, strip and cache pipeline | implemented, run against SEC |
| Company roster | 20 companies, all CIKs verified against loaded data |
| Question generation (L1 to L5) | implemented, run; output below the target set size |
| Drop log and question set serialization | implemented |
| Passage annotator (`src/annotate/`) | implemented, not yet run |
| Model call cache, run manifest, backend interface | implemented, exercised by real runs |
| Agent tools over the filing cache | implemented, exercised by real runs |
| Structured answer parsing | implemented, exercised by real runs |
| Agent A, the single-agent ReAct loop | implemented, run against a local model |
| Agent B, the multi-agent system | implemented, not yet run against a model |
| Run harness and run artifacts | implemented, run |
| Metrics with Wilson intervals | implemented |
| `test_no_ixbrl_leak.py` | written, passing |
| `test_grader.py` | written, passing |
| `test_normalizers.py` | mechanical cases passing; hand-verified-figure cases skipped, see below |
| `verify_results.py` | implemented, wired into CI, generates the results table |

The skipped cases in `test_normalizers.py` are the one place the suite is knowingly incomplete. They require figures read by hand off a real filing with the source recorded, and the table of them is empty. The skip reason says so, which is what stops an empty table from reading as a passing suite.

The spec's Week 2 exit criterion, 600 or more validated questions with a dropped-question log, is not met. Two quarters of DERA data yield 94 questions across levels 1, 2 and 5, because only four of the twenty companies filed an annual report inside that window and levels 3 and 4 need a prior-year filing to compare against. That is a data coverage limit rather than a generator limit, and it is why the quarter range in the command below is eight quarters rather than two.

## Architecture

```
src/
  ingest/      DERA loader, EDGAR fetcher, iXBRL stripper
  questions/   generators per difficulty level, filters, validation
  annotate/    which passages of a filing state each answer
  agents/      system_a.py (single agent), system_b.py (multi-agent), tools.py
  eval/        grader, normalizers, metrics, error taxonomy
  report/      figures, tables, verify_results.py
  core/        run manifests, LLM call cache, rate-limited HTTP client
```

The boundary is checked structurally. `tests/test_agent_boundary.py` parses every module under `src/agents/` and fails the build if any of them imports `ingest`, `eval`, `questions`, `annotate` or `sqlite3`. A comment saying the modules must stay separate is not an invariant; an import graph checked on every commit is.

`annotate/` is the one package that sees both sides, so it has a guard of its own. It is handed truth values as plain floats and filing text as strings, and the same test fails the build if it imports the database or the question repository. It exists because filing-level retrieval scoring is close to meaningless here: with twenty companies, finding the right 10-K is a lookup, and that metric sits near 1.0 whatever the agent does. It is reported as `correct_filing_rate`, named for what it measures. `passage_hit_rate` is the number that moves, and it counts a task only when the passage stating its answer was located, so the table prints the denominator alongside the rate.

A passage hit means the agent read a passage where the figure appears, not that it read the right one. A figure printed in both a note and the statement it rolls into annotates both, and the agent gets credit for either. That is a deliberate overcount rather than an undercount: the alternative penalizes an agent for reading a correct passage that happened not to be the one the annotator picked.

Every rate is reported with a 95% Wilson interval rather than as a point estimate. Per-level sample sizes are small, and the difference between 68% and 73% on a hundred questions is not a difference. Wilson rather than the normal approximation because the normal interval misbehaves exactly where this benchmark lives: small samples and proportions near zero, where it can include negative numbers.

A failed task is not a wrong answer. A task that errored produced no answer at all, so counting it as incorrect would credit the agent with an attempt it never made, and dropping it silently would inflate accuracy. Failures are excluded from accuracy and reported as their own count.

Retrieval is deterministic lexical scoring, BM25 over fixed overlapping chunks, with no embeddings. Embeddings would likely retrieve better, but a run would stop being reproducible unless the embedding model, its version and its normalization were all pinned, and reproducibility is the property this benchmark is built around. An embedding retriever is a cleaner ablation to add later than a default to start with.

`src/core/` is a deliberate addition to the layout in `docs/SPEC.md`. It holds the run infrastructure that both `agents/` and `report/` need, specifically the content-addressed model call cache and the run manifest writer. Keeping it separate is what lets `agents/` stay free of imports from `eval/` and `ingest/`.

The boundary invariant in code: agent tools read cached filing text, the grader reads the XBRL database, and no module imports from both. Both live outside the repository by default, under `~/xbrl-data/`.

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
passages:         missing, run make passages
ollama:           2 model(s)
```

## Data

Run on the host machine, not in a sandbox. SEC enforces a rate limit of roughly 10 requests per second and requires a descriptive User-Agent containing a contact email. Every filing is fetched once and cached to disk, which is also what makes runs reproducible.

The database lives outside the repository by default, at `~/xbrl-data/xbrl.db`. That is deliberate rather than tidiness: a multi-gigabyte SQLite write lasting several minutes is exactly the workload iCloud, Dropbox and similar services interact with worst, and a checkout on the Desktop sits inside one. A file rewritten underneath SQLite mid-write kills the process with a bus error that says nothing about its cause. Override with `DB=`.

```
make data QUARTERS=2023q1,2023q2,2023q3,2023q4,2024q1,2024q2,2024q3,2024q4
```

That runs four stages in order, and each can also be run on its own:

```
python -m ingest.dera_loader --quarters 2024q3,2024q4 --db ~/xbrl-data/xbrl.db
python -m ingest.filings --db ~/xbrl-data/xbrl.db
python -m questions.generate --db ~/xbrl-data/xbrl.db
python -m annotate.run
```

The fourth stage records which passages of each cached filing state each answer, which is what `passage_hit_rate` is scored against. It reads the generated questions as JSON and the filings as text, and touches no database.

Then run the benchmark:

```
python -m eval.run --system A --model qwen3:4b --limit 100
python -m eval.run --system B --model qwen3:4b --limit 100
```

The two systems share their tools, their answer schema and their abstention wording, and differ only in orchestration. System A is one model in a ReAct loop. System B plans the question into lookups, runs a retriever per lookup, writes the answer from the findings, and has a verifier check it against those findings before accepting it.

The expectation, stated before any run: B costs several times more for an accuracy difference inside the confidence interval. That result is worth publishing either way, which is why cost per correct answer is reported alongside accuracy.

`make results` regenerates the README table from whatever is under `runs/` and refuses to write if any run's `metrics.json` disagrees with its own task records, so the table cannot be used to launder a corrupt artifact.

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
