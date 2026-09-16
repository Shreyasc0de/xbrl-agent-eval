# CLAUDE.md

Project: `xbrl-agent-eval`. A benchmark that grades agentic financial analysts against SEC XBRL ground truth.

Read `docs/SPEC.md` before working on anything in this repo.

## The one invariant

Ground truth comes from structured XBRL. The agent under test only ever sees the human-readable filing.

These two paths must never touch. Agent tools read from `data/filings/`. The grader reads from `data/xbrl.db`. No module may import from both. If a change would let an agent reach XBRL facts directly, stop and flag it rather than implementing it, because it silently invalidates every result the benchmark has ever produced.

## Evidence rules

These exist because generated pipelines in this portfolio have previously produced confident numbers that traced back to nothing. They are not style preferences.

1. **No fabricated results, ever.** Do not write example numbers, placeholder metrics, or illustrative values into any file under `runs/`, `README.md`, or `src/report/`. If a number is not computed from a real run, it does not get written down.

2. **Assert the work happened.** Before any run writes a results file, assert that the LLM call count for that run is greater than zero and that the cache-miss count is consistent with the run type. A run that silently made zero model calls must crash, not produce output.

3. **No silent fallbacks.** Never catch an exception and substitute a default value. Never use `or 0`, `or ""`, or a bare `except:` to keep a pipeline moving. A failed model call, a missing fact, or an unparseable filing is recorded as an error for that task and the task is marked failed. Partial runs are fine. Fake completeness is not.

4. **Every number is regenerable.** `src/report/verify_results.py` reads only raw artifacts under `runs/` and recomputes every number and figure claimed in `README.md`. It runs in CI and fails the build on any mismatch. If you add a claim to the README, you add its check to that script in the same commit.

5. **Raw artifacts are the source of truth.** Dashboards, notebooks, and summary tables are derived views. A notebook never produces a published number. If asked whether a result holds, answer from the run JSON, not from a summary.

6. **Manifests on every run.** Each run writes model id, temperature, seed, prompt template hash, question set hash, git commit, timestamp, wall clock, token counts, and total cost. A run without a complete manifest is not a valid run.

## Test-first ordering

These tests get written before the code they cover:

- `tests/test_no_ixbrl_leak.py`. Asserts no `ix:` namespace, no `contextRef`, and no `unitRef` attribute survives filing preprocessing. This is the boundary invariant, so it is the first test in the repo.
- `tests/test_normalizers.py`. Unit and scale normalization, sign conventions from the `pre.txt` negating flag, fiscal period alignment. Test against figures hand-verified from actual filings, with the source filing and page recorded in the test docstring.
- `tests/test_grader.py`. Numeric tolerance, abstention scoring, error taxonomy assignment.

Do not build agents before these pass.

## Reproducibility

Cache every model call keyed by `sha256(model_id + prompt + params)`. Store request, response, token counts, latency, and cost. A rerun with an unchanged cache must reproduce results exactly. When a prompt template changes, its hash changes, and affected cache entries are invalidated rather than reused.

SEC fetches are rate limited to under 10 requests per second with a descriptive User-Agent containing a contact email. Every filing is fetched once and cached to disk. Do not re-fetch in a loop.

## Code conventions

Python, type hints throughout. `ruff` and `mypy` clean before commit. Dependencies pinned.

Structured output only from agents: `{answer_value, unit, scale, answer_type, source_quote, confidence}` where `answer_type` is `numeric` or `not_disclosed`. Free-text answers are ungradeable and must be rejected at parse time, not coerced.

Keep `src/ingest/`, `src/questions/`, `src/agents/`, `src/eval/`, and `src/report/` independent. Agents must not import from `eval` or `ingest` internals.

## Writing style

No em dashes in README or documentation prose. Use commas, colons, or separate sentences.

README leads with the results table. Architecture comes after. State null and negative results plainly in the main body, never softened or buried. If the multi-agent system does not beat the single agent, the README says so in the results table without hedging.

Label anything simulated, synthetic, or partial explicitly at the point where it is reported. Never let a reader assume a component is real when it is not.

## When uncertain

If a design choice would affect the validity of the benchmark, stop and ask rather than picking a default. Fiscal period semantics, restatement handling, tolerance thresholds, and abstention scoring all fall in this category. Guessing wrong on any of them produces an answer key that is confidently incorrect, which is worse than a crash because nothing surfaces it.

## Git

This repo belongs to the `Shreyasc0de` account. Before the first commit:

```
git config user.email 194825367+Shreyasc0de@users.noreply.github.com
```

Do not commit anything under `data/`. Run artifacts under `runs/` are committed, since they are the evidence for published claims.
