# SEC XBRL Ground-Truth Evaluator — Build Spec

**Working title:** `xbrl-agent-eval`
**Scope:** 4–6 weeks, portfolio-grade
**Backends:** local Ollama (qwen3:4b or larger) + one cheap hosted model
**One-line pitch:** A benchmark that grades agentic financial analysts against SEC XBRL ground truth, with a published hallucination rate.

---

## 1. The core idea

SEC XBRL data gives you the exact, machine-readable answer to thousands of financial questions. The benchmark works because of one asymmetry:

> **Ground truth comes from structured XBRL. The agent only ever sees the human-readable filing.**

If the agent can query XBRL directly, the task is a lookup and the project is worthless. Enforce this boundary in code, not in a prompt. The agent's toolset reads rendered 10-K/10-Q text. The grader reads XBRL. They never touch the same store.

Everything valuable downstream (accuracy by difficulty, hallucination rate, cost per correct answer, multi-agent ablations) falls out of that one design choice.

---

## 2. Data

### 2.1 Ground truth: SEC Financial Statement Data Sets

Quarterly ZIPs published by SEC DERA. Each contains four tab-delimited files:

| File | Contents | You need |
|---|---|---|
| `sub.txt` | One row per submission: `adsh`, `cik`, `name`, `form`, `period`, `fy`, `fp`, `filed` | Yes |
| `num.txt` | One row per numeric fact: `adsh`, `tag`, `version`, `ddate`, `qtrs`, `uom`, `value` | Yes, this is the answer key |
| `pre.txt` | Presentation: statement, line order, `negating` flag | Yes, for sign handling |
| `tag.txt` | Tag metadata: `custom`, `abstract`, `datatype`, `crdr` | Yes, to filter |

Load all four into SQLite. One quarter is roughly 300–500 MB unzipped; eight quarters fits comfortably on disk and queries fast with indexes on `(adsh, tag, ddate, qtrs)`.

### 2.2 Agent input: the filing itself

Fetch the primary document via EDGAR:
- `https://data.sec.gov/submissions/CIK##########.json` to list filings by accession number
- `https://www.sec.gov/Archives/edgar/data/{cik}/{adsh_nodash}/{primary_doc}` for the HTML

Strip inline XBRL attributes from the HTML before the agent sees it. iXBRL filings embed the tagged values in `ix:` attributes, which would leak the answer key straight into the agent's context. This is the single most likely way to accidentally invalidate your own benchmark, so write a test for it: assert no `ix:` namespace survives preprocessing.

Convert to clean text or markdown, chunk by section, and expose it through tools.

### 2.3 Compliance

SEC requires a descriptive `User-Agent` header with a contact email and enforces ~10 requests/second. Build a rate-limited fetcher with on-disk caching from day one. You will fetch each filing once and then never again, which also makes your runs reproducible.

### 2.4 Scope for 4–6 weeks

- 20 companies, deliberately mixed: large-cap tech, a bank, a retailer, a biotech, an industrial. Different industries surface different tagging conventions, which is where agents break.
- 8 quarters of filings each, roughly 160 filings.
- Target 600–800 questions after filtering.

---

## 3. Question generation

Generate programmatically from XBRL facts. Never hand-write questions; the whole point is that the answer key is derived, not typed.

Five difficulty levels:

**L1 — Single fact retrieval.** "What was Company X's total revenue for fiscal Q3 2024?" One `num.txt` row. Baseline competence.

**L2 — Derived ratio.** Gross margin, current ratio, R&D as a share of revenue, debt-to-equity. Two to four facts from the same filing, combined by a formula you define. Tests arithmetic and correct concept selection.

**L3 — Temporal.** YoY change, multi-year CAGR, sequential growth. Spans filings, so the agent must align fiscal periods. This is where fiscal-year misalignment bugs surface in agents, and it is genuinely hard.

**L4 — Cross-company comparison.** "Which of A, B, C had the highest operating margin in FY2024?" Multi-document, multi-hop. Also gives you a multiple-choice variant with a random baseline you can report against.

**L5 — Unanswerable.** Ask for a concept that is genuinely not tagged in that filing. The correct answer is an explicit "not disclosed."

**L5 is your headline result.** Accuracy on L1–L4 is a number everyone has. A measured hallucination rate — how often the agent invents a figure when none exists — is the number hiring managers actually care about, because it is the exact failure mode blocking agent deployment in regulated settings. Make L5 about 20% of the question set and report it as a first-class metric, not a footnote.

Constructing valid L5 questions takes care. A concept must be absent from the filing's tagged facts *and* plausibly expected for that company and industry. Verify absence against `num.txt` for that `adsh`, and sanity-check a sample by hand before trusting the generator.

### Filtering

Drop questions where:
- The tag is a custom extension (`tag.txt: custom = 1`), since those are company-specific and ambiguous
- The fact was restated in a later filing (same `cik`, same `tag`, same `ddate`, different `adsh` with a later `filed`)
- The value is zero or null, which makes tolerance-based grading meaningless
- Multiple facts match the question template, making the answer ambiguous

Log every dropped question with its reason. That log is itself evidence of rigor.

---

## 4. Agent design

Build two systems so you have something to compare.

**System A — single agent with tools.** One LLM, a ReAct-style loop, tools for `search_filing(query)`, `read_section(section_id)`, `list_filings(company, period)`, and `calculate(expression)`. Hard cap on tool calls per task.

**System B — multi-agent.** A planner that decomposes the question, one or more retrieval agents, a calculator agent, and a verifier that checks the final answer against retrieved snippets.

Report both. The likely finding is that System B costs three to five times more for a small or statistically insignificant accuracy gain. **Publish that result.** A candidate who measured the multi-agent premium and found it unjustified is more credible than one who assumed it was worth it. If B does win, you have a defensible reason why, tied to specific question levels.

Every agent must emit structured output: `{answer_value, unit, scale, answer_type, source_quote, confidence}`. `answer_type` is one of `numeric` or `not_disclosed`. Free-text answers are ungradeable; do not accept them.

---

## 5. Grading

This is where most benchmarks quietly fail. Be strict and explicit.

**Unit and scale normalization.** Filings report in thousands, millions, or units, inconsistently and sometimes within one document. Normalize both prediction and truth to base units before comparing. A large share of apparent agent errors are scale errors, and separating "wrong number" from "right number, wrong scale" is a genuinely useful finding for your writeup. Report them as distinct error classes.

**Sign conventions.** XBRL stores many values with a natural sign that the presentation layer flips via the `negating` flag in `pre.txt`. Costs and expenses are the common trap. Decide one convention, apply it to the answer key, and document it.

**Numeric tolerance.** Relative tolerance of 0.5% for derived ratios, exact match after scale normalization for directly reported facts. Justify the choice in the README rather than tuning it until results look good.

**Abstention scoring.** For L5, correct means the agent said `not_disclosed`. Any numeric answer is a hallucination. Also track the inverse: false abstention on answerable questions, where the agent gives up on something it should have found. Both matter, and an agent can game one by sacrificing the other, so always report the pair.

### Metrics to publish

- Accuracy by difficulty level, with 95% confidence intervals (Wilson intervals; your sample sizes per level are small enough that this matters)
- Hallucination rate on L5
- False abstention rate on L1–L4
- Error taxonomy: wrong concept, wrong period, scale error, arithmetic error, retrieval miss, hallucination
- Cost per correct answer, in dollars and in tokens
- Median and p95 latency
- Tool calls per task, and retrieval precision (did it open the filing that actually contains the answer)

---

## 6. Reproducibility and anti-fabrication

Given how easily generated pipelines produce numbers that trace back to nothing, build the audit trail into the system rather than checking for it afterward.

**Cache every LLM call** keyed by `sha256(model_id + prompt + params)`. Store request, response, token counts, latency, and cost. Reruns hit cache and reproduce exactly.

**No silent fallbacks.** If a model call fails, the task fails and is recorded as an error. Never substitute a default value, never catch broadly and continue. Assert that the LLM call count for a run is greater than zero before writing any results file.

**Every run writes a manifest**: model id, temperature, seed, prompt template hash, question set hash, git commit, timestamp, wall clock, total cost.

**`verify_results.py`.** A script that reads only the raw run artifacts and regenerates every number in the README and every figure. Wire it into CI so the build fails if a claimed number and the logs disagree. This closes the gap where a summary drifts from the underlying data.

**Publish the raw artifacts.** Ship the run JSONs in the repo, or a sample if they are large. A reviewer who can recompute your headline number from your logs will trust everything else you say.

---

## 7. Repository structure

```
xbrl-agent-eval/
├── README.md                  # results table first, architecture second
├── data/
│   ├── raw/                   # SEC zips, gitignored
│   ├── filings/               # cached filing HTML/text, gitignored
│   └── xbrl.db                # SQLite ground truth
├── src/
│   ├── ingest/                # DERA loader, EDGAR fetcher, iXBRL stripper
│   ├── questions/             # generators per level, filters, validation
│   ├── agents/                # system_a.py, system_b.py, tools.py
│   ├── eval/                  # grader, normalizers, metrics, taxonomy
│   └── report/                # figures, tables, verify_results.py
├── runs/                      # per-run manifests + artifacts
├── tests/
│   ├── test_no_ixbrl_leak.py  # the boundary test
│   ├── test_normalizers.py
│   └── test_grader.py
├── notebooks/                 # exploration only, never the source of a claim
└── .github/workflows/ci.yml   # tests + verify_results.py
```

Two rules that keep this credible: notebooks never produce a published number, and the README leads with a results table rather than an architecture diagram.

Note on git identity: portfolio repos live under the `Shreyasc0de` account, so configure `user.email` to `194825367+Shreyasc0de@users.noreply.github.com` in this repo before the first commit, otherwise commits attribute to the other account.

---

## 8. Backends

With a tight budget, this configuration gives you a real cost-accuracy story:

- **Local:** qwen3:4b via Ollama with `think:false`, capped tokens, and a hard timeout, matching what already works for you. Consider adding a 7–8B model if hardware allows; the 4B-to-8B jump on multi-hop retrieval is usually visible and makes the scaling point concrete.
- **Hosted, cheap:** one small frontier model. Budget the full run: 700 questions × roughly 8 tool calls × context length. Estimate this before you spend anything, and cache aggressively so reruns are free.

Plot cost per correct answer against accuracy for each backend. If the local 4B model reaches, say, 70% of hosted accuracy at a fraction of the cost, that is a legitimate and interesting result about where small models are sufficient.

---

## 9. Milestones

**Week 1 — Data foundation.** DERA loader into SQLite, EDGAR fetcher with rate limiting and caching, iXBRL stripper plus its leak test. Exit criterion: you can pull any tagged fact for any of your 20 companies in one query, and the corresponding filing text is cached and demonstrably clean.

**Week 2 — Question generation.** All five levels, filters, dedup. Hand-validate 50 questions across levels; if more than two or three are wrong, fix the generator before proceeding. Exit criterion: 600+ validated questions with a dropped-question log.

**Week 3 — Agent A and grading.** Single agent, tools, structured output, full grading pipeline with normalizers and error taxonomy. Exit criterion: an end-to-end run on 100 questions producing a metrics JSON.

**Week 4 — Agent B and full runs.** Multi-agent system, then complete runs of both systems on both backends. Exit criterion: four run manifests with complete artifacts.

**Week 5 — Analysis and writeup.** Confidence intervals, error taxonomy breakdown, cost-accuracy plot, `verify_results.py`, README. Exit criterion: CI green, every README number regenerable from logs.

**Week 6 — Buffer.** Something in weeks 1–3 will take longer than planned. Fiscal period alignment in L3 is the usual candidate.

---

## 10. Known traps

**Fiscal period semantics.** `ddate` is the period end date and `qtrs` is the duration: 0 for instantaneous balance-sheet items, 1 for a quarter, 4 for a full year. Getting this wrong silently produces an answer key that is confidently wrong, which is worse than a crash. Write unit tests against a handful of figures you have verified by hand from an actual filing.

**Fiscal year ≠ calendar year.** Apple's FY2024 ends in September, Nvidia's in January. Your L3 and L4 questions must align on fiscal periods, not calendar dates.

**Restatements and amendments.** 10-K/A filings restate prior figures. Always take the original filing as truth for a question about that filing's period, and exclude any fact later restated.

**Custom extension tags.** Companies extend the taxonomy. These are ambiguous and should be filtered out, but check how many you are dropping. If it is a large share for a given company, that company may be a poor benchmark candidate.

**Context limits.** A 10-K can run 200+ pages. Retrieval has to be real; you cannot stuff the filing into context, especially with a 4B local model. Chunking and retrieval quality will drive a meaningful share of your accuracy differences, so measure retrieval precision separately from answer accuracy. Otherwise you cannot tell whether a failure was retrieval or reasoning, and that distinction is most of the interesting analysis.

---

## 11. What makes this worth putting in front of a recruiter

The finished artifact is not "I built an agent." It is:

> "I built a benchmark of 700 auto-generated questions with XBRL-derived ground truth, measured a 14% hallucination rate on unanswerable questions, found that multi-agent orchestration cost 4x more for a 2-point accuracy gain inside the confidence interval, and every number in the README regenerates from the raw logs in CI."

Two of those clauses are negative results. That is the point. It is also the same discipline that produced a defensible null result in your RAG evaluation project, applied to a domain that maps directly onto quant and ML engineering interviews.

---

## Next steps

Tell me which piece to go deeper on and I can draft it: the question generator templates, the grader and normalizer logic, the tool interface for the agents, or the README results-table format.
