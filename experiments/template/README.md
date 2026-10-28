# Experiment template

Copy this folder to start a stream's experiment:

```bash
cp -r experiments/template experiments/a-benchmark
```

Then edit `__init__.py` (rename `NAME` and `RESULTS_SUBDIR`, write your arms, scoring and four numbers) and replace the files in `data/`. Run `python -m pytest -q` and `signet --experiment a-benchmark check-questions` before spending anything.

## The experiment contract

An experiment is a Python package (a folder with an `__init__.py`). The harness loads it by path, so relative imports between its own files work (`from . import arms`).

**Required**

| Name | What it is |
|---|---|
| `NAME` | short id; goes in cache keys and every log line |
| `PROMPT_VERSION` | bump it whenever a prompt or arm changes, so cached results aren't reused |
| `ARMS` | `{arm name: fn(ctx) -> str}`, or `-> {"answer": str, "flag": [reasons], ...}` when code makes the flag |
| `load_questions()` | `-> list[Question]`, usually `signet_harness.data.load_questions(path)` |
| `load_corpus()` | `-> Corpus`, usually `signet_harness.data.Corpus(path)` |
| `score(rec, question, judge)` | `-> dict` with at least `outcome` and `flags`, plus `key, model, arm, qid, case, seed, condition, cost_usd, latency_s, judge_cost_usd`. `judge` is `None` unless you define `JUDGE_SYSTEM` / `JUDGE_SCHEMA` |
| `four_numbers(rows)` | `-> {"detection": (rate, n), "false_flags": (rate, n), "coverage": (rate, n), "cost_per_100": float}`; `signet_harness.metrics.rate` and `cost_per_100` do the arithmetic |

**Optional**

| Name | Default |
|---|---|
| `RESULTS_SUBDIR` | `NAME`: runs go to `results/<this>/<run name>/` |
| `DATA_FILES` | none: list the files whose sha256 should go in `meta.json` (do list them) |
| `STAGES` | one stage, `run`: every question x every arm x the configured seeds, condition `normal`. Format: `{stage: (question filter or None, arms, seeds, condition)}` |
| `FORCE_GOLD` | `True`: gold passages are always retrieved (retrieval isn't under test). Set `False` for a product pipeline |
| `retrieve(q, corpus, embedder)` | hybrid BM25 + dense top-k from `corpus.pool_for(q)`; define it to use your own retrieval or a defense's |
| `resources(condition)` | `{}`: handed to every arm as `ctx.resources` (a registry, a reference set, a defense model loaded once) |
| `trace_summary(trace)` | `{}`: extra fields for the results row, e.g. whether a check ran |
| `JUDGE_SYSTEM`, `JUDGE_SCHEMA`, `JUDGE_FALLBACK` | no judge. With them, `score` gets a cached `Judge`; call `judge.label(prompt)` |
| `build_report(scored_path)` | the four-number table |
| `stage_runs`, `CALLS_PER_RUN`, `ESTIMATE_NOTE` | estimates from `STAGES` at one call per run |
| `register_cli(subparsers, common)` | extra commands, e.g. Track 2's `select-confirm` |

**Inside an arm** you have `ctx.question`, `ctx.passages`, `ctx.llm`, `ctx.seed`, `ctx.condition` and `ctx.resources`. Make model calls with `signet_harness.tools.single_call(ctx, system, user_text)` or `tool_loop(ctx, system, user_text, decls, handlers)`, so they are logged in the trace. Log anything code does with `ctx.log_tool(role, name, args, result)`. Test arms offline with `signet_harness.testing.FakeLLM`; see `tests/test_template.py`.

## Data formats

### `questions.json`

```json
{"questions": [
  {"id": "irs-01", "case": "C1", "family": "IRS-1040-INSTR",
   "question": "What is the standard deduction for a single filer?",
   "corpus": ["irs-2024", "irs-2025"],
   "gold_passages": ["irs-2024-p1", "irs-2025-p1"],
   "gold_answer": "$15,750 (2025 Instructions for Form 1040)",
   "alt_answer": "$14,600 (2024 instructions)",
   "accept": ["\\b15750\\b"], "alt": ["\\b14600\\b"],
   "verified": true}
]}
```

`id`, `case` and `question` are required. `case` is any string, so attack cases can be named for what they are (`single_source_edit`, `forged_label`, `agreeing_copies`, `number_reformatting`, `clean`, ...). `corpus` lists the documents "in the library" for this question (empty = all of them). `accept` / `alt` are regexes matched against the answer, lower-cased with `$` and thousands commas removed in Track 2's scorer. `verified: false` prints a warning on `run`. **Any other field is kept** in `question.extra`, so an attack set can carry `attack`, `target_doc`, `edited_value` and so on.

### `docs.json`

```json
{"docs": [
  {"doc_id": "irs-2025", "family": "IRS-1040-INSTR", "default_pool": true,
   "filename": "i1040gi_2025.pdf", "title": "2025 Instructions for Form 1040",
   "source_url": "https://www.irs.gov/pub/irs-pdf/i1040gi.pdf",
   "ingest": {"date": "2025", "version": "Tax Year 2025", "status": "active"},
   "passages": [{"pid": "irs-2025-p1", "locator": "What's New", "text": "..."}]}
]}
```

`default_pool: true` documents are added to every other family's questions as distractors. `ingest` is the metadata a pipeline would have recorded at ingestion (Track 2's A2 arm shows it to the model).

### `registry.json` (for the freshness check)

```json
{"families": {
  "IRS-1040-INSTR": {
    "aliases": ["form 1040 instructions", "i1040gi"],
    "latest": "Tax Year 2025",
    "versions": [
      {"version": "Tax Year 2024", "aliases": ["2024", "tax year 2024"], "effective": "2024",
       "status": "superseded", "superseded_by": "Tax Year 2025"},
      {"version": "Tax Year 2025", "aliases": ["2025"], "effective": "2025", "status": "current"}
    ]}
}}
```

A lookup matches a family when one of its aliases appears in the (lower-cased) title or filename, then a version when one of its aliases appears as a whole token. Versions can also carry `withdrawn` (a date) and `changes` (`[{"release": "5.2.0", "date": "2025-08-27", ...}]`) for in-place amendments.

## Freezing a shared set

When a question set is frozen (e.g. the attack set on Oct 30), commit it, tag it, and list it in `DATA_FILES`. `signet check-questions --file <path>` prints its sha256, and every run's `meta.json` records the hash, so anyone can tell which version a result used.
