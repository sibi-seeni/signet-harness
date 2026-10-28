# signet-harness

Team Signet's shared experiment harness. Every stream (Track 1, A defense benchmark, B trust-layer prototype) runs through it, so results are comparable and reproducible:

- **One OpenRouter client** with the model as a setting, every model **pinned** to a dated version (preflight fails if an id silently starts pointing at a new one), reasoning and provider settings fixed per model, and OpenRouter's reported cost logged on every call.
- **One log format**: `shared.jsonl`, one line per run with `question_id, case, arm, model, retrieved_passages, answer, flag, tokens, cost, latency` plus what reproducing it needs.
- **The same four numbers** for every result: detection, false flags, coverage, cost per 100 questions.
- **The Track 2 freshness check as a library** (`signet_harness.checks.freshness`), byte-identical to the A4 arm that closed Track 2.

It grew out of the Track 2 harness ([trust-the-source](https://github.com/sibi-seeni/trust-the-source) at `389b781`). Track 2 now lives here as the reference experiment, with its runs in [`results/track2/`](results/track2) and its verdict in [`results/track2/CLOSURE_REPORT.md`](results/track2/CLOSURE_REPORT.md).

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/sibi-seeni/signet-harness.git && cd signet-harness
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest -q                 # offline, no key needed
export OPENROUTER_API_KEY=...       # https://openrouter.ai/settings/keys (set a credit limit on it)
signet info --experiment template   # settings, models, arms
signet preflight                    # models listed and pinned, tool calls, judge JSON, embeddings (< 1 cent)
```

A separate repo (e.g. B's prototype) can depend on a tagged release instead:

```bash
pip install "signet-harness @ git+https://github.com/sibi-seeni/signet-harness@v0.1.0"
```

## Models

Set in [`signet.toml`](signet.toml) and logged in every run's `meta.json`. **Default pair: DeepSeek V4.1 Flash + Gemma 4 31B**, run in parallel; GPT-6 Luna (low effort) is the judge.

| Alias | OpenRouter id → pinned version | Tier | Reasoning | $ / 1M in / out |
|---|---|---|---|---|
| `deepseek-flash` | `deepseek/deepseek-v4.1-flash` → `-20260910` | budget, **default** | model default | 0.14 / 0.42 |
| `gemma-4-31b` | `google/gemma-4-31b-it` → `-20260402` | budget, **default** | on; fp8-or-better providers only | 0.20 / 0.40 |
| `gpt-6-luna` | `openai/gpt-6-luna` → `-20260922` | budget, **judge** | low | 0.10 / 0.50 |
| `nemotron-3.5-super-vl` | `nvidia/nemotron-3.5-super-vl` → `-20261015` | budget | default | 0.15 / 0.50 |
| `glm-5.3` | `z-ai/glm-5.3` → `-20260816` | budget | low | 0.07 / 1.40 |
| `gpt-6-sol` | `openai/gpt-6-sol` → `-20260922` | funded | low | 2.00 / 10.00 |
| `gemini-4-argon` | `google/gemini-4-argon` → `-20261013` | funded | default | 2.00 / 10.00 |
| `gemini-3.8-flash` | `google/gemini-3.8-flash` → `-20260902` | funded | default | 0.83 / 4.13 |

**Switching is a settings change, not a code change.** Once more credits are approved, change `models` / `judge_model` in `signet.toml`, or for one run use `--models gemini-4-argon` or `SIGNET_MODELS=...` / `SIGNET_JUDGE_MODEL=gpt-6-sol`. Run `signet preflight --models <alias>` first. If you switch mid-stream, rerun every arm you compare on the new model; never mix models inside one table. A model that isn't in the table goes under `[models_table.<alias>]` in `signet.toml` (example in the file).

Two things Track 2 learned that matter for your arms:

- **Gemma almost never calls optional tools** (1 of 45 A3 runs, never the skill or the sub-agent). An arm that relies on the model choosing to call a tool will look broken on Gemma. Do the check in code (like A4) or report the tool-call rate next to the result.
- **Nemotron's listing doesn't confirm tool calling everywhere.** `preflight` checks `tools` / `tool_choice` in `supported_parameters` for every model under test.

## Running an experiment

An experiment is a folder under [`experiments/`](experiments): its data, its arms, its scoring, and what its four numbers mean. Start from the template:

```bash
cp -r experiments/template experiments/a-benchmark       # then edit NAME, RESULTS_SUBDIR, data, arms, score, four_numbers
signet --experiment a-benchmark check-questions          # validate the question set against the corpus
signet --experiment a-benchmark estimate                 # cost per stage and model, no API calls
signet --experiment a-benchmark run --stage smoke --models deepseek-flash
signet --experiment a-benchmark run --name week2-main    # every question x arm x seed, on both models in parallel
```

[`experiments/template/README.md`](experiments/template/README.md) documents the experiment contract and the `questions.json` / `docs.json` / `registry.json` formats. Set `SIGNET_EXPERIMENT=a-benchmark` to drop the flag.

| Command | What it does | Key? |
|---|---|---|
| `info` | settings, models, the experiment's arms and stages | no |
| `preflight [--models ...] [--retrieval bm25]` | models listed and pinned, tool support, a tool round-trip, judge JSON, embeddings | yes |
| `check-questions [--file F] [--no-corpus]` | validates a question set (e.g. the frozen attack set) and prints its sha256 | no |
| `estimate [--models] [--judges] [--questions N] [--from-run R]` | cost table per stage and model | no |
| `run [--stage S] [--arms] [--questions] [--seeds] [--models] [--name]` | runs, then scores, then writes `report.md` and `shared.jsonl` | yes |
| `score --run R` | re-scores from `results.jsonl` (judge calls are cached) | if judged |
| `report --run R` / `metrics --run R` | rebuilds `report.md` / prints the four numbers (no API calls) | no |
| `export-shared --run R` | rebuilds and validates `shared.jsonl` | no |

Useful `run` flags: `--workers 8` (runs in flight across all models), `--rpm 10` (per-model rate cap), `--retrieval bm25` (skip embeddings), `--no-cache`, `--no-score`. Track 2 adds `select-confirm` and `handcheck`.

**Caching:** runs are cached by (experiment, model, reasoning effort, arm, question, seed, condition, prompt version, retrieved passages) in `.signet-cache/`, so an interrupted run is resumed by running the same command again, and failed runs are retried. Bump your experiment's `PROMPT_VERSION` whenever a prompt or an arm changes.

## Output of a run

Each run lands in `results/<stream>/<run name>/`. Each stream only writes to its own folder.

| File | Contents |
|---|---|
| `meta.json` | experiment, harness version, models with pinned versions, judge, temperature, top-k, arms, seeds, questions, prompt version, **sha256 of every data file**, git commit (with `-dirty` if uncommitted), start time |
| `results.jsonl` | one line per run with the **full trace**: every request and response, tool call and registry lookup |
| `scored.jsonl` | one line per run: outcome, flags, judge labels and cost |
| `shared.jsonl` | **the team log format** (below) |
| `report.md` | the experiment's report (default: the four-number table) |

### The shared log format

| Field | Meaning |
|---|---|
| `question_id`, `case`, `arm`, `model` | `model` is the OpenRouter id that was sent |
| `retrieved_passages` | passage ids, in the order the model saw them (data files are hashed in `meta.json`) |
| `answer` | the final answer text |
| `flag` | list of flag reasons; `[]` = no flag; `null` = not scored and no flag made in code |
| `tokens` | `{prompt, output, thoughts, cached}` summed over the run's model calls |
| `cost`, `latency` | USD from OpenRouter, seconds |
| extras | `experiment, run, seed, condition, model_alias, model_versions, reasoning_effort, prompt_version, harness_version, outcome, judge_cost, error, timestamp, key` |

`signet_harness.logschema.validate(line)` checks a line. **The flag vocabulary isn't agreed yet.** `PROPOSED_FLAG_REASONS` in [`logschema.py`](src/signet_harness/logschema.py) is the proposal for the Friday check-in: `outdated, conflicting_value, no_independent_reference, unverifiable, inconsistent, unable_to_verify` (Track 2's judge labels plus B's planned reasons).

### The four numbers

Every report shows them together, per model and arm (`signet metrics --run R`):

| Number | Each experiment defines it in `four_numbers(rows)` | Track 2's definition |
|---|---|---|
| Detection | share of attacked / stale cases handled | C1–C2 answered from the right version or correctly flagged |
| False flags | share of clean or legitimately changed cases that got flagged | C5–C6 false flags |
| Coverage | share of cases the check actually ran on | C2–C4 runs where the registry was queried |
| Cost / 100 questions | models under test only; the judge is reported separately | same |

Write down what coverage means for your stream: it's the number most likely to differ between streams.

## Using the freshness check (B)

```python
from signet_harness.checks import freshness

registry = freshness.MockRegistry.from_file("experiments/track2_freshness/data/registry.json")
statuses = freshness.assess(passages, registry)        # doc_id -> {"status", "latest", "in_place_changes", "annotation", ...}
annotations = freshness.annotate(passages, registry)   # pid -> the "Registry check" line A4 put on each passage
system = f"{base_system} {freshness.ANNOTATION_NOTE}"
answer = freshness.with_unverified_warning(answer, annotations)   # registry down -> warning added in code
```

`status` is one of `current, superseded, withdrawn, unverifiable, ambiguous, not_found, unable_to_verify`. Its known limits, from the closure report: documents amended in place under the same version label (NIST Rev 5, releases 5.1.1 and 5.2.0) come back as `current`, with `in_place_changes` listed, so a flag has to use that; and undated or self-contradictory documents can be flagged but not fixed. The template's `forced-check` arm shows it wired in with a code-made flag.

## Wrapping a published defense (A)

Keep each defense's code and heavy dependencies inside your stream (vendor it or install it in your own venv) and call it from an arm. Use its README defaults. If it makes its own LLM calls, point its OpenAI client at OpenRouter with the same pinned model:

```python
from signet_harness.llm import openai_compatible
cfg = openai_compatible("deepseek-flash")    # base_url, api_key, model id, temperature, extra_body (reasoning, provider)
```

Calls made this way aren't in the trace; record their cost in the arm's output if the tool exposes it. Defenses that run local models (e.g. MiniCheck) log cost 0 and their latency. An arm can return `{"answer": ..., "flag": [...], ...}` when the defense, not the model, decides the flag; anything else in the dict is kept in `arm_output`.

## Layout

```
signet.toml                      team-wide settings (models, judge, temperature, top-k, seeds)
src/signet_harness/
  settings.py, models.py         settings loading; the pinned model table
  llm.py                         OpenRouter client: tool calls, reasoning, provider filter, retries, rate limit, cost
  tools.py                       RunContext and the tool loop every arm uses
  data.py, retrieval.py          questions and corpus; BM25 + dense hybrid retrieval
  runner.py                      parallel multi-model runner, caching, results.jsonl
  judge.py                       cached schema-JSON LLM judge
  logschema.py, metrics.py       shared log format; four numbers; hand-check export
  preflight.py, estimate.py      pre-spend checks; cost estimates
  checks/freshness.py            the Track 2 A4 freshness check
  testing.py                     FakeLLM, for offline tests of your own arms
  cli.py                         the `signet` command
experiments/
  track2_freshness/              reference experiment: arms A0–A7, scoring, closure rule, data, skill
  template/                      copy-me example
results/<stream>/                each stream's runs; track2 has the closure runs and report
tests/                           offline tests, including parity with the Track 2 harness
```

## Versions and the freeze

Releases are git tags (`v0.1.0`, ...); every run records the harness version. Results that go in the report should all name one version.

- **v0.1.0 (Oct 31):** this release.
- **v0.2.0 (~Nov 5):** the freshness API settled with B, the agreed flag vocabulary, fixes from A's first week.
- **Freeze on Nov 8:** after that, patch releases only. Any change to prompts, scoring or retrieval bumps the minor version and is called out in [`CHANGELOG.md`](CHANGELOG.md).

## Tests

`python -m pytest -q` runs offline. Besides unit tests, [`tests/test_parity.py`](tests/test_parity.py) re-runs every arm of all 2,376 logged Track 2 runs with a fake model and checks that the first request (system prompt, user message, tools), and for A4 every registry lookup, is byte-identical to what the old harness sent. It also checks that all five Track 2 reports and the hand-check sample rebuild byte for byte.
