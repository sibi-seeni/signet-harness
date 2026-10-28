# Changelog

Any change to prompts, scoring or retrieval bumps the minor version and is listed here, so a result can always be traced to the harness that produced it (every run records `harness_version` in `meta.json` and `shared.jsonl`).

## v0.1.0 (Oct 2026)

The first shared release, built from the Track 2 harness (trust-the-source at `389b781`).

**What's in it**

- `signet` command and `signet_harness` package: OpenRouter client with pinned model versions, preflight checks, parallel multi-model runner with caching, hybrid retrieval, cached LLM judge, cost estimates.
- Settings in `signet.toml` (+ `SIGNET_*` variables; the old `TTS_*` names still work). **Default pair: DeepSeek V4.1 Flash + Gemma 4 31B**, judge GPT-6 Luna at low effort. Nemotron 3.5, GLM 5.3, and the funded models (GPT-6 Sol, Gemini 4 Argon, Gemini 3.8 Flash) stay in the table, pinned, so switching once credits are approved is one line of config.
- The shared log format, `shared.jsonl`, written by every run, plus `signet export-shared` and `logschema.validate()`.
- The four numbers (`signet metrics`), defined per experiment.
- `signet_harness.checks.freshness`: the Track 2 A4 check as a library, plus the harness-written "couldn't verify" warning from closure recommendation 3.
- `experiments/track2_freshness` (reference, with all five Track 2 runs and the closure report in `results/track2/`) and `experiments/template` (copy-me).
- `signet check-questions` for validating and hashing a frozen question set.

**Parity with Track 2:** the prompts, tools and A4 registry lookups are byte-identical to what the old harness sent, checked against all 2,376 logged runs. All five reports, the hand-check sample and the cost estimates rebuild byte for byte.

**Changes from the Track 2 harness**

- Run cache keys now include the experiment name, so old `cache/` entries aren't reused. Results are unaffected.
- The embedding cache is written atomically.
- `retrieve()` takes `force_gold=False` for pipelines where retrieval is under test.
- `meta.json` adds the experiment, harness version, pinned versions of the models and judge, top-k, max tool steps, `force_gold`, the sha256 of every data file, and a `-dirty` marker on the git commit.

**Things to know before you build on it**

- **Gemma almost never calls optional tools** (1 of 45 A3 runs in Track 2). Arms that wait for the model to decide to call a tool will look broken on Gemma; do the check in code, or report the tool-call rate next to the result.
- **The flag vocabulary and each stream's coverage definition aren't agreed yet.** `logschema.PROPOSED_FLAG_REASONS` is the proposal for the Friday check-in.
- **The wheel contains only the library.** Experiments, data and results live in the repo, so a separate repo that installs from the tag copies whatever data it needs (e.g. the registry file).

**First live check (needs `OPENROUTER_API_KEY`, under one cent):**

```bash
signet preflight
signet --experiment template run --stage smoke --retrieval bm25 --name smoke
```
