# Track 2: freshness check closure experiment

Does a harness-level freshness check make stale documents a non-issue? **Yes, if the harness does the check (arm A4): stale answers on C1–C2 drop from 44% to 0% on both models, with no false flags.** The full verdict is in [`results/track2/CLOSURE_REPORT.md`](../../results/track2/CLOSURE_REPORT.md), and the design, with diagrams of every arm, is in [`docs.md`](docs.md).

This is the reference experiment for the shared harness. Its prompts, arms and scoring are unchanged from the original harness (trust-the-source at `389b781`); `tests/test_parity.py` checks that against all 2,376 logged runs.

| File | What's in it |
|---|---|
| `__init__.py` | stages (pilot, sweep, unavailable, live, confirm), resources (the mock registry), four numbers, extra commands |
| `arms.py` | arms A0–A7 and A3-web |
| `prompts.py` | prompt text shared by the arms |
| `scoring.py` | rule match, judge instructions and schema, case → outcome mapping |
| `report.py` | headline tables, closure rule, confirmation-set selection |
| `data/` | 38 questions, 15 documents, the version registry, and source page excerpts in `data/files/` |
| `skills/version-check/` | the skill arm A5 loads |

## Reproducing

```bash
signet --experiment track2_freshness report --run sweep-20261028-010735      # rebuild a report, no API calls
signet --experiment track2_freshness metrics --run sweep-20261028-010735     # the four numbers
signet --experiment track2_freshness run --stage pilot --name pilot-rerun    # ~$0.15 per model
```

Stages: `pilot` (11 questions x A0–A7), `sweep` (38 x A0–A7 x 3 seeds), `unavailable` (C1–C2 x A3–A7, registry times out), `live` (20 x A3 with web search), `confirm` (after `select-confirm --run <sweep>`). The cases (C1–C7), outcomes and closure rule are described in `docs.md` and the closure report.
