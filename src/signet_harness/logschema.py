"""The team's shared log format: one JSON line per run, so any stream's results can
be re-scored and merged without rerunning them.

Required fields (from the team plan): question_id, case, arm, model,
retrieved_passages, answer, flag, tokens, cost, latency. Extra fields record what
reproducing a run needs (seed, condition, served model version, prompt and harness
version, ...). `signet score` writes shared.jsonl next to results.jsonl and
scored.jsonl; `signet export-shared` rebuilds it for older runs.

`flag` is a list of flag reasons (empty = the answer carries no flag), or null if
the run hasn't been scored and its arm made no flag in code. The reason vocabulary
isn't agreed yet; PROPOSED_FLAG_REASONS is the proposal for the team (Track 2's
judge labels plus B's planned flag reasons). validate() doesn't enforce it.
"""

import json
from pathlib import Path

from . import __version__

# What every new run records as its harness version.
CURRENT_HARNESS = f"signet-harness {__version__}"

REQUIRED = ("question_id", "case", "arm", "model", "retrieved_passages", "answer", "flag", "tokens", "cost", "latency")
EXTRA = ("experiment", "run", "seed", "condition", "model_alias", "model_versions", "reasoning_effort",
         "prompt_version", "harness_version", "outcome", "judge_cost", "error", "timestamp", "key")

PROPOSED_FLAG_REASONS = (
    "outdated",                   # old or superseded version (Track 2 "outdated"; B "old version")
    "conflicting_value",          # retrieved sources disagree on the value (B, Experiment 3)
    "no_independent_reference",   # single-source value with nothing to check it against (B, Track 1)
    "unverifiable",               # no date / version / reference to check (Track 2, B)
    "inconsistent",               # a document's labels and content disagree (Track 2 C7)
    "unable_to_verify",           # the check itself failed, e.g. registry down (Track 2)
)


def _flag(rec: dict, scored: dict | None):
    code_flag = rec.get("flag")
    if code_flag is not None:
        return [code_flag] if isinstance(code_flag, str) else list(code_flag)
    if scored is not None:
        f = scored.get("flags", scored.get("flag"))
        if f is not None:
            return [f] if isinstance(f, str) else list(f)
    return None


def to_shared(rec: dict, scored: dict | None = None, meta: dict | None = None) -> dict:
    """A results.jsonl row (and its scored.jsonl row, if scored) -> one shared log line."""
    meta = meta or {}
    harness = rec.get("harness_version") or meta.get("harness_version")
    if not harness and meta.get("git_commit"):
        harness = f"trust-the-source@{meta['git_commit'][:7]}"   # Track 2 runs, before the package existed
    return {
        "question_id": rec["qid"],
        "case": rec["case"],
        "arm": rec["arm"],
        "model": rec["model"],
        "retrieved_passages": list(rec.get("retrieved") or []),
        "answer": rec.get("answer"),
        "flag": _flag(rec, scored),
        "tokens": dict(rec.get("usage") or {}),
        "cost": rec.get("cost_usd"),
        "latency": rec.get("latency_s"),
        "experiment": rec.get("experiment") or meta.get("experiment"),
        "run": meta.get("run"),
        "seed": rec.get("seed"),
        "condition": rec.get("condition"),
        "model_alias": rec.get("model_alias"),
        "model_versions": rec.get("model_versions"),
        "reasoning_effort": rec.get("reasoning_effort"),
        "prompt_version": rec.get("prompt_version"),
        "harness_version": harness,
        "outcome": (scored or {}).get("outcome"),
        "judge_cost": (scored or {}).get("judge_cost_usd"),
        "error": rec.get("error"),
        "timestamp": rec.get("timestamp"),
        "key": rec.get("key"),
    }


def validate(line: dict) -> list[str]:
    """Problems with one shared log line. Empty list = valid."""
    errs = [f"missing {k}" for k in REQUIRED if k not in line]
    if errs:
        return errs
    for k in ("question_id", "case", "arm", "model"):
        if not isinstance(line[k], str) or not line[k]:
            errs.append(f"{k} must be a non-empty string")
    if not isinstance(line["retrieved_passages"], list) or not all(isinstance(p, str) for p in line["retrieved_passages"]):
        errs.append("retrieved_passages must be a list of passage ids")
    if line["answer"] is not None and not isinstance(line["answer"], str):
        errs.append("answer must be a string or null")
    if line["flag"] is not None and (not isinstance(line["flag"], list) or not all(isinstance(f, str) for f in line["flag"])):
        errs.append("flag must be a list of strings or null")
    if not isinstance(line["tokens"], dict) or not all(isinstance(v, int) for v in line["tokens"].values()):
        errs.append("tokens must be a dict of integer counts")
    for k in ("cost", "latency"):
        if line[k] is not None and not isinstance(line[k], (int, float)):
            errs.append(f"{k} must be a number or null")
    return errs


def write_shared(run_dir: Path, experiment: str | None = None) -> int:
    """Join results.jsonl (+ scored.jsonl if present) into shared.jsonl. Returns the line count.
    `experiment` fills in the name for runs logged before it was recorded."""
    from .metrics import latest_per_job, load_jsonl
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
    meta = {"experiment": experiment, **meta, "run": run_dir.name}
    rows = [r for r in latest_per_job(load_jsonl(run_dir / "results.jsonl")) if not r.get("error")]
    scored = {}
    if (run_dir / "scored.jsonl").exists():
        scored = {_job(s): s for s in latest_per_job(load_jsonl(run_dir / "scored.jsonl"))}
    with open(run_dir / "shared.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(to_shared(r, scored.get(_job(r)), meta)) + "\n")
    return len(rows)


def _job(r: dict) -> tuple:
    return (r["model"], r["arm"], r["qid"], r["seed"], r["condition"])

