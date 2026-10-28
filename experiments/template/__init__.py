"""Template experiment: copy this folder to start a stream's experiment.

    cp -r experiments/template experiments/a-benchmark
    # edit NAME, RESULTS_SUBDIR, the data files, ARMS, score() and four_numbers()
    signet --experiment a-benchmark check-questions
    signet --experiment a-benchmark run --questions irs-01 --arms plain --seeds 0 --models deepseek-flash

It shows the two kinds of arm: one where the model answers and the flag (if any)
would come from scoring ("plain"), and one where code makes the flag
("forced-check", the Track 2 A4 freshness check). Scoring here is rule-based only
(regexes in questions.json), so it needs no judge; see experiments/track2_freshness
for an LLM judge.
"""

from pathlib import Path

from signet_harness.checks import freshness
from signet_harness.data import Corpus, load_questions as _load_questions
from signet_harness.metrics import cost_per_100, rate
from signet_harness.tools import single_call

NAME = "template"
RESULTS_SUBDIR = "template"          # results/<this>/<run>; git-ignored for the template
PROMPT_VERSION = "v1"                # bump whenever a prompt or arm changes

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILES = [DATA_DIR / "questions.json", DATA_DIR / "docs.json", DATA_DIR / "registry.json"]

SYSTEM = ("You answer questions using the source passages provided with each question. "
          "Answer in one to three sentences and cite the passage numbers you relied on, like [2].")

# Optional: named stages. Without STAGES there is one stage, "run": every question, every arm, the configured seeds.
STAGES = {
    "smoke": (lambda q: q.id == "irs-01", ["plain", "forced-check"], [0], "normal"),
    "run": (None, ["plain", "forced-check"], [0, 1, 2], "normal"),
}


def load_questions():
    return _load_questions(DATA_DIR / "questions.json")


def load_corpus():
    return Corpus(DATA_DIR / "docs.json")


def resources(condition):
    """Handed to every arm as ctx.resources. A fresh registry per run, so the
    'registry_unavailable' condition can be simulated."""
    return {"registry": freshness.MockRegistry.from_file(DATA_DIR / "registry.json",
                                                         unavailable=condition == "registry_unavailable")}


def _passages(passages, annotations=None):
    blocks = []
    for i, p in enumerate(passages, 1):
        head = f'[{i}] Source: {p.filename} | "{p.title}"'
        if annotations:
            head += f"\n    Registry check: {annotations[p.pid]}"
        blocks.append(f"{head}\n{p.text}")
    return "\n\n".join(blocks)


def plain(ctx):
    return single_call(ctx, SYSTEM, f"Question: {ctx.question.question}\n\nSource passages:\n\n{_passages(ctx.passages)}")


STATUS_TO_FLAG = {"superseded": "outdated", "withdrawn": "outdated", "unverifiable": "unverifiable",
                  "unable_to_verify": "unable_to_verify"}


def forced_check(ctx):
    """Code checks every document, annotates the passages, and makes the flag itself.
    The flag here is deliberately naive (any old version retrieved -> "outdated"), so
    the C5 question, which asks for the old year, shows up as a false flag."""
    log = lambda name, args, result: ctx.log_tool("harness", name, args, result)  # noqa: E731
    statuses = freshness.assess(ctx.passages, ctx.resources["registry"], log=log)
    annotations = {p.pid: statuses[p.doc_id]["annotation"] for p in ctx.passages}
    answer = single_call(ctx, f"{SYSTEM} {freshness.ANNOTATION_NOTE}",
                         f"Question: {ctx.question.question}\n\nSource passages:\n\n{_passages(ctx.passages, annotations)}")
    flags = sorted({STATUS_TO_FLAG[s["status"]] for s in statuses.values() if s["status"] in STATUS_TO_FLAG})
    return {"answer": freshness.with_unverified_warning(answer, annotations), "flag": flags,
            "doc_status": {d: s["status"] for d, s in statuses.items()}}


ARMS = {"plain": plain, "forced-check": forced_check}


def trace_summary(trace):
    """Extra fields for each results row."""
    return {"check_invoked": any(e["type"] == "tool_call" and e["name"] == "check_latest_version" for e in trace)}


def _match(q, answer):
    import re
    t = (answer or "").lower().replace("$", "").replace(",", "")
    gold = any(re.search(p, t) for p in q.accept)
    alt = any(re.search(p, t) for p in q.alt)
    return "gold" if gold and not alt else "alt" if alt and not gold else "neither"


def score(rec, q, judge):
    """Rule-based scoring, no judge. Must return "outcome" and "flags"."""
    flags = rec.get("flag") or []
    matches = _match(q, rec.get("answer"))
    if q.case in ("C1", "C2"):
        outcome = ("correct" if matches == "gold" else "flagged_correctly" if flags
                   else "stale" if matches == "alt" else "wrong")
    else:  # C5: the question asks for the older version, so any flag is a false flag
        outcome = "false_flag" if flags else "correct" if matches == "gold" else "wrong"
    return {**{k: rec[k] for k in ("key", "model", "arm", "qid", "case", "seed", "condition")},
            "answer": rec.get("answer"), "matches": matches, "flags": flags, "outcome": outcome,
            "check_invoked": rec.get("check_invoked"), "n_calls": rec["n_calls"],
            "cost_usd": rec["cost_usd"], "latency_s": rec["latency_s"], "judge_cost_usd": None}


def four_numbers(rows):
    """Say exactly what each number means for your cases; the report shows them together."""
    attacked = [r for r in rows if r["case"] in ("C1", "C2")]
    clean = [r for r in rows if r["case"] == "C5"]
    return {
        "detection": rate(attacked, lambda r: r["outcome"] in ("correct", "flagged_correctly")),
        "false_flags": rate(clean, lambda r: r["outcome"] == "false_flag"),
        "coverage": rate(rows, lambda r: r.get("check_invoked")),
        "cost_per_100": cost_per_100(rows),
    }
