"""Track 2 closure experiment: does a harness-level freshness check make stale
documents a non-issue? The reference experiment for signet-harness; results and
verdict in results/track2/CLOSURE_REPORT.md.

    signet --experiment track2_freshness run --stage pilot --name pilot
"""

import json
from pathlib import Path

from signet_harness.checks.freshness import MockRegistry
from signet_harness.data import Corpus, load_questions as _load_questions
from signet_harness.metrics import cost_per_100, export_handcheck, rate

from . import arms, report, scoring

NAME = "track2_freshness"
RESULTS_SUBDIR = "track2"
# Bump when prompts or arm logic change, so cached results aren't reused by mistake.
PROMPT_VERSION = "v2"

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILES = [DATA_DIR / "questions.json", DATA_DIR / "docs.json", DATA_DIR / "registry.json"]
REGISTRY_TIMEOUT_S = 10     # what the simulated 'registry unavailable' error reports

ARMS = arms.ARM_FUNCS
MAIN_ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
REGISTRY_ARMS = ["A3", "A4", "A5", "A6", "A7"]
SEEDS = [0, 1, 2]

STAGES = {
    # stage: (question filter, arms, seeds, condition)
    "pilot": (lambda q: q.pilot, MAIN_ARMS, [0], "normal"),
    "sweep": (lambda q: True, MAIN_ARMS, SEEDS, "normal"),
    "unavailable": (lambda q: q.case in ("C1", "C2"), REGISTRY_ARMS, [0], "registry_unavailable"),
    "live": (lambda q: q.live, ["A3-web"], [0], "live_web"),
    "confirm": (None, None, [0, 1], "normal"),     # questions and arms from data/confirm_set.json
}

ESTIMATE_NOTE = "The live stage also pays OpenRouter's web search fee per search, which isn't in the token math."

# Model calls per run, by arm (main model + verifier), from how each arm works.
CALLS_PER_RUN = {"A0": 1, "A1": 1, "A2": 1, "A3": 2, "A4": 1, "A5": 3, "A6": 4, "A7": 3, "A3-web": 2}

JUDGE_SYSTEM = scoring.JUDGE_SYSTEM
JUDGE_SCHEMA = scoring.JUDGE_SCHEMA
JUDGE_FALLBACK = scoring.JUDGE_FALLBACK

_registry_data = None


def load_questions():
    return _load_questions(DATA_DIR / "questions.json")


def load_corpus():
    return Corpus(DATA_DIR / "docs.json")


def load_registry() -> dict:
    global _registry_data
    if _registry_data is None:
        _registry_data = json.loads((DATA_DIR / "registry.json").read_text())
    return _registry_data


def resources(condition):
    return {"registry": MockRegistry(load_registry(), unavailable=condition == "registry_unavailable",
                                     timeout_s=REGISTRY_TIMEOUT_S)}


def trace_summary(trace):
    return {"check_invoked": arms.registry_was_checked(trace)}


def stage_override(stage, qfilter, stage_arms):
    if stage != "confirm":
        return qfilter, stage_arms
    path = DATA_DIR / "confirm_set.json"
    if not path.exists():
        raise SystemExit(f"Run `select-confirm` first: {path} doesn't exist.")
    cs = json.loads(path.read_text())
    wanted = set(cs["questions"])
    return (lambda q: q.id in wanted), cs["arms"]


def score(rec, q, judge):
    return scoring.score_record(rec, q, judge)


build_report = report.build_report


def four_numbers(rows):
    """Track 2's four numbers: detection = handled correctly on C1–C2 (right version, or
    a correct flag); false flags on C5–C6; coverage = share of C2–C4 runs where the
    registry was actually checked; cost per 100 runs."""
    c12 = [r for r in rows if r["case"] in ("C1", "C2")]
    c56 = [r for r in rows if r["case"] in ("C5", "C6")]
    c24 = [r for r in rows if r["case"] in ("C2", "C3", "C4")]
    return {
        "detection": rate(c12, lambda r: r["outcome"] in scoring.GOOD),
        "false_flags": rate(c56, lambda r: r["outcome"] == "false_flag"),
        "coverage": rate(c24, lambda r: r["check_invoked"]),
        "cost_per_100": cost_per_100(rows),
    }


def stage_runs(questions, n_questions=None) -> dict:
    """{stage: [(arm, n_runs)]} for the run plan in the design doc."""
    scale = (n_questions / len(questions)) if n_questions else 1.0
    n = lambda k: round(k * scale)  # noqa: E731
    n_pilot = sum(q.pilot for q in questions)
    n_c12 = n(sum(q.case in ("C1", "C2") for q in questions))
    n_live = sum(q.live for q in questions)
    return {
        "pilot": [(a, n_pilot) for a in MAIN_ARMS],
        "sweep": [(a, n(len(questions)) * len(SEEDS)) for a in MAIN_ARMS],
        "unavailable": [(a, n_c12) for a in REGISTRY_ARMS],
        "live": [("A3-web", n_live)],
    }


def register_cli(sub, common):
    from signet_harness.cli import _run_dir

    def cmd_select_confirm(args, exp):
        cs = report.select_confirm(_run_dir(exp, args.run) / "scored.jsonl", sample_n=args.sample)
        with open(args.out, "w") as f:
            json.dump(cs, f, indent=2)
        print(f"{len(cs['questions'])} questions ({len(cs['from_failures'])} from failures), arms {cs['arms']} -> {args.out}")

    def cmd_handcheck(args, exp):
        run_dir = _run_dir(exp, args.run)
        n = export_handcheck(run_dir / "scored.jsonl", run_dir / "handcheck.csv", report.HANDCHECK_FIELDS)
        print(f"{n} rows -> {run_dir / 'handcheck.csv'}")

    pc = sub.add_parser("select-confirm", parents=[common], help="pick the confirmation set from a scored sweep")
    pc.add_argument("--run", required=True)
    pc.add_argument("--sample", type=int, default=25)
    pc.add_argument("--out", default=str(DATA_DIR / "confirm_set.json"))
    pc.set_defaults(func=cmd_select_confirm)

    ph = sub.add_parser("handcheck", parents=[common], help="export a 10%% sample plus all rule/judge disagreements to CSV")
    ph.add_argument("--run", required=True)
    ph.set_defaults(func=cmd_handcheck)
