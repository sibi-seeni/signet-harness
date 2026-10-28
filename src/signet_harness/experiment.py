"""Loading an experiment. An experiment is a Python package (a folder with an
__init__.py) that defines:

    NAME              short id, used in cache keys and logs
    PROMPT_VERSION    bump when prompts or arm logic change, so cached results aren't reused
    ARMS              {arm name: fn(ctx: RunContext) -> answer str, or {"answer": str, "flag": [...], ...}}
    load_questions()  -> list[Question]
    load_corpus()     -> Corpus
    score(rec, question, judge) -> dict with at least "outcome" and "flags" (judge may be None)
    four_numbers(rows) -> {"detection", "false_flags", "coverage": (rate, n), "cost_per_100": float}

and optionally:

    RESULTS_SUBDIR    folder under results/ (default NAME)
    DATA_FILES        files whose sha256 goes in every run's meta.json (default: none)
    STAGES            {stage: (question filter or None, arms, seeds, condition)}; default one "run" stage
    FORCE_GOLD        include gold passages in retrieval (default True)
    JUDGE_SYSTEM, JUDGE_SCHEMA, JUDGE_FALLBACK   to score with an LLM judge
    resources(condition) -> dict      handed to every arm as ctx.resources
    trace_summary(trace) -> dict      extra fields for the results row (e.g. check_invoked)
    build_report(scored_path) -> str  report.md (default: the four-number table)
    stage_runs(questions, n_questions), CALLS_PER_RUN   for `signet estimate`
    register_cli(subparsers)          extra commands

See experiments/template/ for a small working example.
"""

import importlib.util
import sys
from pathlib import Path

from . import settings

REQUIRED = ("NAME", "PROMPT_VERSION", "ARMS", "load_questions", "load_corpus", "score", "four_numbers")


def resolve_path(spec: str) -> Path:
    s = settings.get()
    for candidate in (Path(spec), s.workspace / spec, s.workspace / "experiments" / spec):
        if (candidate / "__init__.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No experiment package at {spec!r} (looked in . , {s.workspace} and "
                            f"{s.workspace / 'experiments'}). An experiment is a folder with an __init__.py.")


def load(spec: str | None = None):
    spec = spec or settings.get().experiment
    if not spec:
        raise SystemExit("No experiment given: pass --experiment or set experiment = \"...\" in signet.toml.")
    path = resolve_path(spec)
    name = f"signet_experiment_{path.name.replace('-', '_')}"
    if name in sys.modules:
        return sys.modules[name]
    mod_spec = importlib.util.spec_from_file_location(name, path / "__init__.py",
                                                      submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[name] = module
    mod_spec.loader.exec_module(module)
    missing = [k for k in REQUIRED if not hasattr(module, k)]
    if missing:
        raise SystemExit(f"Experiment {path} is missing {missing}. See signet_harness/experiment.py.")
    module.PATH = path
    return module


def stages(exp) -> dict:
    return getattr(exp, "STAGES", None) or {"run": (None, list(exp.ARMS), list(settings.get().seeds), "normal")}


def results_dir(exp) -> Path:
    s = settings.get()
    return s.runs_dir or s.workspace / "results" / getattr(exp, "RESULTS_SUBDIR", exp.NAME)
