"""Command line: signet [--experiment PATH] <command> [options]. See README.md."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import __version__, experiment, logschema, models, settings
from .data import check_questions, file_sha256
from .estimate import estimate, measured_profile, stage_runs
from .judge import Judge, make_judge_llm
from .llm import OpenRouterLLM
from .metrics import four_numbers_table, latest_per_job, load_jsonl
from .retrieval import OpenRouterEmbedder
from .runner import interleave, run_jobs


def _require_key():
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("Set OPENROUTER_API_KEY (create one at https://openrouter.ai/settings/keys).")


def _git_commit(cwd):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=cwd)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd).stdout.strip()
        return (out.stdout.strip() + ("-dirty" if dirty else "")) or None
    except OSError:
        return None


def _model_list(arg):
    return [m.strip() for m in arg.split(",") if m.strip()] if arg else list(settings.get().models)


def _run_dir(exp, arg):
    p = Path(arg)
    return p if p.is_dir() else experiment.results_dir(exp) / arg


def make_judge(exp):
    if not hasattr(exp, "JUDGE_SYSTEM"):
        return None
    return Judge(make_judge_llm(), exp.JUDGE_SYSTEM, exp.JUDGE_SCHEMA, version=exp.PROMPT_VERSION,
                 cache_dir=settings.get().cache / exp.NAME / "judge", fallback=getattr(exp, "JUDGE_FALLBACK", None))


def cmd_info(args, exp):
    s = settings.get()
    print(f"signet-harness {__version__}")
    print(f"config file: {s.config_file or '(none, defaults)'}\nworkspace:   {s.workspace}")
    if exp:
        print(f"experiment:  {exp.NAME} ({exp.PATH}), prompt version {exp.PROMPT_VERSION}, "
              f"results -> {experiment.results_dir(exp)}")
        print(f"arms:        {', '.join(exp.ARMS)}")
        print(f"stages:      {', '.join(experiment.stages(exp))}")
    print(f"temperature {s.temperature}, top_k {s.top_k}, seeds {list(s.seeds)}, max tool steps {s.max_tool_steps}")
    print("\nModels under test:")
    for m in s.models:
        spec = models.resolve(m)
        print(f"  {spec['alias']:24} {spec['slug']:36} pinned {spec.get('canonical') or '-'}  "
              f"reasoning {spec['reasoning'] or 'default'}  ({spec['tier']})")
    j = models.resolve(s.judge_model)
    print(f"Judge: {j['alias']} -> {j['slug']} (effort {s.judge_reasoning_override or j['reasoning'] or 'default'})")
    print(f"Embeddings: {s.embed_model}")
    print("\nOther models in the table (switch with --models, SIGNET_MODELS or signet.toml):")
    for alias, spec in models.table().items():
        if alias not in s.models and alias != s.judge_model:
            print(f"  {alias:24} {spec['slug']:36} {spec['tier']}")


def cmd_preflight(args, exp):
    from .preflight import run_preflight
    _require_key()
    ok = run_preflight(_model_list(args.models), getattr(exp, "JUDGE_SCHEMA", None), args.retrieval)
    sys.exit(0 if ok else 1)


def cmd_run(args, exp):
    _require_key()
    s = settings.get()
    questions = exp.load_questions()
    stage_table = experiment.stages(exp)
    if args.stage not in stage_table:
        sys.exit(f"Unknown stage {args.stage!r}; this experiment has {list(stage_table)}.")
    qfilter, stage_arms, seeds, condition = stage_table[args.stage]
    if hasattr(exp, "stage_override"):
        qfilter, stage_arms = exp.stage_override(args.stage, qfilter, stage_arms)
    selected = [q for q in questions if qfilter is None or qfilter(q)]
    if args.questions:
        ids = set(args.questions.split(","))
        selected = [q for q in selected if q.id in ids]
    arm_list = args.arms.split(",") if args.arms else list(stage_arms)
    seed_list = [int(x) for x in args.seeds.split(",")] if args.seeds else list(seeds)
    unknown = [a for a in arm_list if a not in exp.ARMS]
    if unknown:
        sys.exit(f"Unknown arms: {unknown}")
    unverified = [q.id for q in selected if not q.verified]
    if unverified:
        print(f"NOTE: {len(unverified)} questions have verified=false. Verify gold answers and freeze the data in Git "
              "before a full run.")

    llms = [OpenRouterLLM(model=m, reasoning_effort=s.reasoning_override, rpm=args.rpm) for m in _model_list(args.models)]
    jobs = interleave(llms, [(q, a, sd, condition) for q in selected for a in arm_list for sd in seed_list])
    judge = make_judge(exp) if not args.no_score else None
    name = args.name or f"{args.stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = experiment.results_dir(exp) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    judge_spec = models.resolve(s.judge_model)
    (run_dir / "meta.json").write_text(json.dumps({
        "experiment": exp.NAME, "experiment_path": str(exp.PATH.relative_to(s.workspace)
                                                       if exp.PATH.is_relative_to(s.workspace) else exp.PATH),
        "harness_version": logschema.CURRENT_HARNESS,
        "stage": args.stage,
        "models": [{"alias": m.alias, "slug": m.model, "canonical": m.canonical,
                    "reasoning_effort": m.reasoning_effort or "default",
                    **({"provider_filter": m.provider} if m.provider else {})} for m in llms],
        "temperature": s.temperature, "top_k": s.top_k, "max_tool_steps": s.max_tool_steps,
        "judge_model": judge_spec["slug"] if hasattr(exp, "JUDGE_SYSTEM") else None,
        "judge_canonical": judge_spec.get("canonical") if hasattr(exp, "JUDGE_SYSTEM") else None,
        "judge_reasoning_effort": (judge.llm.reasoning_effort if judge else
                                   s.judge_reasoning_override or judge_spec["reasoning"]),
        "embed_model": s.embed_model if args.retrieval == "hybrid" else None,
        "retrieval": args.retrieval, "force_gold": getattr(exp, "FORCE_GOLD", True),
        "arms": arm_list, "seeds": seed_list, "condition": condition,
        "questions": [q.id for q in selected], "prompt_version": exp.PROMPT_VERSION,
        "data_sha256": {str(Path(p).name): file_sha256(p) for p in getattr(exp, "DATA_FILES", [])},
        "git_commit": _git_commit(s.workspace), "config_file": str(s.config_file) if s.config_file else None,
        "started": datetime.now().isoformat(),
    }, indent=2))
    print(f"{len(jobs)} runs ({len(selected)} questions x {len(arm_list)} arms x {len(seed_list)} seeds "
          f"x {len(llms)} models: {', '.join(m.alias for m in llms)}) -> {run_dir}")

    embedder = OpenRouterEmbedder(llms[0].http) if args.retrieval == "hybrid" else None
    run_jobs(exp, jobs, exp.load_corpus(), run_dir / "results.jsonl", embedder=embedder,
             workers=args.workers, use_cache=not args.no_cache)
    if not args.no_score:
        score(exp, run_dir, args.workers, judge)
    else:
        logschema.write_shared(run_dir, exp.NAME)


def score(exp, run_dir, workers=4, judge=None):
    qs = {q.id: q for q in exp.load_questions()}
    rows = [r for r in latest_per_job(load_jsonl(run_dir / "results.jsonl")) if not r.get("error")]
    judge = judge or make_judge(exp)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(lambda r: exp.score(r, qs[r["qid"]], judge), rows))
    with open(run_dir / "scored.jsonl", "w") as f:
        for row in scored:
            f.write(json.dumps(row) + "\n")
    report = write_report(exp, run_dir)
    n = logschema.write_shared(run_dir, exp.NAME)
    print("\n" + report + f"\n{n} lines -> {run_dir / 'shared.jsonl'}")


def write_report(exp, run_dir) -> str:
    scored_path = run_dir / "scored.jsonl"
    if hasattr(exp, "build_report"):
        report = exp.build_report(scored_path)
    else:
        report = four_numbers_table(load_jsonl(scored_path), exp.four_numbers, run_dir.name)
    (run_dir / "report.md").write_text(report)
    return report


def cmd_score(args, exp):
    if hasattr(exp, "JUDGE_SYSTEM"):
        _require_key()
    score(exp, _run_dir(exp, args.run), workers=args.workers)


def cmd_report(args, exp):
    print(write_report(exp, _run_dir(exp, args.run)))


def cmd_metrics(args, exp):
    run_dir = _run_dir(exp, args.run)
    table = four_numbers_table(load_jsonl(run_dir / "scored.jsonl"), exp.four_numbers, run_dir.name)
    (run_dir / "four_numbers.md").write_text(table)
    print(table)


def cmd_export_shared(args, exp):
    run_dir = _run_dir(exp, args.run)
    n = logschema.write_shared(run_dir, exp.NAME)
    bad = [(i, e) for i, line in enumerate(load_jsonl(run_dir / "shared.jsonl"), 1) if (e := logschema.validate(line))]
    print(f"{n} lines -> {run_dir / 'shared.jsonl'}" + (f"; {len(bad)} invalid, first: {bad[0]}" if bad else ", all valid"))
    if bad:
        sys.exit(1)


def cmd_check_questions(args, exp):
    from .data import load_questions
    qs = load_questions(args.file) if args.file else exp.load_questions()
    problems = check_questions(qs, exp.load_corpus() if not args.no_corpus else None)
    cases = {}
    for q in qs:
        cases[q.case] = cases.get(q.case, 0) + 1
    print(f"{len(qs)} questions; cases: {dict(sorted(cases.items()))}")
    if args.file:
        print(f"sha256 {file_sha256(args.file)}")
    for p in problems:
        print("  " + p)
    if problems:
        sys.exit(1)


def cmd_estimate(args, exp):
    questions = exp.load_questions()
    s = settings.get()
    model_list = _model_list(args.models)
    judges = args.judges.split(",") if args.judges else [s.judge_model, "gpt-6-sol"]
    profile = measured_profile(_run_dir(exp, args.from_run) / "results.jsonl") if args.from_run else None
    if hasattr(exp, "stage_runs"):
        plan = exp.stage_runs(questions, args.questions)
    else:
        plan = stage_runs(questions, experiment.stages(exp), args.questions)
    n = args.questions or len(questions)
    print(f"# Cost estimate ({n} questions{', measured from ' + args.from_run if args.from_run else ''})\n")
    kw = {"note": exp.ESTIMATE_NOTE} if hasattr(exp, "ESTIMATE_NOTE") else {}
    print(estimate(plan, model_list, judges, getattr(exp, "CALLS_PER_RUN", None), profile=profile, **kw))


def build_parser(exp=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--experiment", default=argparse.SUPPRESS,
                        help="experiment folder (default: `experiment` in signet.toml)")
    p = argparse.ArgumentParser(prog="signet", parents=[common])
    p.add_argument("--version", action="version", version=f"signet-harness {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = settings.get()
    default_models = ",".join(s.models)

    pi = sub.add_parser("info", parents=[common], help="show settings, models and the experiment")
    pi.set_defaults(func=cmd_info, needs_exp=False)

    pf = sub.add_parser("preflight", parents=[common], help="check the models, tool calling and JSON output before spending money")
    pf.add_argument("--models", help=f"comma-separated aliases or slugs (default: {default_models})")
    pf.add_argument("--retrieval", choices=["hybrid", "bm25"], default="hybrid")
    pf.set_defaults(func=cmd_preflight, needs_exp=False)

    pr = sub.add_parser("run", parents=[common], help="run a stage on all models in parallel, then score it")
    pr.add_argument("--stage", default="run", help="a stage from the experiment's STAGES (default: run)")
    pr.add_argument("--models", help=f"comma-separated aliases or OpenRouter slugs, run in parallel (default: {default_models})")
    pr.add_argument("--arms", help="comma-separated (default: the stage's arms)")
    pr.add_argument("--seeds", help="comma-separated, e.g. 0,1,2")
    pr.add_argument("--questions", help="comma-separated question ids to restrict to")
    pr.add_argument("--name", help="run folder name (default: <stage>-<timestamp>)")
    pr.add_argument("--workers", type=int, default=s.workers, help="parallel runs across all models")
    pr.add_argument("--rpm", type=float, default=None, help="cap on requests per minute, per model")
    pr.add_argument("--retrieval", choices=["hybrid", "bm25"], default="hybrid")
    pr.add_argument("--no-cache", action="store_true")
    pr.add_argument("--no-score", action="store_true")
    pr.set_defaults(func=cmd_run)

    pe = sub.add_parser("estimate", parents=[common], help="estimate cost per stage and model (no API calls)")
    pe.add_argument("--models", help=f"comma-separated aliases (default: {default_models})")
    pe.add_argument("--judges", help=f"comma-separated aliases (default: {s.judge_model},gpt-6-sol)")
    pe.add_argument("--questions", type=int, help="scale to this many questions")
    pe.add_argument("--from-run", help="use measured tokens and calls from this run")
    pe.set_defaults(func=cmd_estimate)

    for name, func, help_text in (
            ("score", cmd_score, "(re)score a run from its results.jsonl, then write report.md and shared.jsonl"),
            ("report", cmd_report, "rebuild report.md from scored.jsonl (no API calls)"),
            ("metrics", cmd_metrics, "print the four numbers per model and arm, and write four_numbers.md"),
            ("export-shared", cmd_export_shared, "write and validate shared.jsonl, the team log format")):
        sp = sub.add_parser(name, parents=[common], help=help_text)
        sp.add_argument("--run", required=True, help="run folder name or path")
        if name == "score":
            sp.add_argument("--workers", type=int, default=4)
        sp.set_defaults(func=func)

    pq = sub.add_parser("check-questions", parents=[common], help="validate a question set (e.g. the frozen attack set)")
    pq.add_argument("--file", help="questions.json to check (default: the experiment's own)")
    pq.add_argument("--no-corpus", action="store_true", help="don't check docs and gold passages against the corpus")
    pq.set_defaults(func=cmd_check_questions)

    if exp is not None and hasattr(exp, "register_cli"):
        exp.register_cli(sub, common)
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--experiment")
    known, _ = pre.parse_known_args(argv)
    exp = None
    try:
        exp = experiment.load(known.experiment) if (known.experiment or settings.get().experiment) else None
    except FileNotFoundError as e:
        sys.exit(str(e))
    args = build_parser(exp).parse_args(argv)
    if exp is None and getattr(args, "needs_exp", True):
        sys.exit("No experiment given: pass --experiment or set experiment = \"...\" in signet.toml.")
    args.func(args, exp)


if __name__ == "__main__":
    main()
