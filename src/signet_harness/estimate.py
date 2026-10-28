"""Cost estimates per stage and model, from assumed token counts or from a run's
measured usage (e.g. the pilot)."""

from collections import defaultdict

from . import models, settings
from .metrics import latest_per_job, load_jsonl


# Tokens per call. The base prompt (system + 5 passages) measures ~670 tokens;
# tool declarations and tool results push later turns higher. Output includes
# reasoning tokens, which is what really swings the cost.
SCENARIOS = {
    "low": {"in": 1100, "out": 400},
    "high": {"in": 1600, "out": 1500},
}
# Judge: ~900 prompt tokens (instructions + question + answer); output is a short
# JSON label plus reasoning, which grows with the effort setting (low, high).
JUDGE_IN = 900
JUDGE_OUT = {"low": (250, 600), "medium": (500, 1200), "high": (1000, 2500), "default": (500, 1200)}


def stage_runs(questions, stages, n_questions=None) -> dict:
    """{stage: [(arm, n_runs)]} from an experiment's STAGES (stages without a
    question filter, like a confirmation round, are left out)."""
    scale = (n_questions / len(questions)) if n_questions else 1.0
    plan = {}
    for name, (qfilter, arms, seeds, _condition) in stages.items():
        if qfilter is None and name != "run":
            continue
        n = round(sum(1 for q in questions if qfilter is None or qfilter(q)) * scale)
        plan[name] = [(a, n * len(seeds)) for a in arms]
    return plan


def measured_profile(results_path) -> dict:
    """Per-arm average calls per run and tokens per call, from a results.jsonl."""
    rows = [r for r in latest_per_job(load_jsonl(results_path)) if not r.get("error")]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    profile = {}
    for arm, rs in by_arm.items():
        calls = sum(r["n_calls"] for r in rs)
        profile[arm] = {
            "calls": calls / len(rs),
            "in": sum(r["usage"]["prompt"] for r in rs) / max(calls, 1),
            "out": sum(r["usage"]["output"] + r["usage"]["thoughts"] for r in rs) / max(calls, 1),
        }
    return profile


def estimate(plan, model_list, judges, calls_per_run=None, profile=None,
             note="Web search fees, where an arm uses them, aren't in the token math.") -> str:
    """`plan` is {stage: [(arm, n_runs)]}; `calls_per_run` is {arm: model calls per run} (default 1)."""
    calls_per_run = calls_per_run or {}
    stages = list(plan)
    lines = []
    scenarios = {"measured": None} if profile else SCENARIOS
    for scen_name, scen in scenarios.items():
        lines += [f"### {scen_name} scenario" + ("" if profile else
                  f" ({scen['in']} in / {scen['out']} out tokens per call)"), "",
                  "| Model | $/1M in/out | " + " | ".join(stages) + " | Total |",
                  "|---|---|" + "---|" * len(stages) + "---|"]
        grand = 0.0
        for alias in model_list:
            p_in, p_out = models.resolve(alias)["price"] or (0, 0)
            costs = []
            for stage in stages:
                c = 0.0
                for arm, runs in plan[stage]:
                    prof = (profile or {}).get(arm) or {"calls": calls_per_run.get(arm, 1), **(scen or SCENARIOS["low"])}
                    calls = runs * prof["calls"]
                    c += calls * (prof["in"] * p_in + prof["out"] * p_out) / 1e6
                costs.append(c)
            total = sum(costs)
            grand += total
            lines.append(f"| {alias} | ${p_in}/${p_out} | " + " | ".join(f"${c:.2f}" for c in costs)
                         + f" | **${total:.2f}** |")
        lines.append("")

        judged_runs = len(model_list) * sum(runs for s in stages for _, runs in plan[s])
        lines += [f"Judge, one call per run ({judged_runs} runs across {len(model_list)} models):", ""]
        level = 0 if scen_name in ("low", "measured") else 1
        for alias in judges:
            spec = models.resolve(alias)
            effort = settings.get().judge_reasoning_override or spec["reasoning"] or "default"
            out = JUDGE_OUT.get(effort, JUDGE_OUT["default"])[level]
            price = spec["price"] or (0, 0)
            c = judged_runs * (JUDGE_IN * price[0] + out * price[1]) / 1e6
            lines.append(f"- {alias} (effort {effort}, ~{out} output tokens per label): ${c:.2f}"
                         + (f" -> **all models + this judge: ${grand + c:.2f}**" if alias == judges[0] else ""))
        lines.append("")
    total_runs = {s: sum(r for _, r in plan[s]) for s in stages}
    lines.append("Runs per stage: " + ", ".join(f"{s} {n}" for s, n in total_runs.items())
                 + ". " + note)
    return "\n".join(lines)
