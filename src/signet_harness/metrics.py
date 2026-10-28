"""The four numbers every stream reports together (team plan): detection or
security, false flags, coverage, and cost per 100 questions. Each experiment
defines what the first three mean for its cases in its four_numbers(rows); this
module groups, formats and exports them.

Cost per 100 is the models under test only (OpenRouter's reported cost, averaged
per run, times 100). Judge cost is reported separately."""

import csv
import json
import random
from collections import defaultdict


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_per_job(rows):
    """results.jsonl is append-only; keep the last row per job."""
    out = {}
    for r in rows:
        out[(r["model"], r["arm"], r["qid"], r["seed"], r["condition"])] = r
    return list(out.values())


def rate(rows, pred):
    """(share of rows where pred holds, n), or (None, 0) for no rows."""
    return (sum(1 for r in rows if pred(r)) / len(rows), len(rows)) if rows else (None, 0)


def fmt(rate_n):
    r, n = rate_n
    return "n/a" if r is None else f"{r:.0%} (n={n})"


def cost_per_100(rows):
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    return 100 * sum(costs) / len(costs) if costs else None


def four_numbers_table(rows, four_numbers, title="") -> str:
    """Markdown table of the four numbers per model, condition and arm."""
    groups = defaultdict(list)
    for r in latest_per_job(rows):
        groups[(r["model"], r["condition"], r["arm"])].append(r)
    lines = [f"# Four numbers{': ' + title if title else ''}", "",
             "| Model | Condition | Arm | Runs | Detection ↑ | False flags ↓ | Coverage ↑ | Cost / 100 Qs |",
             "|---|---|---|---|---|---|---|---|"]
    for (model, condition, arm), rs in sorted(groups.items()):
        n = four_numbers(rs)
        cost = "n/a" if n["cost_per_100"] is None else f"${n['cost_per_100']:.3f}"
        lines.append(f"| {model} | {condition} | {arm} | {len(rs)} | {fmt(n['detection'])} | {fmt(n['false_flags'])} "
                     f"| {fmt(n['coverage'])} | {cost} |")
    judge = sum(r.get("judge_cost_usd") or 0 for r in rows)
    return "\n".join(lines + ["", f"Judge cost (not in the cost column): ${judge:.4f}", ""])


def export_handcheck(scored_path, out_csv, fields, frac=0.10, seed=0) -> int:
    """A random `frac` sample plus every row with a rule/judge disagreement, with empty
    human_outcome / human_note columns to fill in."""
    rows = latest_per_job(load_jsonl(scored_path))
    rng = random.Random(seed)
    sample = rng.sample(rows, max(1, round(frac * len(rows)))) if rows else []
    picked = {id(r): "sample" for r in sample}
    for r in rows:
        if r.get("disagreement"):
            picked[id(r)] = "disagreement" if id(r) not in picked else "sample+disagreement"
    chosen = [r for r in rows if id(r) in picked]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["why", *fields, "human_outcome", "human_note"], extrasaction="ignore")
        w.writeheader()
        for r in chosen:
            w.writerow({**r, "why": picked[id(r)],
                        **{k: ";".join(v) for k, v in r.items() if isinstance(v, list)},
                        "human_outcome": "", "human_note": ""})
    return len(chosen)
