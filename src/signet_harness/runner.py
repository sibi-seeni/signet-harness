"""Running jobs (question x arm x seed x condition) for any experiment, on several
models in parallel, with caching and JSONL logs."""

import hashlib
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from . import settings
from .logschema import CURRENT_HARNESS
from .retrieval import retrieve
from .tools import RunContext


def job_key(experiment, model, reasoning, arm, qid, seed, condition, prompt_version, retrieved) -> str:
    blob = json.dumps([experiment, model, reasoning or "default", arm, qid, seed, condition, prompt_version, retrieved])
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def run_one(exp, llm, q, arm, seed, condition, passages) -> dict:
    resources = exp.resources(condition) if hasattr(exp, "resources") else {}
    ctx = RunContext(question=q, passages=passages, llm=llm, seed=seed, condition=condition, resources=resources)
    t0 = time.monotonic()
    error, answer, extra = None, None, {}
    try:
        out = exp.ARMS[arm](ctx)
        if isinstance(out, dict):     # an arm that makes its own flag in code
            extra = {k: v for k, v in out.items() if k != "answer"}
            answer = out.get("answer")
        else:
            answer = out
    except Exception as e:  # logged and left out of the cache, so a rerun retries it
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    llm_calls = [e for e in ctx.trace if e["type"] == "llm_call"]
    usage = {k: sum(c["usage"].get(k, 0) for c in llm_calls)
             for k in ("prompt", "output", "thoughts", "cached")}
    costs = [c["cost_usd"] for c in llm_calls]
    return {
        "experiment": exp.NAME, "harness_version": CURRENT_HARNESS,
        "model": llm.model, "model_alias": llm.alias, "reasoning_effort": llm.reasoning_effort or "default",
        "arm": arm, "qid": q.id, "case": q.case, "family": q.family, "seed": seed, "condition": condition,
        "prompt_version": exp.PROMPT_VERSION,
        "retrieved": [p.pid for p in passages],
        "answer": answer, "flag": extra.pop("flag", None), "arm_output": extra or None, "error": error,
        **(exp.trace_summary(ctx.trace) if hasattr(exp, "trace_summary") else {}),
        "n_calls": len(llm_calls),
        "usage": usage,
        "cost_usd": None if any(c is None for c in costs) else sum(costs),
        "latency_s": round(time.monotonic() - t0, 3),
        "model_versions": sorted({c["model_version"] for c in llm_calls if c.get("model_version")}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace": ctx.trace,
    }


def interleave(llms, jobs) -> list:
    """(llm, question, arm, seed, condition) for every model, alternating models so
    they all progress together instead of one finishing before the next starts."""
    return [(llm, *job) for job in jobs for llm in llms]


def retrieve_for(exp, q, corpus, embedder=None):
    if hasattr(exp, "retrieve"):
        return exp.retrieve(q, corpus, embedder)
    return retrieve(q, corpus.pool_for(q), embedder=embedder, force_gold=getattr(exp, "FORCE_GOLD", True))


def run_jobs(exp, jobs, corpus, out_path, embedder=None, workers=None, use_cache=True, quiet=False) -> list[dict]:
    """jobs: list of (llm, question, arm, seed, condition); see interleave(). Appends
    one JSON line per run to out_path. Models run in parallel in one worker pool,
    each with its own rate limiter."""
    workers = workers or settings.get().workers
    retrieved = {}
    for _, q, *_ in jobs:
        if q.id not in retrieved:
            retrieved[q.id] = retrieve_for(exp, q, corpus, embedder)

    cache_dir = settings.get().cache / exp.NAME / "runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results = []

    def work(job):
        llm, q, arm, seed, condition = job
        passages = retrieved[q.id]
        key = job_key(exp.NAME, llm.model, llm.reasoning_effort, arm, q.id, seed, condition, exp.PROMPT_VERSION,
                      [p.pid for p in passages])
        cpath = cache_dir / f"{key}.json"
        if use_cache and cpath.exists():
            rec = json.loads(cpath.read_text())
            rec["from_cache"] = True
        else:
            rec = run_one(exp, llm, q, arm, seed, condition, passages)
            rec["key"] = key
            if rec["error"] is None:
                cpath.write_text(json.dumps(rec))
            rec["from_cache"] = False
        with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            results.append(rec)
        return rec

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            done += 1
            if quiet:
                continue
            status = "ERROR" if rec["error"] else ("cached" if rec["from_cache"] else f"${rec['cost_usd'] or 0:.4f}")
            print(f"[{done}/{len(jobs)}] {rec.get('model_alias', rec['model'])[:22]:22} {rec['arm']:6} {rec['qid']:10} seed={rec['seed']} "
                  f"{rec['condition']:20} calls={rec['n_calls']} {status}", flush=True)
            if rec["error"]:
                print("   " + rec["error"].splitlines()[0], flush=True)
    return results
