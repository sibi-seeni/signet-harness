"""Checks to run before spending money (design doc section 10): every model is
listed on OpenRouter, still points at its pinned version, supports tool calling
(and the judge structured output), a real tool call round-trips, the judge
returns schema JSON, and embeddings work. Costs a fraction of a cent."""

import json

from . import models, settings
from .checks.freshness import CHECK_LATEST_VERSION_DECL
from .judge import make_judge_llm
from .llm import OpenRouterLLM, list_endpoints, list_models
from .retrieval import OpenRouterEmbedder

# Used when the experiment has no judge schema of its own.
SMOKE_SCHEMA = {"type": "object", "properties": {"answer_matches": {"type": "string", "enum": ["gold", "alt", "neither"]}},
                "required": ["answer_matches"], "additionalProperties": False}


def _check_listing(alias, listing, need):
    """Design doc section 10: the model must be listed and support what we need."""
    spec = models.resolve(alias)
    info = listing.get(spec["slug"])
    if info is None:
        print(f"  FAIL {alias}: {spec['slug']} isn't listed on OpenRouter /api/v1/models")
        return False
    pinned, served = spec.get("canonical"), info.get("canonical_slug")
    if pinned and served != pinned:
        print(f"  FAIL {alias}: {spec['slug']} now resolves to {served}, not the pinned {pinned}. "
              f"Update 'canonical' (and the price) in the model table only if the new version is intended.")
        return False
    params = set(info.get("supported_parameters") or [])
    pr = info.get("pricing") or {}
    live = (float(pr.get("prompt") or 0) * 1e6, float(pr.get("completion") or 0) * 1e6)
    table = spec["price"]
    note = "" if table and all(abs(a - b) < 1e-6 for a, b in zip(live, table)) else f"  (config table says {table})"
    print(f"{alias} -> {spec['slug']} ({served}): ${live[0]:.2f}/${live[1]:.2f} per 1M{note}; "
          f"reasoning={'reasoning' in params} effort={spec['reasoning'] or 'default'}")
    missing = [n for n in need if not ({n} if isinstance(n, str) else set(n)) & params]
    for n in missing:
        print(f"  FAIL {alias}: supported_parameters lacks {n}")
    return not missing


# OpenRouter's quantization filter counts these variants as their base format.
_QUANT_FAMILY = {"mxfp8": "fp8", "mxfp4": "fp4", "nvfp4": "fp4"}


def _check_endpoints(alias, endpoints, need):
    """A model with a provider quantization filter must still have at least one
    endpoint that passes it and supports what we need."""
    allowed = set((models.resolve(alias).get("provider") or {}).get("quantizations") or [])
    if not allowed:
        return True
    ok = [e for e in endpoints
          if (q := e.get("quantization") or "unknown") in allowed or _QUANT_FAMILY.get(q) in allowed
          if set(need) <= set(e.get("supported_parameters") or [])]
    names = ", ".join(f"{e['provider_name']} ({e.get('quantization')})" for e in ok)
    if not ok:
        print(f"  FAIL {alias}: no endpoint serves it at {sorted(allowed)} with {need}")
    else:
        print(f"  {alias}: {len(ok)} of {len(endpoints)} endpoints pass the quantization filter: {names}")
    return bool(ok)


def run_preflight(model_list, judge_schema=None, retrieval="hybrid") -> bool:
    """Fail fast before spending: models are listed with tool / JSON support, a tool
    call round-trips, the judge returns schema JSON, embeddings work."""
    s = settings.get()
    listing = list_models()
    ok = all([_check_listing(m, listing, ["tools", "tool_choice"]) for m in model_list])
    ok &= all([_check_endpoints(m, list_endpoints(models.resolve(m)["slug"]), ["tools", "tool_choice"])
               for m in model_list if models.resolve(m).get("provider")])
    ok &= _check_listing(s.judge_model, listing, [("structured_outputs", "response_format")])
    if not ok:
        return False

    for alias in model_list:
        llm = OpenRouterLLM(model=alias, reasoning_effort=s.reasoning_override)
        sys_msg = "You are a helpful assistant."
        messages = [{"role": "user", "content": (
            "Use check_latest_version to look up 'NIST SP 800-53 Rev 4', then tell me its status in one sentence.")}]
        turn = llm.generate(system=sys_msg, messages=messages, tools=[CHECK_LATEST_VERSION_DECL], seed=0)
        if not turn.tool_calls:
            print(f"  FAIL {alias}: no tool call returned (text: {turn.text[:200]!r})")
            ok = False
            continue
        tc = turn.tool_calls[0]
        messages += [turn.message, {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(
            {"family": "NIST-SP-800-53", "version": "Rev 4", "status": "withdrawn", "latest": "Rev 5"})}]
        turn2 = llm.generate(system=sys_msg, messages=messages, tools=[CHECK_LATEST_VERSION_DECL], seed=0)
        cost = (turn.record["cost_usd"] or 0) + (turn2.record["cost_usd"] or 0)
        thoughts = turn.usage["thoughts"] + turn2.usage["thoughts"]
        print(f"{alias}: tool round-trip OK via {turn.record['provider']} ({turn.record['model_version']}): "
              f"{turn2.text.strip()[:120]!r} | {thoughts} reasoning tokens | ${cost:.5f}")
        if llm.reasoning_effort == "on" and not thoughts:
            print(f"  FAIL {alias}: reasoning is set to 'on' but no reasoning tokens came back")
            ok = False

    jt = make_judge_llm().generate(
        system="You label answers.", seed=0,
        messages=[{"role": "user", "content": "QUESTION: What is 2+2?\nREFERENCE answer: 4\nALTERNATIVE answer: 5\n"
                                              "ANSWER TO LABEL:\nIt's 4."}],
        json_schema=judge_schema or SMOKE_SCHEMA)
    try:
        print(f"judge {s.judge_model}: JSON OK {json.loads(jt.text)} via {jt.record['provider']}, "
              f"${jt.record['cost_usd'] or 0:.5f}")
    except json.JSONDecodeError:
        print(f"  FAIL judge didn't return JSON: {jt.text[:200]!r}")
        ok = False

    if retrieval == "hybrid":
        vec = OpenRouterEmbedder(make_judge_llm().http).embed(["preflight check"])
        print(f"embeddings OK: {s.embed_model} dim={vec.shape[1]}")
    return ok
