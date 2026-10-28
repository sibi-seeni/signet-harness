"""The eight ways of wiring in the freshness check (design doc section 6), plus
the live web-search variant of A3. Each arm takes a RunContext and returns the
final answer text; everything it does is appended to ctx.trace. The registry is
ctx.resources["registry"]."""

import json
import re
from pathlib import Path

from signet_harness.checks import freshness
from signet_harness.checks.freshness import CHECK_LATEST_VERSION_DECL
from signet_harness.tools import RunContext, single_call, tool_loop

from . import prompts

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def _system(*extra: str) -> str:
    return " ".join([prompts.BASE_SYSTEM, *extra])


def registry_handler(ctx: RunContext):
    return lambda args: ctx.resources["registry"].lookup(str(args.get("title_or_id", "")))


def run_verifier(ctx: RunContext) -> dict:
    """Separate verifier LLM call(s): sees only the passages and the registry."""
    decls = [CHECK_LATEST_VERSION_DECL]
    handlers = {"check_latest_version": registry_handler(ctx)}
    raw = tool_loop(ctx, prompts.VERIFIER_SYSTEM,
                    "Retrieved passages:\n\n" + prompts.format_passages(ctx.passages),
                    decls, handlers, role="verifier")
    report = parse_json_array(raw)
    return {"report": report} if report is not None else {"report_text": raw, "parse_error": True}


def parse_json_array(text: str):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, list) else None
    except json.JSONDecodeError:
        return None


# ---- arms -----------------------------------------------------------------

def a0_plain(ctx):
    return single_call(ctx, _system(), prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages)))


def a1_system_prompt(ctx):
    return single_call(ctx, _system(prompts.FRESHNESS_LINE),
                       prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages)))


def a2_metadata(ctx):
    return single_call(ctx, _system(prompts.FRESHNESS_LINE),
                       prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages, metadata=True)))


def a3_tool(ctx):
    return tool_loop(ctx, _system(prompts.FRESHNESS_LINE),
                     prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages)),
                     [CHECK_LATEST_VERSION_DECL], {"check_latest_version": registry_handler(ctx)})


def a3_web(ctx):
    """Live condition: A3 with OpenRouter's web search server tool in place of the mock registry."""
    return tool_loop(ctx, _system(prompts.FRESHNESS_LINE),
                     prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages)),
                     [], {}, web_search=True)


def a4_forced_check(ctx):
    annotations = freshness.annotate(ctx.passages, ctx.resources["registry"],
                                     log=lambda name, args, result: ctx.log_tool("harness", name, args, result))
    return single_call(ctx, _system(prompts.FRESHNESS_LINE, prompts.A4_NOTE),
                       prompts.user_message(ctx.question.question,
                                            prompts.format_passages(ctx.passages, annotations=annotations)))


def a5_skill(ctx):
    skill_text = (SKILLS_DIR / "version-check" / "SKILL.md").read_text()
    decls = [prompts.LOAD_SKILL_DECL]
    handlers = {"check_latest_version": registry_handler(ctx)}

    def load_skill(args):
        if str(args.get("name", "")).strip().lower() != "version-check":
            return {"error": f"Unknown skill '{args.get('name')}'. Available: version-check."}
        if CHECK_LATEST_VERSION_DECL not in decls:
            decls.append(CHECK_LATEST_VERSION_DECL)
        return {"skill": skill_text, "unlocked_tools": ["check_latest_version"]}

    handlers["load_skill"] = load_skill
    return tool_loop(ctx, _system(prompts.FRESHNESS_LINE),
                     prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages)),
                     decls, handlers)


def a6_subagent(ctx):
    return tool_loop(ctx, _system(prompts.FRESHNESS_LINE),
                     prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages)),
                     [prompts.VERIFY_SOURCES_DECL], {"verify_sources": lambda args: run_verifier(ctx)})


def a7_forced_subagent(ctx):
    result = run_verifier(ctx)
    ctx.log_tool("harness", "verify_sources", {}, result)
    report = json.dumps(result.get("report", result.get("report_text")), indent=1)
    return single_call(ctx, _system(prompts.FRESHNESS_LINE, prompts.A7_NOTE),
                       prompts.user_message(ctx.question.question, prompts.format_passages(ctx.passages),
                                            extra=f"Source verification report:\n{report}"))


ARM_FUNCS = {
    "A0": a0_plain, "A1": a1_system_prompt, "A2": a2_metadata, "A3": a3_tool,
    "A4": a4_forced_check, "A5": a5_skill, "A6": a6_subagent, "A7": a7_forced_subagent,
    "A3-web": a3_web,
}


def registry_was_checked(trace: list) -> bool:
    """Did anyone actually query the registry during this run?"""
    return any(e["type"] == "tool_call" and e["name"] == "check_latest_version" for e in trace)
