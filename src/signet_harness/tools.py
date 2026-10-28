"""What an arm works with: the run context (question, passages, model, trace) and
the tool loop. Every model call and tool call an arm makes goes into ctx.trace,
which is logged in full with the run."""

import json
from dataclasses import dataclass, field

from . import settings
from .data import Passage, Question


@dataclass
class RunContext:
    question: Question
    passages: list[Passage]
    llm: object
    seed: int
    condition: str = "normal"
    # Whatever the experiment's resources(condition) returned, e.g. {"registry": MockRegistry(...)}.
    resources: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)

    def log_llm(self, role: str, turn):
        self.trace.append({"type": "llm_call", "role": role, **turn.record})

    def log_tool(self, role: str, name: str, args: dict, result):
        self.trace.append({"type": "tool_call", "role": role, "name": name, "args": args, "result": result})


def tool_loop(ctx: RunContext, system: str, user_text: str, decls: list, handlers: dict,
              role: str = "main", web_search: bool = False, max_steps: int | None = None) -> str:
    """Run model turns, executing function calls, until the model answers in text.
    On the last allowed step function calling is switched off to force an answer.
    `decls` may be extended by a handler (that's how a skill unlocks a tool)."""
    max_steps = settings.get().max_tool_steps if max_steps is None else max_steps
    messages = [{"role": "user", "content": user_text}]
    for step in range(max_steps + 1):
        final = step == max_steps
        turn = ctx.llm.generate(system=system, messages=messages, tools=list(decls) or None,
                                web_search=web_search, seed=ctx.seed,
                                tool_choice="none" if final and decls else None)
        ctx.log_llm(role, turn)
        if not turn.tool_calls or final:
            return turn.text
        messages.append(turn.message)
        for tc in turn.tool_calls:
            handler = handlers.get(tc.name)
            result = handler(tc.args) if handler else {"error": f"Unknown tool '{tc.name}'."}
            ctx.log_tool(role, tc.name, tc.args, result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return ""


def single_call(ctx: RunContext, system: str, user_text: str, role: str = "main") -> str:
    return tool_loop(ctx, system, user_text, [], {}, role=role)
