"""A scripted stand-in for OpenRouterLLM, so arms and pipelines can be tested
offline with no API key:

    llm = FakeLLM(lambda system, messages, tool_names, kw: ("$15,750 [1]", []))
    llm = FakeLLM(lambda *a: ("", [("check_latest_version", {"title_or_id": "Rev 4"})]))
"""

import json

from .llm import LLMTurn, ToolCall


class FakeLLM:
    """Calls `script(system, messages, tool_names, kwargs)` -> (text, [(name, args), ...])."""

    def __init__(self, script, model="fake-model"):
        self.script, self.model, self.alias, self.reasoning_effort = script, model, model, None
        self.provider = None
        self.canonical = None
        self.calls = []

    def generate(self, *, system, messages, tools=None, **kw):
        names = [d["name"] for d in tools or []]
        self.calls.append({"system": system, "messages": list(messages), "decls": names, **kw})
        text, fcalls = self.script(system, messages, names, kw)
        calls = [ToolCall(id=f"call-{i}", name=n, args=a) for i, (n, a) in enumerate(fcalls)]
        message = {"role": "assistant", "content": text or None,
                   "tool_calls": [{"id": c.id, "type": "function",
                                   "function": {"name": c.name, "arguments": json.dumps(c.args)}} for c in calls] or None}
        usage = {"prompt": 100, "output": 10, "thoughts": 5, "cached": 0}
        record = {"model": self.model, "model_version": "fake-1", "provider": "fake", "usage": usage,
                  "cost_usd": 0.001, "latency_s": 0.0, "response_message": message,
                  "request_messages": [{"role": "system", "content": system}, *messages]}
        return LLMTurn(message=message, text=text, tool_calls=calls, usage=usage, latency_s=0.0, record=record)
