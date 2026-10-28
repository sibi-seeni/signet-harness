"""An LLM judge that returns schema-checked JSON labels, cached on disk by
(judge model, effort, experiment prompt version, instructions, prompt)."""

import hashlib
import json
from pathlib import Path

from . import settings
from .llm import OpenRouterLLM


def make_judge_llm(model: str | None = None) -> OpenRouterLLM:
    s = settings.get()
    return OpenRouterLLM(model=model or s.judge_model, reasoning_effort=s.judge_reasoning_override)


class Judge:
    def __init__(self, llm, system: str, schema: dict, version: str = "", cache_dir: Path | None = None,
                 fallback: dict | None = None):
        """`fallback` is used as the labels if the judge's reply isn't valid JSON."""
        self.llm, self.system, self.schema, self.version = llm, system, schema, version
        self.fallback = fallback or {}
        self.cache_dir = Path(cache_dir or settings.get().cache / "judge")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def label(self, prompt: str) -> dict:
        """-> {"labels": {...}, "usage", "cost_usd", "model_version"}"""
        key = hashlib.sha256(json.dumps([self.llm.model, self.llm.reasoning_effort, self.version,
                                         self.system, prompt]).encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        turn = self.llm.generate(system=self.system, messages=[{"role": "user", "content": prompt}],
                                 seed=0, json_schema=self.schema)
        try:
            labels = json.loads(turn.text)
        except json.JSONDecodeError:
            labels = {**self.fallback, "rationale": "JUDGE PARSE ERROR", "parse_error": True}
        out = {"labels": labels, "usage": turn.usage, "cost_usd": turn.record["cost_usd"],
               "model_version": turn.record["model_version"]}
        path.write_text(json.dumps(out))
        return out
