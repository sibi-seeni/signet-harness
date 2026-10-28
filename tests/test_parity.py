"""Parity with the Track 2 harness (trust-the-source@389b781): the package must send
exactly what the experiment sent, and rebuild exactly what it reported.

For every run logged in results/track2, the arm is re-run on the logged passages
with a fake model, and its first request (system prompt, user message, tools) must
match the request in the log byte for byte. For A4 the harness's registry lookups
must match too, so B's pipeline gets the check that was validated."""

import pytest

from signet_harness import experiment
from signet_harness.metrics import latest_per_job, load_jsonl
from signet_harness.testing import FakeLLM
from signet_harness.tools import RunContext

from conftest import REPO

RUNS = sorted(p for p in (REPO / "results/track2").iterdir() if (p / "results.jsonl").exists())


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.name)
def test_first_request_of_every_logged_run_is_rebuilt_exactly(run):
    t2 = experiment.load("track2_freshness")
    qs = {q.id: q for q in t2.load_questions()}
    corpus = t2.load_corpus()
    rows = [r for r in latest_per_job(load_jsonl(run / "results.jsonl")) if not r.get("error")]
    assert rows
    for rec in rows:
        logged = next(e for e in rec["trace"] if e["type"] == "llm_call")
        llm = FakeLLM(lambda *a: ("x", []))
        ctx = RunContext(question=qs[rec["qid"]], passages=corpus.get(rec["retrieved"]), llm=llm,
                         seed=rec["seed"], condition=rec["condition"], resources=t2.resources(rec["condition"]))
        t2.ARMS[rec["arm"]](ctx)
        first = llm.calls[0]
        where = (rec["arm"], rec["qid"], rec["seed"], rec["condition"])
        assert [{"role": "system", "content": first["system"]}, *first["messages"]] == logged["request_messages"][:2], where
        logged_tools = [t for t in logged["config"]["tools"] if t != "openrouter:web_search"]
        assert first["decls"] == logged_tools, where
        assert first.get("web_search", False) == ("openrouter:web_search" in logged["config"]["tools"]), where
        if rec["arm"] == "A4":
            lookups = [(e["name"], e["args"], e["result"]) for e in ctx.trace if e["type"] == "tool_call"]
            logged_lookups = [(e["name"], e["args"], e["result"]) for e in rec["trace"]
                              if e["type"] == "tool_call" and e["role"] == "harness"]
            assert lookups == logged_lookups, where


@pytest.mark.parametrize("run", RUNS, ids=lambda p: p.name)
def test_report_is_rebuilt_byte_for_byte(run):
    t2 = experiment.load("track2_freshness")
    assert t2.build_report(run / "scored.jsonl") == (run / "report.md").read_text()


def test_handcheck_sample_is_unchanged(tmp_path):
    from signet_harness.metrics import export_handcheck

    t2 = experiment.load("track2_freshness")
    run = REPO / "results/track2/sweep-20261028-010735"
    export_handcheck(run / "scored.jsonl", tmp_path / "hc.csv", t2.report.HANDCHECK_FIELDS)
    assert (tmp_path / "hc.csv").read_text() == (run / "handcheck.csv").read_text()
