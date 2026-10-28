"""The copy-me template runs end to end offline: run -> score -> report ->
shared.jsonl, through the same code path as `signet run`."""

import json

from signet_harness import cli, experiment, logschema, runner
from signet_harness.metrics import load_jsonl
from signet_harness.testing import FakeLLM


def test_template_end_to_end(tmp_path):
    tpl = experiment.load("template")
    qs = tpl.load_questions()
    llm = FakeLLM(lambda *a: ("The standard deduction is $15,750 [2].", []), model="fake-a")
    jobs = runner.interleave([llm], [(q, a, 0, "normal") for q in qs for a in tpl.ARMS])
    run_dir = tmp_path / "run"
    results = runner.run_jobs(tpl, jobs, tpl.load_corpus(), run_dir / "results.jsonl", quiet=True)
    assert not any(r["error"] for r in results)

    forced = [r for r in results if r["arm"] == "forced-check"]
    assert all(r["check_invoked"] and r["arm_output"]["doc_status"] for r in forced)
    by_q = {r["qid"]: r for r in forced}
    assert by_q["irs-03"]["flag"] == ["outdated"]      # only the 2024 instructions are in its corpus
    assert by_q["irs-05"]["flag"] == ["outdated"]      # the naive flag also fires on the C5 question

    (run_dir / "meta.json").write_text(json.dumps({"experiment": tpl.NAME}))
    cli.score(tpl, run_dir, workers=2)
    scored = {(s["arm"], s["qid"]): s for s in load_jsonl(run_dir / "scored.jsonl")}
    assert scored[("plain", "irs-01")]["outcome"] == "correct"
    assert scored[("forced-check", "irs-05")]["outcome"] == "false_flag"
    assert "Detection" in (run_dir / "report.md").read_text()
    lines = load_jsonl(run_dir / "shared.jsonl")
    assert len(lines) == len(results) and all(logschema.validate(x) == [] for x in lines)
    assert all(x["experiment"] == "template" and x["harness_version"].startswith("signet-harness") for x in lines)


def test_template_registry_down_adds_the_warning_in_code():
    tpl = experiment.load("template")
    q = next(q for q in tpl.load_questions() if q.id == "irs-01")
    llm = FakeLLM(lambda *a: ("$15,750 [2].", []))
    rec = runner.run_one(tpl, llm, q, "forced-check", 0, "registry_unavailable", tpl.load_corpus().get(["irs-2025-p1"]))
    assert rec["flag"] == ["unable_to_verify"] and rec["answer"].endswith("version registry was unavailable.")


def test_signet_run_command(tmp_path, monkeypatch, capsys):
    from signet_harness import settings

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    settings.configure(runs_dir=tmp_path / "results")

    def fake(model, **kw):
        llm = FakeLLM(lambda *a: ("$15,750 [1].", []), model=f"vendor/{model}")
        llm.alias, llm.canonical = model, f"vendor/{model}-20261101"
        return llm

    monkeypatch.setattr(cli, "OpenRouterLLM", fake)
    cli.main(["--experiment", "template", "run", "--stage", "smoke", "--retrieval", "bm25", "--name", "smoke",
              "--models", "deepseek-flash,gemma-4-31b"])
    run_dir = tmp_path / "results" / "smoke"
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["experiment"] == "template" and meta["harness_version"].startswith("signet-harness")
    assert [m["canonical"] for m in meta["models"]] == ["vendor/deepseek-flash-20261101", "vendor/gemma-4-31b-20261101"]
    assert set(meta["data_sha256"]) == {"questions.json", "docs.json", "registry.json"}
    assert meta["judge_model"] is None and meta["force_gold"] is True
    lines = load_jsonl(run_dir / "shared.jsonl")
    assert len(lines) == 4 and {x["model"] for x in lines} == {"vendor/deepseek-flash", "vendor/gemma-4-31b"}
    cli.main(["--experiment", "template", "metrics", "--run", str(run_dir)])
    assert "| forced-check |" in capsys.readouterr().out
