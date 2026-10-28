"""Core harness tests: the OpenRouter client (mocked transport), preflight checks,
the model table, settings, the freshness check module, the shared log format and
the command line. No API key needed."""

import json

import pytest

from signet_harness import cli, experiment, logschema, models, settings
from signet_harness.checks import freshness
from signet_harness.data import Passage, Question, check_questions
from signet_harness.judge import Judge

from conftest import REPO


def test_openrouter_judge_request_and_usage(monkeypatch, tmp_path):
    """OpenRouter judge client against a mocked HTTP transport (no network)."""
    import httpx

    from signet_harness.llm import OpenRouterLLM

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    sent, replies = [], [httpx.Response(429, json={"error": "rate limited"})]
    labels = {"rationale": "r", "answer_matches": "alt", "flags": ["outdated"], "mentions_newer_version": True,
              "undermines_answer": False}

    def handler(request):
        sent.append(request)
        if replies:
            return replies.pop(0)
        return httpx.Response(200, json={
            "model": "openai/gpt-6-sol-20260922", "provider": "OpenAI",
            "choices": [{"message": {"role": "assistant", "content": json.dumps(labels)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 120, "cost": 0.0022,
                      "completion_tokens_details": {"reasoning_tokens": 80}}})

    monkeypatch.setattr("signet_harness.llm.time.sleep", lambda s: None)
    llm = OpenRouterLLM(model="gpt-6-sol", transport=httpx.MockTransport(handler))
    judge = Judge(llm, "You label answers.", {"type": "object"}, version="v2", cache_dir=tmp_path)
    prompt = "QUESTION: q\n\nANSWER TO LABEL:\nThe 2024 figure is $21,900, but these instructions may be outdated."
    out = judge.label(prompt)

    assert out["labels"]["flags"] == ["outdated"] and out["cost_usd"] == 0.0022
    assert out["usage"]["thoughts"] == 80 and out["usage"]["output"] == 40
    body = json.loads(sent[-1].content)
    assert len(sent) == 2  # retried after the 429
    assert sent[-1].headers["authorization"] == "Bearer test-key"
    assert body["model"] == "openai/gpt-6-sol" and body["reasoning"] == {"effort": "low"}
    assert body["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "ANSWER TO LABEL" in body["messages"][1]["content"]
    # cached: no second request
    judge.label(prompt)
    assert len(sent) == 2


def test_openrouter_tool_call_round_trip(monkeypatch):
    """Tool declarations go out in OpenAI format; the assistant message (with
    reasoning_details) comes back intact for the next turn."""
    import httpx

    from signet_harness.llm import OpenRouterLLM
    from signet_harness.checks.freshness import CHECK_LATEST_VERSION_DECL
    VERIFY_SOURCES_DECL = {"name": "verify_sources", "description": "d"}

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    sent = []
    reasoning_details = [{"type": "reasoning.encrypted", "data": "sig"}]

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "deepseek/deepseek-v4.1-flash-20260910", "provider": "DeepSeek",
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "reasoning_details": reasoning_details,
                "tool_calls": [{"id": "tc1", "type": "function", "function": {
                    "name": "check_latest_version", "arguments": '{"title_or_id": "Rev 4"}'}}]}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 300,
                      "completion_tokens_details": {"reasoning_tokens": 200}}})

    llm = OpenRouterLLM(model="deepseek-flash", transport=httpx.MockTransport(handler))
    turn = llm.generate(system="s", messages=[{"role": "user", "content": "q"}],
                        tools=[CHECK_LATEST_VERSION_DECL, VERIFY_SOURCES_DECL], web_search=True,
                        seed=2, tool_choice="none")
    body = sent[0]
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"][0] == {"role": "system", "content": "s"}
    assert body["temperature"] == 0.0 and body["seed"] == 2 and body["tool_choice"] == "none"
    assert body["tools"][0]["function"]["name"] == "check_latest_version"
    assert body["tools"][1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert body["tools"][2] == {"type": "openrouter:web_search"}
    assert "reasoning" not in body  # deepseek-flash uses the model default
    assert "provider" not in body and turn.record["config"]["provider_filter"] is None
    assert turn.tool_calls[0].id == "tc1" and turn.tool_calls[0].args == {"title_or_id": "Rev 4"}
    assert turn.message["reasoning_details"] == reasoning_details
    # no OpenRouter cost reported -> price table: 1000*0.14 + 300*0.42 per 1M
    assert turn.record["cost_usd"] == pytest.approx((1000 * 0.14 + 300 * 0.42) / 1e6)
    json.dumps(turn.record)


def test_gemma_request_switches_reasoning_on_and_filters_quantization(monkeypatch):
    import httpx

    from signet_harness.llm import OpenRouterLLM

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "google/gemma-4-31b-it-20260402", "provider": "Crusoe",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    llm = OpenRouterLLM(model="gemma-4-31b", transport=httpx.MockTransport(handler))
    turn = llm.generate(system="s", messages=[{"role": "user", "content": "q"}], seed=0)
    body = sent[0]
    assert body["model"] == "google/gemma-4-31b-it"
    assert body["reasoning"] == {"enabled": True}
    assert body["provider"] == {"quantizations": ["fp8", "bf16", "fp16", "fp32"]}
    assert turn.record["config"]["reasoning_effort"] == "on"
    assert turn.record["config"]["provider_filter"] == body["provider"]


def test_preflight_endpoints_respect_the_quantization_floor(capsys):
    from signet_harness.preflight import _check_endpoints

    def ep(name, quant, params=("tools", "tool_choice")):
        return {"provider_name": name, "quantization": quant, "supported_parameters": list(params)}

    need = ["tools", "tool_choice"]
    assert _check_endpoints("gemma-4-31b", [ep("A", "fp4"), ep("B", "bf16"), ep("C", "mxfp8")], need)
    out = capsys.readouterr().out
    assert "B (bf16)" in out and "C (mxfp8)" in out and "A (fp4)" not in out
    # fp4, unknown, or fp8 without tool support don't count
    assert not _check_endpoints("gemma-4-31b", [ep("A", "nvfp4"), ep("B", "unknown"), ep("C", "fp8", ["seed"])], need)
    assert "FAIL gemma-4-31b" in capsys.readouterr().out
    assert _check_endpoints("deepseek-flash", [], need)  # no filter configured -> nothing to check


def test_model_aliases_resolve():
    s = settings.get()
    assert models.resolve("glm-5.3")["slug"] == "z-ai/glm-5.3"
    assert models.resolve("glm-5.3")["reasoning"] == "low"
    assert models.resolve("google/gemini-4-argon")["alias"] == "gemini-4-argon"
    assert models.resolve(s.judge_model)["slug"] == "openai/gpt-6-luna"
    assert models.resolve(s.judge_model)["reasoning"] == "low"
    assert s.models == ("deepseek-flash", "gemma-4-31b")
    assert models.resolve("gemma-4-31b")["slug"] == "google/gemma-4-31b-it"
    assert models.resolve("google/gemma-4-31b-it-20260402")["alias"] == "gemma-4-31b"
    assert models.resolve("nemotron-3.5-super-vl")["slug"] == "nvidia/nemotron-3.5-super-vl"
    # a pinned canonical slug resolves to its alias, and requests still send the id
    assert models.resolve("deepseek/deepseek-v4.1-flash-20260910")["slug"] == "deepseek/deepseek-v4.1-flash"
    custom = models.resolve("someone/new-model")
    assert custom["slug"] == "someone/new-model" and custom["price"] is None
    # every model stays switchable, funded tier included
    for alias in ("deepseek-flash", "gemma-4-31b", "nemotron-3.5-super-vl", "glm-5.3", "gpt-6-luna", "gpt-6-sol",
                  "gemini-4-argon", "gemini-3.8-flash"):
        assert alias in models.MODELS and models.MODELS[alias].get("canonical")
    assert {m["tier"] for m in models.MODELS.values()} == {"budget", "funded"}


def test_preflight_listing_uses_ids_and_checks_the_pin(capsys):
    from signet_harness.preflight import _check_listing

    def listing(canonical):
        return {"deepseek/deepseek-v4.1-flash": {
            "id": "deepseek/deepseek-v4.1-flash", "canonical_slug": canonical,
            "pricing": {"prompt": "0.00000014", "completion": "0.00000042"},
            "supported_parameters": ["tools", "tool_choice", "reasoning"]}}

    assert _check_listing("deepseek-flash", listing("deepseek/deepseek-v4.1-flash-20260910"), ["tools", "tool_choice"])
    assert "(config table says" not in capsys.readouterr().out
    assert not _check_listing("deepseek-flash", listing("deepseek/deepseek-v4.1-flash-20261201"), ["tools"])
    assert "not the pinned deepseek/deepseek-v4.1-flash-20260910" in capsys.readouterr().out


def test_openrouter_embedder(tmp_path):
    import httpx

    from signet_harness.retrieval import OpenRouterEmbedder

    def handler(request):
        inp = json.loads(request.content)["input"]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(len(t)), 1.0]}
                                                  for i, t in reversed(list(enumerate(inp)))]})

    http = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))
    vecs = OpenRouterEmbedder(http, cache_path=tmp_path / "e.json").embed(["ab", "abcd", "ab"])
    assert vecs[:, 0].tolist() == [2.0, 4.0, 2.0]


def test_estimate_table():
    from signet_harness.estimate import estimate

    t2 = experiment.load("track2_freshness")
    questions = t2.load_questions()
    plan = t2.stage_runs(questions)
    assert sum(r for _, r in plan["sweep"]) == len(questions) * 8 * 3
    md = estimate(t2.stage_runs(questions, 90), list(settings.get().models), ["gpt-6-luna"], t2.CALLS_PER_RUN)
    assert "| gemma-4-31b |" in md and "gpt-6-luna (effort low" in md


def test_generic_stage_plan_from_stages():
    from signet_harness.estimate import stage_runs

    tpl = experiment.load("template")
    plan = stage_runs(tpl.load_questions(), tpl.STAGES)
    assert plan["smoke"] == [("plain", 1), ("forced-check", 1)]
    assert plan["run"] == [("plain", 9), ("forced-check", 9)]


# ---- settings ---------------------------------------------------------------

def test_settings_from_toml_env_and_extra_models(tmp_path, monkeypatch):
    (tmp_path / "signet.toml").write_text(
        'models = ["deepseek-flash"]\njudge_model = "gpt-6-sol"\nruns_dir = "out"\nseeds = [0]\n'
        '[models_table.my-model]\nslug = "vendor/my-model"\ncanonical = "vendor/my-model-20261101"\n'
        'price = [0.1, 0.4]\nreasoning = "low"\ntier = "funded"\n')
    monkeypatch.setenv("SIGNET_CONFIG", str(tmp_path / "signet.toml"))
    s = settings.configure(settings.load())
    assert s.workspace == tmp_path and s.runs_dir == tmp_path / "out"
    assert s.models == ("deepseek-flash",) and s.judge_model == "gpt-6-sol" and s.seeds == (0,)
    assert models.resolve("my-model")["slug"] == "vendor/my-model"
    assert models.resolve("vendor/my-model-20261101")["alias"] == "my-model"
    assert models.price_for("vendor/my-model") == (0.1, 0.4)
    # environment beats the file; the old TTS_* names still work
    monkeypatch.setenv("TTS_MODELS", "gemini-4-argon,gpt-6-sol")
    monkeypatch.setenv("SIGNET_JUDGE_MODEL", "gpt-6-luna")
    s = settings.configure(settings.load())
    assert s.models == ("gemini-4-argon", "gpt-6-sol") and s.judge_model == "gpt-6-luna"


def test_switching_to_funded_models_is_config_only(monkeypatch):
    import httpx

    from signet_harness.llm import OpenRouterLLM

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("SIGNET_MODELS", "gemini-4-argon")
    settings.configure(settings.load())
    sent = []
    transport = httpx.MockTransport(lambda r: sent.append(json.loads(r.content)) or httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}))
    llm = OpenRouterLLM(settings.get().models[0], transport=transport)
    llm.generate(system="s", messages=[{"role": "user", "content": "q"}], seed=0)
    assert sent[0]["model"] == "google/gemini-4-argon" and llm.canonical == "google/gemini-4-argon-20261013"


def test_openai_compatible_settings_for_external_tools(monkeypatch):
    from signet_harness.llm import openai_compatible

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    cfg = openai_compatible("gemma-4-31b")
    assert cfg["base_url"] == "https://openrouter.ai/api/v1" and cfg["model"] == "google/gemma-4-31b-it"
    assert cfg["extra_body"] == {"reasoning": {"enabled": True},
                                 "provider": {"quantizations": ["fp8", "bf16", "fp16", "fp32"]}}
    assert openai_compatible("deepseek-flash")["extra_body"] == {}


# ---- data -------------------------------------------------------------------

def test_question_keeps_unknown_fields_in_extra():
    q = Question.from_dict({"id": "a1", "case": "single_source_edit", "question": "q?", "attack": "edit",
                            "target_doc": "d1"})
    assert q.extra == {"attack": "edit", "target_doc": "d1"} and q.corpus == []


def test_check_questions_reports_problems():
    t2 = experiment.load("track2_freshness")
    qs = [Question.from_dict({"id": "x", "case": "C1", "question": "q", "corpus": ["nope"], "accept": ["("]}),
          Question.from_dict({"id": "x", "case": "", "question": "q"})]
    problems = check_questions(qs, t2.load_corpus())
    assert any("duplicate" in p for p in problems) and any("bad regex" in p for p in problems)
    assert any("unknown doc nope" in p for p in problems) and any("missing case" in p for p in problems)
    assert check_questions(t2.load_questions(), t2.load_corpus()) == []


def test_retrieval_without_forced_gold():
    from signet_harness.retrieval import retrieve

    t2 = experiment.load("track2_freshness")
    corpus = t2.load_corpus()
    q = next(q for q in t2.load_questions() if q.id == "irs-01")
    got = retrieve(q, corpus.pool_for(q), top_k=2, force_gold=False)
    assert len(got) == 2


# ---- freshness check module (the hand-off to B) -----------------------------

def _passage(doc_id, title, filename):
    return Passage(pid=f"{doc_id}-p1", doc_id=doc_id, family="", filename=filename, title=title, text="t",
                   locator="", ingest={})


REG = json.loads((REPO / "experiments/track2_freshness/data/registry.json").read_text())


def test_assess_gives_machine_readable_statuses():
    ps = [_passage("r4", "NIST SP 800-53 Revision 4", "NIST.SP.800-53r4.pdf"),
          _passage("r5", "NIST SP 800-53 Revision 5", "NIST.SP.800-53r5.pdf"),
          _passage("ud", "Instructions for Form 1040", "i1040gi.pdf"),
          _passage("zz", "Some unrelated manual", "manual.pdf")]
    logged = []
    st = freshness.assess(ps, freshness.MockRegistry(REG), log=lambda *a: logged.append(a))
    assert st["r4"]["status"] == "withdrawn" and st["r4"]["latest"] == "Rev 5"
    assert st["r5"]["status"] == "current" and st["r5"]["in_place_changes"]   # amended in place (C4)
    assert st["ud"]["status"] == "unverifiable"
    assert st["zz"]["status"] == "not_found"
    assert len(logged) == 4 and logged[0][0] == "check_latest_version"
    assert "SUPERSEDED by Rev 5" in st["r4"]["annotation"]


def test_unverified_warning_is_added_in_code():
    ps = [_passage("r4", "NIST SP 800-53 Revision 4", "NIST.SP.800-53r4.pdf")]
    down = freshness.annotate(ps, freshness.MockRegistry(REG, unavailable=True))
    assert freshness.unverified(down)
    out = freshness.with_unverified_warning("Rev 4 has 18 families [1].", down)
    assert out.endswith(freshness.UNVERIFIED_WARNING)
    assert freshness.with_unverified_warning(out, down) == out          # not added twice
    up = freshness.annotate(ps, freshness.MockRegistry(REG))
    assert freshness.with_unverified_warning("x", up) == "x"
    assert freshness.assess(ps, freshness.MockRegistry(REG, unavailable=True))["r4"]["status"] == "unable_to_verify"


# ---- shared log format --------------------------------------------------------

def test_shared_line_from_a_code_flag_arm():
    rec = {"qid": "q1", "case": "single_source_edit", "arm": "B-pipeline", "model": "deepseek/deepseek-v4.1-flash",
           "retrieved": ["d1-p1"], "answer": "42", "flag": "conflicting_value", "usage": {"prompt": 10, "output": 2},
           "cost_usd": 0.0001, "latency_s": 1.2, "seed": 0, "condition": "normal", "harness_version": "signet-harness 0.1.0"}
    line = logschema.to_shared(rec)
    assert logschema.validate(line) == []
    assert line["flag"] == ["conflicting_value"] and line["question_id"] == "q1" and line["tokens"]["prompt"] == 10
    unscored = logschema.to_shared({**rec, "flag": None})
    assert unscored["flag"] is None and logschema.validate(unscored) == []
    scored = logschema.to_shared({**rec, "flag": None}, {"flags": [], "outcome": "correct"})
    assert scored["flag"] == [] and scored["outcome"] == "correct"


def test_validate_catches_bad_lines():
    assert "missing flag" in logschema.validate({k: None for k in logschema.REQUIRED if k != "flag"})
    good = logschema.to_shared({"qid": "q", "case": "c", "arm": "a", "model": "m", "retrieved": [], "answer": "x",
                                "usage": {"prompt": 1}, "cost_usd": None, "latency_s": 0.1})
    assert logschema.validate(good) == []
    assert logschema.validate({**good, "tokens": {"prompt": "1"}}) == ["tokens must be a dict of integer counts"]
    assert logschema.validate({**good, "flag": "outdated"}) == ["flag must be a list of strings or null"]


def test_every_track2_run_exports_valid_shared_lines():
    from signet_harness.metrics import load_jsonl

    for run in sorted((REPO / "results/track2").iterdir()):
        if not (run / "results.jsonl").exists():
            continue
        lines = load_jsonl(run / "shared.jsonl")
        assert lines, run.name
        for line in lines:
            assert logschema.validate(line) == [], (run.name, line["question_id"])
            assert line["experiment"] == "track2_freshness" and line["harness_version"].startswith("trust-the-source@")


# ---- command line --------------------------------------------------------------

def test_cli_info_and_report(capsys, tmp_path):
    cli.main(["info", "--experiment", "track2_freshness"])
    out = capsys.readouterr().out
    assert "deepseek-flash" in out and "gemini-4-argon" in out and "funded" in out
    cli.main(["--experiment", "track2_freshness", "check-questions"])
    assert "38 questions" in capsys.readouterr().out


def test_cli_check_questions_on_a_file(capsys, tmp_path):
    f = tmp_path / "attack.json"
    f.write_text(json.dumps({"questions": [{"id": "a1", "case": "forged_label", "question": "q?", "attack": "x"}]}))
    cli.main(["--experiment", "template", "check-questions", "--file", str(f), "--no-corpus"])
    assert "sha256" in capsys.readouterr().out
