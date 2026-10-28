"""The Track 2 experiment on the shared harness: data integrity, registry, every
arm's tool loop (scripted fake model), scoring, the report. No API key needed."""

import json
import re

import pytest

from signet_harness import experiment, logschema, runner, settings
from signet_harness.checks import freshness
from signet_harness.checks.freshness import MockRegistry
from signet_harness.judge import Judge
from signet_harness.metrics import export_handcheck, latest_per_job, load_jsonl
from signet_harness.retrieval import retrieve
from signet_harness.testing import FakeLLM

from conftest import REPO

T2 = experiment.load("track2_freshness")
arms, scoring, report = T2.arms, T2.scoring, T2.report
QUESTIONS = T2.load_questions()
CORPUS = T2.load_corpus()
REGISTRY = T2.load_registry()


def q_by_id(qid):
    return next(q for q in QUESTIONS if q.id == qid)


def ctx_for(q, llm, unavailable=False):
    return arms.RunContext(question=q, passages=retrieve(q, CORPUS.pool_for(q)), llm=llm, seed=0,
                           resources={"registry": MockRegistry(REGISTRY, unavailable=unavailable)})


# ---- data integrity -------------------------------------------------------

def test_question_ids_unique_and_cases_valid():
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert {q.case for q in QUESTIONS} <= {"C1", "C2", "C3", "C4", "C5", "C6", "C7"}


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q.id)
def test_question_corpus_and_gold_passages(q):
    for d in q.corpus:
        assert d in CORPUS.docs, d
        assert CORPUS.docs[d]["family"] == q.family
    for pid in q.gold_passages:
        assert CORPUS.passages[pid].doc_id in q.corpus
    got = retrieve(q, CORPUS.pool_for(q))
    assert set(q.gold_passages) <= {p.pid for p in got}
    assert len(got) == max(settings.get().top_k, len(q.gold_passages))
    for pattern in q.accept + q.alt:
        re.compile(pattern)


def test_c2_corpus_excludes_newer_version():
    for q in QUESTIONS:
        if q.case == "C2":
            pool_docs = {p.doc_id for p in CORPUS.pool_for(q)}
            newest = [d for d in CORPUS.docs.values() if d["family"] == q.family and d.get("default_pool")]
            assert not any(d["doc_id"] in pool_docs for d in newest), q.id


def test_every_doc_family_is_in_registry():
    for d in CORPUS.docs.values():
        assert d["family"] in REGISTRY["families"]


# ---- registry -------------------------------------------------------------

@pytest.mark.parametrize("query,version,status", [
    ("NIST SP 800-53 Revision 4 (NIST.SP.800-53r4.pdf)", "Rev 4", "withdrawn"),
    ("2024 Instructions for Form 1040 (i1040gi_2024.pdf)", "Tax Year 2024", "superseded"),
    ("pandas 2.0 API reference (pandas-docs-2.0.pdf)", "2.0", "current"),
    ("NumPy 1.26 reference (numpy-docs-1.26.pdf)", "1.26", "superseded"),
    ("GSA Per Diem Rates, Fiscal Year 2025 (FY2025_PerDiem_Rates.pdf)", "FY2025", "current"),
])
def test_registry_lookup_by_title_and_filename(query, version, status):
    r = MockRegistry(REGISTRY).lookup(query)
    assert r["version"] == version and r["status"] == status


def test_registry_undated_doc_returns_unknown_version():
    r = MockRegistry(REGISTRY).lookup("Instructions for Form 1040 (i1040gi.pdf)")
    assert r["version"] == "unknown" and r["latest"] == "Tax Year 2025"


def test_registry_unavailable_and_unknown():
    assert "error" in MockRegistry(REGISTRY, unavailable=True).lookup("NIST SP 800-53 Rev 4")
    assert MockRegistry(REGISTRY).lookup("Some unrelated manual")["found"] is False


def test_forced_check_annotation_mentions_supersession():
    r = MockRegistry(REGISTRY).lookup("NIST SP 800-53 Revision 4 (NIST.SP.800-53r4.pdf)")
    line = freshness.summarize_lookup(r)
    assert "WITHDRAWN" in line and "SUPERSEDED by Rev 5" in line


# ---- arms -----------------------------------------------------------------

def test_a0_has_no_freshness_line_and_a1_does():
    q = q_by_id("irs-01")
    for arm, expected in (("A0", False), ("A1", True)):
        llm = FakeLLM(lambda *a: ("$15,750 [1]", []))
        arms.ARM_FUNCS[arm](ctx_for(q, llm))
        assert (arms.prompts.FRESHNESS_LINE in llm.calls[0]["system"]) == expected


def test_a2_shows_metadata():
    llm = FakeLLM(lambda *a: ("answer", []))
    arms.a2_metadata(ctx_for(q_by_id("nist-03"), llm))
    user_text = llm.calls[0]["messages"][0]["content"]
    assert "status=withdrawn" in user_text


def test_a3_tool_loop_returns_matching_function_response():
    def script(system, contents, decls, kw):
        if len(contents) == 1:
            return "", [("check_latest_version", {"title_or_id": "NIST SP 800-53 Rev 4"})]
        return "Rev 4 is withdrawn; Rev 5 has 20 families.", []
    llm = FakeLLM(script)
    ctx = ctx_for(q_by_id("nist-03"), llm)
    answer = arms.a3_tool(ctx)
    assert answer.startswith("Rev 4 is withdrawn")
    tool_msg = llm.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call-0"
    assert json.loads(tool_msg["content"])["status"] == "withdrawn"
    assert llm.calls[1]["messages"][1]["tool_calls"][0]["id"] == "call-0"   # assistant turn echoed back
    assert arms.registry_was_checked(ctx.trace)


def test_a3_forces_final_answer_after_max_steps():
    llm = FakeLLM(lambda s, c, d, kw: ("final" if kw.get("tool_choice") == "none" else "",
                                       [] if kw.get("tool_choice") == "none" else [("check_latest_version", {"title_or_id": "x"})]))
    assert arms.a3_tool(ctx_for(q_by_id("irs-01"), llm)) == "final"
    assert len(llm.calls) == settings.get().max_tool_steps + 1


def test_a4_annotates_every_passage_without_model_tools():
    llm = FakeLLM(lambda *a: ("answer", []))
    ctx = ctx_for(q_by_id("irs-03"), llm)
    arms.a4_forced_check(ctx)
    assert llm.calls[0]["decls"] == []
    assert llm.calls[0]["messages"][0]["content"].count("Registry check:") == len(ctx.passages)
    assert arms.registry_was_checked(ctx.trace)


def test_a5_skill_unlocks_registry_tool():
    def script(system, contents, decls, kw):
        n = len(contents)
        if n == 1:
            return "", [("load_skill", {"name": "version-check"})]
        if n == 3:
            return "", [("check_latest_version", {"title_or_id": "2024 Instructions for Form 1040"})]
        return "Superseded; 2025 value is $23,625.", []
    llm = FakeLLM(script)
    ctx = ctx_for(q_by_id("irs-03"), llm)
    arms.a5_skill(ctx)
    assert llm.calls[0]["decls"] == ["load_skill"]
    assert "check_latest_version" in llm.calls[1]["decls"]
    skill_reply = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert "Version check" in skill_reply["skill"]
    assert arms.registry_was_checked(ctx.trace)


def test_a6_and_a7_run_verifier():
    def script(system, contents, decls, kw):
        if system == arms.prompts.VERIFIER_SYSTEM:
            if len(contents) == 1:
                return "", [("check_latest_version", {"title_or_id": "NIST.SP.800-53r4.pdf"})]
            return '[{"doc": "NIST.SP.800-53r4.pdf", "status": "withdrawn", "latest_value": "Rev 5", "confidence": "high"}]', []
        if "verify_sources" in decls and len(contents) == 1:
            return "", [("verify_sources", {})]
        return "Rev 4 is withdrawn.", []
    for arm in ("A6", "A7"):
        llm = FakeLLM(script)
        ctx = ctx_for(q_by_id("nist-03"), llm)
        assert arms.ARM_FUNCS[arm](ctx) == "Rev 4 is withdrawn."
        roles = [e["role"] for e in ctx.trace if e["type"] == "llm_call"]
        assert "verifier" in roles and "main" in roles
        assert arms.registry_was_checked(ctx.trace)
    final_prompt = llm.calls[-1]["messages"][0]["content"]
    assert "Source verification report" in final_prompt and "withdrawn" in final_prompt


# ---- scoring --------------------------------------------------------------

def test_rule_match_normalises_currency():
    q = q_by_id("irs-01")
    assert scoring.rule_match(q, "It is $15,750 for 2025.") == "gold"
    assert scoring.rule_match(q, "It is $14,600.") == "alt"
    assert scoring.rule_match(q, "Now $15,750, up from $14,600.") is None


@pytest.mark.parametrize("case,matches,flags,expected", [
    ("C1", "gold", [], "correct"),
    ("C1", "alt", [], "stale"),
    ("C2", "alt", [], "stale"),
    ("C2", "alt", ["outdated"], "flagged_correctly"),
    ("C2", "gold", [], "correct"),
    ("C3", "alt", [], "missed_flag"),
    ("C3", "alt", ["unverifiable"], "flagged_correctly"),
    ("C4", "alt", [], "stale"),
    ("C5", "gold", [], "correct"),
    ("C5", "gold", ["outdated"], "false_flag"),
    ("C5", "alt", [], "false_flag"),
    ("C6", "gold", ["outdated"], "false_flag"),
    ("C6", "gold", [], "correct"),
    ("C7", "alt", ["outdated"], "missed_flag"),
    ("C7", "alt", ["inconsistent"], "flagged_correctly"),
])
def test_outcome_mapping(case, matches, flags, expected):
    assert scoring.outcome(case, matches, flags) == expected


@pytest.mark.parametrize("case,matches,flags,undermines,expected", [
    # a note that leaves the requested value standing is context, not a false flag
    ("C5", "gold", ["outdated"], False, "correct"),
    ("C6", "gold", ["outdated"], False, "correct"),
    # doubting the value still counts
    ("C5", "gold", ["outdated"], True, "false_flag"),
    ("C6", "gold", ["outdated"], True, "false_flag"),
    # swapping in the newer value counts, flag or not
    ("C5", "alt", [], False, "false_flag"),
    ("C5", "alt", ["outdated"], False, "false_flag"),
    # refusing (or hedging between versions) with a freshness warning counts
    ("C5", "no_answer", ["outdated"], False, "false_flag"),
    ("C6", "both", ["outdated"], False, "false_flag"),
    # no flag, no value: wrong, not a false flag
    ("C5", "no_answer", [], False, "wrong"),
    # other cases ignore it: in C2 the same note is the warning we want
    ("C2", "alt", ["outdated"], False, "flagged_correctly"),
])
def test_outcome_note_vs_flag(case, matches, flags, undermines, expected):
    assert scoring.outcome(case, matches, flags, undermines=undermines) == expected


def test_registry_unavailable_outcome():
    assert scoring.outcome("C1", "gold", ["unable_to_verify"], "registry_unavailable") == "flagged_correctly"
    assert scoring.outcome("C1", "gold", [], "registry_unavailable") == "missed_flag"


# ---- end to end (fake model and judge) ------------------------------------

def test_pipeline_end_to_end(tmp_path):
    qs = [q for q in QUESTIONS if q.pilot]
    models = [FakeLLM(lambda *a: ("The answer is $15,750 [1].", []), model="fake-a"),
              FakeLLM(lambda *a: ("It is $14,600 [2].", []), model="fake-b")]
    jobs = runner.interleave(models, [(q, a, 0, "normal") for q in qs for a in T2.MAIN_ARMS])
    assert [j[0].model for j in jobs[:4]] == ["fake-a", "fake-b", "fake-a", "fake-b"]
    out = tmp_path / "run" / "results.jsonl"
    results = runner.run_jobs(T2, jobs, CORPUS, out, workers=2, quiet=True)
    assert len(results) == len(jobs) == 2 * len(qs) * len(T2.MAIN_ARMS)
    assert not any(r["error"] for r in results)
    assert {r["model"] for r in results} == {"fake-a", "fake-b"}
    assert all(r["experiment"] == "track2_freshness" and "check_invoked" in r for r in results)

    # second pass is served from cache
    again = runner.run_jobs(T2, jobs, CORPUS, tmp_path / "run" / "again.jsonl", workers=2, quiet=True)
    assert all(r["from_cache"] for r in again)

    judge_llm = FakeLLM(lambda *a: (json.dumps({"rationale": "x", "answer_matches": "gold", "flags": [],
                                                  "mentions_newer_version": False, "undermines_answer": False}), []),
                        model="fake-judge")
    judge = Judge(judge_llm, T2.JUDGE_SYSTEM, T2.JUDGE_SCHEMA, cache_dir=tmp_path / "judge", fallback=T2.JUDGE_FALLBACK)
    qmap = {q.id: q for q in QUESTIONS}
    scored = [T2.score(r, qmap[r["qid"]], judge) for r in latest_per_job(results)]
    scored_path = tmp_path / "run" / "scored.jsonl"
    scored_path.write_text("\n".join(json.dumps(s) for s in scored) + "\n")
    md = report.build_report(scored_path)
    assert "| A4 |" in md and "Closure rule" in md
    assert "## fake-a" in md and "## fake-b" in md  # one table per model
    cs = report.select_confirm(scored_path)
    assert cs["arms"][:2] == ["A1", "A4"] and cs["questions"]
    assert export_handcheck(scored_path, tmp_path / "hc.csv", report.HANDCHECK_FIELDS) >= 1
    assert logschema.write_shared(tmp_path / "run") == len(results)
    for line in load_jsonl(tmp_path / "run" / "shared.jsonl"):
        assert logschema.validate(line) == [] and line["outcome"] in scoring.OUTCOMES


def test_judge_parse_error_uses_fallback(tmp_path):
    judge = Judge(FakeLLM(lambda *a: ("not json", [])), T2.JUDGE_SYSTEM, T2.JUDGE_SCHEMA,
                  cache_dir=tmp_path, fallback=T2.JUDGE_FALLBACK)
    rec = {"key": "k", "model": "m", "arm": "A0", "qid": "irs-01", "case": "C1", "family": "f", "seed": 0,
           "condition": "normal", "answer": "no idea", "check_invoked": False, "n_calls": 1, "cost_usd": 0.0,
           "latency_s": 0.0}
    s = T2.score(rec, q_by_id("irs-01"), judge)
    assert s["outcome"] == "wrong" and s["judge_rationale"] == "JUDGE PARSE ERROR"


def test_judge_flags_outside_the_vocabulary_are_dropped(tmp_path):
    labels = {"rationale": "x", "answer_matches": "gold", "flags": ["outdated", "made_up"],
              "mentions_newer_version": False, "undermines_answer": False}
    judge = Judge(FakeLLM(lambda *a: (json.dumps(labels), [])), T2.JUDGE_SYSTEM, T2.JUDGE_SCHEMA, cache_dir=tmp_path)
    rec = {"key": "k", "model": "m", "arm": "A0", "qid": "irs-03", "case": "C2", "family": "f", "seed": 0,
           "condition": "normal", "answer": "$21,900, may be outdated", "check_invoked": False, "n_calls": 1,
           "cost_usd": 0.0, "latency_s": 0.0}
    assert T2.score(rec, q_by_id("irs-03"), judge)["flags"] == ["outdated"]


def test_four_numbers_on_the_sweep():
    rows = latest_per_job(load_jsonl(REPO / "results/track2/sweep-20261028-010735/scored.jsonl"))
    a4 = [r for r in rows if r["arm"] == "A4" and r["model"] == "google/gemma-4-31b-it"]
    n = T2.four_numbers(a4)
    assert n["detection"] == (1.0, 48) and n["false_flags"] == (0.0, 36) and n["coverage"] == (1.0, 45)
