"""Offline tests for the paired, development-only prompt comparison."""

import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, call

import httpx
import pytest
from eval import compare_prompts as comparison
from groq import RateLimitError


def rate_limit_error(headers=None):
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return RateLimitError("Rate limit exceeded", response=response, body=None)


@pytest.fixture
def fake_clock(monkeypatch):
    now = [0.0]
    sleeper = Mock(side_effect=lambda delay: now.__setitem__(0, now[0] + delay))
    monkeypatch.setattr(comparison.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(comparison.time, "sleep", sleeper)
    return now, sleeper


@pytest.fixture
def paired_setup(monkeypatch, tmp_path, fake_clock):
    holdout = {f"H{index:02}" for index in range(12)}
    records = [{"id": item} for item in sorted(holdout)]
    records += [
        {
            "id": f"A{index:02}",
            "question": f"A{index:02}",
            "type": "factoid",
            "answerable": True,
            "verified": True,
            "reference_answer": "Reference rule",
            "gold_sources": [{"doc_id": "shared-doc", "page": 1}],
        }
        for index in range(41)
    ]
    records += [
        {
            "id": f"U{index:02}",
            "question": f"U{index:02}",
            "type": "unanswerable",
            "answerable": False,
            "verified": True,
            "reference_answer": "not found in the provided documents",
            "gold_sources": [],
        }
        for index in range(7)
    ]
    monkeypatch.setattr(comparison.run_retrieval, "load_golden", lambda: records)
    monkeypatch.setattr(comparison.run_retrieval, "load_holdout", lambda ids: holdout)
    monkeypatch.setattr(
        comparison,
        "get_settings",
        lambda: SimpleNamespace(
            llm_provider="groq",
            llm_model="openai/gpt-oss-120b",
            bm25_collection="regulatrace_bm25",
        ),
    )
    monkeypatch.setattr(comparison, "RESULTS_DIR", tmp_path / "results")
    candidate = {
        "text": "Complete regulatory clause.",
        "score": 2.0,
        "source_file": "rule.pdf",
        "page": 1,
        "metadata": {"doc_id": "shared-doc", "title": "Rule", "chunk_index": 4},
    }
    retrieval = Mock(return_value=[candidate])
    monkeypatch.setattr(comparison, "retrieve_bm25", retrieval)
    ranking = Mock(
        side_effect=lambda question, candidates, k, model: [
            {**item, "rerank_score": 0.8} for item in candidates
        ][:k]
    )
    monkeypatch.setattr(comparison, "rerank", ranking)
    loader = Mock(return_value=object())
    monkeypatch.setattr(comparison, "_get_reranker", loader)
    llm = Mock()
    llm.generate.return_value = "The rule applies [shared-doc p.1]."
    llm_factory = Mock(return_value=llm)
    monkeypatch.setattr(comparison, "get_llm", llm_factory)
    return records, retrieval, ranking, loader, llm_factory, llm


def test_shared_evidence_and_one_retrieval_per_development_question(paired_setup, fake_clock):
    _, retrieval, ranking, loader, llm_factory, llm = paired_setup

    report = comparison.compare_prompts()

    assert report["summary"]["questions_evaluated"] == 48
    assert report["summary"]["gate_refusals"] == 0
    assert report["summary"]["v1"]["answers_generated"] == 48
    assert report["summary"]["v2"]["answers_generated"] == 48
    assert report["summary"]["v1"]["citations_to_supplied_sources"] == 48
    assert retrieval.call_args_list == [call(f"A{index:02}", k=20) for index in range(41)] + [
        call(f"U{index:02}", k=20) for index in range(7)
    ]
    assert ranking.call_count == 48
    assert llm.generate.call_count == 96
    assert fake_clock[1].call_count > 0  # The same pacing window spans questions and versions.
    loader.assert_called_once_with()
    llm_factory.assert_called_once()
    assert all(
        entry.kwargs == {"temperature": 0, "max_completion_tokens": 1024}
        for entry in llm.generate.call_args_list
    )
    first = report["per_question"][0]
    assert first["id"] == "A00"
    assert first["reference_answer"] == "Reference rule"
    assert first["evidence"]["sources"][0]["doc_id"] == "shared-doc"
    assert "Complete regulatory clause." in first["evidence"]["context"]
    v1_prompt, v2_prompt = (entry.args[0] for entry in llm.generate.call_args_list[:2])
    assert "Keep regulatory categories distinct" in v1_prompt
    assert "Each cited passage must support" in v2_prompt
    assert first["evidence"]["context"] in v1_prompt
    assert first["evidence"]["context"] in v2_prompt
    assert first["answers"]["v1"]["prompt_tokens"] <= 7680
    assert first["answers"]["v2"]["prompt_tokens"] <= 7680
    assert [row["id"] for row in report["per_question"]].count("H00") == 0

    path = comparison.save_results(report)
    assert json.loads(path.read_text(encoding="utf-8"))["per_question"] == report["per_question"]
    with pytest.raises(FileExistsError):
        comparison.save_results(report)


def test_gate_and_evidence_refusals_skip_both_generations(paired_setup, monkeypatch):
    _, retrieval, ranking, _, _, llm = paired_setup
    retrieval.side_effect = lambda question, k: [] if question == "U00" else retrieval.return_value
    llm.generate.side_effect = lambda prompt, **kwargs: (
        comparison.rag._NOT_FOUND
        if "QUESTION\nA02" in prompt
        else "The rule applies [shared-doc p.1]."
    )
    ranking.side_effect = lambda question, candidates, k, model: [
        {**item, "rerank_score": 0.59 if question == "A00" else 0.8} for item in candidates
    ][:k]
    original_budget = comparison.rag._budget_evidence
    monkeypatch.setattr(
        comparison.rag,
        "_budget_evidence",
        lambda question, chunks, prompts: (
            {"context": "", "sources": []}
            if question == "A01"
            else original_budget(question, chunks, prompts)
        ),
    )

    report = comparison.compare_prompts()

    assert report["summary"]["gate_refusals"] == 2
    assert report["summary"]["evidence_refusals"] == 1
    assert report["summary"]["v1"]["answers_generated"] == 45
    assert report["summary"]["v2"]["answers_generated"] == 45
    assert report["summary"]["v1"]["not_found_responses"] == 1
    assert report["summary"]["v2"]["not_found_responses"] == 1
    assert llm.generate.call_count == 90
    assert report["per_question"][0]["stage"] == "gate_refusal"
    assert report["per_question"][0]["answers"] == {"v1": None, "v2": None}
    assert report["per_question"][1]["stage"] == "evidence_refusal"
    assert report["per_question"][1]["refusal_answer"] == comparison.rag._NOT_FOUND
    assert report["per_question"][41]["stage"] == "gate_refusal"


def test_citation_check_only_validates_supplied_document_pages():
    references = comparison.check_citations(
        "Rule [doc p.2]; another [other p.3]; malformed [Source 1].",
        [{"doc_id": "doc", "page": 2}],
    )
    assert references == [
        {"doc_id": "doc", "page": 2, "in_supplied_sources": True},
        {"doc_id": "other", "page": 3, "in_supplied_sources": False},
    ]


def test_invalid_development_split_fails_before_model_or_retrieval(paired_setup):
    records, retrieval, _, loader, llm_factory, _ = paired_setup
    records[-1]["verified"] = False
    with pytest.raises(ValueError, match="48 verified development"):
        comparison.compare_prompts()
    retrieval.assert_not_called()
    loader.assert_not_called()
    llm_factory.assert_not_called()


def test_rate_limit_retries_generation_without_repeating_retrieval(
    paired_setup, fake_clock, capsys
):
    _, retrieval, ranking, _, _, llm = paired_setup
    llm.generate.side_effect = [
        rate_limit_error(
            {
                "retry-after": "2.5",
                "x-ratelimit-remaining-tokens": "0",
                "x-ratelimit-reset-tokens": "7.66s",
            }
        )
    ] + ["The rule applies [shared-doc p.1]."] * 96

    report = comparison.compare_prompts()

    assert report["summary"]["v1"]["answers_generated"] == 48
    assert report["summary"]["v2"]["answers_generated"] == 48
    assert retrieval.call_count == ranking.call_count == 48
    assert llm.generate.call_count == 97
    fake_clock[1].assert_any_call(2.5)
    output = capsys.readouterr().out
    assert "waiting 2.5s before retry 2/5" in output
    assert "Groq HTTP 429: Rate limit exceeded" in output
    assert "retry-after=2.5" in output
    assert "x-ratelimit-remaining-tokens=0" in output
    assert "x-ratelimit-reset-tokens=7.66s" in output


def test_rate_limit_fallback_and_exhaustion_reraises(fake_clock, capsys):
    errors = [rate_limit_error({"retry-after": "invalid"}) for _ in range(5)]
    llm = Mock()
    llm.generate.side_effect = errors

    with pytest.raises(RateLimitError) as caught:
        comparison._generate_with_retry(llm, "same prompt", 2, deque())

    assert caught.value is errors[-1]
    assert llm.generate.call_count == 5
    assert fake_clock[1].call_args_list == [call(15.0)] * 4
    assert capsys.readouterr().out.count("Groq HTTP 429:") == 5


def test_pacing_tracks_all_requests_until_their_60_second_expiry(fake_clock):
    now, sleeper = fake_clock
    recent = deque()
    for timestamp in (0, 10, 20, 30):
        now[0] = timestamp
        comparison._pace_generation(1000, recent)

    # Four requests need 8,096 estimated tokens; the oldest must expire first.
    sleeper.assert_called_once_with(30)
    assert list(recent) == [(10, 2024), (20, 2024), (60, 2024)]


def test_request_too_large_for_pacing_fails_before_generation(fake_clock):
    llm = Mock()
    with pytest.raises(ValueError, match="waiting cannot make it fit"):
        comparison._generate_with_retry(llm, "prompt", comparison._PACING_TOKEN_BUDGET, deque())
    llm.generate.assert_not_called()
    fake_clock[1].assert_not_called()


def test_limit_saves_two_completed_questions_and_prints_progress(paired_setup, monkeypatch, capsys):
    _, retrieval, ranking, _, _, llm = paired_setup
    monkeypatch.setattr("sys.argv", ["compare_prompts", "--limit", "2"])

    comparison.main()

    assert retrieval.call_args_list == [call("A00", k=20), call("A01", k=20)]
    assert ranking.call_count == 2
    assert llm.generate.call_count == 4
    (progress_path,) = comparison.RESULTS_DIR.glob("prompt_comparison_*.jsonl")
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["A00", "A01"]
    assert all(row["answers"]["v1"] and row["answers"]["v2"] for row in rows)
    (report_path,) = comparison.RESULTS_DIR.glob("prompt_comparison_*.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["questions_evaluated"] == 2
    assert report["per_question"] == rows
    output = capsys.readouterr().out
    assert "Completed 1/2" in output
    assert "Completed 2/2" in output


def test_interrupted_run_keeps_only_completed_jsonl_rows(paired_setup, monkeypatch):
    _, retrieval, _, _, _, llm = paired_setup
    llm.generate.side_effect = [
        "First answer [shared-doc p.1].",
        "Second answer [shared-doc p.1].",
        RuntimeError("Generation stopped"),
    ]
    monkeypatch.setattr("sys.argv", ["compare_prompts", "--limit", "2"])

    with pytest.raises(SystemExit) as caught:
        comparison.main()
    assert caught.value.code == 1

    (progress_path,) = comparison.RESULTS_DIR.glob("prompt_comparison_*.jsonl")
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["A00"]
    assert rows[0]["answers"]["v1"]["text"] == "First answer [shared-doc p.1]."
    assert rows[0]["answers"]["v2"]["text"] == "Second answer [shared-doc p.1]."
    assert retrieval.call_args_list == [call("A00", k=20), call("A01", k=20)]
    assert list(comparison.RESULTS_DIR.glob("prompt_comparison_*.json")) == []
