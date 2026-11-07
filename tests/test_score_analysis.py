"""Offline checks for development-only reranker-score analysis."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, call
from xml.etree import ElementTree

import pytest
from eval import analyze_rerank_scores as analysis


@pytest.fixture
def score_setup(monkeypatch, tmp_path):
    holdout = {f"H{index:02}" for index in range(12)}
    records = [
        {"id": item, "question": "hold-out question", "answerable": True, "verified": True}
        for item in sorted(holdout)
    ]
    records += [
        {"id": f"A{index:02}", "question": f"A{index:02}", "answerable": True, "verified": True}
        for index in range(41)
    ]
    records += [
        {"id": f"U{index:02}", "question": f"U{index:02}", "answerable": False, "verified": True}
        for index in range(7)
    ]
    monkeypatch.setattr(analysis.run_retrieval, "load_golden", lambda: records)
    monkeypatch.setattr(analysis.run_retrieval, "load_holdout", lambda ids: holdout)
    monkeypatch.setattr(
        analysis, "get_settings", lambda: SimpleNamespace(bm25_collection="regulatrace_bm25")
    )
    monkeypatch.setattr(analysis, "RESULTS_DIR", tmp_path / "results")

    def candidates(question, k):
        count = 0 if question == "A00" else 3 if question == "U00" else k
        return [{"text": f"Candidate {index}", "score": 0.5} for index in range(count)]

    retriever = Mock(side_effect=candidates)
    monkeypatch.setattr(analysis, "retrieve_bm25", retriever)
    model = Mock()
    model.activation_fn = lambda value: value
    model.predict.side_effect = lambda pairs, **_kwargs: (
        [0.8 if pairs[0][0].startswith("A") else -1.5] + [-2.0] * (len(pairs) - 1)
    )
    loader = Mock(return_value=model)
    monkeypatch.setattr(analysis, "_get_reranker", loader)
    return records, retriever, loader, model


def test_development_only_scores_allow_short_and_empty_candidate_lists(score_setup):
    _, retriever, loader, model = score_setup
    report = analysis.analyze_scores()

    assert report["counts"] == {
        "development": 48,
        "answerable": 41,
        "unanswerable": 7,
        "holdout_excluded": 12,
    }
    assert [row["id"] for row in report["per_question"]] == [
        *(f"A{index:02}" for index in range(41)),
        *(f"U{index:02}" for index in range(7)),
    ]
    assert report["per_question"][0]["top_rerank_score"] is None
    assert report["per_question"][41]["top_rerank_score"] == -1.5
    assert report["score_behavior"]["observed_min"] == -1.5
    assert report["score_behavior"]["observed_max"] == 0.8
    assert report["thresholds"][0]["correct_refusals"] == 7
    assert report["thresholds"][0]["wrong_refusals"] == 1
    assert retriever.call_args_list == [call(f"A{index:02}", k=20) for index in range(41)] + [
        call(f"U{index:02}", k=20) for index in range(7)
    ]
    loader.assert_called_once_with()
    assert model.predict.call_count == 47  # No model call for the empty result.
    assert sorted(len(entry.args[0]) for entry in model.predict.call_args_list) == [3] + [20] * 46
    assert not analysis.RESULTS_DIR.exists()


def test_threshold_counts_use_strict_boundary_and_refuse_without_evidence():
    rows = [
        {"answerable": True, "top_rerank_score": None},
        {"answerable": True, "top_rerank_score": 0.2},
        {"answerable": False, "top_rerank_score": None},
        {"answerable": False, "top_rerank_score": 0.2},
    ]
    assert analysis.threshold_counts(rows, 0.2) == {
        "threshold": 0.2,
        "correct_refusals": 1,
        "missed_refusals": 1,
        "wrong_refusals": 1,
        "correct_acceptances": 1,
        "correct_refusal_rate": 0.5,
        "wrong_refusal_rate": 0.5,
    }


def test_saves_one_report_and_readable_score_plot(score_setup):
    report = analysis.analyze_scores()
    json_path, plot_path = analysis.save_analysis(report)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["per_question"] == report["per_question"]
    assert saved["configuration"]["candidate_k"] == 20
    assert saved["configuration"]["reranker_model"] == analysis.MODEL_NAME
    svg = plot_path.read_text(encoding="utf-8")
    ElementTree.fromstring(svg)
    assert "Answerable (41)" in svg and "Unanswerable (7)" in svg
    assert "1 no-candidate" in svg
    with pytest.raises(FileExistsError):
        analysis.save_analysis(report)


def test_unverified_development_record_fails_before_retrieval(score_setup):
    records, retriever, loader, _ = score_setup
    records[-1]["verified"] = False
    with pytest.raises(ValueError, match="48 verified development"):
        analysis.analyze_scores()
    loader.assert_not_called()
    retriever.assert_not_called()
