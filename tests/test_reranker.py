"""Offline tests for cross-encoder reranking of existing retrieval results."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from regulatrace import reranker


def candidate(text, score, page):
    return {
        "text": text,
        "score": score,
        "source_file": "circular.pdf",
        "page": page,
        "metadata": {"doc_id": "document", "page": page, "regulator": "RBI"},
    }


def test_reranks_truncates_and_preserves_retrieval_data():
    candidates = [
        candidate("First passage", 0.9, 1),
        candidate("Second passage", 0.7, 2),
        candidate("Third passage", 0.5, 3),
    ]
    original = deepcopy(candidates)
    model = Mock()
    model.predict.return_value = [0.1, 2.5, 1.0]

    results = reranker.rerank("What is the rule?", candidates, k=2, model=model)

    model.predict.assert_called_once_with(
        [
            ("What is the rule?", "First passage"),
            ("What is the rule?", "Second passage"),
            ("What is the rule?", "Third passage"),
        ],
        show_progress_bar=False,
    )
    assert results == [
        {**original[1], "rerank_score": 2.5},
        {**original[2], "rerank_score": 1.0},
    ]
    assert candidates == original


def test_tied_scores_keep_original_candidate_order():
    candidates = [candidate("A", 0.3, 1), candidate("B", 0.2, 2), candidate("C", 0.1, 3)]
    model = Mock()
    model.predict.return_value = [1.0, 2.0, 2.0]

    assert [item["text"] for item in reranker.rerank("Question", candidates, model=model)] == [
        "B",
        "C",
        "A",
    ]


def test_empty_candidates_do_not_load_or_call_model(monkeypatch):
    factory = Mock(side_effect=AssertionError("Model must not load"))
    monkeypatch.setattr(reranker, "CrossEncoder", factory)
    model = Mock()

    assert reranker.rerank("Question", [], model=model) == []
    assert reranker.rerank("Question", []) == []
    model.predict.assert_not_called()
    factory.assert_not_called()


@pytest.mark.parametrize("question", [None, 123, "", " \n "])
def test_invalid_question_rejected_before_model(question):
    model = Mock()
    with pytest.raises((TypeError, ValueError)):
        reranker.rerank(question, [candidate("Evidence", 0.5, 1)], model=model)
    model.predict.assert_not_called()


@pytest.mark.parametrize("k", [None, True, 0, -1, 1.5])
def test_invalid_k_rejected_before_model(k):
    model = Mock()
    with pytest.raises((TypeError, ValueError)):
        reranker.rerank("Question", [candidate("Evidence", 0.5, 1)], k=k, model=model)
    model.predict.assert_not_called()


@pytest.mark.parametrize(
    "candidates", [None, {}, "text", [None], [{"text": None}], [{"text": "  "}]]
)
def test_invalid_candidates_or_missing_text_rejected(candidates):
    model = Mock()
    with pytest.raises((TypeError, ValueError)):
        reranker.rerank("Question", candidates, model=model)
    model.predict.assert_not_called()


def test_default_model_is_loaded_once_and_reused(monkeypatch):
    reranker._get_reranker.cache_clear()
    model = Mock()
    model.predict.side_effect = [[0.5], [0.7]]
    factory = Mock(return_value=model)
    monkeypatch.setattr(reranker, "CrossEncoder", factory)
    try:
        reranker.rerank("First question", [candidate("Evidence", 0.2, 1)])
        reranker.rerank("Second question", [candidate("Evidence", 0.2, 1)])
        factory.assert_called_once_with("BAAI/bge-reranker-base")
        assert model.predict.call_count == 2
    finally:
        reranker._get_reranker.cache_clear()


@pytest.mark.parametrize("scores", [[0.1], [float("nan"), 0.4]])
def test_invalid_model_scores_are_rejected(scores):
    model = Mock()
    model.predict.return_value = scores
    with pytest.raises(ValueError):
        reranker.rerank("Question", [candidate("A", 0.2, 1), candidate("B", 0.1, 2)], model=model)


@pytest.mark.parametrize(
    ("score", "abstain"),
    [(0.59, True), (0.60, False), (0.61, False)],
)
def test_abstention_threshold(score, abstain):
    result = reranker.decide_abstention([{"rerank_score": score, "score": 0.99}])
    assert result["abstain"] is abstain
    assert result["top_rerank_score"] == score
    assert result["message"] == (
        reranker.ABSTENTION_MESSAGE
        if abstain
        else "Retrieved evidence meets the score threshold; answer correctness is not assured."
    )


def test_no_candidates_abstains():
    assert reranker.decide_abstention([]) == {
        "abstain": True,
        "top_rerank_score": None,
        "message": reranker.ABSTENTION_MESSAGE,
    }


def test_abstention_uses_highest_score_without_changing_results():
    results = [{"rerank_score": 0.2}, {"rerank_score": 0.72}, {"rerank_score": 0.4}]
    original = deepcopy(results)
    decision = reranker.decide_abstention(results)
    assert decision["abstain"] is False
    assert decision["top_rerank_score"] == 0.72
    assert results == original


@pytest.mark.parametrize(
    "results",
    [
        None,
        {},
        [None],
        [{}],
        [{"rerank_score": None}],
        [{"rerank_score": True}],
        [{"rerank_score": "0.8"}],
        [{"rerank_score": float("nan")}],
        [{"rerank_score": float("inf")}],
        [{"rerank_score": 0.8}, {}],
    ],
)
def test_invalid_reranked_results_are_rejected(results):
    with pytest.raises((TypeError, ValueError), match="reranked"):
        reranker.decide_abstention(results)
