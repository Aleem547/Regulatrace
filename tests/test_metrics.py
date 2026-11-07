"""Hand-made examples for page-level retrieval metrics; no live retrieval."""

from copy import deepcopy

import pytest
from eval.metrics import hit_at_k, recall_at_k, reciprocal_rank


def result(doc_id="doc-a", page=1):
    return {
        "text": "Evidence",
        "score": 0.8,
        "source_file": "document.pdf",
        "page": page,
        "metadata": {"doc_id": doc_id, "page": page},
    }


GOLD = [{"doc_id": "doc-a", "page": 1}]
METRICS = (hit_at_k, recall_at_k, reciprocal_rank)


@pytest.mark.parametrize(
    "results, k, expected",
    [
        ([result()], 1, 1.0),
        ([result("other", 1)], 1, 0.0),
        ([result("other", 1), result()], 1, 0.0),
        ([result("other", 1), result()], 2, 1.0),
    ],
)
def test_hit_at_k_respects_original_top_k(results, k, expected):
    assert hit_at_k(results, GOLD, k) == expected


def test_recall_partial_and_complete():
    gold = [
        {"doc_id": "doc-a", "page": 1},
        {"doc_id": "doc-a", "page": 2},
        {"doc_id": "doc-b", "page": 1},
    ]
    results = [result("doc-a", 1), result("doc-b", 1), result("doc-a", 2)]
    assert recall_at_k(results, gold, 2) == pytest.approx(2 / 3)
    assert recall_at_k(results, gold, 3) == 1.0


def test_recall_counts_distinct_gold_pages_once():
    gold = [*GOLD, *GOLD, {"doc_id": "doc-b", "page": 2}]
    results = [result(), result(), result("doc-b", 2)]
    assert recall_at_k(results, gold, 2) == 0.5
    assert recall_at_k(results, gold, 3) == 1.0


def test_duplicate_chunks_occupy_ranks_before_k_cutoff():
    gold = [*GOLD, {"doc_id": "doc-b", "page": 2}]
    results = [result(), result(), result("doc-b", 2)]
    assert recall_at_k(results, gold, 2) == 0.5
    assert reciprocal_rank(results, [{"doc_id": "doc-b", "page": 2}], 2) == 0.0
    assert reciprocal_rank(results, [{"doc_id": "doc-b", "page": 2}], 3) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "results, k, expected",
    [
        ([result()], 1, 1.0),
        ([result("other", 1), result()], 2, 0.5),
        ([result("other", 1), result()], 1, 0.0),
        ([result("other", 1)], 2, 0.0),
    ],
)
def test_reciprocal_rank_uses_first_matching_position(results, k, expected):
    assert reciprocal_rank(results, GOLD, k) == expected


@pytest.mark.parametrize("metric", METRICS)
def test_same_page_in_different_document_does_not_match(metric):
    assert metric([result("doc-b", 1)], GOLD, 1) == 0.0


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"doc_id": "doc-a"},
        {"page": 1},
        {"doc_id": "", "page": 1},
        {"doc_id": "doc-a", "page": 0},
        {"doc_id": "doc-a", "page": -1},
        {"doc_id": "doc-a", "page": True},
        {"doc_id": "doc-a", "page": "1"},
    ],
)
def test_missing_or_invalid_retrieved_metadata_cannot_match(metadata):
    retrieved = {**result(), "metadata": metadata}
    for metric in METRICS:
        assert metric([retrieved], GOLD, 1) == 0.0


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "k, error", [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_k(metric, k, error):
    with pytest.raises(error, match="k must be a positive integer"):
        metric([result()], GOLD, k)


@pytest.mark.parametrize("metric", METRICS)
def test_empty_gold_sources_are_rejected(metric):
    with pytest.raises(ValueError, match="gold_sources must be nonempty"):
        metric([result()], [], 1)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize(
    "invalid_source",
    [{"doc_id": "doc-a"}, {"doc_id": "doc-a", "page": 0}, {"doc_id": None, "page": 1}],
)
def test_invalid_gold_source_is_rejected(metric, invalid_source):
    with pytest.raises(ValueError, match="gold_sources must contain"):
        metric([result()], [*GOLD, invalid_source], 1)


@pytest.mark.parametrize("metric", METRICS)
def test_metrics_do_not_mutate_inputs(metric):
    results = [result("other", 1), result()]
    gold = [*GOLD, {"doc_id": "doc-b", "page": 2}]
    original_results, original_gold = deepcopy(results), deepcopy(gold)

    metric(results, gold, 2)

    assert results == original_results
    assert gold == original_gold
