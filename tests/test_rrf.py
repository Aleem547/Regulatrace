"""Offline toy rankings for manually implemented reciprocal rank fusion."""

import pytest

from regulatrace.rrf import reciprocal_rank_fusion


def test_formula_shared_and_unique_ids():
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["b", "c", "d"])

    assert [chunk_id for chunk_id, _ in fused] == ["b", "c", "a", "d"]
    assert dict(fused) == {
        "a": pytest.approx(1 / 61),
        "b": pytest.approx(1 / 62 + 1 / 61),
        "c": pytest.approx(1 / 63 + 1 / 62),
        "d": pytest.approx(1 / 63),
    }


@pytest.mark.parametrize(
    ("dense", "sparse", "expected"),
    [([], [], []), (["a"], [], [("a", 1 / 61)]), ([], ["b"], [("b", 1 / 61)])],
)
def test_empty_rankings(dense, sparse, expected):
    assert reciprocal_rank_fusion(dense, sparse) == expected


def test_duplicate_ids_use_first_original_rank_once():
    fused = reciprocal_rank_fusion(["a", "a", "b"], ["b", "b", "a"])

    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert dict(fused) == {
        "a": pytest.approx(1 / 61 + 1 / 63),
        "b": pytest.approx(1 / 63 + 1 / 61),
    }


def test_tied_scores_keep_first_seen_order():
    assert [chunk_id for chunk_id, _ in reciprocal_rank_fusion(["a", "b"], ["b", "a"])] == [
        "a",
        "b",
    ]


def test_non_default_smoothing_constant():
    fused = reciprocal_rank_fusion(["a", "b"], ["b"], smoothing=1)

    assert [chunk_id for chunk_id, _ in fused] == ["b", "a"]
    assert dict(fused) == {"a": pytest.approx(1 / 2), "b": pytest.approx(1 / 3 + 1 / 2)}


@pytest.mark.parametrize("smoothing", [0, -1, True, 1.5])
def test_invalid_smoothing_constant(smoothing):
    with pytest.raises(ValueError, match="smoothing must be a positive integer"):
        reciprocal_rank_fusion(["a"], ["b"], smoothing=smoothing)
