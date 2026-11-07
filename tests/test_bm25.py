"""Offline tests for BM25 collection setup and backfill; no live Qdrant calls."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient, models

from regulatrace import bm25
from regulatrace.config import Settings


@pytest.fixture(autouse=True)
def bm25_settings(monkeypatch):
    settings = Settings(
        _env_file=None,
        qdrant_collection="regulatrace_chunks",
        bm25_collection="regulatrace_bm25",
    )
    monkeypatch.setattr(bm25, "get_settings", lambda: settings)


def collection_info(vectors=None, sparse=None):
    return SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors, sparse_vectors=sparse))
    )


def test_creates_sparse_only_collection_with_idf():
    client = Mock()
    client.collection_exists.return_value = False

    bm25.init_bm25_collection(client=client)

    client.collection_exists.assert_called_once_with("regulatrace_bm25")
    client.get_collection.assert_not_called()
    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "regulatrace_bm25"
    assert "vectors_config" not in kwargs
    assert set(kwargs["sparse_vectors_config"]) == {"bm25"}
    assert kwargs["sparse_vectors_config"]["bm25"].modifier == models.Modifier.IDF


def test_existing_compatible_collection_is_unchanged():
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = collection_info(
        vectors={},
        sparse={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )

    bm25.init_bm25_collection(client=client)
    bm25.init_bm25_collection(client=client)

    assert client.get_collection.call_count == 2
    client.create_collection.assert_not_called()
    client.delete_collection.assert_not_called()


def test_sparse_only_schema_works_in_memory():
    client = QdrantClient(":memory:")

    bm25.init_bm25_collection(client=client)
    bm25.init_bm25_collection(client=client)

    params = client.get_collection("regulatrace_bm25").config.params
    assert params.vectors in (None, {})
    assert set(params.sparse_vectors) == {"bm25"}
    assert params.sparse_vectors["bm25"].modifier == models.Modifier.IDF


@pytest.mark.parametrize(
    "vectors, sparse",
    [
        (
            {"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        ),
        ({}, {"bm25": models.SparseVectorParams()}),
        (
            {},
            {
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
                "extra": models.SparseVectorParams(),
            },
        ),
    ],
)
def test_incompatible_collection_raises_without_changes(vectors, sparse):
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = collection_info(vectors=vectors, sparse=sparse)

    with pytest.raises(ValueError, match="incompatible schema"):
        bm25.init_bm25_collection(client=client)

    client.create_collection.assert_not_called()
    client.delete_collection.assert_not_called()


def test_dense_collection_name_is_rejected_before_qdrant_call():
    client = Mock()

    with pytest.raises(ValueError, match="must differ from the dense collection"):
        bm25.init_bm25_collection(client=client, collection_name="regulatrace_chunks")

    client.collection_exists.assert_not_called()


@pytest.fixture
def backfill_client():
    client = Mock()
    client.cloud_inference = True
    client.collection_exists.return_value = True
    client.get_collection.return_value = collection_info(
        vectors={}, sparse={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}
    )
    return client


def test_backfill_paginates_and_preserves_ids_payloads(backfill_client):
    source = [
        SimpleNamespace(
            id=index, payload={"text": f"Chunk {index}", "doc_id": "doc", "page": index}
        )
        for index in (1, 2, 3)
    ]
    backfill_client.scroll.side_effect = [(source[:2], 3), (source[2:], None)]

    assert bm25.backfill_bm25(client=backfill_client, batch_size=2) == 3

    assert [call.kwargs["offset"] for call in backfill_client.scroll.call_args_list] == [None, 3]
    for call in backfill_client.scroll.call_args_list:
        assert call.kwargs["collection_name"] == "regulatrace_chunks"
        assert call.kwargs["limit"] == 2
        assert call.kwargs["with_payload"] is True
        assert call.kwargs["with_vectors"] is False
    assert backfill_client.upsert.call_count == 2
    written = []
    for call in backfill_client.upsert.call_args_list:
        assert call.kwargs["collection_name"] == "regulatrace_bm25"
        assert call.kwargs["wait"] is True
        written.extend(call.kwargs["points"])
    assert [point.id for point in written] == [point.id for point in source]
    assert [point.payload for point in written] == [point.payload for point in source]
    for point, original in zip(written, source, strict=True):
        assert point.vector == {
            "bm25": models.Document(text=original.payload["text"], model="qdrant/bm25")
        }
    backfill_client.create_collection.assert_not_called()
    backfill_client.delete.assert_not_called()


@pytest.mark.parametrize("text", [None, "  "])
def test_backfill_rejects_missing_text_before_batch_write(backfill_client, text):
    backfill_client.scroll.return_value = (
        [
            SimpleNamespace(id=1, payload={"text": "Valid"}),
            SimpleNamespace(id=2, payload={"text": text}),
        ],
        None,
    )

    with pytest.raises(ValueError, match="Dense point 2 has missing or empty text"):
        bm25.backfill_bm25(client=backfill_client)

    backfill_client.upsert.assert_not_called()


def test_backfill_uses_server_inference_client(monkeypatch, backfill_client):
    factory = Mock(return_value=backfill_client)
    monkeypatch.setattr(bm25, "QdrantClient", factory)
    backfill_client.scroll.return_value = ([], None)

    assert bm25.backfill_bm25() == 0

    factory.assert_called_once_with(url="http://localhost:6333", api_key=None, cloud_inference=True)
    backfill_client.upsert.assert_not_called()


def test_backfill_rejects_same_source_and_target_before_qdrant_calls(monkeypatch, backfill_client):
    settings = Settings(
        _env_file=None, qdrant_collection="regulatrace_chunks", bm25_collection="regulatrace_chunks"
    )
    monkeypatch.setattr(bm25, "get_settings", lambda: settings)

    with pytest.raises(ValueError, match="must differ from the dense collection"):
        bm25.backfill_bm25(client=backfill_client)

    backfill_client.collection_exists.assert_not_called()
    backfill_client.scroll.assert_not_called()
    backfill_client.upsert.assert_not_called()


def test_retrieve_bm25_preserves_order_scores_and_metadata(monkeypatch):
    client = Mock()
    client.query_points.return_value.points = [
        models.ScoredPoint(
            id=2,
            version=1,
            score=0.4,
            payload={
                "text": "First chunk",
                "source_file": "first.pdf",
                "doc_id": "first-document",
                "page": 3,
                "section": "Reporting",
                "chunk_index": 1,
                "extra": "not canonical metadata",
            },
        ),
        models.ScoredPoint(id=1, version=1, score=1.2, payload={"text": "Second chunk"}),
    ]
    factory = Mock(return_value=client)
    monkeypatch.setattr(bm25, "QdrantClient", factory)

    results = bm25.retrieve_bm25("monthly return", k=2)

    factory.assert_called_once_with(url="http://localhost:6333", api_key=None, cloud_inference=True)
    client.query_points.assert_called_once()
    search = client.query_points.call_args.kwargs
    assert search == {
        "collection_name": "regulatrace_bm25",
        "query": models.Document(text="monthly return", model="qdrant/bm25"),
        "using": "bm25",
        "limit": 2,
        "with_payload": True,
    }
    assert results == [
        {
            "text": "First chunk",
            "score": 0.4,
            "source_file": "first.pdf",
            "page": 3,
            "metadata": {
                "doc_id": "first-document",
                "section": "Reporting",
                "page": 3,
                "chunk_index": 1,
            },
        },
        {
            "text": "Second chunk",
            "score": 1.2,
            "source_file": None,
            "page": None,
            "metadata": {},
        },
    ]
    client.upsert.assert_not_called()
    client.create_collection.assert_not_called()


def test_retrieve_bm25_empty_results(monkeypatch):
    client = Mock()
    client.query_points.return_value.points = []
    monkeypatch.setattr(bm25, "QdrantClient", Mock(return_value=client))

    assert bm25.retrieve_bm25("unmatched question") == []
    assert client.query_points.call_args.kwargs["limit"] == 5


@pytest.mark.parametrize("regulator", ["RBI", "SEBI"])
def test_retrieve_bm25_prefilters_regulator(monkeypatch, regulator):
    client = Mock()
    client.query_points.return_value.points = [
        models.ScoredPoint(
            id=1,
            version=1,
            score=0.7,
            payload={"text": "Evidence", "regulator": regulator, "doc_id": "doc", "page": 2},
        )
    ]
    monkeypatch.setattr(bm25, "QdrantClient", Mock(return_value=client))

    result = bm25.retrieve_bm25("question", regulator=regulator)

    assert client.query_points.call_args.kwargs["query_filter"] == models.Filter(
        must=[models.FieldCondition(key="regulator", match=models.MatchValue(value=regulator))]
    )
    assert result[0]["metadata"] == {"doc_id": "doc", "regulator": regulator, "page": 2}
    client.upsert.assert_not_called()


def test_retrieve_bm25_combines_regulator_and_date(monkeypatch):
    client = Mock()
    client.query_points.return_value.points = []
    monkeypatch.setattr(bm25, "QdrantClient", Mock(return_value=client))
    assert (
        bm25.retrieve_bm25(
            "question", regulator="SEBI", start_date="2026-08-24", end_date="2026-09-01"
        )
        == []
    )
    conditions = client.query_points.call_args.kwargs["query_filter"].must
    assert [condition.key for condition in conditions] == ["regulator", "issue_date_iso"]
    assert conditions[0].match.value == "SEBI"
    assert conditions[1].range.lt.isoformat() == "2026-09-02T00:00:00+00:00"


def test_retrieve_bm25_rejects_bad_date_before_client(monkeypatch):
    factory = Mock(side_effect=AssertionError("Invalid date must not contact Qdrant"))
    monkeypatch.setattr(bm25, "QdrantClient", factory)
    with pytest.raises(ValueError, match="end_date"):
        bm25.retrieve_bm25("question", end_date="bad")
    factory.assert_not_called()


@pytest.mark.parametrize("regulator", ["rbi", "OTHER", "", 123, True])
def test_retrieve_bm25_rejects_invalid_regulator_before_client(monkeypatch, regulator):
    factory = Mock(side_effect=AssertionError("Invalid filter must not contact Qdrant"))
    monkeypatch.setattr(bm25, "QdrantClient", factory)

    with pytest.raises(ValueError, match="regulator must be 'RBI' or 'SEBI'"):
        bm25.retrieve_bm25("question", regulator=regulator)

    factory.assert_not_called()


@pytest.mark.parametrize(
    "question, k, error",
    [
        (None, 5, TypeError),
        (" \n\t ", 5, ValueError),
        ("question", True, TypeError),
        ("question", 5.0, TypeError),
        ("question", 0, ValueError),
    ],
)
def test_retrieve_bm25_validates_before_client_creation(monkeypatch, question, k, error):
    factory = Mock(side_effect=AssertionError("Invalid input must not contact Qdrant"))
    monkeypatch.setattr(bm25, "QdrantClient", factory)

    with pytest.raises(error):
        bm25.retrieve_bm25(question, k)

    factory.assert_not_called()
