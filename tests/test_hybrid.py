"""Offline tests for safe hybrid collection setup; no live Qdrant calls."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient, models

from regulatrace import hybrid
from regulatrace.config import Settings


@pytest.fixture(autouse=True)
def hybrid_settings(monkeypatch):
    settings = Settings(
        _env_file=None,
        qdrant_collection="regulatrace_chunks",
        bm25_collection="regulatrace_bm25",
        hybrid_collection="regulatrace_hybrid",
        qdrant_api_key=None,
    )
    monkeypatch.setattr(hybrid, "get_settings", lambda: settings)


def collection_info(vectors, sparse):
    return SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors, sparse_vectors=sparse))
    )


def test_create_collection_with_exact_hybrid_schema():
    client = Mock()
    client.collection_exists.return_value = False

    hybrid.init_hybrid_collection(client=client)

    client.collection_exists.assert_called_once_with("regulatrace_hybrid")
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "regulatrace_hybrid"
    assert set(kwargs["vectors_config"]) == {"dense"}
    assert kwargs["vectors_config"]["dense"].size == 384
    assert kwargs["vectors_config"]["dense"].distance == models.Distance.COSINE
    assert set(kwargs["sparse_vectors_config"]) == {"bm25"}
    assert kwargs["sparse_vectors_config"]["bm25"].modifier == models.Modifier.IDF
    client.get_collection.assert_not_called()
    client.delete_collection.assert_not_called()


def test_existing_compatible_collection_is_unchanged():
    client = QdrantClient(":memory:")

    hybrid.init_hybrid_collection(client=client)
    hybrid.init_hybrid_collection(client=client)

    params = client.get_collection("regulatrace_hybrid").config.params
    assert set(params.vectors) == {"dense"}
    assert params.vectors["dense"].size == 384
    assert params.vectors["dense"].distance == models.Distance.COSINE
    assert set(params.sparse_vectors) == {"bm25"}
    assert params.sparse_vectors["bm25"].modifier == models.Modifier.IDF


@pytest.mark.parametrize(
    ("vectors", "sparse"),
    [
        (
            {"dense": models.VectorParams(size=385, distance=models.Distance.COSINE)},
            {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        ),
        (
            {"dense": models.VectorParams(size=384, distance=models.Distance.DOT)},
            {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        ),
        (
            {"other": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        ),
        (
            {"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            {"bm25": models.SparseVectorParams()},
        ),
        (
            {"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
            {"other": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        ),
    ],
)
def test_incompatible_schema_is_rejected_without_changes(vectors, sparse):
    client = Mock()
    client.collection_exists.return_value = True
    client.get_collection.return_value = collection_info(vectors, sparse)

    with pytest.raises(ValueError, match="incompatible schema"):
        hybrid.init_hybrid_collection(client=client)

    client.create_collection.assert_not_called()
    client.delete_collection.assert_not_called()
    client.upsert.assert_not_called()


@pytest.mark.parametrize("name", ["regulatrace_chunks", "regulatrace_bm25"])
def test_baseline_collection_name_collision_is_rejected_before_qdrant_call(name):
    client = Mock()

    with pytest.raises(ValueError, match="must differ from dense and BM25"):
        hybrid.init_hybrid_collection(client=client, collection_name=name)

    client.collection_exists.assert_not_called()
    client.create_collection.assert_not_called()
    client.delete_collection.assert_not_called()


@pytest.fixture
def backfill_client():
    client = Mock()
    client.cloud_inference = True
    client.collection_exists.return_value = True
    client.get_collection.return_value = collection_info(
        {"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    return client


def source_point(point_id, text="Evidence", vector=None):
    payload = {"text": text, "doc_id": f"document-{point_id}", "page": point_id, "extra": [1, 2]}
    return SimpleNamespace(id=point_id, payload=payload, vector=vector)


def test_backfill_paginates_and_copies_stored_vectors_and_payloads(backfill_client):
    source = [
        source_point(1, "First evidence", {"dense": [0.1] * 384}),
        source_point(2, "Second evidence", {"dense": [0.2] * 384}),
        source_point(3, "Third evidence", {"dense": [0.3] * 384}),
    ]
    backfill_client.scroll.side_effect = [(source[:2], 3), (source[2:], None)]

    assert hybrid.backfill_hybrid(client=backfill_client, batch_size=2) == 3

    assert [call.kwargs["offset"] for call in backfill_client.scroll.call_args_list] == [None, 3]
    for call in backfill_client.scroll.call_args_list:
        assert call.kwargs["collection_name"] == "regulatrace_chunks"
        assert call.kwargs["limit"] == 2
        assert call.kwargs["with_payload"] is True
        assert call.kwargs["with_vectors"] == ["dense"]
    assert backfill_client.upsert.call_count == 2
    written = []
    for call in backfill_client.upsert.call_args_list:
        assert call.kwargs["collection_name"] == "regulatrace_hybrid"
        assert call.kwargs["wait"] is True
        assert len(call.kwargs["points"]) <= 2
        written.extend(call.kwargs["points"])
    assert [point.id for point in written] == [point.id for point in source]
    assert [point.payload for point in written] == [point.payload for point in source]
    for point, original in zip(written, source, strict=True):
        assert set(point.vector) == {"dense", "bm25"}
        assert point.vector["dense"] == original.vector["dense"]
        assert point.vector["bm25"] == models.Document(
            text=original.payload["text"], model="qdrant/bm25"
        )
    backfill_client.create_collection.assert_not_called()
    backfill_client.delete_collection.assert_not_called()
    backfill_client.delete.assert_not_called()


def test_repeating_backfill_uses_the_same_point_id(backfill_client):
    source = source_point(7, vector={"dense": [0.1] * 384})
    backfill_client.scroll.return_value = ([source], None)

    assert hybrid.backfill_hybrid(client=backfill_client) == 1
    assert hybrid.backfill_hybrid(client=backfill_client) == 1

    assert [call.kwargs["points"][0].id for call in backfill_client.upsert.call_args_list] == [
        7,
        7,
    ]
    assert all(
        call.kwargs["collection_name"] == "regulatrace_hybrid"
        for call in backfill_client.upsert.call_args_list
    )


@pytest.mark.parametrize("text", [None, "", " \n ", 123])
def test_backfill_rejects_missing_or_invalid_text_before_batch_write(backfill_client, text):
    backfill_client.scroll.return_value = (
        [
            source_point(1, vector={"dense": [0.1] * 384}),
            source_point(2, text=text, vector={"dense": [0.2] * 384}),
        ],
        None,
    )

    with pytest.raises(ValueError, match="Dense point 2 has missing or empty text"):
        hybrid.backfill_hybrid(client=backfill_client)

    backfill_client.upsert.assert_not_called()


@pytest.mark.parametrize(
    "vector",
    [None, [], {"dense": None}, {"dense": []}, {"dense": [0.1] * 383}],
)
def test_backfill_rejects_missing_or_invalid_dense_vector(backfill_client, vector):
    backfill_client.scroll.return_value = ([source_point(2, vector=vector)], None)

    with pytest.raises(ValueError, match="Dense point 2 has missing or invalid 384-dimensional"):
        hybrid.backfill_hybrid(client=backfill_client)

    backfill_client.upsert.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_backfill_rejects_invalid_batch_size_before_qdrant_call(backfill_client, batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        hybrid.backfill_hybrid(client=backfill_client, batch_size=batch_size)

    backfill_client.collection_exists.assert_not_called()
    backfill_client.scroll.assert_not_called()
    backfill_client.upsert.assert_not_called()


@pytest.mark.parametrize(
    ("dense_name", "bm25_name", "hybrid_name"),
    [
        ("regulatrace_chunks", "regulatrace_bm25", "regulatrace_chunks"),
        ("regulatrace_chunks", "regulatrace_bm25", "regulatrace_bm25"),
        ("same", "same", "regulatrace_hybrid"),
    ],
)
def test_backfill_rejects_collection_name_collisions_before_qdrant_call(
    monkeypatch, backfill_client, dense_name, bm25_name, hybrid_name
):
    settings = Settings(
        _env_file=None,
        qdrant_collection=dense_name,
        bm25_collection=bm25_name,
        hybrid_collection=hybrid_name,
    )
    monkeypatch.setattr(hybrid, "get_settings", lambda: settings)

    with pytest.raises(ValueError, match="must have different names"):
        hybrid.backfill_hybrid(client=backfill_client)

    backfill_client.collection_exists.assert_not_called()
    backfill_client.scroll.assert_not_called()
    backfill_client.upsert.assert_not_called()


def test_backfill_requires_inference_and_existing_compatible_target(backfill_client):
    backfill_client.cloud_inference = False
    with pytest.raises(ValueError, match="requires server-side inference"):
        hybrid.backfill_hybrid(client=backfill_client)
    backfill_client.scroll.assert_not_called()

    backfill_client.cloud_inference = True
    backfill_client.collection_exists.return_value = False
    with pytest.raises(ValueError, match="must already exist"):
        hybrid.backfill_hybrid(client=backfill_client)
    backfill_client.create_collection.assert_not_called()

    backfill_client.collection_exists.return_value = True
    backfill_client.get_collection.return_value = collection_info(
        {"dense": models.VectorParams(size=383, distance=models.Distance.COSINE)},
        {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    with pytest.raises(ValueError, match="incompatible schema"):
        hybrid.backfill_hybrid(client=backfill_client)
    backfill_client.scroll.assert_not_called()
    backfill_client.upsert.assert_not_called()


def test_backfill_default_client_enables_server_inference(monkeypatch, backfill_client):
    settings = Settings(_env_file=None, qdrant_api_key=None)
    monkeypatch.setattr(hybrid, "get_settings", lambda: settings)
    factory = Mock(return_value=backfill_client)
    monkeypatch.setattr(hybrid, "QdrantClient", factory)
    backfill_client.scroll.return_value = ([], None)

    assert hybrid.backfill_hybrid() == 0

    factory.assert_called_once_with(url="http://localhost:6333", api_key=None, cloud_inference=True)
    backfill_client.upsert.assert_not_called()


@pytest.fixture
def retrieval_dependencies(monkeypatch):
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(hybrid, "QdrantClient", factory)
    embedder = Mock()
    embedding = Mock()
    embedding.tolist.return_value = [0.1] * 384
    embedder.encode.return_value = embedding
    embedding_factory = Mock(return_value=embedder)
    monkeypatch.setattr(hybrid, "get_embeddings", embedding_factory)
    return client, factory, embedder, embedding_factory


@pytest.mark.parametrize(("k", "candidate_k"), [(5, 5), (2, 7)])
def test_retrieve_hybrid_uses_native_rrf_and_existing_embedding_method(
    retrieval_dependencies, k, candidate_k
):
    client, client_factory, embedder, embedding_factory = retrieval_dependencies
    client.query_points.return_value.points = [
        models.ScoredPoint(
            id=2,
            version=1,
            score=0.031,
            payload={
                "text": "First fused chunk",
                "source_file": "first.pdf",
                "doc_id": "first-document",
                "page": 3,
                "section": "Reporting",
                "chunk_index": 1,
                "extra": "not canonical metadata",
            },
        ),
        models.ScoredPoint(id=1, version=1, score=0.02, payload={"text": "Second fused chunk"}),
    ]
    question = "What is the reporting deadline?"

    results = hybrid.retrieve_hybrid(question, k=k, candidate_k=candidate_k)

    embedding_factory.assert_called_once_with()
    embedder.encode.assert_called_once_with(question, normalize_embeddings=True)
    client_factory.assert_called_once_with(
        url="http://localhost:6333", api_key=None, cloud_inference=True
    )
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "regulatrace_hybrid"
    assert kwargs["limit"] == k
    assert kwargs["with_payload"] is True
    assert kwargs["prefetch"] == [
        models.Prefetch(query=[0.1] * 384, using="dense", limit=candidate_k),
        models.Prefetch(
            query=models.Document(text=question, model="qdrant/bm25"),
            using="bm25",
            limit=candidate_k,
        ),
    ]
    assert kwargs["query"] == models.RrfQuery(rrf=models.Rrf(k=61))
    assert results == [
        {
            "text": "First fused chunk",
            "score": 0.031,
            "source_file": "first.pdf",
            "page": 3,
            "metadata": {
                "doc_id": "first-document",
                "page": 3,
                "section": "Reporting",
                "chunk_index": 1,
            },
        },
        {
            "text": "Second fused chunk",
            "score": 0.02,
            "source_file": None,
            "page": None,
            "metadata": {},
        },
    ]
    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()
    client.delete.assert_not_called()


def test_retrieve_hybrid_empty_results(retrieval_dependencies):
    client, _, _, _ = retrieval_dependencies
    client.query_points.return_value.points = []

    assert hybrid.retrieve_hybrid("Question") == []
    assert client.query_points.call_args.kwargs["limit"] == 5
    client.upsert.assert_not_called()


@pytest.mark.parametrize("regulator", ["RBI", "SEBI"])
def test_retrieve_hybrid_prefilters_both_branches(retrieval_dependencies, regulator):
    client, _, _, _ = retrieval_dependencies
    client.query_points.return_value.points = [
        models.ScoredPoint(
            id=1,
            version=1,
            score=0.03,
            payload={"text": "Evidence", "regulator": regulator, "doc_id": "doc", "page": 2},
        )
    ]

    result = hybrid.retrieve_hybrid("question", regulator=regulator)

    prefetch = client.query_points.call_args.kwargs["prefetch"]
    expected = models.Filter(
        must=[models.FieldCondition(key="regulator", match=models.MatchValue(value=regulator))]
    )
    assert len(prefetch) == 2
    assert prefetch[0].filter == expected
    assert prefetch[1].filter == expected
    assert prefetch[0].filter is prefetch[1].filter
    assert result[0]["metadata"] == {"doc_id": "doc", "regulator": regulator, "page": 2}
    client.upsert.assert_not_called()


def test_retrieve_hybrid_combines_regulator_and_date_in_both_branches(
    retrieval_dependencies,
):
    client, _, _, _ = retrieval_dependencies
    client.query_points.return_value.points = []
    hybrid.retrieve_hybrid("question", regulator="RBI", start_date="2026-08-24")
    dense, sparse = client.query_points.call_args.kwargs["prefetch"]
    assert dense.filter is sparse.filter
    assert [condition.key for condition in dense.filter.must] == [
        "regulator",
        "issue_date_iso",
    ]
    assert dense.filter.must[1].range.gte.isoformat() == "2026-08-24T00:00:00+00:00"


def test_retrieve_hybrid_rejects_bad_date_before_embedding(retrieval_dependencies):
    client, factory, _, embeddings = retrieval_dependencies
    with pytest.raises(ValueError, match="start_date must not be after end_date"):
        hybrid.retrieve_hybrid("question", start_date="2026-09-01", end_date="2026-08-31")
    embeddings.assert_not_called()
    factory.assert_not_called()
    client.query_points.assert_not_called()


@pytest.mark.parametrize("regulator", ["rbi", "OTHER", "", 123, True])
def test_retrieve_hybrid_rejects_invalid_regulator_before_work(retrieval_dependencies, regulator):
    client, client_factory, _, embedding_factory = retrieval_dependencies

    with pytest.raises(ValueError, match="regulator must be 'RBI' or 'SEBI'"):
        hybrid.retrieve_hybrid("question", regulator=regulator)

    embedding_factory.assert_not_called()
    client_factory.assert_not_called()
    client.query_points.assert_not_called()


@pytest.mark.parametrize(
    ("question", "k", "candidate_k", "error", "message"),
    [
        (None, 5, 5, TypeError, "query must be a string"),
        (" \n ", 5, 5, ValueError, "query must not be empty"),
        ("Question", True, 5, TypeError, "k must be an integer"),
        ("Question", 0, 5, ValueError, "k must be greater than zero"),
        ("Question", 5, True, TypeError, "candidate_k must be an integer"),
        ("Question", 5, 5.0, TypeError, "candidate_k must be an integer"),
        ("Question", 5, 0, ValueError, "candidate_k must be greater than zero"),
        ("Question", 5, -1, ValueError, "candidate_k must be greater than zero"),
        ("Question", 5, 4, ValueError, "candidate_k must be at least k"),
    ],
)
def test_retrieve_hybrid_rejects_invalid_inputs_before_work(
    retrieval_dependencies, question, k, candidate_k, error, message
):
    client, client_factory, _, embedding_factory = retrieval_dependencies

    with pytest.raises(error, match=message):
        hybrid.retrieve_hybrid(question, k=k, candidate_k=candidate_k)

    embedding_factory.assert_not_called()
    client_factory.assert_not_called()
    client.query_points.assert_not_called()
