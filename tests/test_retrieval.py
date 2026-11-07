"""Deterministic tests of the dense retrieval wrapper and terminal output."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from qdrant_client import models
from qdrant_client.models import ScoredPoint

from regulatrace import retrieval, vectorstore
from regulatrace.ingestion.metadata import ChunkMetadata


@pytest.fixture
def search(monkeypatch):
    mock = Mock(return_value=[])
    monkeypatch.setattr(retrieval, "search_chunks", mock)
    return mock


@pytest.mark.parametrize("query", [None, 123, True, ["question"]])
def test_non_string_query(query, search):
    with pytest.raises(TypeError, match="query must be a string"):
        retrieval.retrieve(query)
    search.assert_not_called()


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_blank_query(query, search):
    with pytest.raises(ValueError, match="query must not be empty"):
        retrieval.retrieve(query)
    search.assert_not_called()


@pytest.mark.parametrize("k", [None, "5", 5.0, True, False])
def test_non_integer_k(k, search):
    with pytest.raises(TypeError, match="k must be an integer"):
        retrieval.retrieve("question", k)
    search.assert_not_called()


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k(k, search):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        retrieval.retrieve("question", k)
    search.assert_not_called()


@pytest.mark.parametrize("k", [1, 3, 5])
def test_delegation_and_empty_results(k, search):
    assert retrieval.retrieve(" question ", k) == []
    search.assert_called_once_with(query=" question ", limit=k)


def test_default_k(search):
    assert retrieval.retrieve("question") == []
    search.assert_called_once_with(query="question", limit=5)


@pytest.mark.parametrize("regulator", ["RBI", "SEBI"])
def test_regulator_filter_and_dense_query(regulator, search):
    query_filter = vectorstore.build_regulator_filter(regulator)
    assert query_filter == models.Filter(
        must=[models.FieldCondition(key="regulator", match=models.MatchValue(value=regulator))]
    )
    assert vectorstore.build_regulator_filter(None) is None

    search.return_value = [
        ScoredPoint(
            id=1,
            version=1,
            score=0.8,
            payload={"text": "Evidence", "regulator": regulator, "doc_id": "doc", "page": 2},
        )
    ]
    result = retrieval.retrieve("question", regulator=regulator)
    search.assert_called_once_with(query="question", limit=5, query_filter=query_filter)
    assert result[0]["metadata"] == {"doc_id": "doc", "regulator": regulator, "page": 2}

    client = Mock()
    client.query_points.return_value.points = []
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [0.1, 0.2]
    assert (
        vectorstore.search_chunks(
            "question",
            client=client,
            collection_name="test",
            embedder=embedder,
            query_filter=query_filter,
        )
        == []
    )
    assert client.query_points.call_args.kwargs == {
        "collection_name": "test",
        "query": [0.1, 0.2],
        "using": "dense",
        "limit": 5,
        "query_filter": query_filter,
    }
    client.upsert.assert_not_called()


@pytest.mark.parametrize("regulator", ["rbi", "OTHER", "", 123, True])
def test_invalid_regulator_rejected_before_search(regulator, search):
    with pytest.raises(ValueError, match="regulator must be 'RBI' or 'SEBI'"):
        retrieval.retrieve("question", regulator=regulator)
    search.assert_not_called()


def test_combined_regulator_and_inclusive_dates(search):
    retrieval.retrieve("question", regulator="RBI", start_date="2026-08-24", end_date="2026-08-24")
    query_filter = search.call_args.kwargs["query_filter"]
    assert query_filter == models.Filter(
        must=[
            models.FieldCondition(key="regulator", match=models.MatchValue(value="RBI")),
            models.FieldCondition(
                key="issue_date_iso",
                range=models.DatetimeRange(
                    gte=datetime(2026, 8, 24, tzinfo=timezone.utc),
                    lt=datetime(2026, 8, 25, tzinfo=timezone.utc),
                ),
            ),
        ]
    )


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [
        ("2026-02-30", None),
        ("2026-2-03", None),
        (None, "2026-13-01"),
        ("2026-09-01", "2026-08-31"),
    ],
)
def test_invalid_date_filter_rejected_before_search(search, start_date, end_date):
    with pytest.raises(ValueError):
        retrieval.retrieve("question", start_date=start_date, end_date=end_date)
    search.assert_not_called()


def test_date_only_filter_and_unfiltered_compatibility(search):
    retrieval.retrieve("question", end_date="2026-10-02")
    condition = search.call_args.kwargs["query_filter"].must[0]
    assert condition.key == "issue_date_iso"
    assert condition.range.lt == datetime(2026, 10, 3, tzinfo=timezone.utc)
    search.reset_mock()
    retrieval.retrieve("question")
    search.assert_called_once_with(query="question", limit=5)


def test_index_setup_reuses_existing_and_rejects_incompatible():
    client = Mock()
    client.collection_exists.return_value = True
    schema = {
        "regulator": models.PayloadIndexInfo(data_type=models.PayloadSchemaType.KEYWORD, points=669)
    }
    client.get_collection.return_value.payload_schema = schema
    client.create_payload_index.return_value.status = models.UpdateStatus.COMPLETED
    vectorstore.ensure_filter_indexes(client)
    assert client.create_payload_index.call_count == 3
    for entry in client.create_payload_index.call_args_list:
        assert entry.kwargs["field_name"] == "issue_date_iso"
        assert entry.kwargs["field_schema"] == models.PayloadSchemaType.DATETIME
        assert entry.kwargs["wait"] is True
    client.create_payload_index.reset_mock()
    schema["issue_date_iso"] = models.PayloadIndexInfo(
        data_type=models.PayloadSchemaType.DATETIME, points=669
    )
    vectorstore.ensure_filter_indexes(client)
    client.create_payload_index.assert_not_called()
    schema["issue_date_iso"] = models.PayloadIndexInfo(
        data_type=models.PayloadSchemaType.KEYWORD, points=669
    )
    with pytest.raises(ValueError, match="incompatible"):
        vectorstore.ensure_filter_indexes(client)
    client.create_payload_index.assert_not_called()


def test_result_mapping_preserves_order_scores_and_metadata(search):
    metadata = {
        "doc_id": "document-id",
        "title": "Test circular",
        "regulator": "RBI",
        "circular_no": "RBI/test",
        "issue_date": "Oct 01, 2026",
        "issue_date_iso": "2026-10-01T00:00:00Z",
        "effective_date": None,
        "status": None,
        "supersedes": ["previous-circular"],
        "section": "1. Scope",
        "page": 2,
        "access_level": "public",
        "source_url": "https://example.com/circular",
        "content_hash": "chunk-hash",
        "chunk_index": 1,
    }
    assert set(metadata) == set(ChunkMetadata.model_fields)
    payload = {"text": "Actual chunk text", "source_file": "test.pdf", **metadata, "extra": 42}
    # Deliberately use ascending scores to detect any wrapper re-sorting.
    search.return_value = [
        ScoredPoint(id=2, version=1, score=0.123456789, payload=payload),
        ScoredPoint(id=1, version=1, score=0.9, payload={"text": "Second chunk"}),
    ]
    results = retrieval.retrieve("question", 2)
    assert results[0] == {
        "text": "Actual chunk text",
        "score": 0.123456789,
        "source_file": "test.pdf",
        "page": 2,
        "metadata": metadata,
    }
    assert [result["text"] for result in results] == ["Actual chunk text", "Second chunk"]
    assert [result["score"] for result in results] == [0.123456789, 0.9]
    assert payload["extra"] == 42
    assert payload["effective_date"] is None


@pytest.mark.parametrize("payload", [None, {}, {"text": "Body", "status": None}])
def test_missing_metadata(payload, search):
    search.return_value = [ScoredPoint(id=1, version=1, score=0.2, payload=payload)]
    result = retrieval.retrieve("question")[0]
    assert result["source_file"] is None
    assert result["page"] is None
    assert result["text"] == (payload or {}).get("text")
    assert result["metadata"] == ({"status": None} if payload else {})
    assert "access_level" not in result["metadata"]


def test_terminal_output(search, capsys):
    search.return_value = [
        ScoredPoint(
            id=1,
            version=1,
            score=0.7654321,
            payload={
                "text": "First line\nSecond line",
                "source_file": "circular.pdf",
                "page": 3,
                "title": "Circular title",
                "regulator": "SEBI",
                "status": None,
            },
        ),
        ScoredPoint(id=2, version=1, score=0.1, payload=None),
    ]
    retrieval.print_results(retrieval.retrieve("question"))
    output = capsys.readouterr().out
    for expected in (
        "Rank 1 | Similarity score: 0.765432",
        "Source document: circular.pdf",
        "Page: 3",
        "title: Circular title",
        "regulator: SEBI",
        "status: unavailable",
        "First line\nSecond line",
        "Rank 2",
        "Source document: unavailable",
        "Page: unavailable",
    ):
        assert expected in output
    assert output.index("Rank 1") < output.index("Rank 2")


def test_empty_terminal_output(capsys):
    retrieval.print_results([])
    assert capsys.readouterr().out == "No results.\n"


def test_cli_arguments(search, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["retrieval", "--query", "question", "--k", "3"])
    retrieval.main()
    search.assert_called_once_with(query="question", limit=3)
    assert "No results." in capsys.readouterr().out


def test_cli_search_error(search, monkeypatch, capsys):
    search.side_effect = RuntimeError("Search unavailable")
    monkeypatch.setattr("sys.argv", ["retrieval", "--query", "question"])
    with pytest.raises(SystemExit) as exc:
        retrieval.main()
    assert exc.value.code == 1
    assert "Retrieval error: Search unavailable" in capsys.readouterr().err
