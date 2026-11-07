"""Focused unit tests for Qdrant vector store integration and batch embedding."""

import contextlib
import uuid

import pytest

from regulatrace.ingestion.chunker import chunk_text
from regulatrace.llm import get_embeddings
from regulatrace.vectorstore import (
    embed_chunks,
    generate_point_id,
    get_embedding_dimension,
    get_qdrant_client,
    init_collection,
    search_chunks,
    upsert_chunks,
)


def test_generate_point_id_deterministic():
    """Verify generate_point_id returns identical UUIDv5 for same doc_id and chunk_index."""
    id1 = generate_point_id("doc_123_hash", 1)
    id2 = generate_point_id("doc_123_hash", 1)
    id3 = generate_point_id("doc_123_hash", 2)
    id4 = generate_point_id("doc_456_hash", 1)

    assert id1 == id2
    assert id1 != id3
    assert id1 != id4
    assert len(id1) == 36  # Standard UUID string representation


def test_get_embedding_dimension():
    """Verify vector dimension is retrieved dynamically from the embedding model."""
    embedder = get_embeddings()
    dim = get_embedding_dimension(embedder)
    expected_dim = embedder.get_embedding_dimension()
    assert dim == expected_dim
    assert isinstance(dim, int)
    assert dim > 0


def test_embed_chunks_batching_and_normalization():
    """Verify batch embedding produces unit-normalized vectors with matching count."""
    embedder = get_embeddings()
    expected_dim = get_embedding_dimension(embedder)

    text = (
        "1. Capital Adequacy\n"
        "Banks must maintain a minimum Tier 1 capital ratio of 7 percent.\n\n"
        "2. Liquidity Ratio\n"
        "Banks must maintain a minimum Liquidity Coverage Ratio of 100 percent."
    )
    chunks = chunk_text(text, source_file="test_doc.pdf")
    assert len(chunks) == 2

    vectors = embed_chunks(chunks, embedder=embedder, batch_size=2)
    assert len(vectors) == 2
    for vec in vectors:
        assert len(vec) == expected_dim
        norm_sq = sum(x * x for x in vec)
        assert pytest.approx(norm_sq, rel=1e-3) == 1.0


def test_qdrant_collection_init_and_payload_indexes():
    """Verify collection is created with dense/sparse named vectors and 4 payload indexes."""
    embedder = get_embeddings()
    expected_dim = get_embedding_dimension(embedder)

    client = get_qdrant_client()
    coll_name = f"test_init_{uuid.uuid4().hex[:8]}"

    try:
        init_collection(client=client, collection_name=coll_name, embedder=embedder)
        assert client.collection_exists(coll_name)

        coll_info = client.get_collection(coll_name)
        # Verify named dense vector config matches dynamic model dimension
        assert "dense" in coll_info.config.params.vectors
        assert coll_info.config.params.vectors["dense"].size == expected_dim
        # Verify named sparse vector config
        assert coll_info.config.params.sparse_vectors is not None
        assert "sparse" in coll_info.config.params.sparse_vectors

        # Verify the 4 payload schema indexes exist
        payload_schema = coll_info.payload_schema
        assert "regulator" in payload_schema
        assert "issue_date" in payload_schema
        assert "status" in payload_schema
        assert "access_level" in payload_schema
    finally:
        with contextlib.suppress(Exception):
            client.delete_collection(coll_name)


def test_qdrant_upsert_deduplication_and_query():
    """Verify upserting chunks is idempotent and semantic search returns relevant chunk."""
    client = get_qdrant_client()
    coll_name = f"test_upsert_{uuid.uuid4().hex[:8]}"

    try:
        text1 = (
            "1. Introduction\n"
            "This Master Direction lays down rules for Note Sorting Machines in bank branches."
        )
        text2 = (
            "2. Price Bands for ETFs\n"
            "The Exchange shall implement dynamic price bands for Exchange Traded Funds."
        )
        chunks = chunk_text(
            text1, source_file="473MDD1229693F0604B6D996B1DF75C81E466.PDF", start_chunk_id=1
        ) + chunk_text(text2, source_file="1781524158363.pdf", page_number=1, start_chunk_id=2)
        assert len(chunks) == 2

        # First upsert run
        upserted_1 = upsert_chunks(chunks, client=client, collection_name=coll_name, batch_size=2)
        assert upserted_1 == 2

        count_1 = client.count(coll_name).count
        assert count_1 == 2

        # Second upsert run with identical chunks (idempotent overwrite)
        upserted_2 = upsert_chunks(chunks, client=client, collection_name=coll_name, batch_size=2)
        assert upserted_2 == 2

        count_2 = client.count(coll_name).count
        assert count_2 == 2  # No duplicates created!

        # Semantic query search
        results = search_chunks(
            query="What are the rules for note sorting machines?",
            client=client,
            collection_name=coll_name,
            limit=2,
        )
        assert len(results) > 0
        top_hit = results[0]
        assert "Note Sorting Machines" in top_hit.payload["text"]
        assert top_hit.payload["regulator"] == "RBI"
    finally:
        with contextlib.suppress(Exception):
            client.delete_collection(coll_name)
