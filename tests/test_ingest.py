"""Integration and unit tests for Step 1.6 idempotent ingestion pipeline and document registry."""

import contextlib
import uuid
from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import delete, select

from regulatrace.db.models import Document
from regulatrace.db.session import get_engine, get_session_factory
from regulatrace.ingest import IngestStats, ingest_documents, print_summary
from regulatrace.ingestion.metadata import compute_file_hash, generate_doc_id
from regulatrace.vectorstore import generate_point_id, get_qdrant_client


def create_sample_pdf(path: Path, heading: str, body: str) -> None:
    """Helper to generate a minimal valid PDF with regulatory structure."""
    doc = pymupdf.open()
    page = doc.new_page()
    text = f"{heading}\n\n{body}"
    page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def db_session():
    """Provide a clean database session and cleanup test records after execution."""
    engine = get_engine()
    factory = get_session_factory(engine)
    session = factory()
    test_files: list[str] = []
    try:
        yield session, test_files
    finally:
        if test_files:
            session.execute(delete(Document).where(Document.source_file.in_(test_files)))
            session.commit()
        session.close()


@pytest.fixture
def qdrant_test_collection():
    """Provide an isolated Qdrant collection name and delete it after test."""
    client = get_qdrant_client()
    collection_name = f"test_ingest_{uuid.uuid4().hex[:8]}"
    try:
        yield client, collection_name
    finally:
        with contextlib.suppress(Exception):
            client.delete_collection(collection_name)


def test_compute_file_hash(tmp_path: Path):
    """Verify SHA-256 hash computation is deterministic and detects file modifications."""
    file1 = tmp_path / "test1.pdf"
    file2 = tmp_path / "test2.pdf"

    create_sample_pdf(file1, "1. Applicability", "These rules apply to all commercial banks.")
    file2.write_bytes(file1.read_bytes())

    hash1 = compute_file_hash(file1)
    hash1_again = compute_file_hash(file1)
    hash2 = compute_file_hash(file2)

    assert isinstance(hash1, str)
    assert len(hash1) == 64
    assert hash1 == hash1_again
    assert hash1 == hash2

    # Modify file1 content
    file1.write_bytes(b"Modified binary payload content")
    hash1_mod = compute_file_hash(file1)
    assert hash1 != hash1_mod


def test_document_registry_model(db_session):
    """Verify Document model CRUD and schema constraints in PostgreSQL."""
    session, test_files = db_session
    filename = f"reg_test_{uuid.uuid4().hex[:8]}.pdf"
    test_files.append(filename)

    doc = Document(
        source_file=filename,
        content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        status="active",
        version=1,
        chunk_count=5,
    )
    session.add(doc)
    session.commit()

    retrieved = session.scalars(select(Document).where(Document.source_file == filename)).first()
    assert retrieved is not None
    assert retrieved.source_file == filename
    assert retrieved.version == 1
    assert retrieved.status == "active"
    assert retrieved.chunk_count == 5
    assert retrieved.ingested_at is not None
    assert isinstance(retrieved.id, uuid.UUID)


def test_test_a_ingest_idempotency_run_twice(tmp_path: Path, db_session, qdrant_test_collection):
    """Test A: Run ingestion twice on 3 sample documents.

    Assert second run:
    - Scanned = 3
    - Ingested = 0
    - Skipped = 3
    - Vector count unchanged
    """
    session, test_files = db_session
    q_client, coll_name = qdrant_test_collection

    prefix = uuid.uuid4().hex[:6]
    f1 = tmp_path / f"{prefix}_doc1.pdf"
    f2 = tmp_path / f"{prefix}_doc2.pdf"
    f3 = tmp_path / f"{prefix}_doc3.pdf"

    test_files.extend([f1.name, f2.name, f3.name])

    create_sample_pdf(f1, "1. Applicability", "These rules apply to all commercial banks in India.")
    create_sample_pdf(f2, "2. Capital Adequacy", "Banks must maintain Tier 1 capital ratio of 7%.")
    create_sample_pdf(f3, "3. Liquidity Ratio", "Banks must maintain Liquidity Coverage Ratio.")

    # Run 1: First ingestion
    stats_1 = ingest_documents(
        path=tmp_path,
        session=session,
        qdrant_client=q_client,
        collection_name=coll_name,
    )

    assert stats_1.documents_scanned == 3
    assert stats_1.new_documents == 3
    assert stats_1.updated_documents == 0
    assert stats_1.skipped_documents == 0
    assert stats_1.chunks_indexed == 3

    count_1 = q_client.count(coll_name).count
    assert count_1 == 3

    # Run 2: Re-ingest exact same folder without modifications
    stats_2 = ingest_documents(
        path=tmp_path,
        session=session,
        qdrant_client=q_client,
        collection_name=coll_name,
    )

    assert stats_2.documents_scanned == 3
    assert stats_2.new_documents == 0
    assert stats_2.updated_documents == 0
    assert stats_2.skipped_documents == 3
    assert stats_2.chunks_indexed == 0

    count_2 = q_client.count(coll_name).count
    assert count_2 == count_1 == 3


def test_test_b_modify_one_document_and_reingest(
    tmp_path: Path, db_session, qdrant_test_collection
):
    """Test B: Modify 1 document, re-run ingestion.

    Assert:
    - Scanned = 3
    - Updated = 1
    - Skipped = 2
    - Vector count reflects modified document's chunk count
    - Database version incremented to 2
    """
    session, test_files = db_session
    q_client, coll_name = qdrant_test_collection

    prefix = uuid.uuid4().hex[:6]
    f1 = tmp_path / f"{prefix}_doc1.pdf"
    f2 = tmp_path / f"{prefix}_doc2.pdf"
    f3 = tmp_path / f"{prefix}_doc3.pdf"

    test_files.extend([f1.name, f2.name, f3.name])

    create_sample_pdf(f1, "1. Applicability", "These rules apply to all commercial banks in India.")
    create_sample_pdf(f2, "2. Capital Adequacy", "Banks must maintain Tier 1 capital ratio of 7%.")
    create_sample_pdf(f3, "3. Liquidity Ratio", "Banks must maintain Liquidity Coverage Ratio.")

    # Initial ingestion
    stats_init = ingest_documents(
        path=tmp_path,
        session=session,
        qdrant_client=q_client,
        collection_name=coll_name,
    )
    assert stats_init.new_documents == 3
    assert stats_init.chunks_indexed == 3

    # Modify f2 to contain multiple sections resulting in 2 chunks
    doc2_modified_text = (
        "2. Capital Adequacy\n\n"
        "Updated Tier 1 capital requirement is now 8 percent.\n\n"
        "2.1 Additional Tier 1 Capital\n\n"
        "Additional capital requirements must follow prescribed norms."
    )
    create_sample_pdf(f2, "2. Capital Adequacy", doc2_modified_text)

    # Re-run ingestion
    stats_updated = ingest_documents(
        path=tmp_path,
        session=session,
        qdrant_client=q_client,
        collection_name=coll_name,
    )

    assert stats_updated.documents_scanned == 3
    assert stats_updated.new_documents == 0
    assert stats_updated.updated_documents == 1
    assert stats_updated.skipped_documents == 2
    assert stats_updated.chunks_indexed == 2

    # Vector store count: doc1 (1 chunk) + doc3 (1 chunk) + doc2 (2 chunks) = 4 chunks
    count_after = q_client.count(coll_name).count
    assert count_after == 4

    # Check database record for f2
    doc_record = session.scalars(select(Document).where(Document.source_file == f2.name)).first()
    assert doc_record is not None
    assert doc_record.version == 2
    assert doc_record.chunk_count == 2


def test_test_c_deterministic_chunk_ids():
    """Test C: Chunk IDs are deterministic for identical doc_id and 1-based index."""
    doc_id = generate_doc_id(
        source_file="sample_rbi_circular.pdf",
        regulator="RBI",
        circular_no="RBI/2024/01",
    )
    id_run1_chunk1 = generate_point_id(doc_id=doc_id, chunk_index=1)
    id_run1_chunk2 = generate_point_id(doc_id=doc_id, chunk_index=2)

    # Re-generate
    id_run2_chunk1 = generate_point_id(doc_id=doc_id, chunk_index=1)
    id_run2_chunk2 = generate_point_id(doc_id=doc_id, chunk_index=2)

    assert id_run1_chunk1 == id_run2_chunk1
    assert id_run1_chunk2 == id_run2_chunk2
    assert id_run1_chunk1 != id_run1_chunk2


def test_print_summary(capsys):
    """Verify print_summary outputs the exact expected format."""
    stats = IngestStats(
        documents_scanned=10,
        new_documents=3,
        updated_documents=2,
        skipped_documents=5,
        chunks_indexed=42,
    )
    print_summary(stats)
    captured = capsys.readouterr().out
    assert "Documents scanned: 10" in captured
    assert "New documents: 3" in captured
    assert "Updated documents: 2" in captured
    assert "Skipped documents: 5" in captured
    assert "Chunks indexed: 42" in captured


def test_ingest_missing_directory_raises_error():
    """Verify ingest_documents raises FileNotFoundError on missing directory."""
    with pytest.raises(FileNotFoundError, match="Ingestion directory not found"):
        ingest_documents(path="non_existent_directory_path_12345")
