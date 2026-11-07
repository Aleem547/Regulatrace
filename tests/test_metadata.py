"""Comprehensive unit tests for chunk metadata schema and helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from qdrant_client import models

from regulatrace import issue_date_backfill, vectorstore
from regulatrace.config import Settings
from regulatrace.ingestion.chunker import Chunk, chunk_pages, chunk_text
from regulatrace.ingestion.metadata import (
    ChunkMetadata,
    compute_content_hash,
    generate_doc_id,
    get_document_metadata,
    load_manifest,
    normalize_issue_date,
)
from regulatrace.ingestion.parser import ParsedPage


@pytest.fixture
def date_update_client(monkeypatch):
    settings = Settings(_env_file=None, qdrant_api_key=None)
    monkeypatch.setattr(issue_date_backfill, "get_settings", lambda: settings)
    names = (settings.qdrant_collection, settings.bm25_collection, settings.hybrid_collection)
    points = {
        name: [
            SimpleNamespace(
                id=index,
                payload={"issue_date": "Aug 24, 2026", "text": "Evidence", "doc_id": "doc"},
            )
            for index in range(669)
        ]
        for name in names
    }
    client = Mock()
    client.collection_exists.return_value = True
    client.set_payload.return_value.status = models.UpdateStatus.COMPLETED

    def scroll(*, collection_name, limit, offset, **kwargs):
        start = offset or 0
        page = points[collection_name][start : start + limit]
        next_offset = start + limit if start + limit < len(points[collection_name]) else None
        return page, next_offset

    client.scroll.side_effect = scroll
    return client, points


def test_issue_date_dry_run_validates_all_pages_without_writes(date_update_client):
    client, points = date_update_client
    assert issue_date_backfill.update_issue_dates(client) == {name: 669 for name in points}
    assert client.scroll.call_count == 21
    client.set_payload.assert_not_called()
    client.upsert.assert_not_called()


@pytest.mark.parametrize("bad_date", [None, "Aug 32, 2026"])
def test_issue_date_update_rejects_invalid_date_before_writes(date_update_client, bad_date):
    client, points = date_update_client
    points["regulatrace_hybrid"][668].payload["issue_date"] = bad_date
    with pytest.raises(ValueError, match="Invalid issue_date"):
        issue_date_backfill.update_issue_dates(client, apply=True, confirmed=True)
    client.set_payload.assert_not_called()


def test_issue_date_update_rejects_mismatched_payload_and_ids(date_update_client):
    client, points = date_update_client
    points["regulatrace_bm25"][10].payload["text"] = "Different"
    with pytest.raises(ValueError, match="Payload or date differs"):
        issue_date_backfill.update_issue_dates(client, apply=True, confirmed=True)
    client.set_payload.assert_not_called()
    points["regulatrace_bm25"][10].payload["text"] = "Evidence"
    points["regulatrace_bm25"][10].id = 900
    with pytest.raises(ValueError, match="Point IDs differ"):
        issue_date_backfill.update_issue_dates(client, apply=True, confirmed=True)
    client.set_payload.assert_not_called()


def test_issue_date_apply_updates_only_missing_dates_and_is_retry_safe(date_update_client):
    client, points = date_update_client
    points["regulatrace_chunks"][0].payload["issue_date_iso"] = "2026-08-24T00:00:00Z"
    original_payload = points["regulatrace_chunks"][1].payload.copy()
    with pytest.raises(ValueError, match="explicit confirmation"):
        issue_date_backfill.update_issue_dates(client, apply=True)
    client.scroll.assert_not_called()
    pending = issue_date_backfill.update_issue_dates(client, apply=True, confirmed=True)
    assert pending["regulatrace_chunks"] == 668
    assert client.set_payload.call_count == 21
    for entry in client.set_payload.call_args_list:
        assert entry.kwargs["payload"] == {"issue_date_iso": "2026-08-24T00:00:00Z"}
        assert entry.kwargs["wait"] is True
        assert entry.kwargs["collection_name"] in points
    assert points["regulatrace_chunks"][1].payload == original_payload
    assert all(
        0 not in entry.kwargs["points"].points
        for entry in client.set_payload.call_args_list
        if entry.kwargs["collection_name"] == "regulatrace_chunks"
    )
    client.upsert.assert_not_called()


def test_issue_date_update_rejects_conflicting_existing_value(date_update_client):
    client, points = date_update_client
    points["regulatrace_hybrid"][0].payload["issue_date_iso"] = "2026-08-25T00:00:00Z"
    with pytest.raises(ValueError, match="Conflicting issue_date_iso"):
        issue_date_backfill.update_issue_dates(client, apply=True, confirmed=True)
    client.set_payload.assert_not_called()


def test_issue_date_update_fails_on_uncompleted_write(date_update_client):
    client, _ = date_update_client
    client.set_payload.return_value.status = models.UpdateStatus.ACKNOWLEDGED
    with pytest.raises(RuntimeError, match="did not complete"):
        issue_date_backfill.update_issue_dates(client, apply=True, confirmed=True)


def test_chunk_metadata_model_creation():
    """Verify ChunkMetadata instantiates successfully with valid required and optional fields."""
    meta = ChunkMetadata(
        doc_id="abc123hash",
        title="Master Direction on Note Sorting Machines",
        regulator="RBI",
        circular_no="RBI/DCM/2026-27/473",
        issue_date="Oct 02, 2026",
        issue_date_iso="2026-10-02T00:00:00Z",
        effective_date=None,
        status="active",
        supersedes=None,
        section="1. Introduction",
        page=1,
        access_level="public",
        source_url="https://www.rbi.org.in/sample",
        content_hash="hash456",
        chunk_index=1,
    )
    assert meta.doc_id == "abc123hash"
    assert meta.title == "Master Direction on Note Sorting Machines"
    assert meta.regulator == "RBI"
    assert meta.circular_no == "RBI/DCM/2026-27/473"
    assert meta.issue_date == "Oct 02, 2026"
    assert meta.issue_date_iso == "2026-10-02T00:00:00Z"
    assert meta.effective_date is None
    assert meta.status == "active"
    assert meta.supersedes is None
    assert meta.section == "1. Introduction"
    assert meta.page == 1
    assert meta.access_level == "public"
    assert meta.source_url == "https://www.rbi.org.in/sample"
    assert meta.content_hash == "hash456"
    assert meta.chunk_index == 1


def test_chunk_metadata_all_15_fields_present():
    """Verify that all 15 canonical fields are present on the ChunkMetadata schema."""
    expected_fields = {
        "doc_id",
        "title",
        "regulator",
        "circular_no",
        "issue_date",
        "issue_date_iso",
        "effective_date",
        "status",
        "supersedes",
        "section",
        "page",
        "access_level",
        "source_url",
        "content_hash",
        "chunk_index",
    }
    actual_fields = set(ChunkMetadata.model_fields.keys())
    assert expected_fields == actual_fields


def test_status_valid_values():
    """Verify status accepts 'active', 'superseded', and None."""
    meta_active = ChunkMetadata(
        doc_id="1",
        title="T",
        regulator="RBI",
        page=1,
        content_hash="h",
        chunk_index=1,
        status="active",
    )
    assert meta_active.status == "active"

    meta_superseded = ChunkMetadata(
        doc_id="1",
        title="T",
        regulator="RBI",
        page=1,
        content_hash="h",
        chunk_index=1,
        status="superseded",
    )
    assert meta_superseded.status == "superseded"

    meta_none = ChunkMetadata(
        doc_id="1",
        title="T",
        regulator="RBI",
        page=1,
        content_hash="h",
        chunk_index=1,
        status=None,
    )
    assert meta_none.status is None


def test_status_invalid_value_fails():
    """Verify invalid status values raise a Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        ChunkMetadata(
            doc_id="1",
            title="T",
            regulator="RBI",
            page=1,
            content_hash="h",
            chunk_index=1,
            status="draft",  # Invalid
        )

    with pytest.raises(ValidationError):
        ChunkMetadata(
            doc_id="1",
            title="T",
            regulator="RBI",
            page=1,
            content_hash="h",
            chunk_index=1,
            status="archived",  # Invalid
        )


def test_optional_fields_defaults():
    """Verify optional fields have expected defaults when omitted."""
    meta = ChunkMetadata(
        doc_id="doc1",
        title="Sample Document",
        regulator="SEBI",
        page=1,
        content_hash="hash1",
        chunk_index=1,
    )
    assert meta.circular_no is None
    assert meta.issue_date is None
    assert meta.issue_date_iso is None
    assert meta.effective_date is None
    assert meta.status is None
    assert meta.supersedes is None
    assert meta.section is None
    assert meta.access_level == "public"
    assert meta.source_url is None


def test_page_and_chunk_index_types_and_constraints():
    """Verify page and chunk_index enforce proper integer values and bounds."""
    # page must be >= 1
    with pytest.raises(ValidationError):
        ChunkMetadata(
            doc_id="1",
            title="T",
            regulator="RBI",
            page=0,  # Invalid: page < 1
            content_hash="h",
            chunk_index=1,
        )

    # chunk_index must be >= 0
    with pytest.raises(ValidationError):
        ChunkMetadata(
            doc_id="1",
            title="T",
            regulator="RBI",
            page=1,
            content_hash="h",
            chunk_index=-1,  # Invalid: chunk_index < 0
        )


def test_content_hash_deterministic():
    """Verify compute_content_hash produces deterministic SHA-256 hashes."""
    text1 = "This is a regulatory rule clause for capital requirements."
    text2 = "This is a regulatory rule clause for capital requirements."
    text_whitespace = "  \n  This is a regulatory rule clause for capital requirements. \n\t "
    diff_text = "This is a DIFFERENT regulatory clause."

    hash1 = compute_content_hash(text1)
    hash2 = compute_content_hash(text2)
    hash_ws = compute_content_hash(text_whitespace)
    hash_diff = compute_content_hash(diff_text)

    assert hash1 == hash2
    assert hash1 == hash_ws
    assert hash1 != hash_diff
    assert len(hash1) == 64


def test_doc_id_deterministic():
    """Verify generate_doc_id produces deterministic SHA-256 document IDs."""
    doc_id1 = generate_doc_id(
        source_file="473MDD1229693F0604B6D996B1DF75C81E466.PDF",
        regulator="RBI",
        circular_no="RBI/DCM/2026-27/473",
    )
    doc_id2 = generate_doc_id(
        source_file="473MDD1229693F0604B6D996B1DF75C81E466.PDF",
        regulator="rbi",
        circular_no="rbi/dcm/2026-27/473",
    )
    doc_id_diff = generate_doc_id(
        source_file="other_file.pdf",
        regulator="RBI",
        circular_no="RBI/DCM/2026-27/473",
    )

    assert doc_id1 == doc_id2
    assert doc_id1 != doc_id_diff
    assert len(doc_id1) == 64


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Jan 16, 2023", "2023-01-16T00:00:00Z"),
        ("Feb 29, 2024", "2024-02-29T00:00:00Z"),
        ("Aug 24, 2026", "2026-08-24T00:00:00Z"),
        ("Oct 02, 2026", "2026-10-02T00:00:00Z"),
    ],
)
def test_normalize_issue_date(raw, expected):
    assert normalize_issue_date(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", " ", "Aug 4, 2026", "2026-08-24", "Feb 30, 2026", "Foo 24, 2026", 123],
)
def test_normalize_issue_date_rejects_missing_or_malformed(raw):
    with pytest.raises(ValueError, match="issue_date"):
        normalize_issue_date(raw)


def test_manifest_loading_and_mapping():
    """Verify load_manifest parses manifest.csv and maps all expected fields."""
    manifest = load_manifest("data/manifest.csv")
    assert len(manifest) > 0

    first_file = "473MDD1229693F0604B6D996B1DF75C81E466.PDF"
    assert first_file in manifest
    record = manifest[first_file]
    assert record["regulator"] == "RBI"
    assert record["title"] == "Master Direction on Note Sorting Machines"
    assert record["circular_no"] == "RBI/DCM/2026-27/473"
    assert record["issue_date"] == "Oct 02, 2026"
    assert record["issue_date_iso"] == "2026-10-02T00:00:00Z"
    assert record["effective_date"] is None
    assert record["status"] is None
    assert record["supersedes"] is None
    assert record["access_level"] == "public"
    assert "rbi.org.in" in record["source_url"]


def test_all_manifest_dates_normalize_without_changing_original_values():
    manifest = load_manifest("data/manifest.csv")
    assert len(manifest) == 42
    for record in manifest.values():
        assert record["issue_date_iso"] == normalize_issue_date(record["issue_date"])


@pytest.mark.parametrize("raw", ["", "Feb 30, 2026"])
def test_manifest_rejects_missing_or_invalid_date(tmp_path, raw):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"file,date\ntest.pdf,{raw}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid issue date for test.pdf"):
        load_manifest(manifest)


def test_get_document_metadata_lookup_and_fallback():
    """Verify get_document_metadata matches manifest entries and provides defaults for
    unknown files.
    """
    known_doc = get_document_metadata("473MDD1229693F0604B6D996B1DF75C81E466.PDF")
    assert known_doc["regulator"] == "RBI"
    assert known_doc["title"] == "Master Direction on Note Sorting Machines"
    assert known_doc["doc_id"] is not None
    assert known_doc["issue_date_iso"] == "2026-10-02T00:00:00Z"

    unknown_doc = get_document_metadata("non_existent_circular.pdf")
    assert unknown_doc["regulator"] == "UNKNOWN"
    assert unknown_doc["title"] == "non_existent_circular.pdf"
    assert unknown_doc["doc_id"] is not None
    assert unknown_doc["access_level"] == "public"
    assert unknown_doc["issue_date_iso"] is None


def test_chunk_to_metadata_conversion():
    """Verify Chunk.to_metadata() converts dataclass metadata dict to validated ChunkMetadata."""
    chunk = Chunk(
        text="Sample chunk body text.",
        metadata={
            "doc_id": "test_doc_id",
            "title": "Test Title",
            "regulator": "RBI",
            "circular_no": "RBI/123",
            "issue_date": "Oct 01, 2026",
            "effective_date": None,
            "status": "active",
            "supersedes": None,
            "section": "1. Scope",
            "page": 2,
            "access_level": "public",
            "source_url": None,
            "content_hash": "hash789",
            "chunk_index": 5,
            # Legacy backward-compatible keys
            "source_file": "test.pdf",
            "page_number": 2,
            "chunk_id": 5,
            "section_title": "1. Scope",
        },
    )
    chunk_meta = chunk.to_metadata()
    assert isinstance(chunk_meta, ChunkMetadata)
    assert chunk_meta.doc_id == "test_doc_id"
    assert chunk_meta.page == 2
    assert chunk_meta.chunk_index == 5
    assert chunk_meta.section == "1. Scope"
    assert chunk_meta.issue_date == "Oct 01, 2026"
    assert chunk_meta.issue_date_iso is None  # Historical payloads can omit the new field.
    assert chunk_meta.model_dump()["issue_date_iso"] is None


def test_chunk_text_populates_canonical_metadata():
    """Verify chunk_text produces chunks with all canonical ChunkMetadata fields and
    valid schema.
    """
    text = (
        "1. Introduction\nThese directions outline the regulatory requirements for authentication."
    )
    chunks = chunk_text(
        text,
        source_file="473MDD1229693F0604B6D996B1DF75C81E466.PDF",
        page_number=1,
    )
    assert len(chunks) == 1
    c = chunks[0]

    # Verify Chunk.to_metadata() validates cleanly
    meta = c.to_metadata()
    assert meta.regulator == "RBI"
    assert meta.title == "Master Direction on Note Sorting Machines"
    assert meta.circular_no == "RBI/DCM/2026-27/473"
    assert meta.issue_date == "Oct 02, 2026"
    assert meta.issue_date_iso == "2026-10-02T00:00:00Z"
    assert meta.section == "1. Introduction"
    assert meta.page == 1
    assert meta.chunk_index == 1
    assert meta.content_hash == compute_content_hash(c.text)

    # Verify dict access for canonical keys
    assert c.metadata["doc_id"] == meta.doc_id
    assert c.metadata["page"] == 1
    assert c.metadata["chunk_index"] == 1
    assert c.metadata["content_hash"] == meta.content_hash
    assert c.metadata["issue_date_iso"] == meta.issue_date_iso

    # Verify backward-compatibility for legacy keys
    assert c.metadata["source_file"] == "473MDD1229693F0604B6D996B1DF75C81E466.PDF"
    assert c.metadata["page_number"] == 1
    assert c.metadata["chunk_id"] == 1
    assert c.metadata["section_title"] == "1. Introduction"


def test_chunk_pages_populates_canonical_metadata():
    """Verify chunk_pages attaches canonical metadata across multiple pages with manifest lookup."""
    pages = [
        ParsedPage(
            page_number=1,
            text=(
                "Preamble on general banking regulations.\n\n"
                "1. Scope\nApplicable to all commercial banks."
            ),
            is_low_text_page=False,
        ),
        ParsedPage(
            page_number=2,
            text="Continuing section scope provisions across page boundary.",
            is_low_text_page=False,
        ),
    ]
    chunks = chunk_pages(
        pages,
        source_file="473MDD1229693F0604B6D996B1DF75C81E466.PDF",
    )
    assert len(chunks) >= 3

    for idx, c in enumerate(chunks, start=1):
        meta = c.to_metadata()
        assert meta.regulator == "RBI"
        assert meta.title == "Master Direction on Note Sorting Machines"
        assert meta.chunk_index == idx
        assert meta.content_hash == compute_content_hash(c.text)
        assert meta.page in [1, 2]
        assert meta.issue_date_iso == "2026-10-02T00:00:00Z"


def test_new_chunk_serializes_issue_date_iso_into_qdrant_payload(monkeypatch):
    chunks = chunk_text(
        "Regulatory evidence for a dated circular.",
        source_file="473MDD1229693F0604B6D996B1DF75C81E466.PDF",
    )
    client = Mock()
    monkeypatch.setattr(vectorstore, "init_collection", Mock())
    monkeypatch.setattr(vectorstore, "embed_chunks", Mock(return_value=[[0.1] * 384]))

    assert (
        vectorstore.upsert_chunks(
            chunks, client=client, collection_name="offline_test", embedder=Mock()
        )
        == 1
    )

    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["issue_date"] == "Oct 02, 2026"
    assert payload["issue_date_iso"] == "2026-10-02T00:00:00Z"
