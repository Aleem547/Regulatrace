"""Focused unit tests for structure-aware recursive chunking."""

from regulatrace.ingestion.chunker import (
    Chunk,
    chunk_pages,
    chunk_text,
)
from regulatrace.ingestion.parser import ParsedPage


def test_chunk_creation_and_sizing():
    """Verify recursive chunking splits long text according to configured sizes."""
    text = (
        "This is sentence one of the regulatory direction. "
        "It provides compliance mandates for all regulated entities. "
    ) * 30  # ~2700 chars

    # 500 chars (overlap 50 ~10%)
    chunks_500 = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks_500) >= 5
    for c in chunks_500:
        assert len(c.text) <= 550

    # 1000 chars (overlap 150 ~15%)
    chunks_1000 = chunk_text(text, chunk_size=1000, chunk_overlap=150)
    assert len(chunks_1000) >= 3
    for c in chunks_1000:
        assert len(c.text) <= 1050

    # 1500 chars (overlap 200 ~13%)
    chunks_1500 = chunk_text(text, chunk_size=1500, chunk_overlap=200)
    assert len(chunks_1500) >= 2
    for c in chunks_1500:
        assert len(c.text) <= 1550


def test_chunk_overlap_behavior():
    """Verify consecutive chunks share overlapping content."""
    text = (
        "First segment of the policy document establishing capital adequacy ratios. "
        "Second segment outlining governance structures for risk management. "
        "Third segment detailing supervisory actions and penalties for non-compliance."
    )
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=25)
    assert len(chunks) >= 2
    words_0 = set(chunks[0].text.split())
    words_1 = set(chunks[1].text.split())
    shared_words = words_0.intersection(words_1)
    assert len(shared_words) > 0


def test_metadata_population():
    """Verify chunk metadata contains required keys and types."""
    text = "Simple policy sentence for metadata verification."
    chunks = chunk_text(text, source_file="circular_123.pdf", page_number=2, start_chunk_id=10)
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    meta = chunks[0].metadata

    assert meta["source_file"] == "circular_123.pdf"
    assert meta["page_number"] == 2
    assert meta["chunk_id"] == 10
    assert meta["section_title"] is None


def test_section_title_prepending():
    """Verify section headings are detected, stored in metadata, and prepended to chunk text."""
    text = (
        "1. Introduction\n"
        "Guidelines on installation of Note Sorting Machines conform to RBI standards.\n\n"
        "2. Applicability\n"
        "These directions apply to all commercial banks and non-banking financial companies."
    )
    chunks = chunk_text(text, source_file="master_direction.pdf", page_number=1)
    assert len(chunks) == 2

    assert chunks[0].metadata["section_title"] == "1. Introduction"
    assert chunks[0].text.startswith("1. Introduction\n\nGuidelines on installation")

    assert chunks[1].metadata["section_title"] == "2. Applicability"
    assert chunks[1].text.startswith("2. Applicability\n\nThese directions apply")


def test_chunk_pages_structure_aware():
    """Verify multi-page chunking preserves page metadata and tracks section across pages."""
    pages = [
        ParsedPage(
            page_number=1,
            text=(
                "Preamble text before sections.\n\n1. Scope and Applicability\nPart one on page 1."
            ),
            is_low_text_page=False,
        ),
        ParsedPage(
            page_number=2,
            text="Continuing section from page 1 without new heading.",
            is_low_text_page=False,
        ),
    ]
    chunks = chunk_pages(pages, source_file="doc.pdf")
    assert len(chunks) >= 3

    # Chunk 0: Preamble (no section)
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["section_title"] is None
    assert chunks[0].metadata["chunk_id"] == 1

    # Chunk 1: Section 1 on page 1
    assert chunks[1].metadata["page_number"] == 1
    assert chunks[1].metadata["section_title"] == "1. Scope and Applicability"
    assert chunks[1].metadata["chunk_id"] == 2

    # Chunk 2: Continued on page 2 (carries section from page 1)
    assert chunks[2].metadata["page_number"] == 2
    assert chunks[2].metadata["section_title"] == "1. Scope and Applicability"
    assert chunks[2].metadata["chunk_id"] == 3
    assert chunks[2].text.startswith("1. Scope and Applicability\n\nContinuing section")
