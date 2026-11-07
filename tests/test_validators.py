"""Focused unit tests for chunk validation, noise filtering, and TOC detection."""

from regulatrace.ingestion.chunker import (
    chunk_pages,
    chunk_text,
)
from regulatrace.ingestion.parser import ParsedPage
from regulatrace.ingestion.validators import (
    is_toc_page,
    is_valid_chunk,
)


def test_low_value_chunk_filtering():
    """Verify standalone page numbers, separators, and empty text are discarded."""
    # Low-value artifacts that must be discarded
    assert is_valid_chunk("2") is False
    assert is_valid_chunk("3") is False
    assert is_valid_chunk("") is False
    assert is_valid_chunk("   \n\n   ") is False
    assert is_valid_chunk("---") is False
    assert is_valid_chunk("___") is False
    assert is_valid_chunk("Page 2") is False
    assert is_valid_chunk("Page 2 of 6") is False
    assert is_valid_chunk("- 3 -") is False

    # Valid legal content that must be preserved
    assert is_valid_chunk("Section 45-IA") is True
    assert is_valid_chunk("Clause 4(2)") is True
    assert is_valid_chunk("(6A)") is True
    assert is_valid_chunk("₹5,00,000") is True
    assert is_valid_chunk("October 1, 2026") is True
    assert is_valid_chunk("2023-24") is True
    assert is_valid_chunk("DOR.No.BP.BC.23/21.04.048/2020-21") is True


def test_chunking_discards_standalone_page_numbers():
    """Verify that chunking a page with standalone page numbers discards solitary number chunks."""
    text = "2\n\nSection 45-IA outlines mandatory reserve funds for all non-banking companies."
    chunks = chunk_text(text, source_file="sample.pdf", page_number=2)

    assert len(chunks) == 1
    assert "2" not in [c.text for c in chunks]
    assert "Section 45-IA outlines" in chunks[0].text
    assert chunks[0].metadata["chunk_id"] == 1


def test_toc_page_does_not_contaminate_future_pages():
    """Verify that TOC page headings do not leak into future content pages via carry-forward."""
    pages = [
        ParsedPage(
            page_number=1,
            text="Preamble text before table of contents.",
            is_low_text_page=False,
        ),
        ParsedPage(
            page_number=2,
            text=(
                "Contents\n\n"
                "CHAPTER I\nPRELIMINARY\n\n"
                "CHAPTER II\nPRIOR APPROVAL FOR ACQUISITION\n\n"
                "CHAPTER III\nCONTINUOUS MONITORING ARRANGEMENTS\n\n"
                "CHAPTER IV\nREPEAL AND OTHER PROVISIONS\n\n"
                "Annex I\n"
                "Annex II"
            ),
            is_low_text_page=False,
        ),
        ParsedPage(
            page_number=3,
            text="Substantive regulatory text on page 3 establishing capital requirements.",
            is_low_text_page=False,
        ),
        ParsedPage(
            page_number=4,
            text=(
                "CHAPTER – I\n"
                "PRELIMINARY\n\n"
                "1. Short Title and Commencement\n"
                "These directions shall be called the Master Direction, 2026."
            ),
            is_low_text_page=False,
        ),
    ]

    assert is_toc_page(pages[1].text) is True
    assert is_toc_page(pages[2].text) is False

    chunks = chunk_pages(pages, source_file="master_direction.pdf")
    page_3_chunks = [c for c in chunks if c.metadata["page_number"] == 3]
    assert len(page_3_chunks) >= 1
    # Page 3 must NOT have inherited "Annex II" or any TOC section title
    for c in page_3_chunks:
        assert c.metadata["section_title"] is None
        assert "Annex" not in c.text

    page_4_chunks = [c for c in chunks if c.metadata["page_number"] == 4]
    assert len(page_4_chunks) >= 1
    # Page 4 must detect its own section title, not TOC
    assert page_4_chunks[0].metadata["section_title"] in {
        "CHAPTER I - PRELIMINARY",
        "1. Short Title and Commencement",
    }


def test_tiny_low_value_chunk_filtering():
    """Verify that bare labels, status words, and tiny non-informative chunks are filtered out."""
    # Obvious low-information stubs that must NOT become standalone chunks
    assert is_valid_chunk("Annex I") is False
    assert is_valid_chunk("Withdrawn") is False
    assert is_valid_chunk("Annex II") is False
    assert is_valid_chunk("Form A1") is False
    assert is_valid_chunk("Annex I\n\nWithdrawn") is False
    assert is_valid_chunk("Annex I - Withdrawn") is False
    assert is_valid_chunk("Annex II (Withdrawn)") is False

    # Verify pipeline discards "Annex I\n\nWithdrawn"
    chunks = chunk_text("Annex I\n\nWithdrawn")
    assert len(chunks) == 0

    # Ensure valid legal references and content remain preserved
    assert is_valid_chunk("Section 45-IA") is True
    assert is_valid_chunk("Clause 4(2)") is True
    assert is_valid_chunk("(6A)") is True
    assert is_valid_chunk("₹5,00,000") is True
    assert is_valid_chunk("October 1, 2026") is True
    assert is_valid_chunk("2023-24") is True
    assert is_valid_chunk("DOR.No.BP.BC.23/21.04.048/2020-21") is True
