"""Unit tests for Regulatrace PDF parser using synthetic PDFs."""

from pathlib import Path

import pymupdf as fitz
import pytest

from regulatrace.ingestion.parser import (
    DEFAULT_LOW_TEXT_THRESHOLD,
    EmptyPDFError,
    InvalidPDFError,
    parse_pdf,
)


def create_synthetic_pdf(
    path: Path,
    pages_text: list[str],
) -> Path:
    """Create a synthetic PDF file with specified text per page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def test_successful_parsing_and_page_ordering(tmp_path: Path):
    """Verify parsing a multi-page PDF extracts text in exact page order."""
    pdf_file = tmp_path / "sample_ordered.pdf"
    pages_content = [
        "First page content with regulatory details regarding bank operations.",
        "Second page content covering investment compliance and reporting.",
        "Third page content concluding the statutory framework directions.",
    ]
    create_synthetic_pdf(pdf_file, pages_content)

    results = parse_pdf(pdf_file)

    assert len(results) == 3
    for idx, page in enumerate(results, 1):
        assert page.page_number == idx
        assert pages_content[idx - 1] in page.text
        assert page.is_low_text_page is False


def test_low_text_page_detection(tmp_path: Path):
    """Verify pages with text below the threshold are flagged as low-text."""
    pdf_file = tmp_path / "low_text.pdf"
    pages_content = [
        "A" * 150,  # Normal text (> threshold)
        "Brief",  # Low text (5 chars < default threshold 50)
        "",  # Blank page (0 chars)
    ]
    create_synthetic_pdf(pdf_file, pages_content)

    results = parse_pdf(pdf_file, low_text_threshold=DEFAULT_LOW_TEXT_THRESHOLD)

    assert len(results) == 3
    assert results[0].is_low_text_page is False
    assert results[1].is_low_text_page is True
    assert results[2].is_low_text_page is True


def test_file_not_found():
    """Verify FileNotFoundError is raised when target file does not exist."""
    non_existent = Path("data/raw/non_existent_file_xyz_123.pdf")
    with pytest.raises(FileNotFoundError, match="PDF file not found"):
        parse_pdf(non_existent)


def test_invalid_pdf_raises_error(tmp_path: Path):
    """Verify InvalidPDFError is raised when file contains corrupt or non-PDF data."""
    corrupt_file = tmp_path / "corrupt.pdf"
    corrupt_file.write_bytes(b"NOT_A_VALID_PDF_HEADER_JUST_CORRUPT_BYTES")

    with pytest.raises(InvalidPDFError, match="Failed to open invalid or corrupted PDF"):
        parse_pdf(corrupt_file)


def test_empty_pdf_raises_error(tmp_path: Path):
    """Verify EmptyPDFError is raised when PDF contains zero pages or is empty."""
    # Case 1: Valid PDF structure with 0 pages
    empty_pages_pdf = tmp_path / "zero_pages.pdf"
    pdf_0_pages_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [] /Count 0 >>\n"
        b"endobj\n"
        b"xref\n"
        b"0 3\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"trailer\n"
        b"<< /Size 3 /Root 1 0 R >>\n"
        b"startxref\n"
        b"115\n"
        b"%%EOF"
    )
    empty_pages_pdf.write_bytes(pdf_0_pages_bytes)

    with pytest.raises(EmptyPDFError, match="PDF contains no pages"):
        parse_pdf(empty_pages_pdf)

    # Case 2: Empty 0-byte file
    zero_byte_file = tmp_path / "zero_bytes.pdf"
    zero_byte_file.write_bytes(b"")

    with pytest.raises(EmptyPDFError, match="PDF file is empty"):
        parse_pdf(zero_byte_file)
