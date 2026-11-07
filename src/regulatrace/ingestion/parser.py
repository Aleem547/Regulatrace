"""Reliable PDF parser for regulatory documents using PyMuPDF (fitz).

Extracts page-level text, preserves exact page ordering, and flags low-text
pages for potential scanned document identification without running OCR.
"""

from pathlib import Path

import pymupdf as fitz
from pydantic import BaseModel

DEFAULT_LOW_TEXT_THRESHOLD: int = 50


class PDFParserError(Exception):
    """Base exception for PDF parsing failures."""


class InvalidPDFError(PDFParserError, ValueError):
    """Raised when a file is corrupted, unreadable, or not a valid PDF."""


class EmptyPDFError(PDFParserError, ValueError):
    """Raised when a PDF contains no pages."""


class ParsedPage(BaseModel):
    page_number: int
    text: str
    is_low_text_page: bool


def parse_pdf(
    pdf_path: str | Path,
    low_text_threshold: int = DEFAULT_LOW_TEXT_THRESHOLD,
) -> list[ParsedPage]:
    """Parse a PDF file and extract text page-by-page.

    Args:
        pdf_path: Path to the target PDF file on disk.
        low_text_threshold: Minimum character count (after stripping whitespace)
            required for a page to not be considered low-text.

    Returns:
        A list of ParsedPage objects in ascending page order.

    Raises:
        FileNotFoundError: If the specified file does not exist on disk.
        InvalidPDFError: If the file is not a valid PDF or cannot be opened.
        EmptyPDFError: If the document contains zero pages.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise InvalidPDFError(f"Target path is not a regular file: {path}")

    try:
        doc = fitz.open(path)
    except fitz.EmptyFileError as e:
        raise EmptyPDFError(f"PDF file is empty (0 bytes): '{path}'") from e
    except Exception as e:
        raise InvalidPDFError(f"Failed to open invalid or corrupted PDF '{path}': {e}") from e

    try:
        total_pages = doc.page_count
        if total_pages == 0:
            raise EmptyPDFError(f"PDF contains no pages: {path}")

        parsed_pages: list[ParsedPage] = []
        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            text = page.get_text()
            stripped_length = len(text.strip())
            is_low_text = stripped_length < low_text_threshold

            parsed_pages.append(
                ParsedPage(
                    page_number=page_idx + 1,
                    text=text,
                    is_low_text_page=is_low_text,
                )
            )

        return parsed_pages

    finally:
        doc.close()
