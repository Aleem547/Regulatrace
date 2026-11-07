"""Pure text cleaning and normalization functions for regulatory documents.

Provides small, pure functions for Unicode normalization, repairing hyphenated
line breaks from PDF extraction, and whitespace normalization while strictly
preserving legal meaning (section numbers, dates, monetary figures, and references).
"""

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Apply Unicode normalization (NFKC) using standard library unicodedata."""
    return unicodedata.normalize("NFKC", text)


def fix_hyphenated_line_breaks(text: str) -> str:
    """Repair hyphenated line breaks caused by PDF extraction (e.g. 'regula-\\ntion')."""
    return re.sub(r"([a-zA-Z]+)-[^\S\r\n]*\r?\n[^\S\r\n]*([a-zA-Z]+)", r"\1\2", text)


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace while preserving paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def clean_text(text: str) -> str:
    """Clean and normalize extracted text using small pure functions.

    Applies Unicode normalization, repairs hyphenated line breaks,
    and normalizes excessive whitespace while preserving legal meaning.
    """
    text = normalize_unicode(text)
    text = fix_hyphenated_line_breaks(text)
    text = normalize_whitespace(text)
    return text
