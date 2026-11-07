"""Chunk validation and Table of Contents (TOC) detection for regulatory documents."""

import re

from regulatrace.ingestion.patterns import (
    BARE_LABEL_PATTERN,
    LEGAL_SIGNAL_PATTERN,
    PAGE_NUMBER_PATTERN,
    TOC_MARKERS,
)


def is_toc_page(text: str) -> bool:
    """Detect if page text represents a Table of Contents page."""
    for line in text.split("\n")[:10]:
        stripped = line.strip().lower().rstrip(":-. ")
        if stripped in TOC_MARKERS:
            return True
        if stripped.startswith(("table of contents", "index")) and len(stripped) <= 35:
            return True
    return False


def is_valid_chunk(text: str) -> bool:
    """Determine whether a chunk contains meaningful content rather than noise or bare labels."""
    stripped = text.strip()
    if not stripped:
        return False

    # Discard separator / punctuation-only chunks (e.g. '---', '___', '***')
    if not any(c.isalnum() for c in stripped):
        return False

    # Discard standalone numbers (e.g. '2', '3')
    if stripped.isdigit():
        return False

    # Discard standalone page markers (e.g. 'Page 2', 'Page 2 of 6', '- 3 -')
    if PAGE_NUMBER_PATTERN.match(stripped):
        return False

    # Discard bare structural labels and status stubs (e.g. 'Annex I', 'Annex II', 'Form A1')
    normalized_spaces = re.sub(r"\s+", " ", stripped)
    if BARE_LABEL_PATTERN.match(normalized_spaces):
        return False

    # Filter short chunks (< 25 chars and < 3 words) without legal, financial, or date signals
    words = stripped.split()
    return not (len(stripped) < 25 and len(words) < 3 and not LEGAL_SIGNAL_PATTERN.search(stripped))
