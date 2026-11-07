"""Focused tests for text cleaning and normalization functions."""

from regulatrace.ingestion.cleaner import (
    clean_text,
    fix_hyphenated_line_breaks,
    normalize_unicode,
    normalize_whitespace,
)


def test_normalize_unicode():
    """Verify Unicode normalization handles ligatures and compatibility characters."""
    ligature_text = "The \ufb01nance team met with the \ufb02eet manager."
    assert normalize_unicode(ligature_text) == "The finance team met with the fleet manager."

    nbsp_text = "RBI\u00a0Circular\u00a02024"
    assert normalize_unicode(nbsp_text) == "RBI Circular 2024"


def test_fix_hyphenated_line_breaks():
    """Verify hyphenated line breaks from PDF extraction are properly rejoined."""
    text = "This regula-\ntion applies to all non-banking insti-\ntutions."
    expected = "This regulation applies to all non-banking institutions."
    assert fix_hyphenated_line_breaks(text) == expected

    crlf_text = "compli-\r\nance requirement"
    assert fix_hyphenated_line_breaks(crlf_text) == "compliance requirement"

    preserved = "Section 45-IA, Year 2023-24, intra-day liquidity."
    assert fix_hyphenated_line_breaks(preserved) == preserved


def test_normalize_whitespace():
    """Verify excessive whitespace, multiple blanks, and trailing spaces are cleaned."""
    text = "  Paragraph one    has   extra   spaces.   \n\n\n\n   Paragraph two.  "
    expected = "Paragraph one has extra spaces.\n\nParagraph two."
    assert normalize_whitespace(text) == expected


def test_clean_text_preserves_legal_content():
    """Verify clean_text preserves section numbers, dates, monetary amounts, and references."""
    raw = (
        "Direc-\ntion issued under Section 45-IA of the RBI Act, 1934.\n\n\n"
        "Reference: DOR.No.BP.BC.23/21.04.048/2020-21 dated March 27, 2020.\n"
        "Penalty: ₹5,00,000 for non-compliance."
    )
    cleaned = clean_text(raw)
    assert "Direction issued under Section 45-IA of the RBI Act, 1934." in cleaned
    assert "Reference: DOR.No.BP.BC.23/21.04.048/2020-21 dated March 27, 2020." in cleaned
    assert "Penalty: ₹5,00,000 for non-compliance." in cleaned


def test_clean_text_idempotency():
    """Verify clean_text is strictly idempotent: clean_text(clean_text(x)) == clean_text(x)."""
    samples = [
        "Simple string with no artifacts.",
        "Direc-\ntion with ligatures: \ufb01nance and \ufb02ow.",
        "   Excessive   whitespace   \n\n\n\n   and trailing spaces.   ",
        "Section 45-IA of the RBI Act, 1934. ₹10,00,000 dated 2023-24.",
    ]
    for sample in samples:
        once = clean_text(sample)
        twice = clean_text(once)
        assert once == twice
