"""Compiled regular expression patterns and constants for regulatory document ingestion.

Centralizes all structural division keywords, heading patterns, TOC markers,
page number markers, and legal reference signals used across ingestion modules.
"""

import re

MAX_HEADING_LENGTH: int = 80
MAX_HEADING_WORDS: int = 10

# Structural divisions supported by Indian regulatory frameworks
DIVISION_KEYWORD_REGEX: str = (
    r"(?:CHAPTER|Chapter|PART|Part|SECTION|Section|"
    r"ANNEXURE|Annexure|ANNEX|Annex|SCHEDULE|Schedule|APPENDIX|Appendix)"
)

# Regex patterns for detecting common Indian regulatory section headers
SECTION_HEADING_PATTERNS: list[re.Pattern[str]] = [
    # Explicit structural divisions (supporting hyphen, en-dash, em-dash, or spaces)
    re.compile(
        r"^\s*("
        + DIVISION_KEYWORD_REGEX
        + r"(?:\s*[-–—]\s*|\s+)[IVXLCDM0-9A-Za-z]+(?:\s*[-–—:]\s*[^\n]{1,80})?)\s*$",
        re.IGNORECASE,
    ),
    # Roman numeral headings (e.g. I. Background, II. Scope)
    re.compile(
        r"^\s*([IVXLCDM]+\.\s+[A-Z][^\n]{1,80})\s*$",
    ),
    # Numbered title headings (Title Case words, title connectors, and parentheticals)
    re.compile(
        r"^\s*(\d+(?:\.\d+)*\.?\s+"
        r"(?:[A-Z0-9][\w\/\-]*,?|and|in|of|on|for|to|the|with|by|or|from|a|an)"
        r"(?:\s+(?:[A-Z0-9][\w\/\-]*,?|and|in|of|on|for|to|the|with|by|or|from|a|an|\([A-Za-z0-9\/\-]+\)))*)\s*$"
    ),
    # Amendment notification numbered headings (single-level: <number>. <heading text>)
    re.compile(r"^\s*(\d+\.\s+[A-Z][A-Za-z0-9\s,\/\-\(\)]*?)\s*$"),
]

TRAILING_HEADING_SEPARATOR_PATTERN: re.Pattern[str] = re.compile(r"[\s:\-–—]+$")

BASE_DIVISION_PATTERN: re.Pattern[str] = re.compile(
    r"^" + DIVISION_KEYWORD_REGEX + r"(?:\s*[-–—]\s*|\s+)[IVXLCDM0-9A-Za-z]+$",
    re.IGNORECASE,
)

TITLE_LIKE_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:[A-Z0-9][\w\/\-]*,?|and|in|of|on|for|to|the|with|by|or|from|a|an)"
    r"(?:\s+(?:[A-Z0-9][\w\/\-]*,?|and|in|of|on|for|to|the|with|by|or|from|a|an|\([A-Za-z0-9\/\-]+\)))*$"
)

TOC_MARKERS: tuple[str, ...] = ("table of contents", "contents", "index")

PAGE_NUMBER_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:page\s+\d+(?:\s+of\s+\d+)?|-?\s*\d+\s*-?|\[\s*\d+\s*\])$",
    re.IGNORECASE,
)

BARE_LABEL_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:Annex|Annexure|Appendix|Schedule|Form)\s+[A-Za-z0-9IVXLCDM]+"
    r"(?:\s*[-–—:]?\s*(?:\([A-Za-z0-9\s]+\)|withdrawn|repealed|nil|na|none))?$",
    re.IGNORECASE,
)

LEGAL_SIGNAL_PATTERN: re.Pattern[str] = re.compile(
    r"(?:"
    r"[₹$€£]|Rs\.?|INR"
    r"|\b(?:Section|Sec\.|Clause|Article|Rule|Act|Notification|Circular)\s+[A-Za-z0-9\-]+"
    r"|\([A-Za-z0-9]+\)"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b"
    r"|\b\d{4}[-/]\d{2,4}\b"
    r"|\b[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)+/[A-Za-z0-9\.\/-]+"
    r")",
    re.IGNORECASE,
)
