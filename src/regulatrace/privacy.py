"""Best-effort redaction of obvious personal identifiers in diagnostic messages."""

import re

_PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), "[REDACTED_EMAIL]"),
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I), "[REDACTED_PAN]"),
    (
        re.compile(
            r"(?<!\w)(?:\+[0-9](?:[ ()-]*[0-9]){7,14}"
            r"|[6-9][0-9]{4}[ -]?[0-9]{5}"
            r"|0[0-9]{2,4}[ -]?[0-9]{6,8})(?!\w)"
        ),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(r"(?<!\w)[2-9][0-9]{3}(?:[ -]?[0-9]{4}){2}(?!\w)"),
        "[REDACTED_AADHAAR]",
    ),
)

_TRACE_PATTERNS = (
    re.compile(
        r"\b(?:password|passwd|pwd|api[_-]?key|access[_-]?token|refresh[_-]?token|"
        r"secret|authorization|credential|REGULATRACE_[A-Z_]*(?:KEY|SECRET))\b"
        r"[\"']?\s*(?:[:=]|\bis\b)\s*[^\r\n]*",
        re.I,
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.I),
    re.compile(r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(
        r"\b(?:sk[-_]|pk-lf-|gsk_|hf_|gh[pousr]_|github_pat_|xox[baprs]-|AIza)" r"[A-Za-z0-9_.-]+"
    ),
    re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I),
    re.compile(r"\b[A-Za-z0-9_+/=-]{32,}\b"),
)


def redact_for_logging(message: str) -> str:
    """Mask common email, phone, PAN and Aadhaar formats; this is not ID validation."""
    for pattern, replacement in _PATTERNS:
        message = pattern.sub(replacement, message)
    return message


def redact_for_tracing(message: str) -> str:
    """Redact free text only; document identifiers are recorded separately.

    Credential assignments redact the rest of their line, including spaced secrets.
    These regexes are basic protection, not comprehensive personal-data detection.
    """
    for pattern in _TRACE_PATTERNS:
        message = pattern.sub("[REDACTED]", message)
    return redact_for_logging(message)
