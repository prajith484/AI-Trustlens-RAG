"""Text preprocessing: cleaning that preserves headings/paragraph boundaries."""
from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Remove excessive whitespace/formatting but keep paragraph breaks.

    - Normalises line endings.
    - Collapses 3+ newlines to a paragraph break.
    - Collapses runs of spaces/tabs.
    - Strips control characters (keeps \\n, \\t).
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove null/control chars except newline/tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Tidy hyphenated line-breaks from PDFs: "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", cleaned)]
    return [p for p in paras if p]


def normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())
