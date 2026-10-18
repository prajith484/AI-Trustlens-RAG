"""Paragraph-aware chunking.

Defaults: ~200 words per chunk, 30 words overlap (configurable).
- Short paragraphs are merged, never pointlessly split.
- Long paragraphs are split on sentence boundaries with word overlap.
- Stable chunk IDs: sha1(document_id + position + text hash)[:12].
- PDF page numbers propagated when the loader provides page_texts.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .loaders import RawDocument
from .preprocessing import clean_text

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    title: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    domain: Optional[str] = None
    publication_date: Optional[str] = None
    page_number: Optional[int] = None
    chunk_position: int = 0
    word_count: int = 0

    def to_payload(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "title": self.title,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "domain": self.domain,
            "publication_date": self.publication_date,
            "page_number": self.page_number,
            "chunk_position": self.chunk_position,
        }


def _stable_chunk_id(document_id: str, position: int, text: str) -> str:
    h = hashlib.sha1(f"{document_id}::{position}::{text[:500]}".encode("utf-8")).hexdigest()[:12]
    return f"{document_id}::c{position:04d}::{h}"


def _words(s: str) -> List[str]:
    return s.split()


def _split_long_paragraph(para: str, max_words: int, overlap: int) -> List[str]:
    """Split a long paragraph on sentence boundaries with word overlap."""
    sentences = _SENT_SPLIT.split(para.strip())
    if len(sentences) <= 1:
        # No sentence boundaries: hard split on words
        w = _words(para)
        out, i = [], 0
        step = max(max_words - overlap, 1)
        while i < len(w):
            out.append(" ".join(w[i:i + max_words]))
            if i + max_words >= len(w):
                break
            i += step
        return out
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        slen = len(_words(sent))
        if current and current_len + slen > max_words:
            chunks.append(" ".join(current))
            # overlap: carry last `overlap` words forward
            carry = _words(" ".join(current))[-overlap:] if overlap > 0 else []
            current = [" ".join(carry)] if carry else []
            current_len = len(carry)
        current.append(sent)
        current_len += slen
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def chunk_document(
    doc: RawDocument,
    chunk_size_words: int = 200,
    chunk_overlap_words: int = 30,
) -> List[Chunk]:
    # If PDF page texts exist, chunk per page so page numbers are preserved.
    page_spans: List[tuple[Optional[int], str]] = []
    if doc.page_texts:
        for entry in doc.page_texts:
            t = clean_text(entry.get("text", ""))
            if t.strip():
                page_spans.append((entry.get("page"), t))
    else:
        t = clean_text(doc.text)
        if t.strip():
            page_spans.append((None, t))

    # Gather paragraphs (with page context), then pack into window-sized chunks.
    paras: List[tuple[Optional[int], str]] = []
    for page, text in page_spans:
        for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
            paras.append((page, para))

    # Expand long paragraphs first
    expanded: List[tuple[Optional[int], str]] = []
    for page, para in paras:
        if len(_words(para)) > chunk_size_words:
            for piece in _split_long_paragraph(para, chunk_size_words, chunk_overlap_words):
                expanded.append((page, piece))
        else:
            expanded.append((page, para))

    # Pack short paragraphs up to chunk_size_words
    grouped: List[tuple[Optional[int], str]] = []
    buf: List[str] = []
    buf_len = 0
    buf_page: Optional[int] = None
    for page, para in expanded:
        plen = len(_words(para))
        # If adding this para overflows and buffer non-empty -> flush
        if buf and buf_len + plen > chunk_size_words:
            grouped.append((buf_page, "\n\n".join(buf)))
            # overlap: last `overlap` words of flushed chunk start next buffer
            tail = _words("\n\n".join(buf))[-chunk_overlap_words:] if chunk_overlap_words > 0 else []
            buf = [" ".join(tail)] if tail else []
            buf_len = len(tail)
            buf_page = buf_page if buf else page
        if buf_page is None:
            buf_page = page
        buf.append(para)
        buf_len += plen
    if buf:
        grouped.append((buf_page, "\n\n".join(buf)))

    chunks: List[Chunk] = []
    for pos, (page, text) in enumerate(grouped):
        text = text.strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_id=_stable_chunk_id(doc.document_id, pos, text),
            document_id=doc.document_id,
            text=text,
            title=doc.title,
            source_name=doc.source_name,
            source_url=doc.source_url,
            domain=doc.domain,
            publication_date=doc.publication_date,
            page_number=page,
            chunk_position=pos,
            word_count=len(_words(text)),
        ))
    return chunks
