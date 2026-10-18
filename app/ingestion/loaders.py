"""Document loaders: TXT, PDF, HTML with provenance tracking.

- Never fabricates source info.
- Never assumes trustworthiness.
- Extraction errors are returned, not raised, so batch ingestion continues.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RawDocument:
    document_id: str
    title: str
    text: str
    source_name: str = "unknown"
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    ingestion_date: str = ""
    domain: str = "general"
    document_type: str = "txt"
    content_hash: str = ""
    page_texts: List[Dict] = field(default_factory=list)  # [{"page": int, "text": str}]
    metadata: Dict = field(default_factory=dict)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _doc_id(title: str, content_hash: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in title.lower())[:40].strip("_")
    return f"doc_{safe or 'untitled'}_{content_hash[:8]}"


# Document-provided header fields, e.g.:
#   Source: World Health Organization (WHO)
#   Title: Diabetes
#   URL: https://www.who.int/...
# Only recognised at the very top of a text file (consecutive `Key: value`
# lines, terminated by a blank line or the first non-header line). Headers are
# provenance metadata, NOT factual evidence, so they are stripped from the
# body text used for chunking/embedding.
_HEADER_KEY_MAP = {
    "source": "source_name",
    "title": "title",
    "url": "source_url",
    "link": "source_url",
    "date": "publication_date",
    "publication_date": "publication_date",
    "published": "publication_date",
    "domain": "domain",
}
_HEADER_LINE_RE = re.compile(r"^([A-Za-z_ ]{2,24}):\s*(.+?)\s*$")


def parse_header_block(text: str) -> Tuple[Dict[str, str], str]:
    """Split leading `Key: value` headers from body. Returns (headers, body).

    Never invents values: only explicitly labelled, recognised keys are kept.
    """
    headers: Dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1  # blank line ends the header block
            break
        m = _HEADER_LINE_RE.match(line)
        if not m:
            break
        key = m.group(1).strip().lower()
        if key not in _HEADER_KEY_MAP:
            break
        headers[_HEADER_KEY_MAP[key]] = m.group(2)
        i += 1
    body = "\n".join(lines[i:]).strip()
    return headers, body


def load_txt(path: Path, **overrides) -> RawDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    file_headers, body = parse_header_block(raw)
    text = body if body else raw  # never produce an empty doc from headers alone
    # Precedence: explicit overrides > document headers > defaults.
    title = overrides.get("title") or file_headers.get("title") or path.stem.replace("_", " ")
    chash = _content_hash(raw)  # hash raw file bytes: stable across re-ingestion
    return RawDocument(
        document_id=overrides.get("document_id") or _doc_id(title, chash),
        title=title,
        text=text,
        source_name=overrides.get("source_name") or file_headers.get("source_name") or path.name,
        source_url=overrides.get("source_url") or file_headers.get("source_url"),
        publication_date=overrides.get("publication_date") or file_headers.get("publication_date"),
        ingestion_date=datetime.now(timezone.utc).isoformat(),
        domain=overrides.get("domain") or file_headers.get("domain") or "general",
        document_type="txt",
        content_hash=chash,
        metadata={"file_path": str(path), "headers_parsed": file_headers},
    )


def load_pdf(path: Path, **overrides) -> RawDocument:
    """Extract text page-by-page with pypdf so page numbers are preserved."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required for PDF ingestion: pip install pypdf") from exc
    reader = PdfReader(str(path))
    page_texts: List[Dict] = []
    full_parts: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        page_texts.append({"page": i, "text": t})
        full_parts.append(t)
    text = "\n\n".join(p for p in full_parts if p.strip())
    # Fall back to metadata title if available
    meta_title = ""
    try:
        meta_title = (reader.metadata.title or "") if reader.metadata else ""
    except Exception:
        meta_title = ""
    title = overrides.get("title") or meta_title or path.stem.replace("_", " ")
    chash = _content_hash(text)
    return RawDocument(
        document_id=overrides.get("document_id") or _doc_id(title, chash),
        title=title,
        text=text,
        source_name=overrides.get("source_name") or path.name,
        source_url=overrides.get("source_url"),
        publication_date=overrides.get("publication_date"),
        ingestion_date=datetime.now(timezone.utc).isoformat(),
        domain=overrides.get("domain") or "general",
        document_type="pdf",
        content_hash=chash,
        page_texts=page_texts,
        metadata={"file_path": str(path), "num_pages": len(page_texts)},
    )


def load_html(path: Path, **overrides) -> RawDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # graceful fallback: strip tags naively
        import re
        text = re.sub(r"<[^>]+>", " ", raw)
        title = overrides.get("title") or path.stem.replace("_", " ")
    else:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        # Preserve headings/paragraphs with newlines
        text = soup.get_text(separator="\n")
        title_tag = soup.find("title")
        title = overrides.get("title") or (title_tag.get_text(strip=True) if title_tag else path.stem)
    chash = _content_hash(text)
    return RawDocument(
        document_id=overrides.get("document_id") or _doc_id(title, chash),
        title=title,
        text=text,
        source_name=overrides.get("source_name") or path.name,
        source_url=overrides.get("source_url"),
        publication_date=overrides.get("publication_date"),
        ingestion_date=datetime.now(timezone.utc).isoformat(),
        domain=overrides.get("domain") or "general",
        document_type="html",
        content_hash=chash,
        metadata={"file_path": str(path)},
    )


def load_file(path: str | Path, **overrides) -> RawDocument:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    suffix = p.suffix.lower()
    if suffix == ".txt":
        return load_txt(p, **overrides)
    if suffix == ".pdf":
        return load_pdf(p, **overrides)
    if suffix in (".html", ".htm"):
        return load_html(p, **overrides)
    raise ValueError(f"Unsupported file type: {suffix} (supported: .txt, .pdf, .html)")


def load_directory(directory: str | Path, recursive: bool = False,
                   **overrides) -> List[RawDocument]:
    """Load supported files in a directory. Errors are skipped with a log.

    recursive=False preserves the original behaviour (top level only), so a
    bulk `documents/` ingest never silently pulls in sub-folders such as
    `documents/medical/` unless requested.
    """
    d = Path(directory)
    paths = sorted(d.rglob("*") if recursive else d.iterdir())
    docs: List[RawDocument] = []
    for p in paths:
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".txt", ".pdf", ".html", ".htm"):
            continue
        try:
            docs.append(load_file(p, **overrides))
        except Exception as exc:
            print(f"[ingest] skipping {p.name}: {exc}")
    return docs
