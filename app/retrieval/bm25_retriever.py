"""BM25 keyword retrieval over ingested chunks (local index).

- Consistent tokenisation (lowercase alphanumerics).
- Index persisted to BM25_INDEX_PATH via pickle; rebuilt on startup if missing.
- Rebuild/update supported after ingestion changes.
"""
from __future__ import annotations

import pickle
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, index_path: str = "./bm25_index.pkl"):
        self.index_path = index_path
        self._lock = threading.Lock()
        self.doc_ids: List[str] = []       # chunk_ids in order
        self.payloads: List[Dict] = []     # chunk payloads in order
        self.tokenized: List[List[str]] = []
        self._bm25 = None

    def __len__(self) -> int:
        return len(self.doc_ids)

    def add_or_update(self, payloads: List[Dict]) -> int:
        """Upsert payloads by chunk_id; rebuilds the BM25 model. Returns index size."""
        with self._lock:
            by_id = {p["chunk_id"]: p for p in self.payloads}
            for p in payloads:
                by_id[p["chunk_id"]] = p
            self.payloads = list(by_id.values())
            self.doc_ids = [p["chunk_id"] for p in self.payloads]
            self.tokenized = [tokenize(p.get("text", "")) for p in self.payloads]
            self._build()
            self.save()
            return len(self.doc_ids)

    def remove_document(self, document_id: str) -> int:
        with self._lock:
            keep = [p for p in self.payloads if p.get("document_id") != document_id]
            removed = len(self.payloads) - len(keep)
            self.payloads = keep
            self.doc_ids = [p["chunk_id"] for p in self.payloads]
            self.tokenized = [tokenize(p.get("text", "")) for p in self.payloads]
            self._build()
            self.save()
            return removed

    def _build(self) -> None:
        if not self.tokenized:
            self._bm25 = None
            return
        # BM25Plus: unlike plain Okapi (whose IDF is log(1)=0 on tiny corpora,
        # making every score 0), it stays discriminative on small dev corpora.
        try:
            from rank_bm25 import BM25Plus  # type: ignore
            self._bm25 = BM25Plus(self.tokenized)
        except ImportError:
            from rank_bm25 import BM25Okapi  # type: ignore
            self._bm25 = BM25Okapi(self.tokenized)

    def save(self) -> None:
        if not self.index_path:
            return
        try:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({"payloads": self.payloads}, f)
        except Exception as exc:
            print(f"[bm25] save failed: {exc}")

    def load(self) -> int:
        if not self.index_path or not Path(self.index_path).exists():
            return 0
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            payloads = data.get("payloads", [])
            with self._lock:
                self.payloads = payloads
                self.doc_ids = [p["chunk_id"] for p in self.payloads]
                self.tokenized = [tokenize(p.get("text", "")) for p in self.payloads]
                self._build()
            return len(self.doc_ids)
        except Exception as exc:
            print(f"[bm25] load failed: {exc}")
            return 0

    def clear(self) -> None:
        with self._lock:
            self.doc_ids, self.payloads, self.tokenized, self._bm25 = [], [], [], None

    def search(self, query: str, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        if not query or not query.strip() or self._bm25 is None:
            return []
        qtok = tokenize(query)
        if not qtok:
            return []
        qset = set(qtok)
        with self._lock:
            scores = self._bm25.get_scores(qtok)
            # Gate: only chunks sharing >=1 query term are candidates.
            # (BM25Plus lower-bounds term contributions, so non-matching docs
            # can score > 0; the overlap gate preserves keyword semantics.)
            order = sorted(
                (i for i in range(len(scores)) if qset & set(self.tokenized[i])),
                key=lambda i: scores[i], reverse=True)
            out = []
            for rank, i in enumerate(order[: max(top_k * 3, top_k)]):
                p = self.payloads[i]
                if filters:
                    skip = any(k in filters and filters[k] is not None
                               and p.get(k) != filters[k]
                               for k in ("domain", "document_id", "source_name"))
                    if skip:
                        continue
                d = dict(p)
                d["retrieval_methods"] = ["bm25"]
                d["retrieval_score"] = float(scores[i])
                d["retrieval_rank"] = rank
                d.setdefault("graph_path", [])
                out.append(d)
                if len(out) >= top_k:
                    break
        return out


# Process-wide singleton -------------------------------------------------
_singleton: BM25Index | None = None
_singleton_lock = threading.Lock()


def get_bm25_index(index_path: str | None = None) -> BM25Index:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            from app.config import settings
            _singleton = BM25Index(index_path or settings.bm25_index_path)
            _singleton.load()
        return _singleton


def reset_bm25_index() -> None:
    global _singleton
    with _singleton_lock:
        _singleton = None


def retrieve_bm25_evidence(claim: str, top_k: int = 5,
                           filters: Optional[Dict[str, Any]] = None,
                           index: BM25Index | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    diagnostics: Dict[str, Any] = {"method": "bm25", "error": None}
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    claim = claim.strip()
    try:
        idx = index or get_bm25_index()
        hits = idx.search(claim, top_k=top_k, filters=filters)
    except Exception as exc:
        diagnostics["error"] = str(exc)
        hits = []
    latency = int((time.perf_counter() - t0) * 1000)
    diagnostics["latency_ms"] = latency
    return {"claim": claim, "evidence": hits, "evidence_found": bool(hits),
            "latency_ms": latency, "diagnostics": diagnostics}
