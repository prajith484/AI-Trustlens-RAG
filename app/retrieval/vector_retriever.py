"""Vector retrieval: embed claim -> Qdrant search -> evidence dicts."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..embeddings.embedding_service import EmbeddingService
from ..vector_store.qdrant_store import QdrantStore


def _validate_claim(claim: str) -> str:
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    return claim.strip()


def retrieve_vector_evidence(
    claim: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    store: QdrantStore | None = None,
    embedder: EmbeddingService | None = None,
) -> Dict[str, Any]:
    """Return dict with evidence list, latency, diagnostics.

    NOTE: a retrieval score is a similarity signal, NOT a truth probability.
    """
    from ..embeddings.embedding_service import get_embedding_service
    claim = _validate_claim(claim)
    t0 = time.perf_counter()
    diagnostics: Dict[str, Any] = {"method": "vector", "error": None}
    try:
        from app.config import settings
        from ..vector_store.qdrant_store import get_qdrant_store
        store = store or get_qdrant_store()
        embedder = embedder or get_embedding_service()
        qvec = embedder.embed_query(claim)
        diagnostics["embedding_backend"] = embedder.backend
        diagnostics["embedding_model"] = embedder.model_name
        if qvec.shape[0] != store.dim:
            # adapt store dim on first use (e.g. 384 MiniLM)
            store.dim = int(qvec.shape[0])
        hits = store.search(qvec, top_k=top_k, filters=filters)
    except Exception as exc:
        diagnostics["error"] = str(exc)
        return {"claim": claim, "evidence": [], "evidence_found": False,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "diagnostics": diagnostics}
    evidence = []
    for rank, h in enumerate(hits):
        evidence.append({
            "chunk_id": h.get("chunk_id", ""),
            "document_id": h.get("document_id", ""),
            "text": h.get("text", ""),
            "title": h.get("title"),
            "source_name": h.get("source_name"),
            "source_url": h.get("source_url"),
            "publication_date": h.get("publication_date"),
            "page_number": h.get("page_number"),
            "domain": h.get("domain"),
            "retrieval_methods": ["vector"],
            "retrieval_score": h.get("_score"),
            "retrieval_rank": rank,
            "graph_path": [],
        })
    # dedup by chunk_id (keep best rank)
    seen, deduped = set(), []
    for e in evidence:
        if e["chunk_id"] in seen:
            continue
        seen.add(e["chunk_id"])
        deduped.append(e)
    latency = int((time.perf_counter() - t0) * 1000)
    diagnostics["latency_ms"] = latency
    return {"claim": claim, "evidence": deduped, "evidence_found": bool(deduped),
            "latency_ms": latency, "diagnostics": diagnostics}
