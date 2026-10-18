"""Hybrid retrieval: vector + BM25 + graph, fused with Reciprocal Rank Fusion.

RRF: fused(chunk) = sum_over_methods 1 / (k + rank), k=60 default.
Raw scores from different methods are NEVER directly combined.
Each method runs independently; failures are reported with fallbacks noted.
Modes: vector | bm25 | graph | hybrid.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .bm25_retriever import retrieve_bm25_evidence
from .evidence_ranker import rank_evidence
from .graph_retriever import retrieve_graph_evidence
from .vector_retriever import retrieve_vector_evidence


def _rrf_fuse(method_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    for res in method_results:
        method = res.get("diagnostics", {}).get("method", "?")
        for rank, ev in enumerate(res.get("evidence", [])):
            cid = ev.get("chunk_id", "")
            if not cid:
                continue
            entry = fused.setdefault(cid, dict(ev))
            entry["fused_score"] = float(entry.get("fused_score") or 0.0) + 1.0 / (k + rank + 1)
            methods = set(entry.get("retrieval_methods", [])) | set(ev.get("retrieval_methods", [method]))
            entry["retrieval_methods"] = sorted(methods)
            if ev.get("graph_path") and not entry.get("graph_path"):
                entry["graph_path"] = ev["graph_path"]
            # keep best individual rank for tie-breaking
            if entry.get("retrieval_rank") is None or rank < entry["retrieval_rank"]:
                entry["retrieval_rank"] = rank
    return list(fused.values())


def retrieve_hybrid_evidence(
    claim: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    retrieval_mode: str = "hybrid",
    graph_max_depth: Optional[int] = None,
    per_method_k: int = 10,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    from app.config import settings
    depth = graph_max_depth or settings.graph_max_depth
    mode = (retrieval_mode or "hybrid").lower()
    if mode not in ("vector", "bm25", "graph", "hybrid"):
        raise ValueError(f"unknown retrieval_mode: {retrieval_mode}")

    runs: List[Dict[str, Any]] = []
    fallbacks: List[str] = []
    want = {"vector": mode in ("vector", "hybrid"),
            "bm25": mode in ("bm25", "hybrid"),
            "graph": mode in ("graph", "hybrid")}

    if want["vector"]:
        r = retrieve_vector_evidence(claim, top_k=per_method_k, filters=filters)
        runs.append(r)
        if r["diagnostics"].get("error"):
            fallbacks.append(f"vector failed: {r['diagnostics']['error']}")
    if want["bm25"]:
        r = retrieve_bm25_evidence(claim, top_k=per_method_k, filters=filters)
        runs.append(r)
        if r["diagnostics"].get("error"):
            fallbacks.append(f"bm25 failed: {r['diagnostics']['error']}")
    if want["graph"]:
        r = retrieve_graph_evidence(claim, top_k=per_method_k, max_depth=depth,
                                    filters=filters)
        runs.append(r)
        if r["diagnostics"].get("error"):
            fallbacks.append(f"graph failed: {r['diagnostics']['error']}")

    fused = _rrf_fuse(runs, k=rrf_k)
    ranked = rank_evidence(fused, top_k=top_k, filters=filters)
    # Final safety check: domain filtering must survive fusion even if a
    # backend ignored it (defence in depth; rank_evidence already filters).
    domain = (filters or {}).get("domain")
    if domain is not None:
        ranked = [e for e in ranked if e.get("domain") == domain]
    # strip helper keys, keep schema-clean evidence
    evidence = []
    for e in ranked:
        evidence.append({
            "chunk_id": e.get("chunk_id", ""),
            "document_id": e.get("document_id", ""),
            "text": e.get("text", ""),
            "title": e.get("title"),
            "source_name": e.get("source_name"),
            "source_url": e.get("source_url"),
            "publication_date": e.get("publication_date"),
            "page_number": e.get("page_number"),
            "domain": e.get("domain"),
            "retrieval_methods": e.get("retrieval_methods", []),
            "retrieval_score": e.get("fused_score"),
            "graph_path": e.get("graph_path", []),
        })
    latency = int((time.perf_counter() - t0) * 1000)

    def _status(r: Dict[str, Any]) -> str:
        # Distinguish backend failure from a successful search with no hits.
        if r["diagnostics"].get("error"):
            return "error"
        return "ok" if r.get("evidence") else "empty"

    return {
        "claim": claim.strip(),
        "retrieval_mode": mode,
        "evidence": evidence,
        "evidence_found": bool(evidence),
        "latency_ms": latency,
        "diagnostics": {
            "per_method": [
                {"method": r["diagnostics"].get("method"),
                 "status": _status(r),
                 "count": len(r["evidence"]),
                 "error": r["diagnostics"].get("error"),
                 "latency_ms": r.get("latency_ms", 0)} for r in runs],
            "fallbacks": fallbacks,
            "filters": filters or {},
            "rrf_k": rrf_k,
            "fused_candidates": len(fused),
        },
    }


def retrieve_evidence_for_claims(
    claims: List[str],
    top_k: int = 5,
    retrieval_mode: str = "hybrid",
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Multi-claim retrieval: independent per claim; one failure never drops others."""
    results: List[Dict[str, Any]] = []
    for claim in claims:
        try:
            results.append(retrieve_hybrid_evidence(
                claim, top_k=top_k, filters=filters, retrieval_mode=retrieval_mode))
        except Exception as exc:
            results.append({
                "claim": claim if isinstance(claim, str) else str(claim),
                "retrieval_mode": retrieval_mode,
                "evidence": [], "evidence_found": False, "latency_ms": 0,
                "diagnostics": {"error": str(exc), "per_claim_failure": True},
            })
    return results
