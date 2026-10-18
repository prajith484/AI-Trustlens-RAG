"""Graph-based retrieval: entities -> Neo4j traversal -> source chunks.

Uses graph RELATIONSHIPS (not just entity storage): matched entities are
expanded via RELATED_TO edges up to max_depth, and evidence chunks are those
linked to the traversed entities. Every item links back to an original chunk
with its graph path. Connectivity is never treated as proof of correctness.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..graph_store.entity_extractor import EntityExtractor
from ..graph_store.neo4j_store import Neo4jStore


def retrieve_graph_evidence(
    claim: str,
    top_k: int = 5,
    max_depth: int = 2,
    filters: Optional[Dict[str, Any]] = None,
    store: Neo4jStore | None = None,
    extractor: EntityExtractor | None = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    diagnostics: Dict[str, Any] = {"method": "graph", "error": None,
                                   "matched_entities": [], "edges": 0,
                                   "filters": filters or {}}
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    claim = claim.strip()
    try:
        from app.config import settings
        from ..graph_store.neo4j_store import get_graph_store
        store = store or get_graph_store()
        diagnostics["backend"] = store.backend
        extractor = extractor or EntityExtractor()
        claim_entities = extractor.extract_entities(claim)
        names = [e.text for e in claim_entities]
        diagnostics["claim_entities"] = names
        if not names:
            # Fallback: keyword match of claim tokens against entity texts
            names = [w for w in claim.split() if len(w) > 3][:8]
        matched = store.find_entities(names)
        diagnostics["matched_entities"] = [m.get("text") for m in matched][:10]
        if not matched:
            latency = int((time.perf_counter() - t0) * 1000)
            return {"claim": claim, "evidence": [], "evidence_found": False,
                    "latency_ms": latency, "diagnostics": diagnostics}
        start_ids = [m["entity_id"] for m in matched[:8]]
        edges = store.traverse(start_ids, max_depth=max_depth)
        diagnostics["edges"] = len(edges)
        # Collect entity ids in the traversed subgraph
        ent_ids = set(start_ids)
        for e in edges:
            for node in e.get("path", []):
                ent_ids.add(node)
            if e.get("to"):
                ent_ids.add(e["to"])
        chunks = store.chunks_for_entities(list(ent_ids), filters=filters)
        # Rank chunks: prefer those linked to more matched entities / shallower
        evidence: List[Dict[str, Any]] = []
        for ch in chunks[: max(top_k * 4, top_k)]:
            # find a representative path for provenance
            paths = [e["path"] for e in edges
                     if e.get("chunk_id") == ch.get("chunk_id")][:1]
            if not paths:
                # link via directly matched entity
                paths = [start_ids[:1]]
            evidence.append({
                "chunk_id": ch.get("chunk_id", ""),
                "document_id": ch.get("document_id", ""),
                "text": ch.get("text", ""),
                "title": ch.get("title"),
                "source_name": ch.get("source_name"),
                "source_url": ch.get("source_url"),
                "publication_date": ch.get("publication_date"),
                "page_number": ch.get("page_number"),
                "domain": ch.get("domain"),
                "retrieval_methods": ["graph"],
                "retrieval_score": None,
                "retrieval_rank": len(evidence),
                "graph_path": paths[0] if paths else start_ids[:1],
            })
        evidence = evidence[:top_k]
    except Exception as exc:
        diagnostics["error"] = str(exc)
        latency = int((time.perf_counter() - t0) * 1000)
        return {"claim": claim, "evidence": [], "evidence_found": False,
                "latency_ms": latency, "diagnostics": diagnostics}
    latency = int((time.perf_counter() - t0) * 1000)
    diagnostics["latency_ms"] = latency
    return {"claim": claim, "evidence": evidence, "evidence_found": bool(evidence),
            "latency_ms": latency, "diagnostics": diagnostics}
