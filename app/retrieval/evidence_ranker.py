"""Final evidence selection: dedup, near-dup suppression, filtering, top-k."""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from ..ingestion.preprocessing import normalize_for_dedup


def _near_duplicate(a: str, b: str, threshold: float = 0.88) -> bool:
    if normalize_for_dedup(a) == normalize_for_dedup(b):
        return True
    if abs(len(a) - len(b)) > max(len(a), len(b)) * 0.3:
        return False
    return SequenceMatcher(None, a[:2000], b[:2000]).ratio() >= threshold


def rank_evidence(
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    relevance_threshold: Optional[float] = None,
    dedup_near: bool = True,
) -> List[Dict[str, Any]]:
    """Deduplicate (chunk_id), apply metadata filters, suppress near-identical
    passages, sort fused-score desc, truncate to top_k.

    - Never classifies claims; never converts scores to truth probabilities.
    - Contradictory passages are KEPT (needed by the verification module).
    - relevance_threshold applies only if explicitly configured.
    """
    # filters
    filtered = []
    for c in candidates:
        if filters:
            skip = any(k in filters and filters[k] is not None
                       and c.get(k) != filters[k]
                       for k in ("domain", "document_id", "source_name"))
            if skip:
                continue
        if relevance_threshold is not None and c.get("retrieval_score") is not None:
            if float(c["retrieval_score"]) < relevance_threshold:
                continue
        filtered.append(c)
    # exact dedup by chunk_id (merge retrieval_methods)
    merged: Dict[str, Dict] = {}
    for c in filtered:
        cid = c.get("chunk_id", "")
        if cid in merged:
            prev = merged[cid]
            methods = sorted(set(prev.get("retrieval_methods", [])) | set(c.get("retrieval_methods", [])))
            prev["retrieval_methods"] = methods
            # keep better (lower) rank / higher fused score
            if (c.get("fused_score") is not None and
                    (prev.get("fused_score") is None or c["fused_score"] > prev["fused_score"])):
                prev["fused_score"] = c["fused_score"]
                prev["retrieval_score"] = c.get("retrieval_score", prev.get("retrieval_score"))
            if c.get("graph_path") and not prev.get("graph_path"):
                prev["graph_path"] = c["graph_path"]
        else:
            merged[cid] = dict(c)
    cands = list(merged.values())
    # near-dup suppression (keep first = best ranked)
    if dedup_near:
        kept: List[Dict] = []
        for c in cands:
            if any(_near_duplicate(c.get("text", ""), k.get("text", "")) for k in kept):
                continue
            kept.append(c)
        cands = kept
    cands.sort(key=lambda d: (d.get("fused_score") is not None,
                              d.get("fused_score") or 0,
                              -(d.get("retrieval_rank") or 0)), reverse=True)
    return cands[:top_k]
