"""Retrieval metrics: Recall@k, Precision@k, MRR, latency, coverage.

Retrieval-only metrics (kept separate from classifier accuracy).
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    rel = set(relevant)
    for i, cid in enumerate(retrieved, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def coverage(results: List[Dict]) -> float:
    """Fraction of claims with at least one relevant chunk retrieved."""
    if not results:
        return 0.0
    hit = sum(1 for r in results if r.get("hits", 0) > 0)
    return hit / len(results)


def summarize(per_claim: List[Dict], ks: Sequence[int] = (1, 3, 5)) -> Dict:
    import statistics
    summary: Dict = {}
    for k in ks:
        rs = [recall_at_k(r["retrieved"], r["relevant"], k) for r in per_claim]
        ps = [precision_at_k(r["retrieved"], r["relevant"], k) for r in per_claim]
        summary[f"recall@{k}"] = round(statistics.mean(rs), 4) if rs else 0.0
        summary[f"precision@{k}"] = round(statistics.mean(ps), 4) if ps else 0.0
    summary["mrr"] = round(statistics.mean(
        [reciprocal_rank(r["retrieved"], r["relevant"]) for r in per_claim]), 4)
    lats = [r.get("latency_ms", 0) for r in per_claim]
    summary["mean_latency_ms"] = round(statistics.mean(lats), 2) if lats else 0.0
    summary["coverage"] = round(coverage(per_claim), 4)
    summary["n_claims"] = len(per_claim)
    return summary
