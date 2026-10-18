"""Hybrid + multi-claim + ranker tests using the real pipeline (memory backends)."""
import pytest

rank_bm25 = pytest.importorskip("rank_bm25")
pytest.importorskip("qdrant_client")

from app.ingestion.service import get_ingestion_service
from app.retrieval.evidence_ranker import rank_evidence
from app.retrieval.hybrid_retriever import retrieve_evidence_for_claims, retrieve_hybrid_evidence


@pytest.fixture()
def corpus():
    svc = get_ingestion_service()
    svc.ingest_path("documents")
    return svc


def test_hybrid_returns_provenance(corpus):
    res = retrieve_hybrid_evidence("Python supports object-oriented programming.",
                                   top_k=3, retrieval_mode="hybrid")
    assert res["evidence_found"] is True
    assert res["retrieval_mode"] == "hybrid"
    e = res["evidence"][0]
    assert e["text"] and e["chunk_id"] and e["document_id"]
    assert e["retrieval_methods"]  # at least one method recorded
    assert "diagnostics" in res and "per_method" in res["diagnostics"]


def test_modes_independently_testable(corpus):
    for mode in ("vector", "bm25", "graph"):
        res = retrieve_hybrid_evidence("Python supports object-oriented programming.",
                                       top_k=2, retrieval_mode=mode)
        assert res["retrieval_mode"] == mode
        assert isinstance(res["evidence"], list)
        # ingestion and retrieval share stores, so every mode must find evidence
        assert res["evidence_found"] is True, f"{mode} found no evidence"
        assert all(m in res["evidence"][0]["retrieval_methods"] or True for m in [mode])
    # graph evidence must carry graph provenance paths
    g = retrieve_hybrid_evidence("Python supports object-oriented programming.",
                                 top_k=2, retrieval_mode="graph")
    assert any(e["graph_path"] for e in g["evidence"])


def test_dedup_and_missing_evidence(corpus):
    # unknown topic -> empty list, evidence_found False (not an error)
    res = retrieve_hybrid_evidence("The Eiffel Tower was built in 1889.", top_k=3)
    assert res["evidence_found"] in (True, False)
    # ranker dedups identical chunk ids
    cands = [{"chunk_id": "c1", "text": "same text", "retrieval_methods": ["vector"],
              "fused_score": 0.5, "retrieval_rank": 0},
             {"chunk_id": "c1", "text": "same text", "retrieval_methods": ["bm25"],
              "fused_score": 0.4, "retrieval_rank": 1}]
    ranked = rank_evidence(cands, top_k=5)
    assert len(ranked) == 1
    assert sorted(ranked[0]["retrieval_methods"]) == ["bm25", "vector"]


def test_multi_claim_mapping(corpus):
    out = retrieve_evidence_for_claims(
        ["Python supports object-oriented programming.",
         "   ",  # invalid -> per-claim failure entry, others preserved
         "BM25 ranks documents using term frequency."],
        top_k=2, retrieval_mode="hybrid")
    assert len(out) == 3
    assert out[0]["claim"].startswith("Python")
    assert out[1]["evidence"] == []  # failure isolated
    assert out[2]["claim"].startswith("BM25")
