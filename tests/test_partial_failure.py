"""Partial backend failure: one broken method must not sink the others."""
import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from app.ingestion.service import get_ingestion_service
from app.retrieval.hybrid_retriever import retrieve_hybrid_evidence
from app.retrieval.vector_retriever import retrieve_vector_evidence
from app.vector_store.qdrant_store import QdrantStore


def test_broken_vector_reports_error_not_empty():
    store = QdrantStore(url="http://127.0.0.1:9", collection="x", dim=384, timeout=1)
    import numpy as np
    res = retrieve_vector_evidence("Diabetes is chronic.", store=store,
                                   embedder=None)
    # Either a clear error status or (if something listens) evidence — never
    # a silent fake-empty caused by swallowing the failure.
    if not res["evidence"]:
        assert res["diagnostics"].get("error"), "failure must be reported, not hidden"


def test_hybrid_survives_dead_qdrant_server(monkeypatch):
    get_ingestion_service().ingest_path("documents/medical", domain="Medical")
    from app.vector_store import qdrant_store as q_mod
    from app.ingestion import service as svc_mod
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    q_mod.reset_qdrant_store()
    svc_mod.reset_ingestion_service()
    res = retrieve_hybrid_evidence("Diabetes is a chronic disease.", top_k=3,
                                   retrieval_mode="hybrid",
                                   filters={"domain": "Medical"})
    methods = {m["method"]: m for m in res["diagnostics"]["per_method"]}
    assert methods["vector"]["status"] == "error"
    assert methods["vector"]["error"]
    # BM25/graph evidence still returned despite the vector outage.
    assert res["evidence_found"] is True
    assert any("bm25" in e["retrieval_methods"] or "graph" in e["retrieval_methods"]
               for e in res["evidence"])


def test_dim_mismatch_is_an_error_not_silent(tmp_path):
    store = QdrantStore(url=":memory:", collection="mismatch", dim=8)
    store.ensure_collection()
    store.dim = 16
    with pytest.raises(RuntimeError, match="mismatch"):
        store.ensure_collection()
