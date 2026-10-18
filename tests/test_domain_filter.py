"""Domain filtering: Medical-only queries must exclude General documents."""
import pytest

rank_bm25 = pytest.importorskip("rank_bm25")
pytest.importorskip("qdrant_client")

from app.ingestion.service import get_ingestion_service
from app.retrieval.hybrid_retriever import retrieve_hybrid_evidence


@pytest.fixture()
def mixed_corpus():
    svc = get_ingestion_service()
    svc.ingest_path("documents")  # general docs (top level only)
    svc.ingest_path("documents/medical", domain="Medical")
    return svc


@pytest.mark.parametrize("mode", ["vector", "bm25", "graph", "hybrid"])
def test_medical_only_excludes_general(mixed_corpus, mode):
    res = retrieve_hybrid_evidence(
        "Diabetes is a chronic disease.", top_k=5, retrieval_mode=mode,
        filters={"domain": "Medical"})
    assert res["evidence_found"] is True
    assert res["evidence"], f"{mode} returned no medical evidence"
    assert all(e["domain"] == "Medical" for e in res["evidence"]), \
        f"{mode} leaked: {[e['domain'] for e in res['evidence']]}"


def test_no_domain_preserves_default(mixed_corpus):
    res = retrieve_hybrid_evidence("Python supports object-oriented programming.",
                                   top_k=3, retrieval_mode="hybrid")
    assert res["evidence_found"] is True  # unfiltered default behaviour kept


def test_filters_dict_style_still_works(mixed_corpus):
    res = retrieve_hybrid_evidence("What is hypertension?", top_k=5,
                                   retrieval_mode="hybrid",
                                   filters={"domain": "Medical"})
    assert all(e["domain"] == "Medical" for e in res["evidence"])
