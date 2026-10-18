"""Medical graph retrieval + persistence across 'restarts' (new singletons)."""
import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("rank_bm25")

from app.graph_store.entity_extractor import EntityExtractor
from app.ingestion.service import get_ingestion_service, reset_ingestion_service
from app.retrieval.graph_retriever import retrieve_graph_evidence


def test_medical_entities_extracted():
    ex = EntityExtractor()
    ents = {e.text.lower() for e in ex.extract_entities(
        "Diabetes is a chronic disease. The WHO tracks hypertension.")}
    assert "diabetes" in ents and "hypertension" in ents and "who" in ents


def test_graph_finds_medical_evidence_with_provenance():
    get_ingestion_service().ingest_path("documents/medical", domain="Medical")
    res = retrieve_graph_evidence("Diabetes is a chronic disease.", top_k=3,
                                  filters={"domain": "Medical"})
    assert res["evidence_found"] is True, res["diagnostics"]
    e = res["evidence"][0]
    assert e["text"] and e["graph_path"]
    assert e["source_url"] == "https://www.who.int/news-room/fact-sheets/detail/diabetes"
    assert e["domain"] == "Medical"


def test_graph_survives_restart(tmp_path, monkeypatch):
    from app.graph_store import neo4j_store as g_mod
    from app.retrieval import bm25_retriever
    from app.ingestion import service as svc_mod
    from app.vector_store import qdrant_store as q_mod
    snap = str(tmp_path / "g.json")
    qpath = str(tmp_path / "qdrant")
    reg = str(tmp_path / "reg.json")
    bm = str(tmp_path / "bm.pkl")
    for m in (monkeypatch.setenv("GRAPH_STORE_PATH", snap),
              monkeypatch.setenv("QDRANT_URL", qpath),
              monkeypatch.setenv("DOCUMENTS_PATH", reg),
              monkeypatch.setenv("BM25_INDEX_PATH", bm)):
        pass
    svc = get_ingestion_service()
    svc.ingest_path("documents/medical", domain="Medical")
    n_docs = len(svc.list_documents())
    assert n_docs == 3
    # Simulate API restart: drop all singletons, rebuild from disk.
    bm25_retriever.reset_bm25_index()
    svc_mod.reset_ingestion_service()
    q_mod.reset_qdrant_store()
    g_mod.reset_graph_store()
    svc2 = get_ingestion_service()
    assert len(svc2.list_documents()) == n_docs  # registry reloaded
    res = retrieve_graph_evidence("What is hypertension?", top_k=3,
                                  filters={"domain": "Medical"})
    assert res["evidence_found"] is True
    assert "hypertension" in res["evidence"][0]["text"].lower()
