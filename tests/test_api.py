import pytest

rank_bm25 = pytest.importorskip("rank_bm25")
pytest.importorskip("qdrant_client")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from app.ingestion.service import get_ingestion_service
from app.main import app


@pytest.fixture()
def client():
    get_ingestion_service().ingest_path("documents")
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "qdrant" in body and "neo4j" in body


def test_retrieve_schema(client):
    r = client.post("/retrieve", json={"claim": "Python supports object-oriented programming.",
                                       "top_k": 2, "retrieval_mode": "hybrid"})
    assert r.status_code == 200
    body = r.json()
    assert body["claim"] and body["retrieval_mode"] == "hybrid"
    assert isinstance(body["evidence"], list)
    assert "evidence_found" in body and "latency_ms" in body
    assert "verification" not in body and "trust_score" not in body  # no classifier leakage


def test_retrieve_validation(client):
    r = client.post("/retrieve", json={"claim": "   "})
    assert r.status_code == 422


def test_batch_and_doc_crud(client, tmp_path):
    r = client.post("/retrieve/batch", json={"claims": ["Python supports OOP.", "What is Qdrant?"],
                                             "top_k": 1})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
    p = tmp_path / "extra.txt"
    p.write_text("TrustLens audit evidence must link back to source passages.", encoding="utf-8")
    r2 = client.post("/documents/ingest", json={"path": str(p)})
    assert r2.status_code == 200
    docs = client.get("/documents").json()
    assert len(docs) >= 1
    did = docs[-1]["document_id"]
    assert client.get(f"/documents/{did}").status_code == 200
    assert client.delete(f"/documents/{did}").status_code == 200
    assert client.get(f"/documents/{did}").status_code == 404
