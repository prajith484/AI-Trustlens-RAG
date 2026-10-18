"""Provenance: headers -> ingestion -> retrieval -> API response."""
from pathlib import Path

from fastapi.testclient import TestClient

from app.ingestion.loaders import load_file, parse_header_block
from app.ingestion.service import get_ingestion_service
from app.main import app


def test_parse_header_block(tmp_path):
    headers, body = parse_header_block(
        "Source: World Health Organization (WHO)\nTitle: Diabetes\n"
        "URL: https://www.who.int/x\n\nBody text here.")
    assert headers == {"source_name": "World Health Organization (WHO)",
                       "title": "Diabetes", "source_url": "https://www.who.int/x"}
    assert body == "Body text here."
    # Unknown keys stop the block; nothing invented.
    h2, b2 = parse_header_block("Foo: bar\nReal text")
    assert h2 == {} and "Real text" in b2


def test_txt_headers_become_metadata_not_evidence(tmp_path):
    p = tmp_path / "med.txt"
    p.write_text("Source: WHO\nTitle: Diabetes\nURL: https://www.who.int/x\n\n"
                 "Diabetes is a chronic disease.", encoding="utf-8")
    doc = load_file(p, domain="Medical")
    assert doc.source_url == "https://www.who.int/x"
    assert doc.title == "Diabetes" and doc.domain == "Medical"
    assert "URL:" not in doc.text and "Source:" not in doc.text
    # Explicit overrides win over headers.
    doc2 = load_file(p, domain="Medical", source_url="https://override.example/")
    assert doc2.source_url == "https://override.example/"


def test_stable_ids_on_reingest(tmp_path):
    p = tmp_path / "med.txt"
    p.write_text("Title: T\nURL: https://x.example/\n\nSome body.", encoding="utf-8")
    assert load_file(p).document_id == load_file(p).document_id


def test_source_url_reaches_api_response():
    svc = get_ingestion_service()
    svc.ingest_path("documents/medical", domain="Medical")
    client = TestClient(app)
    r = client.post("/retrieve", json={
        "claim": "Diabetes is a chronic disease.", "top_k": 3,
        "retrieval_mode": "hybrid", "domain": "Medical"})
    assert r.status_code == 200
    body = r.json()
    assert body["evidence_found"] is True
    urls = [e["source_url"] for e in body["evidence"]]
    assert "https://www.who.int/news-room/fact-sheets/detail/diabetes" in urls
    assert all(e["domain"] == "Medical" for e in body["evidence"])
