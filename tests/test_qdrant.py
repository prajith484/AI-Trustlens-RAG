import numpy as np
import pytest

qdrant = pytest.importorskip("qdrant_client")

from app.ingestion.chunking import Chunk
from app.vector_store.qdrant_store import QdrantStore


def _chunk(cid="doc1::c0000::aaa", did="doc1", text="Python supports OOP"):
    return Chunk(chunk_id=cid, document_id=did, text=text, title="T",
                 source_name="s", domain="CS")


def test_upsert_search_filter_delete():
    store = QdrantStore(url=":memory:", collection="test_chunks", dim=8)
    chunks = [_chunk("doc1::c0000::aaa", "doc1", "Python supports object oriented programming"),
              _chunk("doc2::c0000::bbb", "doc2", "Qdrant is a vector database", )]
    for c in chunks:
        c.domain = "CS" if c.document_id == "doc1" else "AI"
    vecs = np.eye(8, dtype=np.float32)[:2]
    assert store.upsert_chunks(chunks, vecs, content_hash="h1") == 2
    hits = store.search(vecs[0], top_k=5)
    assert hits and hits[0]["chunk_id"] == chunks[0].chunk_id
    assert hits[0]["text"] == chunks[0].text  # original text preserved
    filt = store.search(vecs[0], top_k=5, filters={"domain": "AI"})
    assert all(h["domain"] == "AI" for h in filt)
    store.delete_by_document("doc1")
    assert store.health_check()["ok"] is True


def test_health_ok():
    store = QdrantStore(url=":memory:", collection="hc", dim=8)
    assert store.health_check()["ok"] is True
