"""Shared fixtures: isolate singletons + temp backend paths.

Settings are read lazily from the environment, so monkeypatched paths take
effect for every store/singleton created inside a test.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25_test.pkl"))
    monkeypatch.setenv("GRAPH_STORE_PATH", str(tmp_path / "graph_test.json"))
    monkeypatch.setenv("DOCUMENTS_PATH", str(tmp_path / "docs_test.json"))
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    from app.retrieval import bm25_retriever
    from app.ingestion import service as svc_mod
    from app.embeddings import embedding_service as emb_mod
    from app.vector_store import qdrant_store as q_mod
    from app.graph_store import neo4j_store as g_mod
    bm25_retriever.reset_bm25_index()
    svc_mod.reset_ingestion_service()
    emb_mod.reset_embedding_service()
    q_mod.reset_qdrant_store()
    g_mod.reset_graph_store()
    yield
    bm25_retriever.reset_bm25_index()
    svc_mod.reset_ingestion_service()
    emb_mod.reset_embedding_service()
    q_mod.reset_qdrant_store()
    g_mod.reset_graph_store()
