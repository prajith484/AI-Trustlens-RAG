"""AI TrustLens RAG service — evidence retrieval API (no LLM, no classifier)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api.routes import router
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation: fail LOUD, never silently serve a broken backend.
    # 1. BM25 file index (persists across processes already).
    try:
        from .retrieval.bm25_retriever import get_bm25_index
        n = get_bm25_index()
        print(f"[startup] bm25 index: {len(n)} chunks from {settings.bm25_index_path}")
    except Exception as exc:  # pragma: no cover
        print(f"[startup] bm25 warmup warning: {exc}")
    # 2. Embedding model (first load downloads/caches weights; warms here so
    #    the first /retrieve call is not mysteriously slow).
    try:
        from .embeddings.embedding_service import get_embedding_service
        info = get_embedding_service().warmup()
        print(f"[startup] embeddings: backend={info['backend']} model={info['model']}"
              + (f" error={info['error']}" if info["error"] else ""))
    except Exception as exc:  # pragma: no cover
        print(f"[startup] embedding warmup warning: {exc}")
    # 3. Qdrant collection: idempotent create + dim/distance validation.
    try:
        from .vector_store.qdrant_store import get_qdrant_store
        store = get_qdrant_store()
        # Align collection dim with the real embedding model before validating.
        from .embeddings.embedding_service import get_embedding_service as _emb
        try:
            store.dim = _emb().dimension()
        except Exception:
            pass
        result = store.ensure_collection()
        hc = store.health_check()
        print(f"[startup] qdrant: mode={hc['mode']} persistent={hc['persistent']} "
              f"collection={hc['collection']} points={hc['points']} init={result}")
    except Exception as exc:
        print(f"[startup] QDRANT INIT FAILED: {exc} "
              f"(vector retrieval will report errors, not empty results)")
    # 4. Graph snapshot / Neo4j (fallback is supported; just report it).
    try:
        from .graph_store.neo4j_store import get_graph_store
        ghc = get_graph_store().health_check()
        print(f"[startup] graph: backend={ghc['backend']} counts={ghc.get('counts')}")
    except Exception as exc:  # pragma: no cover
        print(f"[startup] graph warmup warning: {exc}")
    # 5. Document registry (rebuilt view of previously ingested documents).
    try:
        from .ingestion.service import get_ingestion_service
        docs = get_ingestion_service().list_documents()
        print(f"[startup] registry: {len(docs)} known documents")
    except Exception as exc:  # pragma: no cover
        print(f"[startup] registry warmup warning: {exc}")
    yield


app = FastAPI(
    title="AI TrustLens RAG — Evidence Retrieval",
    description=("Retrieves traceable evidence passages for factual claims. "
                 "Returns evidence only; verification/classification is handled "
                 "by other teammates' modules."),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root():
    return {"service": "AI TrustLens RAG", "version": "1.0.0",
            "endpoints": ["GET /health", "POST /retrieve",
                          "POST /retrieve/batch", "POST /documents/ingest",
                          "GET /documents", "GET /documents/{id}",
                          "DELETE /documents/{id}"]}
