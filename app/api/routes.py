"""FastAPI routes: retrieval + document management. No LLM, no classifier."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..embeddings.embedding_service import get_embedding_service
from ..ingestion.loaders import load_directory, load_file
from ..ingestion.service import get_ingestion_service
from ..retrieval.bm25_retriever import get_bm25_index
from ..retrieval.hybrid_retriever import retrieve_evidence_for_claims, retrieve_hybrid_evidence
from .schemas import (BatchRetrieveRequest, BatchRetrieveResponse, DocumentInfo,
                      HealthResponse, IngestRequest, RetrieveRequest, RetrieveResponse)

router = APIRouter()


def _effective_filters(req) -> Optional[Dict[str, Any]]:
    """Merge top-level `domain` into filters (explicit filters win)."""
    filters = dict(req.filters or {})
    if getattr(req, "domain", None) is not None and "domain" not in filters:
        filters["domain"] = req.domain
    return filters or None


@router.get("/health", response_model=HealthResponse)
def health():
    from ..embeddings.embedding_service import get_embedding_service
    svc = get_ingestion_service()
    qdrant = svc.qdrant.health_check()
    neo4j = svc.graph.health_check()
    emb = get_embedding_service()
    emb_status = {"backend": emb.backend, "model": emb.model_name}
    warnings = []
    if not qdrant.get("ok"):
        warnings.append(f"qdrant unavailable: {qdrant.get('error')}")
    if qdrant.get("ok") and not qdrant.get("persistent"):
        warnings.append("qdrant running in :memory: mode — data is lost on restart")
    if emb.backend == "hash-fallback":
        warnings.append("embedding hash-fallback active — semantic quality degraded")
    elif emb.backend == "uninitialised":
        warnings.append("embedding model not loaded yet (loads on first use)")
    status = "ok" if qdrant.get("ok") else "degraded"
    return HealthResponse(
        status=status,
        qdrant=qdrant,
        neo4j=neo4j,
        embedding_model=emb.model_name,
        embedding_backend=emb_status["backend"],
        bm25_index_size=len(get_bm25_index()),
        warnings=warnings,
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    try:
        res = retrieve_hybrid_evidence(
            req.claim, top_k=req.top_k, filters=_effective_filters(req),
            retrieval_mode=req.retrieval_mode, graph_max_depth=req.graph_max_depth)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"retrieval failed: {exc}")
    return RetrieveResponse(**res)


@router.post("/retrieve/batch", response_model=BatchRetrieveResponse)
def retrieve_batch(req: BatchRetrieveRequest):
    results = retrieve_evidence_for_claims(
        req.claims, top_k=req.top_k, filters=_effective_filters(req),
        retrieval_mode=req.retrieval_mode)
    return BatchRetrieveResponse(
        results=[RetrieveResponse(**r) for r in results],
        retrieval_mode=req.retrieval_mode)


@router.post("/documents/ingest")
def ingest_documents(req: IngestRequest):
    svc = get_ingestion_service()
    if not req.path:
        raise HTTPException(status_code=422, detail="path is required (file or directory)")
    overrides = {k: v for k, v in {
        "title": req.title, "source_name": req.source_name,
        "source_url": req.source_url, "publication_date": req.publication_date,
        "domain": req.domain, "document_type": req.document_type}.items() if v is not None}
    try:
        results = svc.ingest_path(req.path, **overrides)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"ingestion failed: {exc}")
    return {"ingested": results, "count": len(results)}


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents():
    return [DocumentInfo(**d) for d in get_ingestion_service().list_documents()]


@router.get("/documents/{document_id}", response_model=DocumentInfo)
def get_document(document_id: str):
    doc = get_ingestion_service().get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentInfo(**doc)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    ok = get_ingestion_service().delete_document(document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="document not found")
    return {"deleted": document_id}
