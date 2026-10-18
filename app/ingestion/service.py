"""Ingestion pipeline: load -> clean -> chunk -> embed -> index (Qdrant/BM25/graph).

Supports incremental add, update (re-ingest), delete. Duplicate detection via
content_hash: re-ingesting identical content is a no-op.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..embeddings.embedding_service import get_embedding_service
from ..graph_store.graph_builder import GraphBuilder
from ..graph_store.neo4j_store import Neo4jStore
from ..retrieval.bm25_retriever import get_bm25_index
from ..vector_store.qdrant_store import QdrantStore
from .chunking import Chunk, chunk_document
from .loaders import RawDocument, load_directory, load_file


class IngestionService:
    def __init__(self, qdrant: QdrantStore | None = None,
                 graph: Neo4jStore | None = None):
        from app.config import settings
        self.settings = settings
        if qdrant is not None:
            self.qdrant = qdrant
        else:
            from ..vector_store.qdrant_store import get_qdrant_store
            self.qdrant = get_qdrant_store()
        if graph is not None:
            self.graph = graph
        else:
            from ..graph_store.neo4j_store import get_graph_store
            self.graph = get_graph_store()
        self.builder = GraphBuilder(self.graph)
        self.documents: Dict[str, Dict] = {}  # document_id -> info (hash, meta, chunks)
        self._load_registry()

    # -- registry persistence (documents survive API restarts) ------------
    def _registry_path(self) -> str:
        return self.settings.documents_path

    def _load_registry(self) -> int:
        path = self._registry_path()
        if not path:
            return 0
        try:
            import json
            from pathlib import Path as _P
            p = _P(path)
            if not p.exists():
                return 0
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self.documents = data
            return len(self.documents)
        except Exception as exc:
            print(f"[ingest] registry load failed: {exc}")
            return 0

    def _save_registry(self) -> None:
        path = self._registry_path()
        if not path:
            return
        try:
            import json
            from pathlib import Path as _P
            p = _P(path)
            if str(p.parent) not in (".", ""):
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.documents, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[ingest] registry save failed: {exc}")

    # -- core ------------------------------------------------------------
    def ingest_document(self, doc: RawDocument) -> Dict:
        if not doc.text or not doc.text.strip():
            raise ValueError(f"document {doc.document_id} has no extractable text")
        # duplicate detection
        prev = self.documents.get(doc.document_id)
        if prev and prev.get("content_hash") == doc.content_hash:
            return {"document_id": doc.document_id, "status": "duplicate-skipped",
                    "chunks": prev.get("chunk_count", 0)}
        chunks: List[Chunk] = chunk_document(
            doc, chunk_size_words=self.settings.chunk_size_words,
            chunk_overlap_words=self.settings.chunk_overlap_words)
        if not chunks:
            raise ValueError(f"document {doc.document_id} produced no chunks")
        embedder = get_embedding_service()
        vectors = embedder.embed([c.text for c in chunks])
        # adapt qdrant dim to real model dim on first write
        self.qdrant.dim = int(vectors.shape[1])
        n = self.qdrant.upsert_chunks(chunks, vectors, content_hash=doc.content_hash)
        get_bm25_index().add_or_update([c.to_payload() for c in chunks])
        meta = {"document_id": doc.document_id, "title": doc.title,
                "source_name": doc.source_name, "source_url": doc.source_url,
                "publication_date": doc.publication_date, "domain": doc.domain,
                "document_type": doc.document_type, "content_hash": doc.content_hash,
                "ingestion_date": doc.ingestion_date}
        graph_stats = self.builder.add_document(meta, chunks)
        self.documents[doc.document_id] = {**meta, "chunk_count": len(chunks)}
        self._save_registry()
        return {"document_id": doc.document_id, "status": "ingested",
                "chunks": n, **graph_stats}

    def ingest_path(self, path: str, recursive: bool = False, **overrides) -> List[Dict]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"path not found: {path}")
        docs = load_directory(p, recursive=recursive, **overrides) \
            if p.is_dir() else [load_file(p, **overrides)]
        results = []
        for d in docs:
            try:
                results.append(self.ingest_document(d))
            except Exception as exc:
                results.append({"document_id": getattr(d, "document_id", str(path)),
                                "status": f"error: {exc}", "chunks": 0})
        return results

    def get_document(self, document_id: str) -> Optional[Dict]:
        return self.documents.get(document_id)

    def list_documents(self) -> List[Dict]:
        return list(self.documents.values())

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        try:
            self.qdrant.delete_by_document(document_id)
        except Exception as exc:
            print(f"[ingest] qdrant delete warning: {exc}")
        try:
            get_bm25_index().remove_document(document_id)
        except Exception as exc:
            print(f"[bm25] delete warning: {exc}")
        self.builder.remove_document(document_id)
        self.documents.pop(document_id, None)
        self._save_registry()
        return True


_singleton: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    global _singleton
    if _singleton is None:
        _singleton = IngestionService()
    return _singleton


def reset_ingestion_service() -> None:
    global _singleton
    _singleton = None
