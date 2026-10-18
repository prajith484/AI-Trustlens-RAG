"""Qdrant vector store behind a reusable interface.

Modes (QDRANT_URL):
- Local file mode (DEFAULT, Docker-free): QDRANT_URL=./qdrant_storage
  (persistent on-disk storage via QdrantClient(path=...)). Ingestion in one
  process is visible to the API in another, and survives restarts.
- Memory mode: QDRANT_URL=:memory: — NON-persistent; data is lost when the
  process exits. Suitable for unit tests only.
- Server mode: QDRANT_URL=http://localhost:6333 (docker-compose).

Collection uses COSINE distance. Each point id is a deterministic UUID5 of the
chunk_id; payload stores the full chunk + metadata. Duplicate prevention via
upsert on the same point id + content_hash tracking per document.

A missing collection is created idempotently by ensure_collection(), which
also VALIDATES an existing collection's vector size and distance and raises a
clear error on mismatch (never silently re-creates and drops data).
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..ingestion.chunking import Chunk


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, url: str = "./qdrant_storage", collection: str = "trustlens_chunks",
                 dim: int = 384, path: str = "./qdrant_storage",
                 api_key: str = "", timeout: int = 5):
        self.url = url
        self.collection = collection
        self.dim = dim
        self.path = path
        self.api_key = api_key
        self.timeout = timeout
        self._client = None
        self._last_error: str | None = None
        self._doc_hashes: Dict[str, str] = {}  # document_id -> content_hash

    # -- connection -----------------------------------------------------
    def _connect(self):
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient  # type: ignore
        if self.url in ("", ":memory:", "memory"):
            self._client = QdrantClient(":memory:")
        elif self.url.startswith(("http://", "https://")):
            kwargs: Dict[str, Any] = {"url": self.url, "timeout": self.timeout}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            # Fail fast when no server is running (no long retries).
            kwargs.setdefault("check_compatibility", False)
            self._client = QdrantClient(**kwargs)
        else:
            # treat as local file path (persistent, Docker-free)
            self._client = QdrantClient(path=self.url if self.url != "local" else self.path)
        return self._client

    def is_persistent(self) -> bool:
        return self.url not in ("", ":memory:", "memory")

    def close(self) -> None:
        """Release the client (avoids shutdown noise for local-file mode)."""
        try:
            if self._client is not None and hasattr(self._client, "close"):
                self._client.close()
        except Exception:
            pass
        finally:
            self._client = None

    def health_check(self) -> Dict[str, Any]:
        try:
            client = self._connect()
            try:
                info = client.get_collection(self.collection)
                count = info.points_count
                validation = self._validate_info(info)
            except Exception as exc:
                count = None
                validation = {"exists": False, "note": f"collection missing: {exc}"}
            return {"ok": True, "mode": self._describe_mode(),
                    "persistent": self.is_persistent(),
                    "collection": self.collection,
                    "points": count, "validation": validation, "error": None}
        except Exception as exc:
            self._last_error = str(exc)
            return {"ok": False, "mode": self._describe_mode(),
                    "persistent": self.is_persistent(),
                    "collection": self.collection,
                    "points": None, "validation": {"exists": False},
                    "error": str(exc)}

    def _describe_mode(self) -> str:
        if self.url in ("", ":memory:", "memory"):
            return "memory"
        if self.url.startswith(("http://", "https://")):
            return f"server:{self.url}"
        return f"local:{self.url}"

    # -- collection -----------------------------------------------------
    def _validate_info(self, info: Any) -> Dict[str, Any]:
        """Check an existing collection's vector size/distance against config."""
        try:
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):  # named vectors
                return {"exists": True, "note": "named vectors; size check skipped",
                        "expected_dim": self.dim}
            size = getattr(vectors, "size", None)
            distance = str(getattr(vectors, "distance", "")).split(".")[-1]
            ok = (size == self.dim) and (distance.upper() == "COSINE" or not distance)
            return {"exists": True, "dim": size, "distance": distance,
                    "expected_dim": self.dim, "expected_distance": "COSINE",
                    "valid": ok}
        except Exception as exc:
            return {"exists": True, "valid": False, "note": f"validation failed: {exc}"}

    def ensure_collection(self) -> Dict[str, Any]:
        """Idempotent create + validate. Raises on config mismatch (no silent drop)."""
        from qdrant_client.models import Distance, VectorParams  # type: ignore
        client = self._connect()
        try:
            info = client.get_collection(self.collection)
            validation = self._validate_info(info)
            if validation.get("exists") and validation.get("valid") is False:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' config mismatch: "
                    f"has dim={validation.get('dim')} distance={validation.get('distance')}, "
                    f"expected dim={self.dim} COSINE. Delete/reset the collection "
                    f"explicitly if re-indexing with a new model.")
            return {"created": False, **validation}
        except RuntimeError:
            raise
        except Exception:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            return {"created": True, "dim": self.dim, "distance": "COSINE"}

    def reset_collection(self) -> None:
        client = self._connect()
        try:
            client.delete_collection(self.collection)
        except Exception:
            pass
        self.ensure_collection()
        self._doc_hashes.clear()

    # -- writes ----------------------------------------------------------
    def upsert_chunks(self, chunks: List[Chunk], vectors: np.ndarray,
                      content_hash: str = "") -> int:
        """Insert/update chunks. Skips write if same document+hash already stored."""
        if len(chunks) == 0:
            return 0
        if chunks and chunks[0].document_id in self._doc_hashes \
                and content_hash and self._doc_hashes[chunks[0].document_id] == content_hash:
            return 0  # duplicate prevention (same session)
        from qdrant_client.models import PointStruct  # type: ignore
        self.ensure_collection()
        client = self._connect()
        points = []
        for ch, vec in zip(chunks, vectors):
            points.append(PointStruct(
                id=_point_id(ch.chunk_id),
                vector=np.asarray(vec, dtype=float).tolist(),
                payload=ch.to_payload(),
            ))
        # batch in groups of 100
        for i in range(0, len(points), 100):
            client.upsert(collection_name=self.collection, points=points[i:i + 100])
        if chunks and content_hash:
            self._doc_hashes[chunks[0].document_id] = content_hash
        return len(points)

    def delete_by_document(self, document_id: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        client = self._connect()
        try:
            client.delete(
                collection_name=self.collection,
                points_selector=Filter(must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id))]),
            )
        except Exception as exc:
            raise RuntimeError(f"Qdrant delete failed: {exc}") from exc
        self._doc_hashes.pop(document_id, None)
        return 1

    # -- reads -----------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int = 5,
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore
        self.ensure_collection()
        client = self._connect()
        qfilter = None
        if filters:
            conds = []
            for k in ("domain", "document_id", "source_name"):
                if k in filters and filters[k] is not None:
                    conds.append(FieldCondition(key=k, match=MatchValue(value=filters[k])))
            if conds:
                qfilter = Filter(must=conds)
        try:
            t0 = time.perf_counter()
            hits = client.query_points(
                collection_name=self.collection,
                query=np.asarray(query_vector, dtype=float).tolist(),
                limit=top_k,
                with_payload=True,
                query_filter=qfilter,
            )
            latency = (time.perf_counter() - t0) * 1000
            # qdrant-client>=1.10 returns QueryResponse(points=[...])
            points = getattr(hits, "points", hits)
            out = []
            for h in points:
                payload = dict(getattr(h, "payload", {}) or {})
                payload["_score"] = float(getattr(h, "score", 0.0))
                payload["_latency_ms"] = latency
                out.append(payload)
            return out
        except Exception as exc:
            raise RuntimeError(f"Qdrant search failed: {exc}") from exc


# Process-wide singleton: ingestion and retrieval must share one store,
# otherwise :memory:/local modes would diverge between writer and reader.
_singleton: "QdrantStore | None" = None


def get_qdrant_store(url: str | None = None, collection: str | None = None,
                     dim: int | None = None, path: str | None = None,
                     api_key: str | None = None,
                     timeout: int | None = None) -> "QdrantStore":
    global _singleton
    if _singleton is None:
        from app.config import settings
        _singleton = QdrantStore(
            url=url or settings.qdrant_url,
            collection=collection or settings.qdrant_collection,
            dim=dim or settings.embedding_dim,
            path=path or settings.qdrant_path,
            api_key=api_key if api_key is not None else settings.qdrant_api_key,
            timeout=timeout if timeout is not None else settings.qdrant_timeout)
    return _singleton


def reset_qdrant_store() -> None:
    global _singleton
    if _singleton is not None:
        try:
            _singleton.close()
        except Exception:
            pass
    _singleton = None
