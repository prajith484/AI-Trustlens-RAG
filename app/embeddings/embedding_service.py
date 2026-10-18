"""Embedding service: Sentence Transformers (all-MiniLM-L6-v2 default).

- Model loaded ONCE (singleton) and reused for chunks + claims.
- Same model for documents and claims (compatibility enforced).
- Batch embedding with validation of empty/invalid text.
- Offline-safe fallback: if sentence-transformers/torch or the model weights
  are unavailable, a deterministic hash-based embedding is used so tests and
  local development still work. The active backend is exposed via
  ``EmbeddingService.backend`` ("sentence-transformers" | "hash-fallback").

Similarity metric: COSINE (vectors L2-normalised; Qdrant collection uses
COSINE distance). Documented here and in README.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Dict, List

import numpy as np

_lock = threading.Lock()
_instance = None


def _hash_vector(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic pseudo-embedding from token hashes (fallback only)."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h >> 16) % dim] += 0.5
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self.backend = "uninitialised"
        self._model = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> None:
        if self.backend != "uninitialised":
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
            # Infer dim from model if possible (new API name with fallback).
            try:
                if hasattr(self._model, "get_embedding_dimension"):
                    d = self._model.get_embedding_dimension()
                else:  # older sentence-transformers
                    d = self._model.get_sentence_embedding_dimension()
                if d:
                    self.dim = int(d)
            except Exception:
                pass
            self.backend = "sentence-transformers"
        except Exception as exc:  # offline / missing dep -> fallback
            self._model = None
            self._load_error = str(exc)
            self.backend = "hash-fallback"

    def warmup(self) -> Dict[str, str | None]:
        """Load the model eagerly (used at API startup) and report the backend."""
        self._ensure_loaded()
        return {"backend": self.backend, "model": self.model_name,
                "error": self._load_error}

    def status(self) -> Dict[str, str | int | None]:
        return {"backend": self.backend if self.backend != "uninitialised" else "not-loaded",
                "model": self.model_name, "dim": self.dim,
                "error": self._load_error}

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _validate(self, texts: List[str]) -> List[str]:
        cleaned: List[str] = []
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                raise ValueError("embedding text must be a non-empty string")
            cleaned.append(t.strip())
        return cleaned

    def embed(self, texts: List[str]) -> np.ndarray:
        texts = self._validate(texts)
        self._ensure_loaded()
        if self.backend == "sentence-transformers":
            try:
                vecs = self._model.encode(texts, batch_size=32, normalize_embeddings=True,
                                          show_progress_bar=False)
                return np.asarray(vecs, dtype=np.float32)
            except Exception as exc:
                raise RuntimeError(f"embedding inference failed: {exc}") from exc
        return np.stack([_hash_vector(t, self.dim) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def dimension(self) -> int:
        self._ensure_loaded()
        return self.dim


def get_embedding_service(model_name: str | None = None, dim: int = 384) -> EmbeddingService:
    """Process-wide singleton (model loaded once)."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = EmbeddingService(model_name or _default_model_name(), dim)
        elif model_name and _instance.model_name != model_name:
            # Re-create if a different model explicitly requested
            _instance = EmbeddingService(model_name, dim)
        return _instance


def _default_model_name() -> str:
    import os
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def reset_embedding_service() -> None:
    global _instance
    with _lock:
        _instance = None
