"""Central configuration for AI TrustLens RAG subsystem.

All settings are read LAZILY from environment variables (properties evaluated
on each access) so that a `.env` file, test monkeypatching, or process env
changes always take effect. The `.env` file in the project root is loaded on
import via python-dotenv when available.

Docker-free default: Qdrant uses persistent LOCAL file storage
(QDRANT_URL=./qdrant_storage), BM25 uses a pickle file, and the graph uses a
JSON snapshot — so ingest → restart API → retrieve works across processes.
See .env.example.
"""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _getenv(name: str, default: str) -> str:
    return os.getenv(name, default)


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


class Settings:
    """Lazy environment-backed settings (attribute access reads os.environ)."""

    # -- Embeddings --
    @property
    def embedding_model(self) -> str:
        return _getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def embedding_dim(self) -> int:
        return _getenv_int("EMBEDDING_DIM", 384)

    @property
    def similarity_metric(self) -> str:
        return _getenv("SIMILARITY_METRIC", "cosine")  # cosine (documented)

    # -- Chunking (defaults: ~200 words, 30 overlap) --
    @property
    def chunk_size_words(self) -> int:
        return _getenv_int("CHUNK_SIZE", 200)

    @property
    def chunk_overlap_words(self) -> int:
        return _getenv_int("CHUNK_OVERLAP", 30)

    # -- Qdrant --
    # Default is persistent local file storage (no Docker needed).
    # ":memory:" opts into non-persistent mode (data lost on process exit).
    # "http(s)://..." selects server mode. "local" uses QDRANT_PATH.
    @property
    def qdrant_url(self) -> str:
        return _getenv("QDRANT_URL", "./qdrant_storage")

    @property
    def qdrant_api_key(self) -> str:
        return _getenv("QDRANT_API_KEY", "")

    @property
    def qdrant_collection(self) -> str:
        return _getenv("QDRANT_COLLECTION", "trustlens_chunks")

    @property
    def qdrant_path(self) -> str:
        return _getenv("QDRANT_PATH", "./qdrant_storage")

    @property
    def qdrant_timeout(self) -> int:
        return _getenv_int("QDRANT_TIMEOUT", 5)

    # -- Neo4j (optional; memory+JSON-snapshot fallback when unreachable) --
    @property
    def neo4j_uri(self) -> str:
        return _getenv("NEO4J_URI", "bolt://localhost:7687")

    @property
    def neo4j_username(self) -> str:
        return _getenv("NEO4J_USERNAME", "neo4j")

    @property
    def neo4j_password(self) -> str:
        return _getenv("NEO4J_PASSWORD", "trustlens")

    @property
    def neo4j_timeout(self) -> int:
        return _getenv_int("NEO4J_TIMEOUT", 5)

    @property
    def graph_store_path(self) -> str:
        return _getenv("GRAPH_STORE_PATH", "./graph_store.json")

    # -- Retrieval --
    @property
    def top_k(self) -> int:
        return _getenv_int("TOP_K", 5)

    @property
    def retrieval_mode(self) -> str:
        return _getenv("RETRIEVAL_MODE", "hybrid")

    @property
    def graph_max_depth(self) -> int:
        return _getenv_int("GRAPH_MAX_DEPTH", 2)

    @property
    def rrf_k(self) -> int:
        return _getenv_int("RRF_K", 60)

    # -- BM25 index persistence: path, or empty string = in-memory only --
    @property
    def bm25_index_path(self) -> str:
        return _getenv("BM25_INDEX_PATH", "./bm25_index.pkl")

    # -- Document registry persistence (survives API restarts) --
    @property
    def documents_path(self) -> str:
        return _getenv("DOCUMENTS_PATH", "./documents_registry.json")

    # -- API --
    @property
    def api_host(self) -> str:
        return _getenv("API_HOST", "127.0.0.1")

    @property
    def api_port(self) -> int:
        return _getenv_int("API_PORT", 8000)


settings = Settings()
