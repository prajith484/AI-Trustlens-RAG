"""Pydantic schemas: the stable integration contract for LLM/classifier teammates.

The RAG module returns EVIDENCE ONLY — never verification labels, trust
scores, or generated explanations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

RetrievalMode = Literal["vector", "bm25", "graph", "hybrid"]


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str = Field(description="Original source passage, never a paraphrase.")
    title: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    page_number: Optional[int] = None
    domain: Optional[str] = None
    retrieval_methods: List[str] = Field(default_factory=list)
    retrieval_score: Optional[float] = None
    graph_path: List[str] = Field(default_factory=list)


class RetrieveRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_mode: RetrievalMode = "hybrid"
    filters: Optional[Dict[str, Any]] = None
    domain: Optional[str] = Field(default=None,
        description="Convenience filter, e.g. 'Medical'. Merged into filters; "
                    "explicit filters take precedence.")
    graph_max_depth: Optional[int] = Field(default=None, ge=1, le=5)

    @field_validator("claim")
    @classmethod
    def _strip_claim(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("claim must be a non-empty string")
        return v


class RetrieveResponse(BaseModel):
    claim: str
    retrieval_mode: str
    evidence: List[EvidenceItem] = Field(default_factory=list)
    evidence_found: bool = False
    latency_ms: int = 0
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class BatchRetrieveRequest(BaseModel):
    claims: List[str] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=5, ge=1, le=50)
    retrieval_mode: RetrievalMode = "hybrid"
    filters: Optional[Dict[str, Any]] = None
    domain: Optional[str] = Field(default=None,
        description="Convenience filter applied to every claim.")

    @field_validator("claims")
    @classmethod
    def _validate_claims(cls, v: List[str]) -> List[str]:
        cleaned = [c.strip() for c in v]
        if any(not c for c in cleaned):
            raise ValueError("claims must all be non-empty strings")
        return cleaned


class BatchRetrieveResponse(BaseModel):
    results: List[RetrieveResponse]
    retrieval_mode: str


class IngestRequest(BaseModel):
    path: Optional[str] = Field(default=None, description="File or directory path on server")
    title: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    domain: Optional[str] = None
    document_type: Optional[str] = None


class DocumentInfo(BaseModel):
    document_id: str
    title: Optional[str] = None
    source_name: Optional[str] = None
    source_url: Optional[str] = None
    publication_date: Optional[str] = None
    ingestion_date: Optional[str] = None
    domain: Optional[str] = None
    document_type: Optional[str] = None
    content_hash: Optional[str] = None
    chunk_count: int = 0


class HealthResponse(BaseModel):
    status: str
    qdrant: Dict[str, Any] = Field(default_factory=dict)
    neo4j: Dict[str, Any] = Field(default_factory=dict)
    embedding_model: str = ""
    embedding_backend: str = ""
    bm25_index_size: int = 0
    warnings: List[str] = Field(default_factory=list)
