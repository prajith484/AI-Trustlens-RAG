"""Builds the knowledge graph from chunks + extraction results."""
from __future__ import annotations

from typing import Dict, List

from ..ingestion.chunking import Chunk
from .entity_extractor import EntityExtractor, ExtractionResult
from .neo4j_store import Neo4jStore


class GraphBuilder:
    def __init__(self, store: Neo4jStore, extractor: EntityExtractor | None = None):
        self.store = store
        self.extractor = extractor or EntityExtractor()

    def add_document(self, meta: Dict, chunks: List[Chunk]) -> Dict[str, int]:
        """Merge a document, its chunks, entities and relationships. Returns counts."""
        self.store.merge_document(meta)
        n_ent = n_rel = 0
        for ch in chunks:
            self.store.merge_chunk(ch.to_payload())
            res: ExtractionResult = self.extractor.extract(
                ch.text, document_id=ch.document_id, chunk_id=ch.chunk_id)
            for ent in res.entities:
                self.store.merge_entity(ent, ch.chunk_id, ch.document_id)
                n_ent += 1
            for rel in res.relationships:
                self.store.merge_relationship(rel)
                n_rel += 1
        return {"chunks": len(chunks), "entities": n_ent, "relationships": n_rel}

    def remove_document(self, document_id: str) -> None:
        self.store.delete_document(document_id)
