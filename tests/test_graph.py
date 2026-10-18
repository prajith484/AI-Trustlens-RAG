from app.graph_store.entity_extractor import Entity
from app.graph_store.neo4j_store import Neo4jStore


def test_memory_graph_roundtrip():
    store = Neo4jStore(uri="bolt://localhost:1", username="x", password="y")  # unreachable -> fallback
    assert store.backend == "memory"
    assert store.health_check()["ok"] is True
    store.merge_document({"document_id": "d1", "title": "T", "source_name": "s",
                          "domain": "CS", "publication_date": None, "content_hash": "h"})
    store.merge_chunk({"chunk_id": "c1", "document_id": "d1", "text": "Python uses Qdrant",
                       "title": "T", "chunk_position": 0, "page_number": None})
    store.merge_entity(Entity("Python", "ProgrammingLanguage"), "c1", "d1")
    store.merge_entity(Entity("Qdrant", "Technology"), "c1", "d1")
    from app.graph_store.entity_extractor import Relationship
    store.merge_relationship(Relationship("Python", "ProgrammingLanguage", "RELATED_TO",
                                         "Qdrant", "Technology", "Python uses Qdrant",
                                         "d1", "c1"))
    matched = store.find_entities(["python"])
    assert any("Python" in m["text"] for m in matched)
    edges = store.traverse([matched[0]["entity_id"]], max_depth=2)
    assert len(edges) >= 1
    chunks = store.chunks_for_entities([matched[0]["entity_id"]])
    assert any(c["chunk_id"] == "c1" for c in chunks)
    store.delete_document("d1")
    assert store.find_entities(["python"]) == []
