from app.graph_store.entity_extractor import EntityExtractor


def test_extracts_tech_entities():
    ex = EntityExtractor()
    res = ex.extract("Python and FastAPI are used with Qdrant in 2024.", "d1", "c1")
    texts = [e.text.lower() for e in res.entities]
    assert "python" in texts and "qdrant" in texts
    assert any(e.type == "Date" for e in res.entities)


def test_relationships_only_same_sentence():
    ex = EntityExtractor()
    text = "Python is great. Qdrant is separate."
    res = ex.extract(text, "d1", "c1")
    # entities in different sentences must NOT be related
    assert res.relationships == []
    text2 = "Python integrates with Qdrant for retrieval."
    res2 = ex.extract(text2, "d1", "c1")
    assert len(res2.relationships) >= 1
    r = res2.relationships[0]
    assert r.supporting_text and r.document_id == "d1" and r.chunk_id == "c1"


def test_empty_text_no_invention():
    ex = EntityExtractor()
    res = ex.extract("   ", "d1", "c1")
    assert res.entities == [] and res.relationships == []
