from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import RawDocument


def _doc(text, **kw):
    return RawDocument(document_id=kw.get("document_id", "doc_test"),
                       title="T", text=text, content_hash="abc123")


def test_short_doc_single_chunk():
    doc = _doc("Python supports object-oriented programming.")
    chunks = chunk_document(doc, chunk_size_words=200, chunk_overlap_words=30)
    assert len(chunks) == 1
    assert chunks[0].chunk_position == 0
    assert "object-oriented" in chunks[0].text
    assert chunks[0].document_id == "doc_test"


def test_metadata_and_stable_ids():
    text = "Para one about Python.\n\nPara two about Qdrant."
    doc = RawDocument(document_id="doc_1", title="My Title", text=text,
                      source_name="src", source_url="http://x", domain="CS",
                      publication_date="2024-01-01", content_hash="h1")
    c1 = chunk_document(doc)
    c2 = chunk_document(doc)
    assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]  # stable
    assert c1[0].title == "My Title" and c1[0].domain == "CS"
    assert c1[0].source_url == "http://x"


def test_long_paragraph_splits_on_sentences():
    para = " ".join(f"Sentence number {i} about transformers and attention." for i in range(60))
    doc = _doc(para)
    chunks = chunk_document(doc, chunk_size_words=200, chunk_overlap_words=30)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 230
