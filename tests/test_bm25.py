import pytest

rank_bm25 = pytest.importorskip("rank_bm25")

from app.retrieval.bm25_retriever import BM25Index, retrieve_bm25_evidence


def _payload(cid, did, text):
    return {"chunk_id": cid, "document_id": did, "text": text, "title": "T",
            "source_name": "s", "domain": "CS", "page_number": None,
            "source_url": None, "publication_date": None}


def test_index_search_and_update(tmp_path):
    idx = BM25Index(index_path=str(tmp_path / "bm25.pkl"))
    idx.add_or_update([_payload("c1", "d1", "Python supports object-oriented programming"),
                       _payload("c2", "d2", "Qdrant is a vector database")])
    hits = idx.search("object-oriented programming", top_k=5)
    assert hits and hits[0]["chunk_id"] == "c1"
    assert hits[0]["retrieval_methods"] == ["bm25"]
    # update same chunk -> no duplicate growth
    idx.add_or_update([_payload("c1", "d1", "Python supports object-oriented programming")])
    assert len(idx) == 2
    assert idx.remove_document("d1") == 1
    assert len(idx) == 1


def test_empty_query_returns_empty(tmp_path):
    idx = BM25Index(index_path="")
    idx.add_or_update([_payload("c1", "d1", "some text here")])
    assert idx.search("", top_k=5) == []
    assert idx.search("   ", top_k=5) == []


def test_retrieve_validates_claim():
    with pytest.raises(ValueError):
        retrieve_bm25_evidence("   ")
