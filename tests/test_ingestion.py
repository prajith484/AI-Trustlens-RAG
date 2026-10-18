import pytest

from app.ingestion import preprocessing
from app.ingestion.loaders import load_file


def test_clean_preserves_paragraphs():
    raw = "Heading\n\nPara one.   \n\n\nPara two."
    cleaned = preprocessing.clean_text(raw)
    assert "Para one" in cleaned and "Para two" in cleaned
    assert "\n\n" in cleaned


def test_split_paragraphs():
    paras = preprocessing.split_paragraphs("A\n\nB\n\nC")
    assert paras == ["A", "B", "C"]


def test_load_txt(tmp_path):
    p = tmp_path / "hello.txt"
    p.write_text("Hello world", encoding="utf-8")
    doc = load_file(p)
    assert doc.text == "Hello world"
    assert doc.content_hash and doc.document_id.startswith("doc_")


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_file(tmp_path / "nope.txt")


def test_unsupported_type(tmp_path):
    p = tmp_path / "f.csv"
    p.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(p)
