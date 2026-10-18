import pytest

from app.embeddings.embedding_service import EmbeddingService


def test_embed_validates_empty():
    svc = EmbeddingService()
    with pytest.raises(ValueError):
        svc.embed(["   "])


def test_embed_batch_shape_and_norm():
    svc = EmbeddingService(dim=32)  # small dim; may use fallback offline
    vecs = svc.embed(["Python supports OOP.", "Qdrant stores vectors."])
    assert vecs.shape[0] == 2
    assert vecs.shape[1] in (32, 384)  # fallback dim or real model dim
    # rows should be ~unit norm (both backends normalise)
    import numpy as np
    norms = (vecs ** 2).sum(axis=1) ** 0.5
    assert all(abs(n - 1.0) < 0.05 for n in norms)


def test_singleton_loads_once():
    from app.embeddings.embedding_service import get_embedding_service, reset_embedding_service
    reset_embedding_service()
    a = get_embedding_service()
    b = get_embedding_service()
    assert a is b
