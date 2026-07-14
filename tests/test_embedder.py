import math

import numpy as np
import pytest

from greenloop_rag_crew.rag import embedder as embedder_module
from greenloop_rag_crew.rag.embedder import (
    EXPECTED_EMBEDDING_DIMENSION,
    EmbeddingValidationError,
    GreenLoopEmbedder,
)


class FakeSentenceTransformer:
    instances = 0

    def __init__(self, model_name, device=None, trust_remote_code=False):
        FakeSentenceTransformer.instances += 1
        self.model_name = model_name
        self.device = device
        self.trust_remote_code = trust_remote_code

    def encode(self, texts, **kwargs):
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["precision"] == "float32"
        vectors = []
        for index, _text in enumerate(texts):
            vector = np.zeros(EXPECTED_EMBEDDING_DIMENSION, dtype=np.float32)
            vector[index % EXPECTED_EMBEDDING_DIMENSION] = 1.0
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def test_embedder_loads_model_lazily(monkeypatch):
    FakeSentenceTransformer.instances = 0
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    service = GreenLoopEmbedder(model_name="fake-model", device="cpu", batch_size=2)

    assert service.model_loaded is False
    assert FakeSentenceTransformer.instances == 0

    result = service.embed_documents(["hello", "world"])

    assert service.model_loaded is True
    assert FakeSentenceTransformer.instances == 1
    assert len(result) == 2
    assert len(result[0]) == EXPECTED_EMBEDDING_DIMENSION
    assert all(isinstance(value, float) for value in result[0])


def test_embed_query_uses_same_validation(monkeypatch):
    monkeypatch.setattr(embedder_module, "SentenceTransformer", FakeSentenceTransformer)

    service = GreenLoopEmbedder(model_name="fake-model")
    query_embedding = service.embed_query("remote work policy")

    assert len(query_embedding) == EXPECTED_EMBEDDING_DIMENSION
    assert math.isclose(sum(value * value for value in query_embedding), 1.0)


def test_embedding_dimension_and_finite_validation():
    service = GreenLoopEmbedder(model_name="fake-model")

    with pytest.raises(EmbeddingValidationError):
        service._validate_embeddings(np.zeros((1, 10), dtype=np.float32), expected_count=1)

    bad = np.zeros((1, EXPECTED_EMBEDDING_DIMENSION), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(EmbeddingValidationError):
        service._validate_embeddings(bad, expected_count=1)
