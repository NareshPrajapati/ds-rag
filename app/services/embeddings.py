"""Embedding model wrapper (lazy singleton, normalized vectors)."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0].tolist()
