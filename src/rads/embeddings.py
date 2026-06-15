from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class BertEmbedder:
    """Frozen BERT [CLS] feature extractor used by Eq. (1)."""

    def __init__(self, model_name: str = "bert-base-uncased", max_tokens: int = 64):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BERT extraction requires the optional 'deep' dependencies."
            ) from exc
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.max_tokens = max_tokens

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        batches: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(
                    list(texts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                )
                output = self.model(**tokens).last_hidden_state[:, 0, :]
                batches.append(output.cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)

    @staticmethod
    def save(path: str | Path, embeddings: np.ndarray) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)


class HashingEmbedder:
    """Deterministic lightweight substitute for tests, never for paper results."""

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        del batch_size
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).split():
                matrix[row, hash(token) % self.dimensions] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)
