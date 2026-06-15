from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    seed: int = 42
    lookback: int = 10
    test_fraction: float = 0.20
    bert_model: str = "bert-base-uncased"
    bert_batch_size: int = 16
    max_tokens: int = 64
    attention_heads: int = 4
    attention_head_size: int = 64
    feed_forward_dim: int = 64
    lstm_units: int = 64
    dropout: float = 0.20
    forecast_epochs: int = 500
    forecast_batch_size: int = 16
    temporal_epochs: int = 50
    temporal_batch_size: int = 10
    isolation_contamination: float = 0.10
    xgb_estimators: int = 200
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.05
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values: dict[str, Any] = yaml.safe_load(handle) or {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
