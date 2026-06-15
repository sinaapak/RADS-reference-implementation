from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import (
    aggregate_daily_modalities,
    load_market_data,
    make_rads_forecast_windows,
)
from .embeddings import BertEmbedder
from .metrics import regression_metrics
from .model import build_rads_forecaster
from .rads import RADSDetector
from .temporal import fit_normal_temporal_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except (ImportError, RuntimeError):
        pass


def run_experiment(
    data_path: str,
    price_column: str,
    config: ExperimentConfig,
    *,
    date_column: str = "date",
    headline_column: str = "headline",
    label_column: str = "is_fake",
    variant: str = "full",
) -> dict[str, float]:
    set_seed(config.seed)
    prepared = load_market_data(
        data_path,
        price_column,
        date_column=date_column,
        headline_column=headline_column,
        label_column=label_column,
    )
    embedder = BertEmbedder(config.bert_model, config.max_tokens)
    embeddings = embedder.encode(
        prepared.frame[headline_column].tolist(), config.bert_batch_size
    )
    prepared, embeddings = aggregate_daily_modalities(
        prepared,
        embeddings,
        price_column,
        date_column=date_column,
        headline_column=headline_column,
        label_column=label_column,
    )
    frame = prepared.frame
    split_row = int(round(len(frame) * (1.0 - config.test_fraction)))
    split_row = min(max(split_row, config.lookback + 1), len(frame) - 1)
    prepared.price_scaler.fit(frame[[price_column]].iloc[:split_row])
    scaled_price = prepared.price_scaler.transform(
        frame[[price_column]]
    ).ravel().astype(np.float32)
    numerical = frame[prepared.numerical_columns].to_numpy(dtype=np.float32)
    labels = frame[label_column].to_numpy(dtype=int)

    aligned = make_rads_forecast_windows(
        scaled_price,
        embeddings,
        numerical,
        lookback=config.lookback,
    )
    current_rows = aligned["current_rows"]
    target_rows = aligned["target_rows"]
    temporal_windows = aligned["temporal_windows"]
    temporal_actual = aligned["temporal_actual"]
    numerical_windows = aligned["numerical_windows"]
    text_windows = aligned["text_windows"]
    targets = aligned["forecast_targets"]
    current_labels = labels[current_rows]
    train_idx = np.flatnonzero(target_rows < split_row)
    test_idx = np.flatnonzero(target_rows >= split_row)

    temporal_model = fit_normal_temporal_model(
        temporal_windows[train_idx],
        temporal_actual[train_idx],
        current_labels[train_idx],
        epochs=config.temporal_epochs,
        batch_size=config.temporal_batch_size,
    )
    temporal_expected = temporal_model.predict(temporal_windows, verbose=0).ravel()

    detector = RADSDetector(
        contamination=config.isolation_contamination,
        random_state=config.seed,
        xgb_estimators=config.xgb_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
    )
    detector.fit(
        embeddings[current_rows][train_idx],
        temporal_actual[train_idx],
        temporal_expected[train_idx],
        numerical[current_rows][train_idx],
        current_labels[train_idx],
    )
    rads_scores = detector.predict_score(
        embeddings[current_rows],
        temporal_actual,
        temporal_expected,
        numerical[current_rows],
    )

    variants = {
        "full": (True, True, True),
        "without_cross_attention": (True, False, True),
        "without_text_transformer": (False, True, True),
        "without_rads": (True, True, False),
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(variants)}")
    use_text_transformer, use_cross_attention, use_rads = variants[variant]
    forecaster = build_rads_forecaster(
        config.lookback,
        numerical_windows.shape[-1],
        text_windows.shape[-1],
        head_size=config.attention_head_size,
        num_heads=config.attention_heads,
        ff_dim=config.feed_forward_dim,
        lstm_units=config.lstm_units,
        dropout=config.dropout,
        use_text_transformer=use_text_transformer,
        use_cross_attention=use_cross_attention,
        use_rads=use_rads,
    )
    train_inputs = [numerical_windows[train_idx], text_windows[train_idx]]
    test_inputs = [numerical_windows[test_idx], text_windows[test_idx]]
    if use_rads:
        train_inputs.append(rads_scores[train_idx, None])
        test_inputs.append(rads_scores[test_idx, None])
    forecaster.fit(
        train_inputs,
        targets[train_idx],
        validation_split=0.20,
        epochs=config.forecast_epochs,
        batch_size=config.forecast_batch_size,
        shuffle=False,
        verbose=1,
    )
    prediction_scaled = forecaster.predict(
        test_inputs,
        verbose=0,
    ).ravel()
    actual = prepared.price_scaler.inverse_transform(
        targets[test_idx, None]
    ).ravel()
    predicted = prepared.price_scaler.inverse_transform(
        prediction_scaled[:, None]
    ).ravel()
    metrics = regression_metrics(actual, predicted)

    output_dir = Path(config.output_dir) / price_column / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": frame[date_column].iloc[target_rows[test_idx]].to_numpy(),
            "actual": actual,
            "predicted": predicted,
            "attack_label_at_t": current_labels[test_idx],
            "rads_score": rads_scores[test_idx],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    detector.save(str(output_dir / "rads_detector.joblib"))
    forecaster.save(output_dir / "forecaster.keras")
    return metrics
