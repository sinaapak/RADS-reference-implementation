from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .config import ExperimentConfig
from .data import (
    aggregate_daily_modalities,
    load_market_data,
    make_rads_forecast_windows,
)
from .embeddings import BertEmbedder
from .metrics import regression_metrics
from .model import build_rads_forecaster
from .pipeline import set_seed
from .rads import RADSDetector
from .temporal import fit_normal_temporal_model


def run_cross_asset_experiment(
    data_path: str,
    source_column: str,
    target_column: str,
    config: ExperimentConfig,
    *,
    date_column: str = "date",
    headline_column: str = "headline",
    label_column: str = "is_fake",
) -> dict[str, float]:
    """Train on one entity and evaluate another in a later time period."""
    set_seed(config.seed)
    source = load_market_data(
        data_path,
        source_column,
        date_column=date_column,
        headline_column=headline_column,
        label_column=label_column,
    )
    target = load_market_data(
        data_path,
        target_column,
        date_column=date_column,
        headline_column=headline_column,
        label_column=label_column,
    )
    if not source.frame[date_column].equals(target.frame[date_column]):
        raise ValueError("Source and target observations must be date-aligned")

    embeddings = BertEmbedder(config.bert_model, config.max_tokens).encode(
        source.frame[headline_column].tolist(), config.bert_batch_size
    )
    source, source_embeddings = aggregate_daily_modalities(
        source,
        embeddings,
        source_column,
        date_column=date_column,
        headline_column=headline_column,
        label_column=label_column,
    )
    target, target_embeddings = aggregate_daily_modalities(
        target,
        embeddings,
        target_column,
        date_column=date_column,
        headline_column=headline_column,
        label_column=label_column,
    )
    if not np.array_equal(source_embeddings, target_embeddings):
        raise ValueError("Source and target headline aggregation must be identical")
    embeddings = source_embeddings
    frame = source.frame
    split_row = int(round(len(frame) * (1.0 - config.test_fraction)))
    split_row = min(max(split_row, config.lookback + 1), len(frame) - 1)
    scaler = MinMaxScaler().fit(source.frame[[source_column]].iloc[:split_row])
    source_price = scaler.transform(source.frame[[source_column]]).ravel()
    target_price = scaler.transform(target.frame[[target_column]]).ravel()
    source_numerical = source.frame[source.numerical_columns].to_numpy(
        dtype=np.float32
    )
    target_numerical = target.frame[target.numerical_columns].to_numpy(
        dtype=np.float32
    )
    labels = frame[label_column].to_numpy(dtype=int)
    source_aligned = make_rads_forecast_windows(
        source_price, embeddings, source_numerical, lookback=config.lookback
    )
    target_aligned = make_rads_forecast_windows(
        target_price, embeddings, target_numerical, lookback=config.lookback
    )
    current_rows = source_aligned["current_rows"]
    target_rows = source_aligned["target_rows"]
    train_idx = np.flatnonzero(target_rows < split_row)
    test_idx = np.flatnonzero(target_rows >= split_row)
    current_labels = labels[current_rows]

    temporal_model = fit_normal_temporal_model(
        source_aligned["temporal_windows"][train_idx],
        source_aligned["temporal_actual"][train_idx],
        current_labels[train_idx],
        epochs=config.temporal_epochs,
        batch_size=config.temporal_batch_size,
    )
    source_expected = temporal_model.predict(
        source_aligned["temporal_windows"], verbose=0
    ).ravel()
    target_expected = temporal_model.predict(
        target_aligned["temporal_windows"], verbose=0
    ).ravel()

    detector = RADSDetector(
        contamination=config.isolation_contamination,
        random_state=config.seed,
        xgb_estimators=config.xgb_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
    ).fit(
        embeddings[current_rows][train_idx],
        source_aligned["temporal_actual"][train_idx],
        source_expected[train_idx],
        source_numerical[current_rows][train_idx],
        current_labels[train_idx],
    )
    source_rads = detector.predict_score(
        embeddings[current_rows],
        source_aligned["temporal_actual"],
        source_expected,
        source_numerical[current_rows],
    )
    target_rads = detector.predict_score(
        embeddings[current_rows],
        target_aligned["temporal_actual"],
        target_expected,
        target_numerical[current_rows],
    )

    forecaster = build_rads_forecaster(
        config.lookback,
        source_aligned["numerical_windows"].shape[-1],
        source_aligned["text_windows"].shape[-1],
        head_size=config.attention_head_size,
        num_heads=config.attention_heads,
        ff_dim=config.feed_forward_dim,
        lstm_units=config.lstm_units,
        dropout=config.dropout,
    )
    forecaster.fit(
        [
            source_aligned["numerical_windows"][train_idx],
            source_aligned["text_windows"][train_idx],
            source_rads[train_idx, None],
        ],
        source_aligned["forecast_targets"][train_idx],
        validation_split=0.20,
        epochs=config.forecast_epochs,
        batch_size=config.forecast_batch_size,
        shuffle=False,
        verbose=1,
    )
    prediction_scaled = forecaster.predict(
        [
            target_aligned["numerical_windows"][test_idx],
            target_aligned["text_windows"][test_idx],
            target_rads[test_idx, None],
        ],
        verbose=0,
    ).ravel()
    actual = scaler.inverse_transform(
        target_aligned["forecast_targets"][test_idx, None]
    ).ravel()
    predicted = scaler.inverse_transform(prediction_scaled[:, None]).ravel()
    metrics = regression_metrics(actual, predicted)

    output_dir = Path(config.output_dir) / f"{source_column}_to_{target_column}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": frame[date_column].iloc[target_rows[test_idx]].to_numpy(),
            "actual": actual,
            "predicted": predicted,
            "rads_score": target_rads[test_idx],
        }
    ).to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Table 7 cross-asset transfer.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--source-column", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--headline-column", default="headline")
    parser.add_argument("--label-column", default="is_fake")
    args = parser.parse_args()
    metrics = run_cross_asset_experiment(
        args.data,
        args.source_column,
        args.target_column,
        ExperimentConfig.from_yaml(args.config),
        date_column=args.date_column,
        headline_column=args.headline_column,
        label_column=args.label_column,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
