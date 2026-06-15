from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class PreparedData:
    frame: pd.DataFrame
    numerical_columns: list[str]
    price_scaler: MinMaxScaler


def clean_headline(text: object) -> str:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"[^\w\s$%.-]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def load_market_data(
    path: str | Path,
    price_column: str,
    *,
    date_column: str = "date",
    headline_column: str = "headline",
    label_column: str = "is_fake",
    volume_column: str | None = None,
) -> PreparedData:
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    required = {date_column, headline_column, price_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    frame[headline_column] = frame[headline_column].map(clean_headline)
    frame = frame.drop_duplicates(subset=[date_column, headline_column])
    frame = frame.sort_values(date_column).reset_index(drop=True)
    frame[price_column] = pd.to_numeric(frame[price_column], errors="coerce").ffill()

    if label_column not in frame:
        frame[label_column] = 0
    frame[label_column] = frame[label_column].fillna(0).astype(int)

    numerical_columns = [price_column]
    if volume_column and volume_column in frame:
        frame[volume_column] = pd.to_numeric(frame[volume_column], errors="coerce").ffill()
        numerical_columns.append(volume_column)

    scaler = MinMaxScaler()
    return PreparedData(frame, numerical_columns, scaler)


def aggregate_daily_modalities(
    prepared: PreparedData,
    embeddings: np.ndarray,
    price_column: str,
    *,
    date_column: str = "date",
    headline_column: str = "headline",
    label_column: str = "is_fake",
) -> tuple[PreparedData, np.ndarray]:
    """Average same-day headline embeddings as stated in preprocessing."""
    frame = prepared.frame.reset_index(drop=True)
    embeddings = np.asarray(embeddings)
    if len(frame) != len(embeddings):
        raise ValueError("Embedding count must match the observation count")
    rows: list[dict[str, object]] = []
    daily_embeddings: list[np.ndarray] = []
    extra_columns = [
        column
        for column in prepared.numerical_columns
        if column != price_column
    ]
    for date, indices in frame.groupby(date_column, sort=True).indices.items():
        positions = np.asarray(indices, dtype=int)
        group = frame.iloc[positions]
        row: dict[str, object] = {
            date_column: date,
            headline_column: " ".join(group[headline_column].astype(str)),
            price_column: float(group[price_column].iloc[-1]),
            label_column: int(group[label_column].max()),
        }
        for column in extra_columns:
            row[column] = float(group[column].iloc[-1])
        rows.append(row)
        daily_embeddings.append(embeddings[positions].mean(axis=0))

    daily = pd.DataFrame(rows).sort_values(date_column).reset_index(drop=True)
    daily["return_1d"] = daily[price_column].pct_change().fillna(0.0)
    daily["volatility_10d"] = (
        daily["return_1d"].rolling(10, min_periods=2).std().fillna(0.0)
    )
    numerical_columns = [price_column, "return_1d", "volatility_10d", *extra_columns]
    return (
        PreparedData(daily, numerical_columns, MinMaxScaler()),
        np.vstack(daily_embeddings).astype(np.float32),
    )


def chronological_split(n_rows: int, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    split = int(round(n_rows * (1.0 - test_fraction)))
    split = min(max(split, 1), n_rows - 1)
    return np.arange(split), np.arange(split, n_rows)


def make_windows(*arrays: np.ndarray, lookback: int) -> tuple[list[np.ndarray], np.ndarray]:
    if not arrays:
        raise ValueError("At least one array is required")
    n_rows = len(arrays[0])
    if any(len(array) != n_rows for array in arrays):
        raise ValueError("All arrays must have the same length")
    if n_rows <= lookback:
        raise ValueError(f"Need more than {lookback} rows, received {n_rows}")
    windows = [
        np.asarray([array[i : i + lookback] for i in range(n_rows - lookback)])
        for array in arrays
    ]
    targets = np.arange(lookback, n_rows)
    return windows, targets


def make_rads_forecast_windows(
    scaled_price: np.ndarray,
    embeddings: np.ndarray,
    numerical: np.ndarray,
    *,
    lookback: int,
) -> dict[str, np.ndarray]:
    """Align current-day RADS information with a strictly next-day target."""
    price = np.asarray(scaled_price).ravel()
    embeddings = np.asarray(embeddings)
    numerical = np.asarray(numerical)
    if not (len(price) == len(embeddings) == len(numerical)):
        raise ValueError("All modalities must contain the same number of rows")
    if len(price) <= lookback + 1:
        raise ValueError(f"Need more than {lookback + 1} rows")

    current_rows = np.arange(lookback, len(price) - 1)
    temporal_windows = np.asarray(
        [price[row - lookback : row, None] for row in current_rows]
    )
    numerical_windows = np.asarray(
        [numerical[row - lookback + 1 : row + 1] for row in current_rows]
    )
    text_windows = np.asarray(
        [embeddings[row - lookback + 1 : row + 1] for row in current_rows]
    )
    return {
        "current_rows": current_rows,
        "target_rows": current_rows + 1,
        "temporal_windows": temporal_windows,
        "temporal_actual": price[current_rows],
        "numerical_windows": numerical_windows,
        "text_windows": text_windows,
        "forecast_targets": price[current_rows + 1],
    }
