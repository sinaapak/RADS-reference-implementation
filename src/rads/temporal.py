from __future__ import annotations

import numpy as np


def build_temporal_lstm(lookback: int, n_features: int, units: int = 64):
    """Table 3 LSTM used to estimate expected normal temporal behavior."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("Temporal LSTM requires the optional 'deep' dependencies.") from exc
    inputs = tf.keras.Input(shape=(lookback, n_features), name="temporal_input")
    x = tf.keras.layers.LSTM(units, activation="relu", name="temporal_lstm")(inputs)
    output = tf.keras.layers.Dense(1, name="expected_value")(x)
    model = tf.keras.Model(inputs, output, name="t_nds_lstm")
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def fit_normal_temporal_model(
    sequences: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 10,
    verbose: int = 0,
):
    labels = np.asarray(labels).ravel()
    normal = labels == 0
    if normal.sum() < 2:
        raise ValueError("At least two normal sequences are required")
    model = build_temporal_lstm(sequences.shape[1], sequences.shape[2])
    model.fit(
        sequences[normal],
        np.asarray(targets)[normal],
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        shuffle=False,
    )
    return model
