from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    error = predicted - actual
    denominator = np.where(np.abs(actual) > 1e-12, np.abs(actual), np.nan)
    correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else np.nan
    alpha = np.std(predicted) / max(np.std(actual), 1e-12)
    beta = np.mean(predicted) / max(np.mean(actual), 1e-12)
    kge = 1.0 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    nse = 1.0 - np.sum(error**2) / max(
        np.sum((actual - np.mean(actual)) ** 2), 1e-12
    )
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mape_percent": float(np.nanmean(np.abs(error) / denominator) * 100),
        "r2": float(r2_score(actual, predicted)),
        "kge": float(kge),
        "nse": float(nse),
        "bias_factor": float(np.mean(predicted) / max(np.mean(actual), 1e-12)),
    }
