from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def evaluate_rads_component_ablation(
    deviation_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    random_state: int = 42,
    folds: int = 5,
) -> pd.DataFrame:
    """Evaluate Table 5 by excluding one deviation signal at a time."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Component ablation requires xgboost.") from exc

    matrix = StandardScaler().fit_transform(np.asarray(deviation_matrix, dtype=float))
    labels = np.asarray(labels, dtype=int).ravel()
    variants = {
        "RADS": [0, 1, 2],
        "Without S-NDS": [1, 2],
        "Without T-NDS": [0, 2],
        "Without NDS": [0, 1],
    }
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    rows = []
    for name, columns in variants.items():
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
        predictions = cross_val_predict(
            classifier, matrix[:, columns], labels, cv=cv, method="predict"
        )
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )
        rows.append(
            {
                "model_version": name,
                "accuracy": accuracy_score(labels, predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)
