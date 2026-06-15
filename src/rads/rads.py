from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler


def semantic_deviation(
    embeddings: np.ndarray,
    reference_embedding: np.ndarray,
) -> np.ndarray:
    """Eq. (3): cosine distance from the normal-news centroid."""
    embeddings = np.asarray(embeddings, dtype=float)
    reference = np.asarray(reference_embedding, dtype=float).reshape(1, -1)
    return cosine_distances(embeddings, reference).ravel()


def temporal_deviation(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Eq. (5): squared temporal prediction residual used as T-NDS."""
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    if actual.shape != expected.shape:
        raise ValueError("actual and expected must have identical shapes")
    return np.square(actual - expected)


def numerical_deviation(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    """Eq. (6): negative Isolation Forest normality score."""
    return -model.score_samples(np.asarray(features, dtype=float))


@dataclass
class RADSComponents:
    s_nds: np.ndarray
    t_nds: np.ndarray
    nds: np.ndarray

    def matrix(self) -> np.ndarray:
        return np.column_stack([self.s_nds, self.t_nds, self.nds])


class RADSDetector:
    """Fit the three paper-defined deviations and fuse them with XGBoost."""

    def __init__(
        self,
        contamination: float = 0.10,
        random_state: int = 42,
        xgb_estimators: int = 200,
        xgb_max_depth: int = 3,
        xgb_learning_rate: float = 0.05,
    ):
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        self.numerical_scaler = StandardScaler()
        self.deviation_scaler = StandardScaler()
        self.reference_embedding: np.ndarray | None = None
        self.classifier = self._make_classifier(
            random_state,
            xgb_estimators,
            xgb_max_depth,
            xgb_learning_rate,
        )

    @staticmethod
    def _make_classifier(
        random_state: int,
        estimators: int,
        max_depth: int,
        learning_rate: float,
    ):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("RADS fusion requires xgboost.") from exc
        return XGBClassifier(
            n_estimators=estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )

    def fit(
        self,
        embeddings: np.ndarray,
        temporal_actual: np.ndarray,
        temporal_expected: np.ndarray,
        numerical_features: np.ndarray,
        attack_labels: np.ndarray,
    ) -> "RADSDetector":
        labels = np.asarray(attack_labels, dtype=int).ravel()
        normal = labels == 0
        if normal.sum() < 2 or np.unique(labels).size != 2:
            raise ValueError("RADS training requires both labels and at least two normal samples")
        self.reference_embedding = np.asarray(embeddings)[normal].mean(axis=0)
        normal_numerical = np.asarray(numerical_features)[normal]
        self.numerical_scaler.fit(normal_numerical)
        self.isolation_forest.fit(self.numerical_scaler.transform(normal_numerical))
        components = self.components(
            embeddings, temporal_actual, temporal_expected, numerical_features
        )
        scaled = self.deviation_scaler.fit_transform(components.matrix())
        self.classifier.fit(scaled, labels)
        return self

    def components(
        self,
        embeddings: np.ndarray,
        temporal_actual: np.ndarray,
        temporal_expected: np.ndarray,
        numerical_features: np.ndarray,
    ) -> RADSComponents:
        if self.reference_embedding is None:
            raise RuntimeError("RADSDetector has not been fitted")
        return RADSComponents(
            semantic_deviation(embeddings, self.reference_embedding),
            temporal_deviation(temporal_actual, temporal_expected),
            numerical_deviation(
                self.isolation_forest,
                self.numerical_scaler.transform(np.asarray(numerical_features)),
            ),
        )

    def predict_score(
        self,
        embeddings: np.ndarray,
        temporal_actual: np.ndarray,
        temporal_expected: np.ndarray,
        numerical_features: np.ndarray,
    ) -> np.ndarray:
        components = self.components(
            embeddings, temporal_actual, temporal_expected, numerical_features
        )
        scaled = self.deviation_scaler.transform(components.matrix())
        return self.classifier.predict_proba(scaled)[:, 1]

    def save(self, path: str) -> None:
        joblib.dump(self, path)
