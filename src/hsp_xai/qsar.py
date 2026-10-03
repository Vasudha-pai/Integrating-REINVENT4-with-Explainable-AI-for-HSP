"""QSAR activity model for HSP90 inhibition.

A gradient-boosted / random-forest classifier over Morgan fingerprints that
predicts the probability a molecule is an HSP90 inhibitor. The trained model
is the shared backbone for:

- scoring generated molecules inside REINVENT4 (:mod:`hsp_xai.scoring_component`)
- SHAP attribution (:mod:`hsp_xai.explain`)
- counterfactual evaluation (:mod:`hsp_xai.counterfactual`)

Class imbalance (actives are usually the minority) is handled with
``class_weight`` plus optional SMOTE oversampling from imbalanced-learn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .featurization import FeaturizedBatch, FingerprintConfig, featurize_smiles

LOGGER = logging.getLogger(__name__)


@dataclass
class QSARConfig:
    """Hyperparameters and training options for the QSAR model."""

    model_type: str = "random_forest"  # "random_forest" or "gradient_boosting"
    n_estimators: int = 400
    max_depth: Optional[int] = None
    random_state: int = 42
    test_size: float = 0.2
    use_smote: bool = True
    fingerprint: FingerprintConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.fingerprint is None:
            self.fingerprint = FingerprintConfig()


class QSARModel:
    """Thin wrapper around a scikit-learn classifier plus its featurizer."""

    def __init__(self, config: Optional[QSARConfig] = None):
        self.config = config or QSARConfig()
        self.estimator = None
        self.metrics: dict = {}

    # ------------------------------------------------------------------ build
    def _make_estimator(self):
        if self.config.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.config.random_state,
            )
        if self.config.model_type == "gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
            )
        raise ValueError(f"Unknown model_type: {self.config.model_type!r}")

    # ------------------------------------------------------------------ train
    def fit(self, smiles: list[str], labels: np.ndarray) -> "QSARModel":
        """Train on SMILES + binary activity labels, reporting held-out metrics."""
        from sklearn.model_selection import train_test_split

        batch = featurize_smiles(smiles, self.config.fingerprint)
        X = batch.features
        # Realign labels to the molecules that featurized successfully.
        label_by_smiles = dict(zip(smiles, np.asarray(labels)))
        y = np.array([label_by_smiles[s] for s in batch.valid_smiles])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        if self.config.use_smote and len(np.unique(y_train)) > 1:
            X_train, y_train = self._maybe_smote(X_train, y_train)

        self.estimator = self._make_estimator()
        self.estimator.fit(X_train, y_train)
        self._evaluate(X_test, y_test)
        return self

    def _maybe_smote(self, X, y):
        try:
            from imblearn.over_sampling import SMOTE

            minority = min(np.bincount(y))
            # SMOTE needs at least k_neighbors+1 minority samples.
            k = min(5, max(1, minority - 1))
            if minority <= 1:
                LOGGER.warning("Too few minority samples for SMOTE; skipping.")
                return X, y
            return SMOTE(random_state=self.config.random_state, k_neighbors=k).fit_resample(X, y)
        except ImportError:  # pragma: no cover
            LOGGER.warning("imbalanced-learn not installed; skipping SMOTE.")
            return X, y

    def _evaluate(self, X_test, y_test):
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            roc_auc_score,
            matthews_corrcoef,
        )

        proba = self.estimator.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_test, preds)),
            "n_test": int(len(y_test)),
        }
        if len(np.unique(y_test)) > 1:
            self.metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        LOGGER.info("Held-out metrics: %s", self.metrics)

    # ---------------------------------------------------------------- predict
    def predict_proba(self, smiles: list[str]) -> np.ndarray:
        """Return P(active) for each input SMILES (0.0 for invalid molecules)."""
        self._check_fitted()
        batch = featurize_smiles(smiles, self.config.fingerprint)
        valid_set = set(batch.valid_smiles)
        proba_valid = (
            self.estimator.predict_proba(batch.features)[:, 1]
            if len(batch.valid_smiles)
            else np.array([])
        )
        proba_by_smiles = dict(zip(batch.valid_smiles, proba_valid))
        return np.array([proba_by_smiles.get(s, 0.0) if s in valid_set else 0.0 for s in smiles])

    def predict_proba_features(self, features: np.ndarray) -> np.ndarray:
        """Predict directly from a fingerprint matrix (used by SHAP)."""
        self._check_fitted()
        return self.estimator.predict_proba(features)[:, 1]

    # --------------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        import joblib

        self._check_fitted()
        joblib.dump({"estimator": self.estimator, "config": self.config, "metrics": self.metrics}, path)
        LOGGER.info("Saved QSAR model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "QSARModel":
        import joblib

        payload = joblib.load(path)
        model = cls(payload["config"])
        model.estimator = payload["estimator"]
        model.metrics = payload.get("metrics", {})
        return model

    def _check_fitted(self):
        if self.estimator is None:
            raise RuntimeError("QSARModel is not fitted. Call fit() or load() first.")
