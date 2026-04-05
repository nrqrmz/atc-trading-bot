"""Model mixin — ML classifiers for trading signal prediction.

Trains LightGBM, CatBoost, and XGBoost individually, then combines them
in a Voting or Stacking ensemble. Uses TimeSeriesSplit to prevent data
leakage and compares CV vs out-of-sample metrics for overfitting detection.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from atc_trading_bot.config import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_ML_CV_SPLITS,
    DEFAULT_ML_TEST_SIZE,
    DEFAULT_N_ESTIMATORS,
    ML_OVERFIT_THRESHOLD,
)
from atc_trading_bot.pipeline_warning import PipelineWarning


class ModelMixin:
    """Mixin for training ML classifiers and ensembles."""

    def _require_labels(self) -> bool:
        """Check that labels have been computed."""
        if getattr(self, "labels", None) is None:
            warnings.warn("No labels available. Call compute_labels first.", PipelineWarning)
            return False
        return True

    def _require_features(self) -> bool:
        """Check that features have been computed."""
        if getattr(self, "features_pca", None) is None:
            warnings.warn("No features available. Call compute_features first.", PipelineWarning)
            return False
        return True

    def _require_model(self) -> bool:
        """Check that models have been trained."""
        if getattr(self, "active_model", None) is None:
            warnings.warn("No model trained. Call train_models first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trained_models: dict | None = None
        self.active_model = None
        self._model_metrics: list[dict] | None = None

    def train_models(self, test_size: float = DEFAULT_ML_TEST_SIZE,
                     cv_splits: int = DEFAULT_ML_CV_SPLITS,
                     n_estimators: int = DEFAULT_N_ESTIMATORS,
                     learning_rate: float = DEFAULT_LEARNING_RATE,
                     max_depth: int = DEFAULT_MAX_DEPTH):
        """Train LightGBM, CatBoost, XGBoost and ensemble classifiers.

        Uses TimeSeriesSplit for cross-validation (no future leakage).
        Compares CV vs out-of-sample metrics and warns on overfitting.

        Args:
            test_size: Fraction for out-of-sample evaluation. Default: 0.2.
            cv_splits: Number of TimeSeriesSplit folds. Default: 5.
            n_estimators: Trees per GBM. Default: 200.
            learning_rate: GBM learning rate. Default: 0.05.
            max_depth: GBM max tree depth. Default: 6.

        Returns:
            self for method chaining.
        """
        if not self._require_features():
            return
        if not self._require_labels():
            return

        import lightgbm as lgb
        import catboost as cb
        import xgboost as xgb

        # Align features and labels
        features_index = getattr(self, "features_index", None)
        if features_index is not None:
            valid_labels = self.labels.loc[features_index].values
        else:
            valid_labels = self.labels.values[:len(self.features_pca)]

        X = self.features_pca
        # Map labels to non-negative for XGBoost compatibility: -1→0, 0→1, 1→2
        self._label_map = {-1: 0, 0: 1, 1: 2}
        self._label_unmap = {v: k for k, v in self._label_map.items()}
        y = np.array([self._label_map.get(v, 1) for v in valid_labels])

        # Train/test split (temporal, no shuffle)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        # Define models
        models = {
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, random_state=42, verbose=-1,
                class_weight="balanced",
            ),
            "CatBoost": cb.CatBoostClassifier(
                iterations=n_estimators, learning_rate=learning_rate,
                depth=max_depth, random_seed=42, verbose=0,
                auto_class_weights="Balanced",
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, random_state=42, verbosity=0,
                eval_metric="mlogloss",
            ),
        }

        # Train and evaluate each model
        self.trained_models = {}
        self._model_metrics = []

        for name, model in models.items():
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                        scoring="f1_weighted")
            # Fit on full training set
            model.fit(X_train, y_train)
            self.trained_models[name] = model

            # Out-of-sample evaluation
            y_pred = model.predict(X_test)
            oos_metrics = self._compute_classification_metrics(y_test, y_pred)
            oos_metrics["cv_f1_mean"] = float(np.mean(cv_scores))
            oos_metrics["model"] = name
            self._model_metrics.append(oos_metrics)

            # Overfitting check
            if oos_metrics["cv_f1_mean"] > 0.01:
                ratio = oos_metrics["f1"] / oos_metrics["cv_f1_mean"]
                if ratio < ML_OVERFIT_THRESHOLD:
                    warnings.warn(
                        f"{name}: possible overfitting — F1 dropped from "
                        f"CV {oos_metrics['cv_f1_mean']:.3f} to OOS {oos_metrics['f1']:.3f}",
                        PipelineWarning,
                    )

        # Build ensembles
        estimators = [(name, m) for name, m in self.trained_models.items()]

        # Voting ensemble
        voting = VotingClassifier(estimators=estimators, voting="soft")
        voting.fit(X_train, y_train)
        self.trained_models["Voting"] = voting
        y_pred_v = voting.predict(X_test)
        voting_metrics = self._compute_classification_metrics(y_test, y_pred_v)
        voting_metrics["cv_f1_mean"] = np.nan
        voting_metrics["model"] = "Voting"
        self._model_metrics.append(voting_metrics)

        # Stacking ensemble
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=cv_splits,
        )
        stacking.fit(X_train, y_train)
        self.trained_models["Stacking"] = stacking
        y_pred_s = stacking.predict(X_test)
        stacking_metrics = self._compute_classification_metrics(y_test, y_pred_s)
        stacking_metrics["cv_f1_mean"] = np.nan
        stacking_metrics["model"] = "Stacking"
        self._model_metrics.append(stacking_metrics)

        # Select best model by OOS F1
        best = max(self._model_metrics, key=lambda m: m["f1"])
        self.active_model = self.trained_models[best["model"]]

        return self

    def models_summary(self) -> pd.DataFrame | None:
        """Return a comparison of all trained models.

        Returns:
            DataFrame with accuracy, F1, precision, recall, and CV F1
            for each model. Indexed by model name.
        """
        if self._model_metrics is None:
            warnings.warn("No models trained. Call train_models first.", PipelineWarning)
            return

        df = pd.DataFrame(self._model_metrics)
        df = df.set_index("model")
        cols = ["accuracy", "f1", "precision", "recall", "cv_f1_mean"]
        return df[cols].round(4)

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        """Predict labels using the active (best) model.

        Args:
            X: Feature array. Defaults to self.features_pca.

        Returns:
            Array of predicted labels (-1, 0, 1).
        """
        if not self._require_model():
            return

        if X is None:
            X = self.features_pca

        raw_preds = np.ravel(self.active_model.predict(X))
        # Unmap: 0→-1, 1→0, 2→1
        unmap = getattr(self, "_label_unmap", {0: -1, 1: 0, 2: 1})
        return np.array([unmap.get(int(v), 0) for v in raw_preds])

    @staticmethod
    def _compute_classification_metrics(y_true, y_pred) -> dict:
        """Compute classification metrics for trading labels."""
        labels = sorted(set(np.ravel(y_true).tolist()) | set(np.ravel(y_pred).tolist()))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        }
