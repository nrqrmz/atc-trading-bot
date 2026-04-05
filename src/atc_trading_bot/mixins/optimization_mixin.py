"""Optimization mixin — Optuna hyperparameter tuning and walk-forward validation.

Uses Bayesian optimization (TPE sampler) to search for the best GBM
hyperparameters and walk-forward sliding windows to evaluate model
stability across time periods.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from atc_trading_bot.config import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_ML_CV_SPLITS,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_OPTUNA_TRIALS,
    DEFAULT_WF_STEP_SIZE,
    DEFAULT_WF_WINDOW_SIZE,
)
from atc_trading_bot.pipeline_warning import PipelineWarning


class OptimizationMixin:
    """Mixin for hyperparameter optimization and walk-forward validation."""

    def _require_features(self) -> bool:
        """Check that PCA features have been computed."""
        if getattr(self, "features_pca", None) is None:
            warnings.warn(
                "No features available. Call compute_features first.",
                PipelineWarning,
            )
            return False
        return True

    def _require_labels(self) -> bool:
        """Check that labels have been computed."""
        if getattr(self, "labels", None) is None:
            warnings.warn(
                "No labels available. Call compute_labels first.",
                PipelineWarning,
            )
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optuna_study = None

    # ------------------------------------------------------------------
    # Optuna hyperparameter optimization
    # ------------------------------------------------------------------

    def optimize_model(
        self,
        n_trials: int = DEFAULT_OPTUNA_TRIALS,
        model: str = "lightgbm",
        cv_splits: int = DEFAULT_ML_CV_SPLITS,
    ) -> pd.DataFrame | None:
        """Find optimal hyperparameters with Bayesian optimization (Optuna).

        Creates a study using the TPE sampler and optimizes the weighted F1
        score via TimeSeriesSplit cross-validation.  For LightGBM, an Optuna
        pruning callback is attached so unpromising trials are stopped early.

        Args:
            n_trials: Number of optimization trials. Default: 100.
            model: Model type — ``"lightgbm"``, ``"catboost"``, or ``"xgboost"``.
                   Default: ``"lightgbm"``.
            cv_splits: Number of TimeSeriesSplit folds. Default: 5.

        Returns:
            DataFrame with the best parameters (param name as index, value column).
            Stores the study as ``self.optuna_study``.
        """
        if not self._require_features():
            return None
        if not self._require_labels():
            return None

        import optuna

        # Align features and labels
        features_index = getattr(self, "features_index", None)
        if features_index is not None:
            valid_labels = self.labels.loc[features_index].values
        else:
            valid_labels = self.labels.values[: len(self.features_pca)]

        X = self.features_pca
        # Map labels for XGBoost compatibility: -1 -> 0, 0 -> 1, 1 -> 2
        label_map = {-1: 0, 0: 1, 1: 2}
        y = np.array([label_map.get(v, 1) for v in valid_labels])

        tscv = TimeSeriesSplit(n_splits=cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            clf = self._create_model(model, params)
            scores = cross_val_score(
                clf, X, y, cv=tscv, scoring="f1_weighted"
            )
            return float(np.mean(scores))

        # Suppress Optuna logging during optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=42)
        self.optuna_study = optuna.create_study(
            direction="maximize", sampler=sampler
        )

        # Use pruning callback for LightGBM
        callbacks = []
        if model.lower() == "lightgbm":
            callbacks.append(
                optuna.integration.LightGBMPruningCallback(
                    trial=None, metric="multi_logloss"
                )
            )

        self.optuna_study.optimize(
            objective, n_trials=n_trials, show_progress_bar=False
        )

        # Build result DataFrame
        best = self.optuna_study.best_params
        result = pd.DataFrame(
            {"value": best.values()},
            index=pd.Index(best.keys(), name="param"),
        )
        return result

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        window_size: int = DEFAULT_WF_WINDOW_SIZE,
        step_size: int = DEFAULT_WF_STEP_SIZE,
        model: str = "lightgbm",
    ) -> pd.DataFrame | None:
        """Sliding-window walk-forward validation.

        Trains on *window_size* bars, tests on the next *step_size* bars,
        then slides forward by *step_size*. Records per-window classification
        metrics and appends a **Mean** row.

        Args:
            window_size: Training window length (bars). Default: 252.
            step_size: Step / test length (bars). Default: 21.
            model: Model type — ``"lightgbm"``, ``"catboost"``, or ``"xgboost"``.
                   Default: ``"lightgbm"``.

        Returns:
            DataFrame indexed by ``"Window 0"``, ``"Window 1"``, ... , ``"Mean"``
            with columns ``accuracy``, ``f1``, ``precision``, ``recall``.
        """
        if not self._require_features():
            return None
        if not self._require_labels():
            return None

        # Align features and labels
        features_index = getattr(self, "features_index", None)
        if features_index is not None:
            valid_labels = self.labels.loc[features_index].values
        else:
            valid_labels = self.labels.values[: len(self.features_pca)]

        X = self.features_pca
        label_map = {-1: 0, 0: 1, 1: 2}
        y = np.array([label_map.get(v, 1) for v in valid_labels])

        n = len(X)
        rows: list[dict] = []
        win_idx = 0

        start = 0
        while start + window_size + step_size <= n:
            train_end = start + window_size
            test_end = train_end + step_size

            X_train, y_train = X[start:train_end], y[start:train_end]
            X_test, y_test = X[train_end:test_end], y[train_end:test_end]

            params = {
                "n_estimators": DEFAULT_N_ESTIMATORS,
                "learning_rate": DEFAULT_LEARNING_RATE,
                "max_depth": DEFAULT_MAX_DEPTH,
            }
            clf = self._create_model(model, params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            labels = sorted(
                set(np.ravel(y_test).tolist()) | set(np.ravel(y_pred).tolist())
            )
            rows.append(
                {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "f1": float(
                        f1_score(
                            y_test, y_pred, average="weighted",
                            labels=labels, zero_division=0,
                        )
                    ),
                    "precision": float(
                        precision_score(
                            y_test, y_pred, average="weighted",
                            labels=labels, zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            y_test, y_pred, average="weighted",
                            labels=labels, zero_division=0,
                        )
                    ),
                }
            )
            start += step_size
            win_idx += 1

        index = [f"Window {i}" for i in range(len(rows))]
        df = pd.DataFrame(rows, index=index)

        # Append mean row
        mean_row = df.mean()
        mean_df = pd.DataFrame([mean_row], index=["Mean"])
        df = pd.concat([df, mean_df])

        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_model(model: str, params: dict):
        """Instantiate a GBM classifier with the given parameters.

        Args:
            model: One of ``"lightgbm"``, ``"catboost"``, ``"xgboost"``.
            params: Hyperparameter dict.

        Returns:
            Fitted-ready classifier instance.
        """
        model_lower = model.lower()

        if model_lower == "lightgbm":
            import lightgbm as lgb

            return lgb.LGBMClassifier(
                n_estimators=params.get("n_estimators", DEFAULT_N_ESTIMATORS),
                learning_rate=params.get("learning_rate", DEFAULT_LEARNING_RATE),
                max_depth=params.get("max_depth", DEFAULT_MAX_DEPTH),
                min_child_samples=params.get("min_child_samples", 20),
                subsample=params.get("subsample", 1.0),
                colsample_bytree=params.get("colsample_bytree", 1.0),
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 0.0),
                random_state=42,
                verbose=-1,
                class_weight="balanced",
            )

        if model_lower == "catboost":
            import catboost as cb

            return cb.CatBoostClassifier(
                iterations=params.get("n_estimators", DEFAULT_N_ESTIMATORS),
                learning_rate=params.get("learning_rate", DEFAULT_LEARNING_RATE),
                depth=params.get("max_depth", DEFAULT_MAX_DEPTH),
                min_data_in_leaf=params.get("min_child_samples", 20),
                subsample=params.get("subsample", 1.0),
                colsample_bylevel=params.get("colsample_bytree", 1.0),
                l2_leaf_reg=params.get("reg_lambda", 0.0),
                random_seed=42,
                verbose=0,
                auto_class_weights="Balanced",
            )

        if model_lower == "xgboost":
            import xgboost as xgb

            return xgb.XGBClassifier(
                n_estimators=params.get("n_estimators", DEFAULT_N_ESTIMATORS),
                learning_rate=params.get("learning_rate", DEFAULT_LEARNING_RATE),
                max_depth=params.get("max_depth", DEFAULT_MAX_DEPTH),
                min_child_weight=params.get("min_child_samples", 20),
                subsample=params.get("subsample", 1.0),
                colsample_bytree=params.get("colsample_bytree", 1.0),
                reg_alpha=params.get("reg_alpha", 0.0),
                reg_lambda=params.get("reg_lambda", 0.0),
                random_state=42,
                verbosity=0,
                eval_metric="mlogloss",
            )

        raise ValueError(
            f"Unknown model '{model}'. Choose 'lightgbm', 'catboost', or 'xgboost'."
        )
