"""Explainability mixin — SHAP-based model explainability and permutation importance.

Provides interpretable feature attribution using SHAP values and sklearn
permutation importance. Because PCA components (PC1, PC2...) are not
human-readable, this mixin trains a separate lightweight LightGBM model
on the RAW scaled features (with original column names) instead of PCA
features. This "explainability model" is used only for SHAP analysis,
not for actual trading predictions.
"""
import warnings

import numpy as np
import pandas as pd

from atc_trading_bot.pipeline_warning import PipelineWarning


class ExplainabilityMixin:
    """Mixin for SHAP-based feature explainability and permutation importance."""

    def _require_features(self) -> bool:
        """Check that raw features have been computed."""
        if getattr(self, "features", None) is None:
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
        self._explainability_model = None

    def _train_explainability_model(self):
        """Train a lightweight LightGBM on raw scaled features for SHAP.

        This model is separate from the pipeline's active model. It uses
        the original feature names so SHAP values are human-interpretable.
        Labels are mapped from -1/0/1 to 0/1/2 for XGBoost compatibility.
        """
        import lightgbm as lgb

        features_scaled = getattr(self, "features_scaled", None)
        feature_names = self.features.columns.tolist()

        X = pd.DataFrame(features_scaled, columns=feature_names)

        # Align labels with feature index
        features_index = getattr(self, "features_index", None)
        if features_index is not None:
            valid_labels = self.labels.loc[features_index].values
        else:
            valid_labels = self.labels.values[: len(X)]

        # Map labels: -1 -> 0, 0 -> 1, 1 -> 2
        label_map = {-1: 0, 0: 1, 1: 2}
        y = np.array([label_map.get(v, 1) for v in valid_labels])

        model = lgb.LGBMClassifier(
            n_estimators=100,
            verbose=-1,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X, y)
        self._explainability_model = model

    def explain_prediction(self, index: int = -1):
        """Explain a single prediction using SHAP values.

        Trains the explainability model lazily if not already trained.
        Computes SHAP values for the observation at the given index and
        returns a waterfall-style horizontal bar chart of the top 15
        contributing features.

        Positive SHAP values push the prediction toward buy (class 2).
        Negative SHAP values push the prediction toward sell (class 0).

        Args:
            index: Row index into the feature matrix. Default: -1 (last row).

        Returns:
            plotly Figure with top 15 SHAP contributions, or None on error.
        """
        if not self._require_features():
            return None
        if not self._require_labels():
            return None

        import shap
        import plotly.graph_objects as go

        if self._explainability_model is None:
            self._train_explainability_model()

        feature_names = self.features.columns.tolist()
        X = pd.DataFrame(
            getattr(self, "features_scaled", None), columns=feature_names
        )

        explainer = shap.TreeExplainer(self._explainability_model)
        shap_values = explainer.shap_values(X)

        # Handle different SHAP output formats:
        # - list of 2D arrays (older SHAP, one per class)
        # - 3D array (newer SHAP: samples x features x classes)
        # - 2D array (binary/regression: samples x features)
        if isinstance(shap_values, list):
            sv_array = np.stack(shap_values, axis=-1)  # → (samples, features, classes)
        else:
            sv_array = np.array(shap_values)

        if sv_array.ndim == 3:
            # Multiclass: sum absolute values across classes, sign from buy class
            obs = sv_array[index]  # (features, classes)
            observation_shap = np.sum(np.abs(obs), axis=1) * np.sign(obs[:, -1])
        else:
            observation_shap = sv_array[index]  # (features,)

        # Sort by absolute value, take top 15
        top_n = min(15, len(observation_shap))
        abs_vals = np.abs(observation_shap)
        sorted_idx = np.argsort(abs_vals)[-top_n:]
        top_names = [feature_names[i] for i in sorted_idx]
        top_values = observation_shap[sorted_idx]

        # Color: positive (buy) green, negative (sell) red
        colors = [
            "#2ecc71" if v > 0 else "#e74c3c" for v in top_values
        ]

        fig = go.Figure(
            go.Bar(
                x=top_values,
                y=top_names,
                orientation="h",
                marker_color=colors,
            )
        )

        fig.update_layout(
            title=f"SHAP Explanation — Observation {index}",
            xaxis_title="SHAP Value (positive = buy, negative = sell)",
            template="plotly_dark",
            height=max(400, top_n * 28),
        )
        return fig

    def feature_importance_shap(self, top_n: int = 20):
        """Compute global feature importance using mean |SHAP| values.

        Trains the explainability model lazily if not already trained.
        Computes SHAP values across all training observations and
        averages the absolute values per feature.

        Args:
            top_n: Number of top features to display. Default: 20.

        Returns:
            plotly Figure with horizontal bar chart, or None on error.
        """
        if not self._require_features():
            return None
        if not self._require_labels():
            return None

        import shap
        import plotly.graph_objects as go

        if self._explainability_model is None:
            self._train_explainability_model()

        feature_names = self.features.columns.tolist()
        X = pd.DataFrame(
            getattr(self, "features_scaled", None), columns=feature_names
        )

        explainer = shap.TreeExplainer(self._explainability_model)
        shap_values = explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            sv_array = np.stack(shap_values, axis=-1)
        else:
            sv_array = np.array(shap_values)

        if sv_array.ndim == 3:
            abs_shap = np.sum(np.abs(sv_array), axis=2)  # sum across classes
        else:
            abs_shap = np.abs(sv_array)

        mean_importance = abs_shap.mean(axis=0)

        # Sort and take top_n
        sorted_idx = np.argsort(mean_importance)[-top_n:]
        top_names = [feature_names[i] for i in sorted_idx]
        top_values = mean_importance[sorted_idx]

        fig = go.Figure(
            go.Bar(
                x=top_values,
                y=top_names,
                orientation="h",
                marker_color="#3498db",
            )
        )

        fig.update_layout(
            title=f"Global Feature Importance — Mean |SHAP| (top {top_n})",
            xaxis_title="Mean |SHAP Value|",
            template="plotly_dark",
            height=max(400, top_n * 25),
        )
        return fig

    def feature_importance_permutation(self, top_n: int = 20):
        """Compute feature importance using sklearn permutation importance.

        Trains the explainability model lazily if not already trained.
        Shuffles each feature independently and measures the drop in
        accuracy to quantify each feature's contribution.

        Args:
            top_n: Number of top features to display. Default: 20.

        Returns:
            plotly Figure with horizontal bar chart, or None on error.
        """
        if not self._require_features():
            return None
        if not self._require_labels():
            return None

        from sklearn.inspection import permutation_importance
        import plotly.graph_objects as go

        if self._explainability_model is None:
            self._train_explainability_model()

        feature_names = self.features.columns.tolist()
        X = pd.DataFrame(
            getattr(self, "features_scaled", None), columns=feature_names
        )

        # Align labels
        features_index = getattr(self, "features_index", None)
        if features_index is not None:
            valid_labels = self.labels.loc[features_index].values
        else:
            valid_labels = self.labels.values[: len(X)]

        label_map = {-1: 0, 0: 1, 1: 2}
        y = np.array([label_map.get(v, 1) for v in valid_labels])

        result = permutation_importance(
            self._explainability_model,
            X,
            y,
            n_repeats=10,
            random_state=42,
            scoring="accuracy",
        )

        importances = result.importances_mean

        # Sort and take top_n
        sorted_idx = np.argsort(importances)[-top_n:]
        top_names = [feature_names[i] for i in sorted_idx]
        top_values = importances[sorted_idx]

        fig = go.Figure(
            go.Bar(
                x=top_values,
                y=top_names,
                orientation="h",
                marker_color="#e67e22",
            )
        )

        fig.update_layout(
            title=f"Permutation Feature Importance (top {top_n})",
            xaxis_title="Mean Accuracy Decrease",
            template="plotly_dark",
            height=max(400, top_n * 25),
        )
        return fig
