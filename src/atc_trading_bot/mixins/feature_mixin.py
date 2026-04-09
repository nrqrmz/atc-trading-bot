import warnings

import numpy as np
from atc_trading_bot.config import (
    CORRELATION_THRESHOLD,
    DEFAULT_N_COMPONENTS,
    EXCLUDE_COLS,
    NAN_COLUMN_THRESHOLD,
)
from atc_trading_bot.pipeline_warning import PipelineWarning
import pandas as pd
import ta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureMixin:
    """Mixin for computing technical indicators, standardizing, and applying PCA."""

    def _require_data(self) -> bool:
        """Check that OHLCV data has been loaded."""
        if getattr(self, "df", None) is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
            return False
        return True

    def _require_features(self) -> bool:
        """Check that PCA features have been computed."""
        if getattr(self, "features_pca", None) is None:
            warnings.warn("No features available. Call compute_features first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features: pd.DataFrame | None = None
        self.features_scaled: np.ndarray | None = None
        self.features_pca: np.ndarray | None = None
        self.features_index: pd.Index | None = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def compute_features(self, n_components: int = DEFAULT_N_COMPONENTS):
        """Compute TA features, standardize, and apply PCA.

        Args:
            n_components: Number of PCA components to retain. Capped to available features.
        """
        if not self._require_data():
            return

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="ta")
            df_ta = ta.add_all_ta_features(
                self.df.copy(),
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                fillna=False,
            )

        # Exclude problematic features (binary, price-level, accumulative)
        feature_cols = [c for c in df_ta.columns if c not in EXCLUDE_COLS]
        self.features = df_ta[feature_cols].copy()

        # Clean: forward fill, replace inf
        self.features.ffill(inplace=True)
        self.features.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns with >50% NaN, then drop remaining NaN rows (indicator warmup)
        thresh = int(len(self.features) * NAN_COLUMN_THRESHOLD)
        self.features.dropna(axis=1, thresh=thresh, inplace=True)
        self.features.dropna(inplace=True)

        # Store valid index for alignment with self.df in other mixins
        self.features_index = self.features.index

        # Remove constant features (zero variance)
        non_const = self.features.columns[self.features.std() > 0]
        self.features = self.features[non_const]

        # Remove highly correlated features (|r| > threshold)
        corr = self.features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
        self.features = self.features.drop(columns=to_drop)

        # Standardize
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)

        # PCA — cap components to available features
        actual_components = min(n_components, self.features_scaled.shape[1])
        self.pca = PCA(n_components=actual_components)
        self.features_pca = self.pca.fit_transform(self.features_scaled)
        return self

    def features_summary(self, top_n: int = 5) -> pd.DataFrame | None:
        """Return a summary of PCA reduction as a DataFrame.

        Each row represents a principal component with its explained variance
        and the top contributing features (by absolute loading weight).

        Args:
            top_n: Number of top features to show per component. Default: 5.

        Returns:
            DataFrame indexed by component name (``PC1``, ``PC2``, ...)
            with columns ``variance_pct``, ``cumulative_pct``, ``top_features``.
        """
        if not self._require_features():
            return

        explained = self.pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        rows = []
        for i, ratio in enumerate(explained):
            weights = np.abs(self.pca.components_[i])
            top_idx = weights.argsort()[::-1][:top_n]
            top_names = [self.features.columns[j] for j in top_idx]
            rows.append({
                "variance_pct": round(ratio * 100, 1),
                "cumulative_pct": round(cumulative[i] * 100, 1),
                "top_features": ", ".join(top_names),
            })

        df = pd.DataFrame(rows)
        df.index = [f"PC{i+1}" for i in range(len(rows))]
        df.index.name = "component"
        return df
