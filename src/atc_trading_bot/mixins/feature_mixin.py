import warnings

import numpy as np
from atc_trading_bot.pipeline_warning import PipelineWarning
import pandas as pd
import ta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureMixin:
    """Mixin for computing technical indicators, standardizing, and applying PCA."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features: pd.DataFrame | None = None
        self.features_scaled: np.ndarray | None = None
        self.features_pca: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None

    def compute_features(self, n_components: int = 10) -> None:
        """Compute TA features, standardize, and apply PCA.

        Args:
            n_components: Number of PCA components to retain. Capped to available features.
        """
        if self.df is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
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
                fillna=True,
            )

        # Drop original OHLCV columns to keep only indicators
        feature_cols = [c for c in df_ta.columns if c not in ["Open", "High", "Low", "Close", "Volume"]]
        self.features = df_ta[feature_cols].copy()

        # Remove any columns that are constant (zero variance)
        non_const = self.features.columns[self.features.std() > 0]
        self.features = self.features[non_const]

        # Standardize
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(self.features)

        # PCA — cap components to available features
        actual_components = min(n_components, self.features_scaled.shape[1])
        self.pca = PCA(n_components=actual_components)
        self.features_pca = self.pca.fit_transform(self.features_scaled)

    def features_summary(self, top_n: int = 5) -> None:
        """Print a summary of computed features and PCA reduction.

        Args:
            top_n: Number of top features to show per PCA component. Default: 5.
        """
        if self.features is None or self.pca is None:
            warnings.warn("No features computed. Call compute_features first.", PipelineWarning)
            return

        total = self.features.shape[1]
        n_components = self.pca.n_components_
        explained = self.pca.explained_variance_ratio_
        total_var = explained.sum() * 100

        print(f"Total features: {total}")
        print(f"PCA components: {n_components} ({total_var:.1f}% variance explained)\n")

        for i, ratio in enumerate(explained):
            weights = np.abs(self.pca.components_[i])
            top_idx = weights.argsort()[::-1][:top_n]
            top_names = [self.features.columns[j] for j in top_idx]
            print(f"  PC{i+1} ({ratio*100:.1f}%): {', '.join(top_names)}")
