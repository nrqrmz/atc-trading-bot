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
