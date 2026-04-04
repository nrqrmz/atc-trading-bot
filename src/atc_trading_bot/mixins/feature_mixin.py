import warnings

import numpy as np
from atc_trading_bot.pipeline_warning import PipelineWarning
import pandas as pd
import ta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Features to exclude from regime detection
_EXCLUDE_COLS = {
    # Raw OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # Binary (0/1 only — not continuous)
    "volatility_bbhi", "volatility_bbli",
    "volatility_kchi", "volatility_kcli",
    "trend_psar_up_indicator", "trend_psar_down_indicator",
    # Price-level (scale-dependent, redundant with %B/width)
    "trend_ema_fast", "trend_ema_slow",
    "trend_sma_fast", "trend_sma_slow",
    "volatility_bbh", "volatility_bbl", "volatility_bbm",
    "volatility_kch", "volatility_kcl", "volatility_kcc",
    # Accumulative (no ceiling, non-stationary)
    "volume_adi", "volume_obv", "volume_vpt",
    "volume_nvi", "others_cr", "volume_vwap",
}

_CORRELATION_THRESHOLD = 0.95
_NAN_COLUMN_THRESHOLD = 0.5


class FeatureMixin:
    """Mixin for computing technical indicators, standardizing, and applying PCA."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features: pd.DataFrame | None = None
        self.features_scaled: np.ndarray | None = None
        self.features_pca: np.ndarray | None = None
        self.features_index: pd.Index | None = None
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
                fillna=False,
            )

        # Exclude problematic features (binary, price-level, accumulative)
        feature_cols = [c for c in df_ta.columns if c not in _EXCLUDE_COLS]
        self.features = df_ta[feature_cols].copy()

        # Clean: forward fill, replace inf
        self.features.ffill(inplace=True)
        self.features.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop columns with >50% NaN, then drop remaining NaN rows (indicator warmup)
        thresh = int(len(self.features) * _NAN_COLUMN_THRESHOLD)
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
        to_drop = [col for col in upper.columns if any(upper[col] > _CORRELATION_THRESHOLD)]
        self.features = self.features.drop(columns=to_drop)

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
