import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.feature_mixin import FeatureMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class FeatureBot(FeatureMixin):
    """Minimal class using FeatureMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        super().__init__(**kwargs)


class TestFeatureMixin:
    def test_compute_features_generates_ta_columns(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        assert bot.features is not None
        # ta generates many indicators, should have more columns than OHLCV
        assert bot.features.shape[1] > 5

    def test_compute_features_no_nans(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        assert not bot.features.isnull().any().any()

    def test_features_standardized(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        # After standardization, mean should be ~0 and std ~1
        means = bot.features_scaled.mean(axis=0)
        stds = bot.features_scaled.std(axis=0)
        np.testing.assert_array_almost_equal(means, 0, decimal=1)
        np.testing.assert_array_almost_equal(stds, 1, decimal=0)

    def test_pca_reduces_dimensionality(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        n_components = 10
        bot.compute_features(n_components=n_components)

        assert bot.features_pca is not None
        assert bot.features_pca.shape[1] == n_components
        assert bot.features_pca.shape[0] == bot.features.shape[0]

    def test_pca_default_components(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        # Default n_components should be set and features_pca populated
        assert bot.features_pca is not None
        assert bot.features_pca.shape[1] > 0

    def test_scaler_and_pca_objects_stored(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        assert bot.scaler is not None
        assert bot.pca is not None

    def test_warns_without_data(self):
        bot = FeatureBot()
        with pytest.warns(PipelineWarning, match="fetch_data"):
            result = bot.compute_features()
        assert result is None

    def test_features_summary_prints_output(self, sample_ohlcv_data, capsys):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()
        bot.features_summary()

        output = capsys.readouterr().out
        assert "Total features:" in output
        assert "PCA components:" in output
        assert "variance explained" in output
        assert "PC1" in output

    def test_features_summary_warns_without_features(self):
        bot = FeatureBot()
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.features_summary()
        assert result is None

    def test_features_index_stored(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        assert bot.features_index is not None
        # features_index should be a subset of df.index
        assert bot.features_index.isin(sample_ohlcv_data.index).all()

    def test_features_index_shorter_than_df(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        # NaN rows from indicator warmup should be dropped
        assert len(bot.features_index) <= len(sample_ohlcv_data)
        assert len(bot.features_index) == len(bot.features_pca)

    def test_excludes_problematic_features(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        excluded = {"volatility_bbhi", "volatility_bbli", "volume_adi", "volume_obv"}
        remaining_cols = set(bot.features.columns)
        assert remaining_cols.isdisjoint(excluded)

    def test_no_highly_correlated_features(self, sample_ohlcv_data):
        bot = FeatureBot(df=sample_ohlcv_data)
        bot.compute_features()

        corr = bot.features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_corr = upper.max().max()
        assert max_corr <= 0.95
