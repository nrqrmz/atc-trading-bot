"""Tests for OptimizationMixin — Optuna tuning and walk-forward validation."""

import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.optimization_mixin import OptimizationMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class OptBot(OptimizationMixin):
    def __init__(self, **kwargs):
        self.features_pca = kwargs.pop("features_pca", None)
        self.features_index = kwargs.pop("features_index", None)
        self.labels = kwargs.pop("labels", None)
        super().__init__(**kwargs)


@pytest.fixture
def ml_data():
    """Synthetic features + labels for optimization tests."""
    np.random.seed(42)
    n = 300
    n_features = 10
    X = np.random.randn(n, n_features)

    # Labels correlated with first feature (learnable pattern)
    y = np.where(X[:, 0] > 0.5, 1, np.where(X[:, 0] < -0.5, -1, 0))

    dates = pd.date_range("2023-01-01", periods=n, freq="1D")
    labels = pd.Series(y, index=dates, name="label")
    features_index = dates

    return X, labels, features_index


# ------------------------------------------------------------------
# optimize_model
# ------------------------------------------------------------------


class TestOptimizeModel:
    def test_returns_dataframe_with_param_index(self, ml_data):
        X, labels, idx = ml_data
        bot = OptBot(features_pca=X, labels=labels, features_index=idx)
        result = bot.optimize_model(n_trials=5)

        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "param"
        assert "value" in result.columns

    def test_stores_optuna_study(self, ml_data):
        import optuna

        X, labels, idx = ml_data
        bot = OptBot(features_pca=X, labels=labels, features_index=idx)
        bot.optimize_model(n_trials=5)

        assert bot.optuna_study is not None
        assert isinstance(bot.optuna_study, optuna.study.Study)

    def test_fast_with_few_trials(self, ml_data):
        X, labels, idx = ml_data
        bot = OptBot(features_pca=X, labels=labels, features_index=idx)
        result = bot.optimize_model(n_trials=5)

        assert len(result) > 0
        assert len(bot.optuna_study.trials) == 5

    def test_warns_without_features(self):
        labels = pd.Series([0, 1, -1])
        bot = OptBot(labels=labels)
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.optimize_model(n_trials=5)
        assert result is None

    def test_warns_without_labels(self):
        X = np.random.randn(100, 5)
        bot = OptBot(features_pca=X)
        with pytest.warns(PipelineWarning, match="compute_labels"):
            result = bot.optimize_model(n_trials=5)
        assert result is None


# ------------------------------------------------------------------
# walk_forward
# ------------------------------------------------------------------


class TestWalkForward:
    def test_returns_dataframe_with_mean_row(self, ml_data):
        X, labels, idx = ml_data
        bot = OptBot(features_pca=X, labels=labels, features_index=idx)
        result = bot.walk_forward(window_size=100, step_size=21)

        assert isinstance(result, pd.DataFrame)
        assert "Mean" in result.index

    def test_has_expected_columns(self, ml_data):
        X, labels, idx = ml_data
        bot = OptBot(features_pca=X, labels=labels, features_index=idx)
        result = bot.walk_forward(window_size=100, step_size=21)

        assert "accuracy" in result.columns
        assert "f1" in result.columns
        assert "precision" in result.columns
        assert "recall" in result.columns

    def test_warns_without_features(self):
        labels = pd.Series([0, 1, -1])
        bot = OptBot(labels=labels)
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.walk_forward()
        assert result is None

    def test_warns_without_labels(self):
        X = np.random.randn(100, 5)
        bot = OptBot(features_pca=X)
        with pytest.warns(PipelineWarning, match="compute_labels"):
            result = bot.walk_forward()
        assert result is None

    def test_custom_window_size(self, ml_data):
        X, labels, idx = ml_data
        bot = OptBot(features_pca=X, labels=labels, features_index=idx)
        result = bot.walk_forward(window_size=150, step_size=30)

        assert isinstance(result, pd.DataFrame)
        assert "Mean" in result.index
        # With 300 samples, window=150, step=30 -> windows at 0,30,60,90,120
        # 150+30=180 <= 300, 180+30=210 <= 300, ... last valid start: 120 (120+150+30=300)
        n_windows = len(result) - 1  # subtract Mean row
        assert n_windows > 0
        # Verify window labels
        for i in range(n_windows):
            assert f"Window {i}" in result.index
