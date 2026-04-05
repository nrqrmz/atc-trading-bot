"""Tests for ExplainabilityMixin — SHAP and permutation importance."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from atc_trading_bot.mixins.explainability_mixin import ExplainabilityMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class ExplainBot(ExplainabilityMixin):
    def __init__(self, **kwargs):
        self.features = kwargs.pop("features", None)
        self.features_scaled = kwargs.pop("features_scaled", None)
        self.features_index = kwargs.pop("features_index", None)
        self.labels = kwargs.pop("labels", None)
        super().__init__(**kwargs)


@pytest.fixture
def explain_data():
    """Synthetic features + labels for explainability tests.

    200 samples, 15 features with realistic TA indicator names.
    Labels are correlated with a few features so SHAP can find signal.
    """
    np.random.seed(42)
    n = 200
    feature_names = [
        "momentum_rsi",
        "trend_adx",
        "volatility_bbw",
        "trend_macd",
        "trend_macd_signal",
        "momentum_stoch_rsi",
        "volatility_atr",
        "trend_ema_fast",
        "trend_ema_slow",
        "trend_sma_fast",
        "volume_obv",
        "volume_vwap",
        "momentum_uo",
        "trend_ichimoku_a",
        "volatility_kcp",
    ]
    n_features = len(feature_names)
    X = np.random.randn(n, n_features)

    features = pd.DataFrame(X, columns=feature_names)

    # Labels correlated with momentum_rsi (col 0) and trend_adx (col 1)
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.where(score > 0.8, 1, np.where(score < -0.8, -1, 0))

    dates = pd.date_range("2023-01-01", periods=n, freq="1D")
    labels = pd.Series(y, index=dates, name="label")
    features_index = dates

    return features, X, labels, features_index


class TestExplainPrediction:
    def test_returns_plotly_figure(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.explain_prediction(index=0)
        assert isinstance(fig, go.Figure)

    def test_default_index_last_row(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.explain_prediction()  # default index=-1
        assert isinstance(fig, go.Figure)
        assert "-1" in fig.layout.title.text

    def test_uses_plotly_dark_template(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.explain_prediction(index=0)
        assert fig.layout.template.layout.plot_bgcolor is not None

    def test_has_at_most_15_bars(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.explain_prediction(index=0)
        assert len(fig.data[0].y) <= 15


class TestFeatureImportanceShap:
    def test_returns_plotly_figure(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.feature_importance_shap()
        assert isinstance(fig, go.Figure)

    def test_respects_top_n(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.feature_importance_shap(top_n=5)
        assert len(fig.data[0].y) == 5

    def test_uses_plotly_dark_template(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.feature_importance_shap()
        assert fig.layout.template.layout.plot_bgcolor is not None


class TestFeatureImportancePermutation:
    def test_returns_plotly_figure(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.feature_importance_permutation()
        assert isinstance(fig, go.Figure)

    def test_uses_plotly_dark_template(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.feature_importance_permutation()
        assert fig.layout.template.layout.plot_bgcolor is not None

    def test_respects_top_n(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        fig = bot.feature_importance_permutation(top_n=8)
        assert len(fig.data[0].y) == 8


class TestGuards:
    def test_warns_without_features(self):
        labels = pd.Series([0, 1, -1])
        bot = ExplainBot(labels=labels)
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.explain_prediction()
        assert result is None

    def test_warns_without_labels(self):
        features = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        bot = ExplainBot(features=features, features_scaled=np.random.randn(10, 3))
        with pytest.warns(PipelineWarning, match="compute_labels"):
            result = bot.explain_prediction()
        assert result is None

    def test_shap_warns_without_features(self):
        labels = pd.Series([0, 1, -1])
        bot = ExplainBot(labels=labels)
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.feature_importance_shap()
        assert result is None

    def test_permutation_warns_without_labels(self):
        features = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        bot = ExplainBot(features=features, features_scaled=np.random.randn(10, 3))
        with pytest.warns(PipelineWarning, match="compute_labels"):
            result = bot.feature_importance_permutation()
        assert result is None


class TestLazyTraining:
    def test_model_not_trained_on_init(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        assert bot._explainability_model is None

    def test_model_trained_after_explain(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        bot.explain_prediction(index=0)
        assert bot._explainability_model is not None

    def test_model_reused_across_calls(self, explain_data):
        features, X_scaled, labels, idx = explain_data
        bot = ExplainBot(
            features=features, features_scaled=X_scaled,
            labels=labels, features_index=idx,
        )
        bot.explain_prediction(index=0)
        model_id = id(bot._explainability_model)
        bot.feature_importance_shap()
        assert id(bot._explainability_model) == model_id
