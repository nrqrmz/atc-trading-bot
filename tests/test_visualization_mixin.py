"""Tests for the VisualizationMixin chart methods."""

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from atc_trading_bot.mixins.visualization_mixin import VisualizationMixin
from atc_trading_bot.pipeline_warning import PipelineWarning
from atc_trading_bot.strategies.bull_strategy import BullStrategy


class VizBot(VisualizationMixin):
    """Minimal class using VisualizationMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.results = kwargs.pop("results", None)
        self.signals = kwargs.pop("signals", None)
        self.cv_results = kwargs.pop("cv_results", None)
        self.pca = kwargs.pop("pca", None)
        self.features = kwargs.pop("features", None)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 30000 + np.cumsum(np.random.randn(n) * 200)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 50,
        "High": close + 100,
        "Low": close - 100,
        "Close": close,
        "Volume": np.abs(np.random.randn(n) * 1e6) + 5e5,
    }, index=dates)


@pytest.fixture
def sample_results():
    return pd.DataFrame([
        {"metric": "backtest_start", "value": "2023-04-01", "description": ""},
        {"metric": "backtest_end", "value": "2023-07-01", "description": ""},
        {"metric": "total_return", "value": 0.15, "description": ""},
        {"metric": "sharpe_ratio", "value": 1.2, "description": ""},
        {"metric": "max_drawdown", "value": -0.08, "description": ""},
    ])


@pytest.fixture
def sample_cv_results():
    return [
        {"fold": 0, "sharpe_ratio": 1.2, "total_return": 0.1, "max_drawdown": -0.05, "win_rate": 0.6},
        {"fold": 1, "sharpe_ratio": 0.8, "total_return": 0.05, "max_drawdown": -0.08, "win_rate": 0.55},
        {"fold": 2, "sharpe_ratio": 1.5, "total_return": 0.12, "max_drawdown": -0.03, "win_rate": 0.65},
    ]


@pytest.fixture
def sample_pca_and_features():
    np.random.seed(42)
    n = 100
    n_features = 20
    data = np.random.randn(n, n_features)
    cols = [f"feature_{i}" for i in range(n_features)]
    features = pd.DataFrame(data, columns=cols)
    pca = PCA(n_components=5)
    pca.fit(data)
    return pca, features


class TestPlotEquityCurve:
    def test_returns_figure(self, sample_df, sample_results):
        bot = VizBot(df=sample_df, results=sample_results)
        fig = bot.plot_equity_curve()
        assert fig is not None
        assert fig.layout.template.layout.plot_bgcolor is not None  # plotly_dark

    def test_warns_without_results(self):
        bot = VizBot()
        with pytest.warns(PipelineWarning, match="backtest"):
            result = bot.plot_equity_curve()
        assert result is None

    def test_has_equity_and_drawdown_traces(self, sample_df, sample_results):
        bot = VizBot(df=sample_df, results=sample_results)
        fig = bot.plot_equity_curve()
        trace_names = [t.name for t in fig.data]
        assert "Equity" in trace_names
        assert "Drawdown" in trace_names


class TestPlotSignals:
    def test_returns_figure(self, sample_df):
        signals = {"regime": "bull", "strategy": "BullStrategy",
                   "signal": "buy", "confidence": 0.85}
        bot = VizBot(df=sample_df, signals=signals)
        fig = bot.plot_signals()
        assert fig is not None

    def test_warns_without_signals(self, sample_df):
        bot = VizBot(df=sample_df)
        with pytest.warns(PipelineWarning, match="signals"):
            result = bot.plot_signals()
        assert result is None

    def test_warns_without_data(self):
        bot = VizBot(signals={"signal": "buy", "regime": "bull",
                              "strategy": "X", "confidence": 0.9})
        with pytest.warns(PipelineWarning, match="fetch_data"):
            result = bot.plot_signals()
        assert result is None

    def test_shows_signal_marker(self, sample_df):
        signals = {"regime": "bull", "strategy": "BullStrategy",
                   "signal": "sell", "confidence": 0.7}
        bot = VizBot(df=sample_df, signals=signals)
        fig = bot.plot_signals()
        assert len(fig.data) >= 2  # price line + signal marker


class TestPlotCpcvResults:
    def test_returns_figure(self, sample_cv_results):
        bot = VizBot(cv_results=sample_cv_results)
        fig = bot.plot_cpcv_results()
        assert fig is not None

    def test_warns_without_cv_results(self):
        bot = VizBot()
        with pytest.warns(PipelineWarning, match="CPCV"):
            result = bot.plot_cpcv_results()
        assert result is None

    def test_has_four_subplots(self, sample_cv_results):
        bot = VizBot(cv_results=sample_cv_results)
        fig = bot.plot_cpcv_results()
        assert len(fig.data) == 4  # one bar trace per metric


class TestPlotFeatureImportance:
    def test_returns_figure(self, sample_pca_and_features):
        pca, features = sample_pca_and_features
        bot = VizBot(pca=pca, features=features)
        fig = bot.plot_feature_importance()
        assert fig is not None

    def test_warns_without_pca(self):
        bot = VizBot()
        with pytest.warns(PipelineWarning, match="features"):
            result = bot.plot_feature_importance()
        assert result is None

    def test_respects_top_n(self, sample_pca_and_features):
        pca, features = sample_pca_and_features
        bot = VizBot(pca=pca, features=features)
        fig = bot.plot_feature_importance(top_n=5)
        # Bar chart should have 5 bars
        assert len(fig.data[0].y) == 5
