"""Tests for LabelingMixin — Triple-Barrier Method."""

import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.labeling_mixin import LabelingMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class LabelBot(LabelingMixin):
    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        super().__init__(**kwargs)


@pytest.fixture
def trending_up_data():
    """Strong uptrend — should produce mostly buy labels."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 30000 + np.linspace(0, 10000, n) + np.random.randn(n) * 50
    high = close + np.abs(np.random.randn(n) * 100) + 50
    low = close - np.abs(np.random.randn(n) * 100) - 50
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 30,
        "High": high, "Low": low, "Close": close,
        "Volume": np.abs(np.random.randn(n) * 1e6) + 5e5,
    }, index=dates)


@pytest.fixture
def trending_down_data():
    """Strong downtrend — should produce mostly sell labels."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 40000 - np.linspace(0, 10000, n) + np.random.randn(n) * 50
    high = close + np.abs(np.random.randn(n) * 100) + 50
    low = close - np.abs(np.random.randn(n) * 100) - 50
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 30,
        "High": high, "Low": low, "Close": close,
        "Volume": np.abs(np.random.randn(n) * 1e6) + 5e5,
    }, index=dates)


@pytest.fixture
def sideways_data():
    """Range-bound data — should produce mostly hold labels."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 30000 + np.random.randn(n) * 20  # tiny noise
    high = close + 10
    low = close - 10
    return pd.DataFrame({
        "Open": close, "High": high, "Low": low, "Close": close,
        "Volume": np.abs(np.random.randn(n) * 1e6) + 5e5,
    }, index=dates)


class TestComputeLabels:
    def test_returns_self(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        result = bot.compute_labels()
        assert result is bot

    def test_labels_stored(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        assert bot.labels is not None
        assert len(bot.labels) == len(trending_up_data)

    def test_labels_added_to_df(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        assert "label" in bot.df.columns

    def test_labels_are_valid_values(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        assert set(bot.labels.unique()).issubset({-1, 0, 1})

    def test_uptrend_has_buy_labels(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        buy_count = (bot.labels == 1).sum()
        assert buy_count > 0

    def test_downtrend_has_sell_labels(self, trending_down_data):
        bot = LabelBot(df=trending_down_data)
        bot.compute_labels()
        sell_count = (bot.labels == -1).sum()
        assert sell_count > 0

    def test_sideways_has_mostly_hold(self, sideways_data):
        bot = LabelBot(df=sideways_data)
        bot.compute_labels()
        hold_pct = (bot.labels == 0).mean()
        assert hold_pct > 0.5

    def test_warns_without_data(self):
        bot = LabelBot()
        with pytest.warns(PipelineWarning, match="fetch_data"):
            bot.compute_labels()

    def test_custom_parameters(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels(tp_factor=1.0, sl_factor=1.0, max_holding=5)
        assert bot.labels is not None

    def test_tight_barriers_more_labels(self, trending_up_data):
        bot_tight = LabelBot(df=trending_up_data.copy())
        bot_tight.compute_labels(tp_factor=0.5, sl_factor=0.5)
        tight_holds = (bot_tight.labels == 0).sum()

        bot_wide = LabelBot(df=trending_up_data.copy())
        bot_wide.compute_labels(tp_factor=5.0, sl_factor=5.0)
        wide_holds = (bot_wide.labels == 0).sum()

        # Wider barriers → more timeouts → more holds
        assert wide_holds >= tight_holds


class TestLabelsSummary:
    def test_returns_dataframe(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        summary = bot.labels_summary()
        assert isinstance(summary, pd.DataFrame)

    def test_has_expected_index(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        summary = bot.labels_summary()
        assert summary.index.name == "label"
        assert list(summary.index) == ["buy", "sell", "hold"]

    def test_has_count_and_percentage(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        summary = bot.labels_summary()
        assert "count" in summary.columns
        assert "percentage" in summary.columns

    def test_percentages_sum_to_100(self, trending_up_data):
        bot = LabelBot(df=trending_up_data)
        bot.compute_labels()
        summary = bot.labels_summary()
        assert abs(summary["percentage"].sum() - 100.0) < 0.2

    def test_warns_without_labels(self):
        bot = LabelBot()
        with pytest.warns(PipelineWarning, match="compute_labels"):
            result = bot.labels_summary()
        assert result is None
