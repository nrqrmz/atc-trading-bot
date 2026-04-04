import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.backtest_mixin import BacktestMixin
from atc_trading_bot.pipeline_warning import PipelineWarning
from atc_trading_bot.mixins.strategy_mixin import StrategyMixin
from atc_trading_bot.strategies.bull_strategy import BullStrategy


class BacktestBot(BacktestMixin, StrategyMixin):
    """Minimal class using BacktestMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.current_regime = kwargs.pop("current_regime", None)
        self.features_pca = kwargs.pop("features_pca", None)
        super().__init__(**kwargs)


@pytest.fixture
def long_ohlcv_data():
    """Generate longer OHLCV data suitable for backtesting (300 days with trend)."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    # Create a trending market so SMA crossovers generate trades
    trend = np.linspace(0, 5000, n)
    noise = np.cumsum(np.random.randn(n) * 100)
    close = 30000 + trend + noise
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestBacktestMixin:
    def test_backtest_returns_metrics(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest()

        assert isinstance(results, dict)
        expected_keys = [
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "total_return", "buy_and_hold_return",
            "num_trades",
        ]
        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"

    def test_backtest_with_explicit_strategy(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data)
        results = bot.backtest(strategy=BullStrategy)

        assert isinstance(results, dict)
        assert results["num_trades"] >= 0

    def test_backtest_stores_results(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        bot.backtest()

        assert bot.results is not None

    def test_backtest_warns_without_data(self):
        bot = BacktestBot(current_regime="bull")
        bot.select_strategy()
        with pytest.warns(PipelineWarning, match="fetch_data"):
            result = bot.backtest()
        assert result is None

    def test_backtest_warns_without_strategy(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data)
        with pytest.warns(PipelineWarning, match="select_strategy"):
            result = bot.backtest()
        assert result is None

    def test_max_drawdown_is_negative_or_zero(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest()

        assert results["max_drawdown"] <= 0

    def test_cpcv_returns_list_of_metrics(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        cv_results = bot.cross_validate_cpcv(n_splits=3)

        assert isinstance(cv_results, list)
        assert len(cv_results) > 0
        for result in cv_results:
            assert "sharpe_ratio" in result
            assert "fold" in result

    def test_cpcv_stores_results(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        bot.cross_validate_cpcv(n_splits=3)

        assert bot.cv_results is not None
