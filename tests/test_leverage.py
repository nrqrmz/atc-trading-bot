import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.backtest_mixin import BacktestMixin
from atc_trading_bot.mixins.strategy_mixin import StrategyMixin
from atc_trading_bot.strategies.bull_strategy import BullStrategy


class BacktestBot(BacktestMixin, StrategyMixin):
    """Minimal class using BacktestMixin for testing leverage."""

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


class TestLeverage:
    def test_backtest_with_default_leverage(self, long_ohlcv_data):
        """Leverage=1 (default) should produce same results as before."""
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest(leverage=1)

        assert isinstance(results, pd.DataFrame)
        assert list(results.columns) == ["metric", "value", "description"]
        metrics = results["metric"].tolist()
        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics

    def test_backtest_with_leverage_2(self, long_ohlcv_data):
        """Leverage=2 should run successfully with margin=0.5."""
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest(leverage=2)

        assert isinstance(results, pd.DataFrame)
        metrics = results["metric"].tolist()
        assert "sharpe_ratio" in metrics
        assert "total_return" in metrics
        assert "num_trades" in metrics

    def test_leverage_parameter_accepted(self, long_ohlcv_data):
        """The leverage parameter should be accepted without errors."""
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()

        # Should not raise TypeError for unexpected keyword argument
        results = bot.backtest(strategy=BullStrategy, leverage=3)
        assert isinstance(results, pd.DataFrame)

    def test_leverage_1_matches_no_leverage(self, long_ohlcv_data):
        """Leverage=1 should produce identical results to omitting leverage."""
        bot1 = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot1.select_strategy()
        results_default = bot1.backtest()

        bot2 = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot2.select_strategy()
        results_explicit = bot2.backtest(leverage=1)

        # Same metrics should be present
        assert results_default["metric"].tolist() == results_explicit["metric"].tolist()

        # Same numeric values
        for metric in ["sharpe_ratio", "total_return", "num_trades"]:
            val_default = results_default.loc[
                results_default["metric"] == metric, "value"
            ].iloc[0]
            val_explicit = results_explicit.loc[
                results_explicit["metric"] == metric, "value"
            ].iloc[0]
            assert val_default == val_explicit, f"{metric} differs with leverage=1"
