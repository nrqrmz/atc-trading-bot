"""Tests for new strategies: Momentum, Breakout, Volatility."""

import numpy as np
import pandas as pd
import pytest
from backtesting import Strategy
from backtesting.lib import FractionalBacktest as Backtest

from atc_trading_bot.config import DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT, DEFAULT_POSITION_SIZE
from atc_trading_bot.strategies.bull_strategy import BullStrategy
from atc_trading_bot.strategies.bear_strategy import BearStrategy
from atc_trading_bot.strategies.sideways_strategy import SidewaysStrategy
from atc_trading_bot.strategies.momentum_strategy import MomentumStrategy
from atc_trading_bot.strategies.breakout_strategy import BreakoutStrategy
from atc_trading_bot.strategies.volatility_strategy import VolatilityStrategy


@pytest.fixture
def trending_data():
    """Data with a clear uptrend for momentum/breakout strategies."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    trend = np.linspace(0, 8000, n)
    noise = np.cumsum(np.random.randn(n) * 100)
    close = 30000 + trend + noise
    high = close + np.abs(np.random.randn(n) * 300)
    low = close - np.abs(np.random.randn(n) * 300)
    open_ = close + np.random.randn(n) * 100
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def volatile_data():
    """Data with high volatility swings for volatility strategy."""
    np.random.seed(123)
    n = 200
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    # Alternate between low and high volatility periods
    close = 30000.0 + np.zeros(n)
    for i in range(1, n):
        vol = 50 if (i // 40) % 2 == 0 else 500  # alternating vol regimes
        close[i] = close[i - 1] + np.random.randn() * vol
    high = close + np.abs(np.random.randn(n) * 200) + 50
    low = close - np.abs(np.random.randn(n) * 200) - 50
    open_ = close + np.random.randn(n) * 50
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestMomentumStrategy:
    def test_inherits_from_strategy(self):
        assert issubclass(MomentumStrategy, Strategy)

    def test_has_required_parameters(self):
        assert hasattr(MomentumStrategy, "roc_period")
        assert hasattr(MomentumStrategy, "rsi_period")
        assert hasattr(MomentumStrategy, "roc_threshold")

    def test_runs_backtest_without_error(self, trending_data):
        bt = Backtest(trending_data, MomentumStrategy, cash=100_000,
                      commission=0.001, finalize_trades=True)
        stats = bt.run()
        assert stats is not None
        assert stats["# Trades"] >= 0

    def test_generates_trades_in_trending_market(self, trending_data):
        bt = Backtest(trending_data, MomentumStrategy, cash=100_000,
                      commission=0.001, finalize_trades=True)
        stats = bt.run()
        assert stats["# Trades"] > 0


class TestBreakoutStrategy:
    def test_inherits_from_strategy(self):
        assert issubclass(BreakoutStrategy, Strategy)

    def test_has_required_parameters(self):
        assert hasattr(BreakoutStrategy, "channel_period")
        assert hasattr(BreakoutStrategy, "volume_ma_period")

    def test_runs_backtest_without_error(self, trending_data):
        bt = Backtest(trending_data, BreakoutStrategy, cash=100_000,
                      commission=0.001, finalize_trades=True)
        stats = bt.run()
        assert stats is not None
        assert stats["# Trades"] >= 0

    def test_generates_trades_in_trending_market(self, trending_data):
        bt = Backtest(trending_data, BreakoutStrategy, cash=100_000,
                      commission=0.001, finalize_trades=True)
        stats = bt.run()
        assert stats["# Trades"] > 0


class TestVolatilityStrategy:
    def test_inherits_from_strategy(self):
        assert issubclass(VolatilityStrategy, Strategy)

    def test_has_required_parameters(self):
        assert hasattr(VolatilityStrategy, "atr_period")
        assert hasattr(VolatilityStrategy, "atr_ma_period")
        assert hasattr(VolatilityStrategy, "atr_spike_multiplier")

    def test_runs_backtest_without_error(self, volatile_data):
        bt = Backtest(volatile_data, VolatilityStrategy, cash=100_000,
                      commission=0.001, finalize_trades=True)
        stats = bt.run()
        assert stats is not None
        assert stats["# Trades"] >= 0

    def test_generates_trades_in_volatile_market(self, volatile_data):
        bt = Backtest(volatile_data, VolatilityStrategy, cash=100_000,
                      commission=0.001, finalize_trades=True)
        stats = bt.run()
        assert stats["# Trades"] > 0


class TestRiskManagement:
    """Verify all strategies have risk management parameters."""

    ALL_STRATEGIES = [
        BullStrategy, BearStrategy, SidewaysStrategy,
        MomentumStrategy, BreakoutStrategy, VolatilityStrategy,
    ]

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_has_stop_loss(self, strategy_cls):
        assert hasattr(strategy_cls, "stop_loss")
        assert strategy_cls.stop_loss == DEFAULT_STOP_LOSS

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_has_take_profit(self, strategy_cls):
        assert hasattr(strategy_cls, "take_profit")
        assert strategy_cls.take_profit == DEFAULT_TAKE_PROFIT

    @pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES)
    def test_has_position_size(self, strategy_cls):
        assert hasattr(strategy_cls, "position_size")
        assert strategy_cls.position_size == DEFAULT_POSITION_SIZE
