import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.signal_mixin import SignalMixin
from atc_trading_bot.pipeline_warning import PipelineWarning
from atc_trading_bot.strategies.bull_strategy import BullStrategy


class SignalBot(SignalMixin):
    """Minimal class using SignalMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.current_regime = kwargs.pop("current_regime", None)
        self.active_strategy = kwargs.pop("active_strategy", None)
        super().__init__(**kwargs)


@pytest.fixture
def trending_data():
    """Data with a clear uptrend to trigger bull strategy signals."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 30000 + np.linspace(0, 8000, n) + np.random.randn(n) * 100
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 50
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestSignalMixin:
    def test_generate_signals_returns_dict(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        signals = bot.generate_signals()

        assert isinstance(signals, dict)
        assert "regime" in signals
        assert "strategy" in signals
        assert "signal" in signals

    def test_signal_is_valid(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        signals = bot.generate_signals()

        assert signals["signal"] in ("buy", "sell", "hold")

    def test_signal_respects_regime(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        signals = bot.generate_signals()

        assert signals["regime"] == "bull"
        assert signals["strategy"] == "BullStrategy"

    def test_signals_stored(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        bot.generate_signals()

        assert bot.signals is not None

    def test_warns_without_data(self):
        bot = SignalBot(current_regime="bull", active_strategy=BullStrategy)
        with pytest.warns(PipelineWarning, match="fetch_data"):
            result = bot.generate_signals()
        assert result is None

    def test_warns_without_regime(self, trending_data):
        bot = SignalBot(df=trending_data, active_strategy=BullStrategy)
        with pytest.warns(PipelineWarning, match="detect_regime"):
            result = bot.generate_signals()
        assert result is None

    def test_warns_without_strategy(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull")
        with pytest.warns(PipelineWarning, match="select_strategy"):
            result = bot.generate_signals()
        assert result is None
