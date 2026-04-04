import pytest
from backtesting import Strategy

from atc_trading_bot.mixins.strategy_mixin import StrategyMixin
from atc_trading_bot.pipeline_warning import PipelineWarning
from atc_trading_bot.strategies.bear_strategy import BearStrategy
from atc_trading_bot.strategies.bull_strategy import BullStrategy
from atc_trading_bot.strategies.sideways_strategy import SidewaysStrategy


class StrategyBot(StrategyMixin):
    """Minimal class using StrategyMixin for testing."""

    def __init__(self, **kwargs):
        self.current_regime = kwargs.pop("current_regime", None)
        super().__init__(**kwargs)


class TestStrategyMixin:
    def test_select_strategy_bull(self):
        bot = StrategyBot(current_regime="bull")
        strategy = bot.select_strategy()
        assert strategy is BullStrategy

    def test_select_strategy_bear(self):
        bot = StrategyBot(current_regime="bear")
        strategy = bot.select_strategy()
        assert strategy is BearStrategy

    def test_select_strategy_sideways(self):
        bot = StrategyBot(current_regime="sideways")
        strategy = bot.select_strategy()
        assert strategy is SidewaysStrategy

    def test_select_strategy_stores_active(self):
        bot = StrategyBot(current_regime="bull")
        bot.select_strategy()
        assert bot.active_strategy is BullStrategy

    def test_warns_without_regime(self):
        bot = StrategyBot()
        with pytest.warns(PipelineWarning, match="detect_regime"):
            result = bot.select_strategy()
        assert result is None

    def test_get_strategy_for_regime(self):
        bot = StrategyBot()
        assert bot.get_strategy_for_regime("bull") is BullStrategy
        assert bot.get_strategy_for_regime("bear") is BearStrategy
        assert bot.get_strategy_for_regime("sideways") is SidewaysStrategy

    def test_get_strategy_for_invalid_regime(self):
        bot = StrategyBot()
        with pytest.raises(ValueError, match="Unknown regime"):
            bot.get_strategy_for_regime("unknown")

    def test_all_strategies_inherit_from_backtesting_strategy(self):
        for strategy_cls in StrategyMixin.REGIME_STRATEGY_MAP.values():
            assert issubclass(strategy_cls, Strategy)
