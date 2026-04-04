import pytest
from backtesting import Strategy

from atc_trading_bot.mixins.strategy_mixin import (
    STRATEGY_REGISTRY,
    StrategyMeta,
    StrategyMixin,
)
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
        bot.select_strategy()
        assert bot.active_strategy is BullStrategy

    def test_select_strategy_bear(self):
        bot = StrategyBot(current_regime="bear")
        bot.select_strategy()
        assert bot.active_strategy is BearStrategy

    def test_select_strategy_sideways(self):
        bot = StrategyBot(current_regime="sideways")
        bot.select_strategy()
        assert bot.active_strategy is SidewaysStrategy

    def test_select_strategy_returns_self_for_chaining(self):
        bot = StrategyBot(current_regime="bull")
        result = bot.select_strategy()
        assert result is bot

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


class TestStrategyRegistry:
    def test_registry_has_entries(self):
        assert len(STRATEGY_REGISTRY) >= 3

    def test_each_entry_is_strategy_meta(self):
        for meta in STRATEGY_REGISTRY:
            assert isinstance(meta, StrategyMeta)
            assert issubclass(meta.strategy_cls, Strategy)
            assert isinstance(meta.description, str)
            assert len(meta.best_regimes) > 0

    def test_all_regimes_covered(self):
        all_best = set()
        for meta in STRATEGY_REGISTRY:
            all_best.update(meta.best_regimes)
        assert {"bull", "bear", "sideways"}.issubset(all_best)

    def test_list_strategies(self):
        result = StrategyMixin.list_strategies()
        assert len(result) == len(STRATEGY_REGISTRY)
        assert all(isinstance(m, StrategyMeta) for m in result)

    def test_get_strategies_for_regime(self):
        bull_strategies = StrategyMixin.get_strategies_for_regime("bull")
        assert len(bull_strategies) >= 1
        assert all("bull" in m.best_regimes for m in bull_strategies)

    def test_get_strategies_for_unknown_regime(self):
        result = StrategyMixin.get_strategies_for_regime("unknown")
        assert result == []

    def test_regime_map_built_from_registry(self):
        regime_map = StrategyMixin.REGIME_STRATEGY_MAP
        assert "bull" in regime_map
        assert "bear" in regime_map
        assert "sideways" in regime_map
