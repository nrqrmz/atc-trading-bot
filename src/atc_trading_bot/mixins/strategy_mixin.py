import warnings

from backtesting import Strategy

from atc_trading_bot.strategies.bear_strategy import BearStrategy
from atc_trading_bot.strategies.bull_strategy import BullStrategy
from atc_trading_bot.strategies.sideways_strategy import SidewaysStrategy


class StrategyMixin:
    """Mixin for selecting trading strategy based on detected regime."""

    REGIME_STRATEGY_MAP: dict[str, type[Strategy]] = {
        "bull": BullStrategy,
        "bear": BearStrategy,
        "sideways": SidewaysStrategy,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_strategy: type[Strategy] | None = None

    def select_strategy(self) -> type[Strategy]:
        """Select the appropriate strategy based on the current regime."""
        if self.current_regime is None:
            warnings.warn("No regime detected. Call detect_regime first.")
            return

        self.active_strategy = self.REGIME_STRATEGY_MAP[self.current_regime]
        return self.active_strategy

    def get_strategy_for_regime(self, regime: str) -> type[Strategy]:
        """Get the strategy class for a specific regime."""
        if regime not in self.REGIME_STRATEGY_MAP:
            raise ValueError(f"Unknown regime: {regime}. Must be one of {list(self.REGIME_STRATEGY_MAP.keys())}")
        return self.REGIME_STRATEGY_MAP[regime]
