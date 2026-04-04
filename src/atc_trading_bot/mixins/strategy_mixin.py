import warnings
from dataclasses import dataclass

from backtesting import Strategy
from atc_trading_bot.pipeline_warning import PipelineWarning

from atc_trading_bot.strategies.bear_strategy import BearStrategy
from atc_trading_bot.strategies.breakout_strategy import BreakoutStrategy
from atc_trading_bot.strategies.bull_strategy import BullStrategy
from atc_trading_bot.strategies.momentum_strategy import MomentumStrategy
from atc_trading_bot.strategies.sideways_strategy import SidewaysStrategy
from atc_trading_bot.strategies.volatility_strategy import VolatilityStrategy


@dataclass(frozen=True)
class StrategyMeta:
    """Metadata for a registered trading strategy."""
    strategy_cls: type[Strategy]
    description: str
    best_regimes: list[str]
    worst_regimes: list[str]


# Strategy registry — add new strategies here.
STRATEGY_REGISTRY: list[StrategyMeta] = [
    StrategyMeta(
        strategy_cls=BullStrategy,
        description="Trend following with SMA crossovers",
        best_regimes=["bull"],
        worst_regimes=["sideways"],
    ),
    StrategyMeta(
        strategy_cls=BearStrategy,
        description="Defensive short mean reversion on resistance",
        best_regimes=["bear"],
        worst_regimes=["bull"],
    ),
    StrategyMeta(
        strategy_cls=SidewaysStrategy,
        description="Bollinger Bands + RSI mean reversion",
        best_regimes=["sideways"],
        worst_regimes=["bull", "bear"],
    ),
    StrategyMeta(
        strategy_cls=MomentumStrategy,
        description="ROC + RSI momentum following",
        best_regimes=["bull"],
        worst_regimes=["sideways"],
    ),
    StrategyMeta(
        strategy_cls=BreakoutStrategy,
        description="Donchian channel breakout with volume confirmation",
        best_regimes=["bull", "bear"],
        worst_regimes=["sideways"],
    ),
    StrategyMeta(
        strategy_cls=VolatilityStrategy,
        description="ATR mean reversion for volatility cycles",
        best_regimes=["sideways", "bear"],
        worst_regimes=["bull"],
    ),
]


def _build_regime_map(registry: list[StrategyMeta]) -> dict[str, type[Strategy]]:
    """Build regime → strategy mapping from the registry.

    For each regime, picks the first strategy that lists it as a best regime.
    """
    regime_map = {}
    for meta in registry:
        for regime in meta.best_regimes:
            if regime not in regime_map:
                regime_map[regime] = meta.strategy_cls
    return regime_map


class StrategyMixin:
    """Mixin for selecting trading strategy based on detected regime."""

    REGIME_STRATEGY_MAP: dict[str, type[Strategy]] = _build_regime_map(STRATEGY_REGISTRY)

    def _require_regime(self) -> bool:
        """Check that regimes have been detected."""
        if getattr(self, "current_regime", None) is None:
            warnings.warn("No regime detected. Call detect_regime first.", PipelineWarning)
            return False
        return True

    def _require_strategy(self) -> bool:
        """Check that a strategy has been selected."""
        if getattr(self, "active_strategy", None) is None:
            warnings.warn("No strategy selected. Call select_strategy first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_strategy: type[Strategy] | None = None

    def select_strategy(self) -> type[Strategy]:
        """Select the appropriate strategy based on the current regime.

        Returns:
            The selected Strategy class, or None if no regime is detected.
        """
        if not self._require_regime():
            return

        self.active_strategy = self.REGIME_STRATEGY_MAP[self.current_regime]
        return self

    def get_strategy_for_regime(self, regime: str) -> type[Strategy]:
        """Get the default strategy class for a specific regime."""
        if regime not in self.REGIME_STRATEGY_MAP:
            raise ValueError(f"Unknown regime: {regime}. Must be one of {list(self.REGIME_STRATEGY_MAP.keys())}")
        return self.REGIME_STRATEGY_MAP[regime]

    @staticmethod
    def get_strategies_for_regime(regime: str) -> list[StrategyMeta]:
        """Get all strategies suitable for a regime (best_regimes includes it)."""
        return [m for m in STRATEGY_REGISTRY if regime in m.best_regimes]

    @staticmethod
    def list_strategies() -> list[StrategyMeta]:
        """List all registered strategies with their metadata."""
        return list(STRATEGY_REGISTRY)
