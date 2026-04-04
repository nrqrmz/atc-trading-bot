from atc_trading_bot.mixins.backtest_mixin import BacktestMixin
from atc_trading_bot.mixins.data_mixin import DataMixin
from atc_trading_bot.mixins.feature_mixin import FeatureMixin
from atc_trading_bot.mixins.regime_mixin import RegimeMixin
from atc_trading_bot.mixins.signal_mixin import SignalMixin
from atc_trading_bot.mixins.strategy_mixin import StrategyMixin


class Bot(SignalMixin, BacktestMixin, StrategyMixin, RegimeMixin, FeatureMixin, DataMixin):
    """Trading bot composing all mixins.

    Pipeline: fetch_data → compute_features → detect_regime → select_strategy → backtest → generate_signals
    """

    def run_pipeline(self, symbol: str = "BTC", n_components: int = 10,
                     n_regimes: int = 3) -> dict:
        """Execute the full pipeline for a symbol and return signals."""
        self.fetch_data(symbol)
        self.compute_features(n_components=n_components)
        self.detect_regime(n_regimes=n_regimes)
        self.select_strategy()
        self.backtest()
        return self.generate_signals()
