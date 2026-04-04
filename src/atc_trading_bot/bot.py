from atc_trading_bot.mixins.backtest_mixin import BacktestMixin
from atc_trading_bot.mixins.data_mixin import DataMixin
from atc_trading_bot.mixins.feature_mixin import FeatureMixin
from atc_trading_bot.mixins.regime_mixin import RegimeMixin
from atc_trading_bot.mixins.signal_mixin import SignalMixin
from atc_trading_bot.mixins.strategy_mixin import StrategyMixin


class Bot(SignalMixin, BacktestMixin, StrategyMixin, RegimeMixin, FeatureMixin, DataMixin):
    """Trading bot composing all mixins.

    Pipeline: fetch_data → compute_features → detect_regime → select_strategy → backtest → generate_signals

    Args:
        exchange_id: CCXT exchange identifier. Default: "binanceus".
        symbols: List of symbols to trade, e.g. ["BTC", "ETH"]. Case insensitive.
        timeframe: Candlestick timeframe. Default: "1d".
        api_key: Exchange API key. Default: "" (public endpoints only).
        secret: Exchange API secret. Default: "".
        data_dir: Directory for CSV cache. Default: "data/" relative to project root.
    """

    def run_pipeline(self, symbol: str = "BTC", n_components: int = 10,
                     n_regimes: int = 3) -> dict:
        """Execute the full pipeline for a symbol and return signals.

        Args:
            symbol: Trading pair, e.g. "BTC" or "BTC/USDT". Case insensitive. Default: "BTC".
            n_components: Number of PCA components for feature reduction. Default: 10.
            n_regimes: Number of HMM regimes. Default: 3.

        Returns:
            Dict with keys: regime, strategy, signal.
        """
        self.fetch_data(symbol)
        self.compute_features(n_components=n_components)
        self.detect_regime(n_regimes=n_regimes)
        self.select_strategy()
        self.backtest()
        return self.generate_signals()
