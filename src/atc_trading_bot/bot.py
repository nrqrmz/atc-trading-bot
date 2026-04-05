from atc_trading_bot.config import DEFAULT_N_COMPONENTS, DEFAULT_N_REGIMES
from atc_trading_bot.mixins.backtest_mixin import BacktestMixin
from atc_trading_bot.mixins.data_mixin import DataMixin
from atc_trading_bot.mixins.explainability_mixin import ExplainabilityMixin
from atc_trading_bot.mixins.feature_mixin import FeatureMixin
from atc_trading_bot.mixins.labeling_mixin import LabelingMixin
from atc_trading_bot.mixins.model_mixin import ModelMixin
from atc_trading_bot.mixins.optimization_mixin import OptimizationMixin
from atc_trading_bot.mixins.persistence_mixin import PersistenceMixin
from atc_trading_bot.mixins.regime_mixin import RegimeMixin
from atc_trading_bot.mixins.scanner_mixin import ScannerMixin
from atc_trading_bot.mixins.sentiment_mixin import SentimentMixin
from atc_trading_bot.mixins.signal_mixin import SignalMixin
from atc_trading_bot.mixins.strategy_mixin import StrategyMixin
from atc_trading_bot.mixins.trading_mixin import TradingMixin
from atc_trading_bot.mixins.visualization_mixin import VisualizationMixin


class Bot(
    ExplainabilityMixin, OptimizationMixin, ModelMixin, LabelingMixin,
    TradingMixin, ScannerMixin, SentimentMixin, VisualizationMixin,
    PersistenceMixin, SignalMixin, BacktestMixin, StrategyMixin,
    RegimeMixin, FeatureMixin, DataMixin,
):
    """Trading bot composing all mixins.

    Two pipelines available:
    - Rule-based: fetch_data → compute_features → detect_regime → select_strategy → backtest → generate_signals
    - ML-based: fetch_data → compute_features → detect_regime → compute_labels → train_models → predict

    Args:
        exchange_id: CCXT exchange identifier. Default: "binanceus".
        symbols: List of symbols to trade, e.g. ["BTC", "ETH"]. Case insensitive.
        timeframe: Candlestick timeframe. Default: "1d".
        api_key: Exchange API key. Default: "" (public endpoints only).
        secret: Exchange API secret. Default: "".
        data_dir: Directory for CSV cache. Default: "data/" relative to project root.
    """

    def run_pipeline(self, symbol: str = "BTC",
                     n_components: int = DEFAULT_N_COMPONENTS,
                     n_regimes: int = DEFAULT_N_REGIMES) -> dict:
        """Execute the rule-based pipeline for a symbol and return signals.

        Args:
            symbol: Trading pair, e.g. "BTC" or "BTC/USDT". Default: "BTC".
            n_components: Number of PCA components. Default: 10.
            n_regimes: Number of HMM regimes. Default: 3.

        Returns:
            Dict with keys: regime, strategy, signal, confidence.
        """
        self.fetch_data(symbol)
        self.compute_features(n_components=n_components)
        self.detect_regime(n_regimes=n_regimes)
        self.select_strategy()
        self.backtest()
        return self.generate_signals()

    def run_pipeline_ml(self, symbol: str = "BTC",
                        n_components: int = DEFAULT_N_COMPONENTS,
                        n_regimes: int = DEFAULT_N_REGIMES) -> dict:
        """Execute the ML pipeline for a symbol and return predictions.

        Args:
            symbol: Trading pair, e.g. "BTC" or "BTC/USDT". Default: "BTC".
            n_components: Number of PCA components. Default: 10.
            n_regimes: Number of HMM regimes. Default: 3.

        Returns:
            Dict with keys: regime, model, predictions summary.
        """
        self.fetch_data(symbol)
        self.compute_features(n_components=n_components)
        self.detect_regime(n_regimes=n_regimes)
        self.compute_labels()
        self.train_models()

        preds = self.predict()
        last_pred = int(preds[-1]) if preds is not None and len(preds) > 0 else 0
        signal_map = {-1: "sell", 0: "hold", 1: "buy"}

        return {
            "regime": self.current_regime,
            "model": type(self.active_model).__name__,
            "signal": signal_map.get(last_pred, "hold"),
        }
