import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from atc_trading_bot.bot import Bot


@pytest.fixture
def bot_with_data(sample_ohlcv_data):
    """Create a Bot with mocked exchange and pre-loaded data."""
    with patch("ccxt.binanceus") as mock_cls:
        mock_exchange = MagicMock()
        mock_cls.return_value = mock_exchange

        bot = Bot(
            exchange_id="binanceus",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )
        bot.exchange = mock_exchange
        bot.df = sample_ohlcv_data
        return bot


class TestBotIntegration:
    def test_bot_instantiation(self):
        with patch("ccxt.binanceus"):
            bot = Bot(
                exchange_id="binanceus",
                symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                timeframe="1d",
            )
            assert bot.exchange_id == "binanceus"
            assert len(bot.symbols) == 3

    def test_full_pipeline_with_mocked_data(self, bot_with_data):
        bot = bot_with_data

        # Step 1: Features
        bot.compute_features(n_components=5)
        assert bot.features_pca is not None
        assert bot.features_pca.shape[1] == 5

        # Step 2: Regime detection
        bot.detect_regime(n_regimes=3)
        assert bot.current_regime in ("bull", "bear", "sideways")
        assert len(bot.regimes) == len(bot.features_pca)

        # Step 3: Strategy selection
        strategy = bot.select_strategy()
        assert strategy is not None

        # Step 4: Backtest
        results = bot.backtest()
        assert isinstance(results, pd.DataFrame)
        assert "sharpe_ratio" in results["metric"].values
        assert "max_drawdown" in results["metric"].values

        # Step 5: Signals
        signals = bot.generate_signals()
        assert signals["signal"] in ("buy", "sell", "hold")
        assert signals["regime"] == bot.current_regime

    def test_run_pipeline(self, sample_ohlcv_raw):
        with patch("ccxt.binanceus") as mock_cls:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
            mock_cls.return_value = mock_exchange

            bot = Bot(
                exchange_id="binanceus",
                symbols=["BTC/USDT"],
                timeframe="1d",
            )
            bot.exchange = mock_exchange

            signals = bot.run_pipeline("BTC/USDT")

            assert isinstance(signals, dict)
            assert signals["signal"] in ("buy", "sell", "hold")
            assert signals["regime"] in ("bull", "bear", "sideways")

    def test_pipeline_per_symbol(self, sample_ohlcv_raw):
        """Test running pipeline independently for each symbol."""
        with patch("ccxt.binanceus") as mock_cls:
            mock_exchange = MagicMock()
            mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
            mock_cls.return_value = mock_exchange

            bot = Bot(
                exchange_id="binanceus",
                symbols=["BTC/USDT", "ETH/USDT"],
                timeframe="1d",
            )
            bot.exchange = mock_exchange

            all_signals = {}
            for symbol in bot.symbols:
                signals = bot.run_pipeline(symbol)
                all_signals[symbol] = signals

            assert len(all_signals) == 2
            for symbol, signals in all_signals.items():
                assert signals["signal"] in ("buy", "sell", "hold")

    def test_backtest_metrics_are_reasonable(self, bot_with_data):
        bot = bot_with_data
        bot.compute_features(n_components=5)
        bot.detect_regime(n_regimes=3)
        bot.select_strategy()
        results = bot.backtest()

        def _get(name):
            return results.loc[results["metric"] == name, "value"].iloc[0]

        assert -1 <= _get("max_drawdown") <= 0
        assert 0 <= _get("win_rate") <= 1
        assert _get("num_trades") >= 0

    def test_cpcv_integration(self, bot_with_data):
        bot = bot_with_data
        bot.compute_features(n_components=5)
        bot.detect_regime(n_regimes=3)
        bot.select_strategy()

        cv_results = bot.cross_validate_cpcv(n_splits=3, n_components=5)
        assert isinstance(cv_results, pd.DataFrame)
