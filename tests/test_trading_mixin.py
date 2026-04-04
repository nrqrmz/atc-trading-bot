import pytest
from unittest.mock import MagicMock, patch

from atc_trading_bot.mixins.trading_mixin import TradingMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class TradingBot(TradingMixin):
    """Minimal class using TradingMixin for testing."""

    def __init__(self, **kwargs):
        self.signals = kwargs.pop("signals", None)
        super().__init__(**kwargs)


class TestConnectTestnet:
    def test_connect_testnet_sets_exchange(self):
        bot = TradingBot()
        with patch("atc_trading_bot.mixins.trading_mixin.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.binanceus.return_value = mock_exchange

            result = bot.connect_testnet("test_key", "test_secret")

            mock_ccxt.binanceus.assert_called_once()
            mock_exchange.set_sandbox_mode.assert_called_once_with(True)
            assert bot.exchange is mock_exchange
            assert result is bot  # method chaining

    def test_connect_testnet_passes_credentials(self):
        bot = TradingBot()
        with patch("atc_trading_bot.mixins.trading_mixin.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.binanceus.return_value = mock_exchange

            bot.connect_testnet("my_api_key", "my_secret")

            call_args = mock_ccxt.binanceus.call_args[0][0]
            assert call_args["apiKey"] == "my_api_key"
            assert call_args["secret"] == "my_secret"

    def test_connect_testnet_custom_exchange(self):
        bot = TradingBot()
        with patch("atc_trading_bot.mixins.trading_mixin.ccxt") as mock_ccxt:
            mock_exchange = MagicMock()
            mock_ccxt.bybit.return_value = mock_exchange

            bot.connect_testnet("key", "secret", exchange_id="bybit")

            mock_ccxt.bybit.assert_called_once()
            assert bot.exchange is mock_exchange

    def test_connect_testnet_warns_unknown_exchange(self):
        bot = TradingBot()
        with patch("atc_trading_bot.mixins.trading_mixin.ccxt") as mock_ccxt:
            mock_ccxt.nonexistent_exchange = None
            delattr(mock_ccxt, "nonexistent_exchange")

            with pytest.warns(PipelineWarning, match="Unknown exchange"):
                bot.connect_testnet("key", "secret", exchange_id="nonexistent_exchange")

            assert bot.exchange is None


class TestExecuteSignal:
    def test_execute_buy_signal(self):
        signals = {"regime": "bull", "strategy": "BullStrategy", "signal": "buy", "confidence": 0.9}
        bot = TradingBot(signals=signals)
        bot.exchange = MagicMock()
        bot.exchange.create_order.return_value = {"id": "123", "status": "filled"}

        result = bot.execute_signal(symbol="BTC/USDT", amount=0.01)

        bot.exchange.create_order.assert_called_once_with(
            symbol="BTC/USDT", type="market", side="buy", amount=0.01,
        )
        assert bot.last_order == {"id": "123", "status": "filled"}
        assert result is bot  # method chaining

    def test_execute_sell_signal(self):
        signals = {"regime": "bear", "strategy": "BearStrategy", "signal": "sell", "confidence": 0.8}
        bot = TradingBot(signals=signals)
        bot.exchange = MagicMock()
        bot.exchange.create_order.return_value = {"id": "456", "status": "filled"}

        bot.execute_signal()

        bot.exchange.create_order.assert_called_once_with(
            symbol="BTC/USDT", type="market", side="sell", amount=0.001,
        )

    def test_execute_warns_hold_signal(self):
        signals = {"regime": "sideways", "strategy": "SidewaysStrategy", "signal": "hold", "confidence": 0.7}
        bot = TradingBot(signals=signals)
        bot.exchange = MagicMock()

        with pytest.warns(PipelineWarning, match="hold"):
            bot.execute_signal()

        bot.exchange.create_order.assert_not_called()

    def test_execute_warns_without_signals(self):
        bot = TradingBot()
        bot.exchange = MagicMock()

        with pytest.warns(PipelineWarning, match="generate_signals"):
            bot.execute_signal()

        bot.exchange.create_order.assert_not_called()

    def test_execute_warns_without_exchange(self):
        signals = {"regime": "bull", "strategy": "BullStrategy", "signal": "buy", "confidence": 0.9}
        bot = TradingBot(signals=signals)

        with pytest.warns(PipelineWarning, match="connect_testnet"):
            bot.execute_signal()

    def test_execute_warns_on_order_failure(self):
        signals = {"regime": "bull", "strategy": "BullStrategy", "signal": "buy", "confidence": 0.9}
        bot = TradingBot(signals=signals)
        bot.exchange = MagicMock()
        bot.exchange.create_order.side_effect = Exception("Insufficient funds")

        with pytest.warns(PipelineWarning, match="Order execution failed"):
            bot.execute_signal()


class TestGetBalance:
    def test_get_balance_returns_balance(self):
        bot = TradingBot()
        bot.exchange = MagicMock()
        bot.exchange.fetch_balance.return_value = {"USDT": {"free": 10000, "used": 0, "total": 10000}}

        balance = bot.get_balance()

        bot.exchange.fetch_balance.assert_called_once()
        assert balance == {"USDT": {"free": 10000, "used": 0, "total": 10000}}

    def test_get_balance_warns_without_exchange(self):
        bot = TradingBot()

        with pytest.warns(PipelineWarning, match="connect_testnet"):
            result = bot.get_balance()

        assert result is None

    def test_get_balance_warns_on_failure(self):
        bot = TradingBot()
        bot.exchange = MagicMock()
        bot.exchange.fetch_balance.side_effect = Exception("Network error")

        with pytest.warns(PipelineWarning, match="Failed to fetch balance"):
            result = bot.get_balance()

        assert result is None


class TestGetOpenPositions:
    def test_get_open_positions_returns_list(self):
        bot = TradingBot()
        bot.exchange = MagicMock()
        bot.exchange.fetch_positions.return_value = [{"symbol": "BTC/USDT", "side": "long"}]

        positions = bot.get_open_positions()

        bot.exchange.fetch_positions.assert_called_once()
        assert positions == [{"symbol": "BTC/USDT", "side": "long"}]

    def test_get_open_positions_warns_without_exchange(self):
        bot = TradingBot()

        with pytest.warns(PipelineWarning, match="connect_testnet"):
            result = bot.get_open_positions()

        assert result is None
