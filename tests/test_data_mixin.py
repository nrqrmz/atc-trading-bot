import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from atc_trading_bot.mixins.data_mixin import DataMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class DataBot(DataMixin):
    """Minimal class using DataMixin for testing."""

    pass


class TestDataMixin:
    def setup_method(self):
        self.bot = DataBot(
            exchange_id="binanceus",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )

    def test_init_sets_attributes(self):
        assert self.bot.exchange_id == "binanceus"
        assert self.bot.symbols == ["BTC/USDT"]
        assert self.bot.timeframe == "1d"
        assert self.bot.exchange is not None

    @patch("ccxt.binanceus")
    def test_fetch_data_returns_dataframe(self, mock_exchange_cls, sample_ohlcv_raw):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
        mock_exchange_cls.return_value = mock_exchange

        bot = DataBot(
            exchange_id="binanceus",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )
        bot.exchange = mock_exchange

        result = bot.fetch_data("BTC/USDT")

        assert result is bot  # method chaining
        assert isinstance(bot.df, pd.DataFrame)
        assert list(bot.df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(bot.df.index, pd.DatetimeIndex)
        assert len(bot.df) == len(sample_ohlcv_raw)

    @patch("ccxt.binanceus")
    def test_fetch_data_stores_in_self_df(self, mock_exchange_cls, sample_ohlcv_raw):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
        mock_exchange_cls.return_value = mock_exchange

        bot = DataBot(
            exchange_id="binanceus",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )
        bot.exchange = mock_exchange

        bot.fetch_data("BTC/USDT")
        assert bot.df is not None
        assert isinstance(bot.df, pd.DataFrame)

    def test_save_and_load_cache(self, sample_ohlcv_data, tmp_path):
        self.bot.data_dir = str(tmp_path)
        self.bot.df = sample_ohlcv_data

        self.bot._save_cache("BTC/USDT", "1d")

        cache_file = tmp_path / "BTC_USDT_1d.csv"
        assert cache_file.exists()

        loaded = self.bot._load_cache("BTC/USDT", "1d")
        assert loaded is not None
        assert isinstance(loaded, pd.DataFrame)
        assert list(loaded.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert len(loaded) == len(sample_ohlcv_data)

    def test_load_cache_returns_none_when_missing(self, tmp_path):
        self.bot.data_dir = str(tmp_path)
        result = self.bot._load_cache("NONEXISTENT/PAIR", "1d")
        assert result is None

    @patch("ccxt.binanceus")
    def test_fetch_data_uses_cache(self, mock_exchange_cls, sample_ohlcv_data, tmp_path):
        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange

        bot = DataBot(
            exchange_id="binanceus",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )
        bot.exchange = mock_exchange
        bot.data_dir = str(tmp_path)

        # Pre-populate cache
        bot.df = sample_ohlcv_data
        bot._save_cache("BTC/USDT", "1d")
        bot.df = None

        result = bot.fetch_data("BTC/USDT", use_cache=True)

        mock_exchange.fetch_ohlcv.assert_not_called()
        assert result is bot  # method chaining
        assert isinstance(bot.df, pd.DataFrame)
        assert len(bot.df) == len(sample_ohlcv_data)

    def test_normalize_symbol_short(self):
        """Short symbol gets /USDT appended."""
        bot = DataBot(symbols=["BTC", "ETH"])
        assert bot.symbols == ["BTC/USDT", "ETH/USDT"]

    def test_normalize_symbol_already_full(self):
        """Full CCXT symbol is left unchanged."""
        bot = DataBot(symbols=["BTC/USDT", "ETH/USDT"])
        assert bot.symbols == ["BTC/USDT", "ETH/USDT"]

    def test_normalize_symbol_mixed(self):
        """Mix of short and full symbols."""
        bot = DataBot(symbols=["BTC", "ETH/USDT", "SOL"])
        assert bot.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_normalize_symbol_lowercase(self):
        """Lowercase symbols are uppercased."""
        bot = DataBot(symbols=["btc", "eth/usdt", "sol"])
        assert bot.symbols == ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

    def test_normalize_symbol_mixedcase(self):
        """Mixed case symbols are uppercased."""
        bot = DataBot(symbols=["Btc", "Eth/Usdt"])
        assert bot.symbols == ["BTC/USDT", "ETH/USDT"]

    @patch("ccxt.binanceus")
    def test_fetch_data_uses_default_symbol(self, mock_exchange_cls, sample_ohlcv_raw):
        """fetch_data() without symbol uses first configured symbol."""
        bot = DataBot(symbols=["BTC/USDT"])
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
        bot.exchange = mock_exchange

        bot.fetch_data()

        mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1d", since=None, limit=1000)

    def test_fetch_data_warns_without_symbol(self):
        """fetch_data() without symbol and no configured symbols warns."""
        bot = DataBot(symbols=[])
        with pytest.warns(PipelineWarning, match="No symbol provided"):
            result = bot.fetch_data()
        assert result is None

    @patch("ccxt.binanceus")
    def test_pagination_fetches_multiple_batches(self, mock_exchange_cls, sample_ohlcv_raw):
        """Pagination loops when start date is provided and exchange returns full batches."""
        bot = DataBot(symbols=["BTC/USDT"])
        mock_exchange = MagicMock()

        # First call returns exactly `limit` entries (triggers next page), second returns partial
        batch1 = sample_ohlcv_raw[:200]
        batch2 = sample_ohlcv_raw[200:250]
        mock_exchange.fetch_ohlcv.side_effect = [batch1, batch2]
        mock_exchange.parse8601.return_value = 1672531200000
        bot.exchange = mock_exchange

        # Use limit=200 so first batch triggers pagination
        result = bot._fetch_with_pagination("BTC/USDT", "1d", since_ts=1672531200000, limit=200)

        assert mock_exchange.fetch_ohlcv.call_count == 2
        assert len(result) == 250

    @patch("ccxt.binanceus")
    def test_fetch_data_normalizes_symbol(self, mock_exchange_cls, sample_ohlcv_raw):
        """fetch_data accepts short symbols."""
        bot = DataBot(symbols=["BTC"])
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
        bot.exchange = mock_exchange

        bot.fetch_data("BTC")

        mock_exchange.fetch_ohlcv.assert_called_once_with("BTC/USDT", "1d", since=None, limit=1000)
