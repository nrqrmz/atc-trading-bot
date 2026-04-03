import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from atc_trading_bot.mixins.data_mixin import DataMixin


class DataBot(DataMixin):
    """Minimal class using DataMixin for testing."""

    pass


class TestDataMixin:
    def setup_method(self):
        self.bot = DataBot(
            exchange_id="binance",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )

    def test_init_sets_attributes(self):
        assert self.bot.exchange_id == "binance"
        assert self.bot.symbols == ["BTC/USDT"]
        assert self.bot.timeframe == "1d"
        assert self.bot.exchange is not None

    @patch("ccxt.binance")
    def test_fetch_data_returns_dataframe(self, mock_exchange_cls, sample_ohlcv_raw):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
        mock_exchange_cls.return_value = mock_exchange

        bot = DataBot(
            exchange_id="binance",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )
        bot.exchange = mock_exchange

        df = bot.fetch_data("BTC/USDT")

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == len(sample_ohlcv_raw)

    @patch("ccxt.binance")
    def test_fetch_data_stores_in_self_df(self, mock_exchange_cls, sample_ohlcv_raw):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = sample_ohlcv_raw
        mock_exchange_cls.return_value = mock_exchange

        bot = DataBot(
            exchange_id="binance",
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

    @patch("ccxt.binance")
    def test_fetch_data_uses_cache(self, mock_exchange_cls, sample_ohlcv_data, tmp_path):
        mock_exchange = MagicMock()
        mock_exchange_cls.return_value = mock_exchange

        bot = DataBot(
            exchange_id="binance",
            symbols=["BTC/USDT"],
            timeframe="1d",
        )
        bot.exchange = mock_exchange
        bot.data_dir = str(tmp_path)

        # Pre-populate cache
        bot.df = sample_ohlcv_data
        bot._save_cache("BTC/USDT", "1d")
        bot.df = None

        df = bot.fetch_data("BTC/USDT", use_cache=True)

        mock_exchange.fetch_ohlcv.assert_not_called()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_ohlcv_data)
