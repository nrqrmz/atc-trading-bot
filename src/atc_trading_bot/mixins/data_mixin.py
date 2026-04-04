import os
import warnings
from pathlib import Path

import ccxt
import pandas as pd
from atc_trading_bot.pipeline_warning import PipelineWarning


class DataMixin:
    """Mixin for fetching and caching OHLCV data via CCXT."""

    def __init__(self, exchange_id: str = "binanceus", symbols: list[str] | None = None,
                 timeframe: str = "1d", api_key: str = "", secret: str = "",
                 data_dir: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.exchange_id = exchange_id
        self.symbols = [self._normalize_symbol(s) for s in (symbols or [])]
        self.timeframe = timeframe
        self.df: pd.DataFrame | None = None

        exchange_cls = getattr(ccxt, exchange_id)
        config = {}
        if api_key:
            config["apiKey"] = api_key
        if secret:
            config["secret"] = secret
        self.exchange = exchange_cls(config)

        if data_dir is None:
            self.data_dir = str(Path(__file__).resolve().parents[3] / "data")
        else:
            self.data_dir = data_dir

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert short symbol like 'btc' to CCXT format 'BTC/USDT'."""
        symbol = symbol.upper()
        if "/" in symbol:
            return symbol
        return f"{symbol}/USDT"

    def fetch_data(self, symbol: str | None = None, timeframe: str | None = None,
                   since: str | None = None, use_cache: bool = False) -> pd.DataFrame | None:
        """Fetch OHLCV data for a symbol. Uses cache if available and requested.

        Args:
            symbol: Trading pair, e.g. "BTC" or "BTC/USDT". Case insensitive.
                Defaults to the first symbol in self.symbols if omitted.
            timeframe: Candlestick timeframe, e.g. "1h", "1d". Defaults to self.timeframe.
            since: ISO 8601 start date, e.g. "2024-01-01T00:00:00Z". Defaults to exchange default.
            use_cache: If True, load from CSV cache before hitting the exchange.

        Returns:
            DataFrame with OHLCV columns, or None if no symbol is available.
        """
        if symbol is None:
            if self.symbols:
                symbol = self.symbols[0]
            else:
                warnings.warn(
                    "No symbol provided and no default symbols configured. "
                    "Pass a symbol to fetch_data or set symbols in the constructor.",
                    PipelineWarning,
                )
                return
        symbol = self._normalize_symbol(symbol)
        tf = timeframe or self.timeframe

        if use_cache:
            cached = self._load_cache(symbol, tf)
            if cached is not None:
                self.df = cached
                return self.df

        since_ts = None
        if since:
            since_ts = self.exchange.parse8601(since)

        raw = self.exchange.fetch_ohlcv(symbol, tf, since=since_ts)
        self.df = self._raw_to_dataframe(raw)
        self._save_cache(symbol, tf)
        return self.df

    def _raw_to_dataframe(self, raw: list[list]) -> pd.DataFrame:
        """Convert raw CCXT OHLCV data to a DataFrame."""
        df = pd.DataFrame(raw, columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.index.name = None
        return df

    def _cache_filename(self, symbol: str, timeframe: str) -> str:
        """Generate cache filename from symbol and timeframe."""
        safe_symbol = symbol.replace("/", "_")
        return f"{safe_symbol}_{timeframe}.csv"

    def _save_cache(self, symbol: str, timeframe: str) -> None:
        """Save current df to CSV cache."""
        if self.df is None:
            return
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, self._cache_filename(symbol, timeframe))
        self.df.to_csv(path)

    def _load_cache(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Load cached data from CSV. Returns None if not found."""
        path = os.path.join(self.data_dir, self._cache_filename(symbol, timeframe))
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df
