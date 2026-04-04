import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from atc_trading_bot.mixins.scanner_mixin import ScannerMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


def _make_ohlcv(n=100, base_price=30000):
    """Create a simple OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = base_price + np.linspace(0, 5000, n) + np.random.randn(n) * 100
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 50
    volume = np.abs(np.random.randn(n) * 1_000_000) + 500_000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class ScannerBot(ScannerMixin):
    """Minimal class using ScannerMixin for testing."""

    def __init__(self, **kwargs):
        self.symbols = kwargs.pop("symbols", [])
        self.df = kwargs.pop("df", None)
        self.current_regime = kwargs.pop("current_regime", None)
        self.hmm_model = kwargs.pop("hmm_model", None)
        self.features_pca = kwargs.pop("features_pca", None)
        # Track pipeline calls for assertion
        self._fetch_calls = []
        self._feature_calls = []
        self._regime_calls = []
        super().__init__(**kwargs)

    def fetch_data(self, symbol=None, **kwargs):
        self._fetch_calls.append(symbol)
        self.df = _make_ohlcv()

    def compute_features(self, n_components=10, **kwargs):
        self._feature_calls.append(n_components)
        self.features_pca = np.random.randn(len(self.df), n_components)

    def detect_regime(self, n_regimes=3, **kwargs):
        self._regime_calls.append(n_regimes)
        self.current_regime = "bull"
        # Set up a mock HMM for confidence
        mock_hmm = MagicMock()
        posteriors = np.array([[0.85, 0.10, 0.05]] * len(self.features_pca))
        mock_hmm.predict_proba.return_value = posteriors
        mock_hmm.predict.return_value = np.array([0])
        self.hmm_model = mock_hmm


class FailingScannerBot(ScannerMixin):
    """ScannerBot where one symbol fails during the pipeline."""

    def __init__(self, **kwargs):
        self.symbols = kwargs.pop("symbols", [])
        self.df = None
        self.current_regime = None
        self.hmm_model = None
        self.features_pca = None
        self._fail_on = kwargs.pop("fail_on", None)
        super().__init__(**kwargs)

    def fetch_data(self, symbol=None, **kwargs):
        if symbol == self._fail_on:
            raise ConnectionError(f"Exchange error for {symbol}")
        self.df = _make_ohlcv()

    def compute_features(self, n_components=10, **kwargs):
        self.features_pca = np.random.randn(len(self.df), n_components)

    def detect_regime(self, n_regimes=3, **kwargs):
        self.current_regime = "sideways"
        mock_hmm = MagicMock()
        posteriors = np.array([[0.70, 0.20, 0.10]] * len(self.features_pca))
        mock_hmm.predict_proba.return_value = posteriors
        mock_hmm.predict.return_value = np.array([0])
        self.hmm_model = mock_hmm


class TestScanRegimes:
    def test_returns_dataframe_with_expected_columns(self):
        bot = ScannerBot(symbols=["BTC/USDT", "ETH/USDT"])
        result = bot.scan_regimes()

        assert isinstance(result, pd.DataFrame)
        expected_cols = ["symbol", "regime", "confidence", "last_price", "pct_change_24h"]
        assert list(result.columns) == expected_cols

    def test_one_row_per_symbol(self):
        bot = ScannerBot(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        result = bot.scan_regimes()

        assert len(result) == 3

    def test_regime_values_are_valid(self):
        bot = ScannerBot(symbols=["BTC/USDT"])
        result = bot.scan_regimes()

        assert result["regime"].iloc[0] in ("bull", "bear", "sideways")

    def test_confidence_between_zero_and_one(self):
        bot = ScannerBot(symbols=["BTC/USDT"])
        result = bot.scan_regimes()

        assert 0 <= result["confidence"].iloc[0] <= 1

    def test_last_price_is_positive(self):
        bot = ScannerBot(symbols=["BTC/USDT"])
        result = bot.scan_regimes()

        assert result["last_price"].iloc[0] > 0

    def test_pipeline_called_for_each_symbol(self):
        symbols = ["BTC/USDT", "ETH/USDT"]
        bot = ScannerBot(symbols=symbols)
        bot.scan_regimes()

        assert bot._fetch_calls == symbols
        assert len(bot._feature_calls) == 2
        assert len(bot._regime_calls) == 2

    def test_passes_parameters_to_pipeline(self):
        bot = ScannerBot(symbols=["BTC/USDT"])
        bot.scan_regimes(n_components=5, n_regimes=4)

        assert bot._feature_calls == [5]
        assert bot._regime_calls == [4]


class TestScanNoSymbols:
    def test_warns_when_no_symbols_configured(self):
        bot = ScannerBot(symbols=[])
        with pytest.warns(PipelineWarning, match="No symbols configured"):
            result = bot.scan_regimes()
        assert result is None

    def test_returns_none_when_no_symbols(self):
        bot = ScannerBot(symbols=[])
        with pytest.warns(PipelineWarning):
            result = bot.scan_regimes()
        assert result is None


class TestScanErrorHandling:
    def test_skips_failing_symbol(self):
        bot = FailingScannerBot(
            symbols=["BTC/USDT", "FAIL/USDT", "ETH/USDT"],
            fail_on="FAIL/USDT",
        )
        result = bot.scan_regimes()

        assert isinstance(result, pd.DataFrame)
        # The failing symbol should be skipped
        assert len(result) == 2
        assert "FAIL/USDT" not in result["symbol"].values

    def test_all_symbols_fail_returns_empty_dataframe(self):
        bot = FailingScannerBot(
            symbols=["FAIL/USDT"],
            fail_on="FAIL/USDT",
        )
        result = bot.scan_regimes()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        # Columns should still be present even if empty
        expected_cols = ["symbol", "regime", "confidence", "last_price", "pct_change_24h"]
        assert list(result.columns) == expected_cols
