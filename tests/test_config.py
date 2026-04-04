"""Tests for the centralized configuration module."""

from atc_trading_bot import config


class TestConfigConstants:
    """Verify all configuration constants exist and have valid values."""

    def test_data_pipeline_defaults(self):
        assert config.DEFAULT_EXCHANGE_ID == "binanceus"
        assert config.DEFAULT_TIMEFRAME == "1d"

    def test_feature_engineering_defaults(self):
        assert config.DEFAULT_N_COMPONENTS == 10
        assert 0 < config.CORRELATION_THRESHOLD < 1
        assert 0 < config.NAN_COLUMN_THRESHOLD < 1

    def test_exclude_cols_contains_ohlcv(self):
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in config.EXCLUDE_COLS

    def test_exclude_cols_contains_binary_indicators(self):
        assert "volatility_bbhi" in config.EXCLUDE_COLS
        assert "volatility_bbli" in config.EXCLUDE_COLS

    def test_exclude_cols_contains_accumulative(self):
        assert "volume_adi" in config.EXCLUDE_COLS
        assert "volume_obv" in config.EXCLUDE_COLS

    def test_regime_detection_defaults(self):
        assert config.DEFAULT_N_REGIMES == 3
        assert config.DEFAULT_HMM_N_ITER == 100
        assert 0 < config.STICKY_TRANSITION_PROB < 1
        assert config.MIN_REGIME_DURATION > 0

    def test_regime_colors_has_all_regimes(self):
        for regime in ("bull", "bear", "sideways"):
            assert regime in config.REGIME_COLORS
            assert config.REGIME_COLORS[regime].startswith("#")

    def test_backtesting_defaults(self):
        assert config.DEFAULT_CASH > 0
        assert 0 < config.DEFAULT_COMMISSION < 1
        assert 0 < config.DEFAULT_TEST_RATIO < 1

    def test_cpcv_defaults(self):
        assert config.DEFAULT_CPCV_SPLITS >= 3
        assert config.DEFAULT_PURGE_GAP >= 0
        assert 0 < config.DEFAULT_EMBARGO_PCT < 1
        assert config.MIN_TRAIN_BARS > 0
        assert config.MIN_TEST_BARS > 0

    def test_signal_lookback(self):
        assert config.SIGNAL_LOOKBACK > 0

    def test_metric_descriptions_keys(self):
        expected = {
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "total_return", "buy_and_hold_return",
            "num_trades",
        }
        assert set(config.METRIC_DESCRIPTIONS.keys()) == expected

    def test_metric_descriptions_are_strings(self):
        for desc in config.METRIC_DESCRIPTIONS.values():
            assert isinstance(desc, str)
            assert len(desc) > 0
