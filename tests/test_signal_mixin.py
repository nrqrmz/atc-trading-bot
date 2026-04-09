import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from atc_trading_bot.mixins.signal_mixin import SignalMixin
from atc_trading_bot.pipeline_warning import PipelineWarning
from atc_trading_bot.strategies.bull_strategy import BullStrategy


class SignalBot(SignalMixin):
    """Minimal class using SignalMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.current_regime = kwargs.pop("current_regime", None)
        self.active_strategy = kwargs.pop("active_strategy", None)
        self.hmm_model = kwargs.pop("hmm_model", None)
        self.features_pca = kwargs.pop("features_pca", None)
        super().__init__(**kwargs)


@pytest.fixture
def trending_data():
    """Data with a clear uptrend to trigger bull strategy signals."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 30000 + np.linspace(0, 8000, n) + np.random.randn(n) * 100
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 50
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestSignalMixin:
    def test_generate_signals_returns_dict(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        signals = bot.generate_signals()

        assert isinstance(signals, dict)
        assert "regime" in signals
        assert "strategy" in signals
        assert "signal" in signals
        assert "confidence" in signals

    def test_signal_is_valid(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        signals = bot.generate_signals()

        assert signals["signal"] in ("buy", "sell", "hold")

    def test_signal_respects_regime(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        signals = bot.generate_signals()

        assert signals["regime"] == "bull"
        assert signals["strategy"] == "BullStrategy"

    def test_signals_stored(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        bot.generate_signals()

        assert bot.signals is not None

    def test_warns_without_data(self):
        bot = SignalBot(current_regime="bull", active_strategy=BullStrategy)
        with pytest.warns(PipelineWarning, match="fetch_data"):
            result = bot.generate_signals()
        assert result is None

    def test_warns_without_regime(self, trending_data):
        bot = SignalBot(df=trending_data, active_strategy=BullStrategy)
        with pytest.warns(PipelineWarning, match="detect_regime"):
            result = bot.generate_signals()
        assert result is None

    def test_warns_without_strategy(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull")
        with pytest.warns(PipelineWarning, match="select_strategy"):
            result = bot.generate_signals()
        assert result is None


class TestConfidenceThresholding:
    def test_confidence_defaults_to_one_without_hmm(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        confidence = bot._compute_confidence()
        assert confidence == 1.0

    def test_confidence_from_hmm_posteriors(self, trending_data):
        n_features = 5
        n_obs = len(trending_data)
        features_pca = np.random.randn(n_obs, n_features)

        mock_hmm = MagicMock()
        posteriors = np.array([[0.8, 0.1, 0.1]] * n_obs)
        mock_hmm.predict_proba.return_value = posteriors
        mock_hmm.predict.return_value = np.array([0])

        bot = SignalBot(
            df=trending_data, current_regime="bull", active_strategy=BullStrategy,
            hmm_model=mock_hmm, features_pca=features_pca,
        )
        confidence = bot._compute_confidence()
        assert 0 < confidence <= 1.0
        assert confidence == 0.8

    def test_low_confidence_overrides_to_hold(self, trending_data):
        n_obs = len(trending_data)
        features_pca = np.random.randn(n_obs, 5)

        mock_hmm = MagicMock()
        # Very low confidence for predicted state
        posteriors = np.array([[0.35, 0.33, 0.32]] * n_obs)
        mock_hmm.predict_proba.return_value = posteriors
        mock_hmm.predict.return_value = np.array([0])

        bot = SignalBot(
            df=trending_data, current_regime="bull", active_strategy=BullStrategy,
            hmm_model=mock_hmm, features_pca=features_pca,
        )
        signals = bot.generate_signals(confidence_threshold=0.6)

        assert signals["signal"] == "hold"
        assert signals["confidence"] == 0.35

    def test_high_confidence_allows_signal(self, trending_data):
        n_obs = len(trending_data)
        features_pca = np.random.randn(n_obs, 5)

        mock_hmm = MagicMock()
        posteriors = np.array([[0.9, 0.05, 0.05]] * n_obs)
        mock_hmm.predict_proba.return_value = posteriors
        mock_hmm.predict.return_value = np.array([0])

        bot = SignalBot(
            df=trending_data, current_regime="bull", active_strategy=BullStrategy,
            hmm_model=mock_hmm, features_pca=features_pca,
        )
        signals = bot.generate_signals(confidence_threshold=0.6)

        assert signals["confidence"] == 0.9
        # Signal can be buy, sell, or hold — but not forced to hold by confidence
        assert signals["signal"] in ("buy", "sell", "hold")

    def test_zero_threshold_disables_filtering(self, trending_data):
        n_obs = len(trending_data)
        features_pca = np.random.randn(n_obs, 5)

        mock_hmm = MagicMock()
        posteriors = np.array([[0.35, 0.33, 0.32]] * n_obs)
        mock_hmm.predict_proba.return_value = posteriors
        mock_hmm.predict.return_value = np.array([0])

        bot = SignalBot(
            df=trending_data, current_regime="bull", active_strategy=BullStrategy,
            hmm_model=mock_hmm, features_pca=features_pca,
        )
        signals = bot.generate_signals(confidence_threshold=0)
        # With threshold=0, confidence filtering is disabled
        assert signals["confidence"] == 0.35

    def test_confidence_in_output(self, trending_data):
        bot = SignalBot(df=trending_data, current_regime="bull", active_strategy=BullStrategy)
        signals = bot.generate_signals()

        assert "confidence" in signals
        assert isinstance(signals["confidence"], float)
        assert 0 <= signals["confidence"] <= 1


class SignalMLBot(SignalMixin):
    """Minimal class using SignalMixin for ML signal testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.current_regime = kwargs.pop("current_regime", None)
        self.active_model = kwargs.pop("active_model", None)
        self.features_pca = kwargs.pop("features_pca", None)
        self._label_unmap = kwargs.pop("_label_unmap", {0: -1, 1: 0, 2: 1})
        super().__init__(**kwargs)


class TestSignalML:
    def test_generate_signals_ml_returns_dict(self, trending_data):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])  # mapped "buy"
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])

        n_obs = len(trending_data)
        features_pca = np.random.randn(n_obs, 5)

        bot = SignalMLBot(
            df=trending_data, current_regime="bull",
            active_model=mock_model, features_pca=features_pca,
        )
        signals = bot.generate_signals_ml()

        assert isinstance(signals, dict)
        assert "regime" in signals
        assert "model" in signals
        assert "signal" in signals
        assert "confidence" in signals

    def test_signal_is_valid_ml(self, trending_data):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])

        bot = SignalMLBot(
            df=trending_data, current_regime="bull",
            active_model=mock_model, features_pca=np.random.randn(len(trending_data), 5),
        )
        signals = bot.generate_signals_ml()

        assert signals["signal"] in ("buy", "sell", "hold")

    def test_confidence_from_predict_proba(self, trending_data):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])
        mock_model.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])

        bot = SignalMLBot(
            df=trending_data, current_regime="bull",
            active_model=mock_model, features_pca=np.random.randn(len(trending_data), 5),
        )
        signals = bot.generate_signals_ml()

        assert signals["confidence"] == 0.8

    def test_low_confidence_overrides_to_hold_ml(self, trending_data):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])
        mock_model.predict_proba.return_value = np.array([[0.35, 0.33, 0.32]])

        bot = SignalMLBot(
            df=trending_data, current_regime="bull",
            active_model=mock_model, features_pca=np.random.randn(len(trending_data), 5),
        )
        signals = bot.generate_signals_ml(confidence_threshold=0.6)

        assert signals["signal"] == "hold"
        assert signals["confidence"] == 0.35

    def test_warns_without_model(self, trending_data):
        bot = SignalMLBot(
            df=trending_data, current_regime="bull",
            features_pca=np.random.randn(len(trending_data), 5),
        )
        with pytest.warns(PipelineWarning, match="train_models"):
            result = bot.generate_signals_ml()
        assert result is None

    def test_warns_without_features(self, trending_data):
        mock_model = MagicMock()
        bot = SignalMLBot(
            df=trending_data, current_regime="bull",
            active_model=mock_model,
        )
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.generate_signals_ml()
        assert result is None

    def test_fallback_when_no_predict_proba(self, trending_data):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([2])
        mock_model.predict_proba.side_effect = AttributeError

        bot = SignalMLBot(
            df=trending_data, current_regime="bull",
            active_model=mock_model, features_pca=np.random.randn(len(trending_data), 5),
        )
        signals = bot.generate_signals_ml(confidence_threshold=0)

        assert signals["confidence"] == 1.0
        assert signals["signal"] == "buy"

    def test_ml_signals_stored(self, trending_data):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.8, 0.1]])

        bot = SignalMLBot(
            df=trending_data, current_regime="sideways",
            active_model=mock_model, features_pca=np.random.randn(len(trending_data), 5),
        )
        bot.generate_signals_ml()

        assert bot.ml_signals is not None
        assert bot.ml_signals["regime"] == "sideways"
