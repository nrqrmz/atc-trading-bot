"""Tests for PersistenceMixin — save/load trained models."""

import os
import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from atc_trading_bot.mixins.persistence_mixin import PersistenceMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class PersistBot(PersistenceMixin):
    """Minimal class using PersistenceMixin for testing."""

    def __init__(self, **kwargs):
        self.hmm_model = kwargs.pop("hmm_model", None)
        self.pca = kwargs.pop("pca", None)
        self.scaler = kwargs.pop("scaler", None)
        self.features = kwargs.pop("features", None)
        self.current_regime = kwargs.pop("current_regime", None)
        self.regimes = kwargs.pop("regimes", None)
        self.regime_metrics = kwargs.pop("regime_metrics", None)
        self.exchange_id = kwargs.pop("exchange_id", "binanceus")
        self.symbols = kwargs.pop("symbols", ["BTC/USDT"])
        self.timeframe = kwargs.pop("timeframe", "1d")


@pytest.fixture
def mock_hmm():
    np.random.seed(42)
    hmm = GaussianHMM(n_components=3, covariance_type="full", n_iter=5, random_state=42)
    hmm.fit(np.random.randn(50, 3))
    return hmm


@pytest.fixture
def bot_with_model(mock_hmm):
    return PersistBot(
        hmm_model=mock_hmm,
        current_regime="bull",
        regimes=["bull", "bull", "bear"],
        regime_metrics={"log_likelihood": -100.0, "bic": 250.0, "avg_duration": 15.0},
    )


class TestSaveModel:
    def test_saves_file(self, bot_with_model, tmp_path):
        path = str(tmp_path / "model.joblib")
        result = bot_with_model.save_model(path)
        assert os.path.exists(path)
        assert result == str((tmp_path / "model.joblib").resolve())

    def test_creates_parent_directories(self, bot_with_model, tmp_path):
        path = str(tmp_path / "nested" / "dir" / "model.joblib")
        bot_with_model.save_model(path)
        assert os.path.exists(path)

    def test_warns_without_model(self, tmp_path):
        bot = PersistBot()
        with pytest.warns(PipelineWarning, match="trained model"):
            result = bot.save_model(str(tmp_path / "model.joblib"))
        assert result is None


class TestLoadModel:
    def test_roundtrip(self, bot_with_model, tmp_path):
        path = str(tmp_path / "model.joblib")
        bot_with_model.save_model(path)

        new_bot = PersistBot()
        metadata = new_bot.load_model(path)

        assert new_bot.hmm_model is not None
        assert new_bot.current_regime == "bull"
        assert new_bot.regimes == ["bull", "bull", "bear"]
        assert new_bot.regime_metrics["bic"] == 250.0

    def test_metadata_returned(self, bot_with_model, tmp_path):
        path = str(tmp_path / "model.joblib")
        bot_with_model.save_model(path)

        new_bot = PersistBot()
        metadata = new_bot.load_model(path)

        assert "saved_at" in metadata
        assert metadata["exchange_id"] == "binanceus"
        assert metadata["symbols"] == ["BTC/USDT"]
        assert metadata["timeframe"] == "1d"

    def test_warns_on_missing_file(self, tmp_path):
        bot = PersistBot()
        with pytest.warns(PipelineWarning, match="not found"):
            result = bot.load_model(str(tmp_path / "nonexistent.joblib"))
        assert result is None

    def test_restores_regime_metrics(self, bot_with_model, tmp_path):
        path = str(tmp_path / "model.joblib")
        bot_with_model.save_model(path)

        new_bot = PersistBot()
        new_bot.load_model(path)

        assert new_bot.regime_metrics["log_likelihood"] == -100.0
        assert new_bot.regime_metrics["avg_duration"] == 15.0
