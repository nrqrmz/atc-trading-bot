"""Tests for ModelMixin — ML classifiers and ensembles."""

import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.model_mixin import ModelMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class ModelBot(ModelMixin):
    def __init__(self, **kwargs):
        self.features_pca = kwargs.pop("features_pca", None)
        self.features_index = kwargs.pop("features_index", None)
        self.labels = kwargs.pop("labels", None)
        super().__init__(**kwargs)


@pytest.fixture
def ml_data():
    """Synthetic features + labels for ML training."""
    np.random.seed(42)
    n = 300
    n_features = 10
    X = np.random.randn(n, n_features)

    # Labels correlated with first feature (learnable pattern)
    y = np.where(X[:, 0] > 0.5, 1, np.where(X[:, 0] < -0.5, -1, 0))

    dates = pd.date_range("2023-01-01", periods=n, freq="1D")
    labels = pd.Series(y, index=dates, name="label")
    features_index = dates

    return X, labels, features_index


class TestTrainModels:
    def test_returns_self(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        result = bot.train_models()
        assert result is bot

    def test_trains_all_models(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()

        assert "LightGBM" in bot.trained_models
        assert "CatBoost" in bot.trained_models
        assert "XGBoost" in bot.trained_models
        assert "Voting" in bot.trained_models
        assert "Stacking" in bot.trained_models

    def test_selects_best_model(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()
        assert bot.active_model is not None

    def test_warns_without_features(self):
        labels = pd.Series([0, 1, -1])
        bot = ModelBot(labels=labels)
        with pytest.warns(PipelineWarning, match="compute_features"):
            bot.train_models()

    def test_warns_without_labels(self):
        X = np.random.randn(100, 5)
        bot = ModelBot(features_pca=X)
        with pytest.warns(PipelineWarning, match="compute_labels"):
            bot.train_models()

    def test_custom_parameters(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models(n_estimators=50, learning_rate=0.1, max_depth=3)
        assert bot.active_model is not None


class TestModelsSummary:
    def test_returns_dataframe(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()
        summary = bot.models_summary()

        assert isinstance(summary, pd.DataFrame)
        assert summary.index.name == "model"

    def test_has_expected_columns(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()
        summary = bot.models_summary()

        assert "accuracy" in summary.columns
        assert "f1" in summary.columns
        assert "precision" in summary.columns
        assert "recall" in summary.columns
        assert "cv_f1_mean" in summary.columns

    def test_has_all_models(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()
        summary = bot.models_summary()

        assert "LightGBM" in summary.index
        assert "CatBoost" in summary.index
        assert "XGBoost" in summary.index
        assert "Voting" in summary.index
        assert "Stacking" in summary.index

    def test_warns_without_training(self):
        bot = ModelBot()
        with pytest.warns(PipelineWarning, match="train_models"):
            result = bot.models_summary()
        assert result is None


class TestPredict:
    def test_predict_returns_array(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()
        preds = bot.predict()

        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_predict_valid_labels(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()
        preds = bot.predict()

        assert set(preds).issubset({-1, 0, 1})

    def test_predict_custom_input(self, ml_data):
        X, labels, idx = ml_data
        bot = ModelBot(features_pca=X, labels=labels, features_index=idx)
        bot.train_models()

        custom_X = np.random.randn(5, X.shape[1])
        preds = bot.predict(custom_X)
        assert len(preds) == 5

    def test_warns_without_model(self):
        bot = ModelBot()
        with pytest.warns(PipelineWarning, match="train_models"):
            bot.predict()
