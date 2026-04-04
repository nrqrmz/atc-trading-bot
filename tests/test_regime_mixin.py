import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.regime_mixin import RegimeMixin


class RegimeBot(RegimeMixin):
    """Minimal class using RegimeMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.features_pca = kwargs.pop("features_pca", None)
        super().__init__(**kwargs)


def _make_synthetic_features_and_df(n=200):
    """Create synthetic PCA features with clear regime separation and matching df."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="1D")

    # Create 3 distinct clusters in feature space
    n1, n2, n3 = n // 3, n // 3, n - 2 * (n // 3)

    # Bull: positive returns
    bull = np.random.randn(n1, 5) + np.array([2, 1, 0, 0, 0])
    bull_close = 30000 + np.cumsum(np.abs(np.random.randn(n1)) * 200)

    # Bear: negative returns
    bear = np.random.randn(n2, 5) + np.array([-2, -1, 0, 0, 0])
    bear_close = bull_close[-1] - np.cumsum(np.abs(np.random.randn(n2)) * 200)

    # Sideways: near zero
    side = np.random.randn(n3, 5) * 0.3
    side_close = bear_close[-1] + np.cumsum(np.random.randn(n3) * 20)

    features_pca = np.vstack([bull, bear, side])
    close = np.concatenate([bull_close, bear_close, side_close])

    df = pd.DataFrame({"Close": close}, index=dates)
    return features_pca, df


class TestRegimeMixin:
    def test_detect_regime_produces_three_states(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime(n_regimes=3)

        assert bot.regimes is not None
        unique_regimes = set(bot.regimes)
        assert len(unique_regimes) == 3

    def test_regime_labels_are_mapped(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime(n_regimes=3)

        # Regimes should be mapped to string labels
        valid_labels = {"bull", "bear", "sideways"}
        assert set(bot.regimes).issubset(valid_labels)

    def test_regime_length_matches_data(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime()

        assert len(bot.regimes) == len(features_pca)

    def test_hmm_model_stored(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime()

        assert bot.hmm_model is not None

    def test_regime_metrics_computed(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime()

        assert bot.regime_metrics is not None
        assert "log_likelihood" in bot.regime_metrics
        assert "bic" in bot.regime_metrics
        assert "avg_duration" in bot.regime_metrics

    def test_warns_without_features(self):
        bot = RegimeBot(df=pd.DataFrame({"Close": [1, 2, 3]}))
        with pytest.warns(UserWarning, match="compute_features"):
            result = bot.detect_regime()
        assert result is None

    def test_current_regime_is_last(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime()

        assert bot.current_regime == bot.regimes[-1]
