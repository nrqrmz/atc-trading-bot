import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.regime_mixin import RegimeMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


class RegimeBot(RegimeMixin):
    """Minimal class using RegimeMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.features_pca = kwargs.pop("features_pca", None)
        self.features_index = kwargs.pop("features_index", None)
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
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime(n_regimes=3)

        assert bot.regimes is not None
        unique_regimes = set(bot.regimes)
        assert len(unique_regimes) == 3

    def test_regime_labels_are_mapped(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime(n_regimes=3)

        valid_labels = {"bull", "bear", "sideways"}
        assert set(bot.regimes).issubset(valid_labels)

    def test_regime_length_matches_data(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()

        assert len(bot.regimes) == len(features_pca)

    def test_hmm_model_stored(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()

        assert bot.hmm_model is not None

    def test_regime_metrics_computed(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()

        assert bot.regime_metrics is not None
        assert "log_likelihood" in bot.regime_metrics
        assert "bic" in bot.regime_metrics
        assert "avg_duration" in bot.regime_metrics

    def test_warns_without_features(self):
        bot = RegimeBot(df=pd.DataFrame({"Close": [1, 2, 3]}))
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.detect_regime()
        assert result is None

    def test_current_regime_is_last(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()

        assert bot.current_regime == bot.regimes[-1]

    def test_plot_regimes_returns_figure(self):
        from plotly.graph_objects import Figure

        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()
        fig = bot.plot_regimes()

        assert isinstance(fig, Figure)

    def test_plot_regimes_warns_without_regimes(self):
        bot = RegimeBot(df=pd.DataFrame({"Close": [1, 2, 3]}))
        with pytest.warns(PipelineWarning, match="detect_regime"):
            result = bot.plot_regimes()
        assert result is None

    def test_regime_labels_correct_for_clear_regimes(self):
        """With well-separated clusters, bull/bear labels should match the data structure."""
        features_pca, df = _make_synthetic_features_and_df(n=300)
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime(n_regimes=3)

        # The bull cluster (first third) has rising prices
        # The bear cluster (second third) has falling prices
        n_third = 100
        bull_section = bot.regimes[:n_third]
        bear_section = bot.regimes[n_third:2 * n_third]
        assert bull_section.count("bull") > n_third * 0.4
        assert bear_section.count("bear") > n_third * 0.4

    def test_override_regime_sets_current(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()
        original = bot.current_regime

        target = "bear" if original != "bear" else "bull"
        bot.override_regime(target)
        assert bot.current_regime == target

    def test_override_regime_returns_self(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()
        result = bot.override_regime("sideways")
        assert result is bot

    def test_override_regime_invalid_raises(self):
        bot = RegimeBot()
        bot.current_regime = "bull"
        with pytest.raises(ValueError, match="Invalid regime"):
            bot.override_regime("crash")

    def test_override_regime_preserves_regimes_list(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca, features_index=df.index)
        bot.detect_regime()
        original_regimes = list(bot.regimes)
        bot.override_regime("bear")
        assert bot.regimes == original_regimes

    def test_regime_mapping_uses_aligned_returns(self):
        """Regime mapping should use features_index-aligned Close prices."""
        np.random.seed(42)
        n_full = 250
        n_valid = 200  # Simulate NaN dropout from indicator warmup
        dates = pd.date_range("2023-01-01", periods=n_full, freq="1D")

        # Full df with all rows
        close = 30000 + np.cumsum(np.random.randn(n_full) * 200)
        df = pd.DataFrame({"Close": close}, index=dates)

        # features_pca only covers the last n_valid rows (after NaN drop)
        valid_index = dates[n_full - n_valid:]
        features_pca = np.random.randn(n_valid, 5)

        bot = RegimeBot(df=df, features_pca=features_pca, features_index=valid_index)
        bot.detect_regime(n_regimes=3)

        # Should not crash and should have correct length
        assert len(bot.regimes) == n_valid
        assert set(bot.regimes).issubset({"bull", "bear", "sideways"})
