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
        with pytest.warns(PipelineWarning, match="compute_features"):
            result = bot.detect_regime()
        assert result is None

    def test_current_regime_is_last(self):
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime()

        assert bot.current_regime == bot.regimes[-1]

    def test_plot_regimes_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime()
        fig = bot.plot_regimes()

        assert isinstance(fig, Figure)

    def test_plot_regimes_warns_without_regimes(self):
        bot = RegimeBot(df=pd.DataFrame({"Close": [1, 2, 3]}))
        with pytest.warns(PipelineWarning, match="detect_regime"):
            result = bot.plot_regimes()
        assert result is None

    def test_plot_regimes_does_not_call_plt_show(self):
        import matplotlib
        matplotlib.use("Agg")
        from unittest.mock import patch

        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime()

        with patch("matplotlib.pyplot.show") as mock_show:
            bot.plot_regimes()
            mock_show.assert_not_called()

    def test_map_states_labels_follow_pc1_means(self):
        """Labeling should follow HMM PC1 means: highest=bull, lowest=bear."""
        features_pca, df = _make_synthetic_features_and_df()
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime(n_regimes=3)

        # Verify the label mapping follows PC1 means ordering
        pc1_means = bot.hmm_model.means_[:, 0]
        sorted_states = list(np.argsort(pc1_means))

        # Reconstruct expected mapping from the model's own means
        expected_map = {sorted_states[-1]: "bull", sorted_states[0]: "bear"}
        for s in sorted_states:
            if s not in expected_map:
                expected_map[s] = "sideways"

        states = bot.hmm_model.predict(features_pca)
        expected_labels = [expected_map[s] for s in states]
        assert bot.regimes == expected_labels

    def test_regime_labels_correct_for_clear_regimes(self):
        """With well-separated clusters, bull/bear labels should match the data structure."""
        features_pca, df = _make_synthetic_features_and_df(n=300)
        bot = RegimeBot(df=df, features_pca=features_pca)
        bot.detect_regime(n_regimes=3)

        # The bull cluster (first third) has PC1 center at +2
        # The bear cluster (second third) has PC1 center at -2
        # Majority of first third should be "bull"
        n_third = 100
        bull_section = bot.regimes[:n_third]
        bear_section = bot.regimes[n_third:2 * n_third]
        assert bull_section.count("bull") > n_third * 0.4
        assert bear_section.count("bear") > n_third * 0.4
