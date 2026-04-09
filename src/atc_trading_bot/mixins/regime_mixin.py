import warnings

import numpy as np
from atc_trading_bot.config import (
    DEFAULT_HMM_N_ITER,
    DEFAULT_N_REGIMES,
    MIN_REGIME_DURATION,
    REGIME_COLORS,
    STICKY_TRANSITION_PROB,
)
from atc_trading_bot.pipeline_warning import PipelineWarning
from hmmlearn.hmm import GaussianHMM


class RegimeMixin:
    """Mixin for HMM-based market regime detection."""

    REGIME_LABELS = ["bull", "bear", "sideways"]

    def _require_features(self) -> bool:
        """Check that PCA features have been computed."""
        if getattr(self, "features_pca", None) is None:
            warnings.warn("No features available. Call compute_features first.", PipelineWarning)
            return False
        return True

    def _require_regime(self) -> bool:
        """Check that regimes have been detected."""
        if getattr(self, "current_regime", None) is None:
            warnings.warn("No regime detected. Call detect_regime first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.regimes: list[str] | None = None
        self.current_regime: str | None = None
        self.hmm_model: GaussianHMM | None = None
        self.regime_metrics: dict | None = None

    def detect_regime(self, n_regimes: int = DEFAULT_N_REGIMES,
                      n_iter: int = DEFAULT_HMM_N_ITER):
        """Train HMM on PCA features and predict regimes.

        Args:
            n_regimes: Number of hidden states (regimes) for the HMM. Default: 3.
            n_iter: Maximum EM iterations for HMM training. Default: 100.
        """
        if not self._require_features():
            return

        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
            init_params="smc",
            params="stmc",
        )
        # Sticky transition matrix: regimes persist (~20 bars avg duration)
        off_diag = (1 - STICKY_TRANSITION_PROB) / (n_regimes - 1)
        model.transmat_ = np.full((n_regimes, n_regimes), off_diag)
        np.fill_diagonal(model.transmat_, STICKY_TRANSITION_PROB)
        model.fit(self.features_pca)
        self.hmm_model = model

        hidden_states = model.predict(self.features_pca)
        self.regimes = self._map_states_to_labels(hidden_states)
        self.regimes = self._smooth_regimes(self.regimes, min_duration=MIN_REGIME_DURATION)
        self.current_regime = self.regimes[-1]
        self.regime_metrics = self._compute_metrics(hidden_states)
        return self

    def _map_states_to_labels(self, states: np.ndarray) -> list[str]:
        """Map HMM state indices to bull/bear/sideways by mean return per state."""
        close = self.df.loc[self.features_index, "Close"].values
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0.0], returns])

        state_ids = sorted(set(states))
        mean_returns = {s: returns[states == s].mean() for s in state_ids}

        sorted_states = sorted(state_ids, key=lambda s: mean_returns[s])
        label_map = {sorted_states[-1]: "bull", sorted_states[0]: "bear"}
        for s in sorted_states:
            if s not in label_map:
                label_map[s] = "sideways"

        return [label_map[s] for s in states]

    @staticmethod
    def _smooth_regimes(regimes: list[str], min_duration: int = 5) -> list[str]:
        """Replace short regime segments (< min_duration bars) with previous regime."""
        result = list(regimes)

        segments = []
        seg_start = 0
        for i in range(1, len(result)):
            if result[i] != result[seg_start]:
                segments.append((seg_start, i, result[seg_start]))
                seg_start = i
        segments.append((seg_start, len(result), result[seg_start]))

        for idx in range(1, len(segments)):
            start, end, label = segments[idx]
            if (end - start) < min_duration:
                prev_label = segments[idx - 1][2]
                for j in range(start, end):
                    result[j] = prev_label
                segments[idx] = (start, end, prev_label)

        return result

    def _compute_metrics(self, states: np.ndarray) -> dict:
        """Compute regime classification metrics."""
        log_likelihood = self.hmm_model.score(self.features_pca)

        n_params = self._count_params()
        n_samples = len(self.features_pca)
        bic = -2 * log_likelihood + n_params * np.log(n_samples)

        # Average regime duration (consecutive same-state runs)
        durations = []
        current_run = 1
        for i in range(1, len(states)):
            if states[i] == states[i - 1]:
                current_run += 1
            else:
                durations.append(current_run)
                current_run = 1
        durations.append(current_run)

        return {
            "log_likelihood": float(log_likelihood),
            "bic": float(bic),
            "avg_duration": float(np.mean(durations)),
        }

    def _count_params(self) -> int:
        """Count free parameters in the HMM model."""
        n = self.hmm_model.n_components
        n_features = self.features_pca.shape[1]
        # startprob + transmat + means + covariances
        start = n - 1
        trans = n * (n - 1)
        means = n * n_features
        covars = n * n_features * (n_features + 1) // 2
        return start + trans + means + covars

    def override_regime(self, regime: str):
        """Override the detected regime with a manual classification.

        Use this after inspecting ``plot_regimes()`` to correct the HMM's
        classification when you disagree with the detected regime.

        Args:
            regime: One of ``"bull"``, ``"bear"``, ``"sideways"``.

        Returns:
            self for method chaining.
        """
        if regime not in self.REGIME_LABELS:
            raise ValueError(
                f"Invalid regime: '{regime}'. Must be one of {self.REGIME_LABELS}"
            )
        self.current_regime = regime
        return self

    def plot_regimes(self):
        """Plot price with background colored by detected regime.

        Returns:
            matplotlib Figure, or None if no regimes detected.
        """
        if not self._require_regime():
            return

        import plotly.graph_objects as go

        regime_colors = REGIME_COLORS

        # Use aligned data (features_index) for regime spans
        plot_df = self.df.loc[self.features_index]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["Close"],
            mode="lines", name="Price",
            line=dict(color="black", width=1),
        ))

        # Legend traces for regime colors
        for regime, color in regime_colors.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=10, color=color),
                name=regime.capitalize(),
            ))

        # Color background spans by regime
        start = 0
        for i in range(1, len(self.regimes)):
            if self.regimes[i] != self.regimes[start]:
                color = regime_colors.get(self.regimes[start], "#cccccc")
                fig.add_vrect(
                    x0=plot_df.index[start],
                    x1=plot_df.index[i - 1],
                    fillcolor=color, opacity=0.2,
                    layer="below", line_width=0,
                )
                start = i
        # Final segment
        color = regime_colors.get(self.regimes[start], "#cccccc")
        fig.add_vrect(
            x0=plot_df.index[start],
            x1=plot_df.index[-1],
            fillcolor=color, opacity=0.2,
            layer="below", line_width=0,
        )

        fig.update_layout(
            title=f"Regime Detection — Current: {self.current_regime}",
            yaxis_title="Price",
            template="plotly_dark",
            hovermode="x unified",
        )

        return fig
