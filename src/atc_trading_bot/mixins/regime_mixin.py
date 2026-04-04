import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM


class RegimeMixin:
    """Mixin for HMM-based market regime detection."""

    REGIME_LABELS = ["bull", "bear", "sideways"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.regimes: list[str] | None = None
        self.current_regime: str | None = None
        self.hmm_model: GaussianHMM | None = None
        self.regime_metrics: dict | None = None

    def detect_regime(self, n_regimes: int = 3, n_iter: int = 100) -> None:
        """Train HMM on PCA features and predict regimes."""
        if self.features_pca is None:
            warnings.warn("No features available. Call compute_features first.")
            return

        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42,
        )
        model.fit(self.features_pca)
        self.hmm_model = model

        hidden_states = model.predict(self.features_pca)
        self.regimes = self._map_states_to_labels(hidden_states)
        self.current_regime = self.regimes[-1]
        self.regime_metrics = self._compute_metrics(hidden_states)

    def _map_states_to_labels(self, states: np.ndarray) -> list[str]:
        """Map HMM state indices to bull/bear/sideways by mean return per state."""
        close = self.df["Close"].values
        returns = np.diff(close) / close[:-1]
        # Pad returns to match states length
        returns = np.concatenate([[0.0], returns])

        state_ids = sorted(set(states))
        mean_returns = {}
        for s in state_ids:
            mask = states == s
            mean_returns[s] = returns[mask].mean()

        # Sort states by mean return: highest = bull, lowest = bear, middle = sideways
        sorted_states = sorted(state_ids, key=lambda s: mean_returns[s])
        label_map = {
            sorted_states[-1]: "bull",
            sorted_states[0]: "bear",
        }
        for s in sorted_states:
            if s not in label_map:
                label_map[s] = "sideways"

        return [label_map[s] for s in states]

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
