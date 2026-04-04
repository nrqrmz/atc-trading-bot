import warnings

import numpy as np
from atc_trading_bot.config import (
    DEFAULT_CASH,
    DEFAULT_COMMISSION,
    DEFAULT_CONFIDENCE_THRESHOLD,
    SIGNAL_LOOKBACK,
)
from atc_trading_bot.pipeline_warning import PipelineWarning
import pandas as pd
from backtesting.lib import FractionalBacktest as Backtest


class SignalMixin:
    """Mixin for generating paper trading signals based on the full pipeline."""

    def _require_data(self) -> bool:
        """Check that OHLCV data has been loaded."""
        if getattr(self, "df", None) is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
            return False
        return True

    def _require_regime(self) -> bool:
        """Check that regimes have been detected."""
        if getattr(self, "current_regime", None) is None:
            warnings.warn("No regime detected. Call detect_regime first.", PipelineWarning)
            return False
        return True

    def _require_strategy(self) -> bool:
        """Check that a strategy has been selected."""
        if getattr(self, "active_strategy", None) is None:
            warnings.warn("No strategy selected. Call select_strategy first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals: dict | None = None

    def generate_signals(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
        """Generate actionable paper trading signals with confidence filtering.

        Uses HMM posterior probabilities to assess regime confidence.
        Signals below the confidence threshold are overridden to 'hold'.

        Args:
            confidence_threshold: Minimum regime probability to act on a signal.
                Default: 0.6. Set to 0 to disable confidence filtering.

        Returns a dict with:
        - regime: current detected regime
        - strategy: name of the active strategy
        - signal: 'buy', 'sell', or 'hold'
        - confidence: HMM posterior probability for the current regime
        """
        if not self._require_data():
            return
        if not self._require_regime():
            return
        if not self._require_strategy():
            return

        confidence = self._compute_confidence()
        signal = self._determine_signal()

        # Override signal to hold when confidence is below threshold
        if confidence < confidence_threshold:
            signal = "hold"

        self.signals = {
            "regime": self.current_regime,
            "strategy": self.active_strategy.__name__,
            "signal": signal,
            "confidence": round(confidence, 4),
        }
        return self.signals

    def _compute_confidence(self) -> float:
        """Compute regime confidence from HMM posterior probabilities.

        Returns the posterior probability of the current regime for the
        last observation. Falls back to 1.0 if HMM model is unavailable.
        """
        hmm_model = getattr(self, "hmm_model", None)
        features_pca = getattr(self, "features_pca", None)

        if hmm_model is None or features_pca is None:
            return 1.0

        try:
            posteriors = hmm_model.predict_proba(features_pca)
            last_posteriors = posteriors[-1]
            # The predicted state for the last observation
            predicted_state = hmm_model.predict(features_pca[-1:])[-1]
            return float(last_posteriors[predicted_state])
        except Exception:
            return 1.0

    def _determine_signal(self) -> str:
        """Determine buy/sell/hold signal by running a short backtest on recent data."""
        lookback = min(SIGNAL_LOOKBACK, len(self.df))
        recent_df = self.df.iloc[-lookback:].copy()

        try:
            bt = Backtest(recent_df, self.active_strategy, cash=DEFAULT_CASH,
                          commission=DEFAULT_COMMISSION, finalize_trades=True)
            stats = bt.run()
            trades = stats._trades

            if trades.empty:
                return "hold"

            last_trade = trades.iloc[-1]

            # If the last trade is still open (ExitTime is NaT or at the end)
            if pd.isna(last_trade.get("ExitTime")) or last_trade["ExitTime"] == recent_df.index[-1]:
                return "buy" if last_trade["Size"] > 0 else "sell"

            # If last trade closed, check how recently
            bars_since_exit = len(recent_df) - recent_df.index.get_loc(last_trade["ExitTime"]) - 1
            if bars_since_exit <= 2:
                return "hold"

            return "hold"
        except Exception:
            return "hold"
