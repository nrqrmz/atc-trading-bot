"""Labeling mixin — Triple-Barrier Method for realistic trade labeling.

Instead of simple "price went up → buy" labels, the triple-barrier method
sets three exit conditions for each potential trade:

- Upper barrier (take-profit): price rises by tp_factor * ATR → label = 1 (buy)
- Lower barrier (stop-loss): price falls by sl_factor * ATR → label = -1 (sell)
- Vertical barrier (timeout): max_holding bars pass → label = 0 (hold)

The barriers are calibrated dynamically using recent ATR, producing labels
that reflect realistic trading outcomes and naturally reduce class imbalance.
"""
import warnings

import numpy as np
import pandas as pd
from atc_trading_bot.config import (
    DEFAULT_ATR_PERIOD,
    DEFAULT_MAX_HOLDING,
    DEFAULT_SL_FACTOR,
    DEFAULT_TP_FACTOR,
)
from atc_trading_bot.pipeline_warning import PipelineWarning


class LabelingMixin:
    """Mixin for triple-barrier trade labeling."""

    def _require_data(self) -> bool:
        """Check that OHLCV data has been loaded."""
        if getattr(self, "df", None) is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels: pd.Series | None = None

    def compute_labels(self, tp_factor: float = DEFAULT_TP_FACTOR,
                       sl_factor: float = DEFAULT_SL_FACTOR,
                       max_holding: int = DEFAULT_MAX_HOLDING,
                       atr_period: int = DEFAULT_ATR_PERIOD):
        """Compute triple-barrier labels for the loaded data.

        For each bar, looks forward up to max_holding bars and checks
        which barrier is hit first: take-profit (+1), stop-loss (-1),
        or timeout (0).

        Args:
            tp_factor: Take-profit multiplier on ATR. Default: 2.0.
            sl_factor: Stop-loss multiplier on ATR. Default: 2.0.
            max_holding: Maximum bars before timeout. Default: 10.
            atr_period: ATR lookback period for barrier width. Default: 14.

        Returns:
            self for method chaining.
        """
        if not self._require_data():
            return

        df = self.df
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values

        # Compute ATR
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        tr = np.concatenate([[high[0] - low[0]], tr])
        atr = pd.Series(tr).rolling(atr_period).mean().values

        n = len(close)
        labels = np.zeros(n, dtype=int)

        for i in range(n):
            if np.isnan(atr[i]) or atr[i] == 0:
                labels[i] = 0
                continue

            upper = close[i] + tp_factor * atr[i]
            lower = close[i] - sl_factor * atr[i]
            end = min(i + max_holding, n)

            label = 0  # timeout by default
            for j in range(i + 1, end):
                if close[j] >= upper:
                    label = 1  # take-profit hit
                    break
                if close[j] <= lower:
                    label = -1  # stop-loss hit
                    break

            labels[i] = label

        self.labels = pd.Series(labels, index=df.index, name="label")
        self.df["label"] = self.labels
        return self

    def labels_summary(self) -> pd.DataFrame | None:
        """Return a summary of label distribution.

        Returns:
            DataFrame with label counts and percentages, indexed by label name.
        """
        if self.labels is None:
            warnings.warn("No labels computed. Call compute_labels first.", PipelineWarning)
            return

        label_names = {1: "buy", -1: "sell", 0: "hold"}
        counts = self.labels.value_counts()

        rows = []
        total = len(self.labels)
        for val in [1, -1, 0]:
            count = counts.get(val, 0)
            rows.append({
                "count": count,
                "percentage": round(count / total * 100, 1),
            })

        df = pd.DataFrame(rows, index=["buy", "sell", "hold"])
        df.index.name = "label"
        return df
