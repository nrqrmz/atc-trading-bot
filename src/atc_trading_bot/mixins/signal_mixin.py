import numpy as np
import pandas as pd
from backtesting import Backtest


class SignalMixin:
    """Mixin for generating paper trading signals based on the full pipeline."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.signals: dict | None = None

    def generate_signals(self) -> dict:
        """Run the full pipeline and generate actionable paper trading signals.

        Returns a dict with:
        - symbol: the traded symbol
        - regime: current detected regime
        - strategy: name of the active strategy
        - signal: 'buy', 'sell', or 'hold'
        - confidence: regime consensus across symbols (if multiple)
        """
        if self.df is None:
            raise ValueError("No data available. Call fetch_data first.")
        if self.current_regime is None:
            raise ValueError("No regime detected. Call detect_regime first.")
        if self.active_strategy is None:
            raise ValueError("No strategy selected. Call select_strategy first.")

        signal = self._determine_signal()

        self.signals = {
            "regime": self.current_regime,
            "strategy": self.active_strategy.__name__,
            "signal": signal,
        }
        return self.signals

    def _determine_signal(self) -> str:
        """Determine buy/sell/hold signal by running a short backtest on recent data."""
        # Use last 100 bars (or all available) for signal generation
        lookback = min(100, len(self.df))
        recent_df = self.df.iloc[-lookback:].copy()

        try:
            bt = Backtest(recent_df, self.active_strategy, cash=100_000, commission=0.001)
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
                # Recently closed — signal might flip
                return "hold"

            return "hold"
        except Exception:
            return "hold"
