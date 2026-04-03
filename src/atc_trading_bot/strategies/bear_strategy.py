import pandas as pd
from backtesting import Strategy


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class BearStrategy(Strategy):
    """Defensive strategy for bear markets. Short mean reversion on resistance."""

    sma_period = 20
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

    def init(self):
        close = pd.Series(self.data.Close)
        self.sma = self.I(lambda: close.rolling(self.sma_period).mean().values, name="SMA")
        self.rsi = self.I(lambda: _compute_rsi(close, self.rsi_period).values, name="RSI")

    def next(self):
        if self.rsi[-1] > self.rsi_overbought and self.data.Close[-1] < self.sma[-1]:
            if self.position.is_long:
                self.position.close()
            self.sell()
        elif self.rsi[-1] < self.rsi_oversold:
            if self.position.is_short:
                self.position.close()
