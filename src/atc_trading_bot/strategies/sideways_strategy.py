import pandas as pd
from backtesting import Strategy


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class SidewaysStrategy(Strategy):
    """Mean reversion strategy for sideways markets using Bollinger Bands + RSI."""

    bb_period = 20
    bb_std = 2
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30

    def init(self):
        close = pd.Series(self.data.Close)
        sma = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        self.sma = self.I(lambda: sma.values, name="BB_SMA")
        self.bb_upper = self.I(lambda: (sma + self.bb_std * std).values, name="BB_upper")
        self.bb_lower = self.I(lambda: (sma - self.bb_std * std).values, name="BB_lower")
        self.rsi = self.I(lambda: _compute_rsi(close, self.rsi_period).values, name="RSI")

    def next(self):
        if self.data.Close[-1] <= self.bb_lower[-1] and self.rsi[-1] < self.rsi_oversold:
            if not self.position.is_long:
                self.position.close()
                self.buy()
        elif self.data.Close[-1] >= self.bb_upper[-1] and self.rsi[-1] > self.rsi_overbought:
            if self.position.is_long:
                self.position.close()
            self.sell()
