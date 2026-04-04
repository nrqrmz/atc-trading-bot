import pandas as pd
from backtesting import Strategy

from atc_trading_bot.config import DEFAULT_POSITION_SIZE, DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT


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
    stop_loss = DEFAULT_STOP_LOSS
    take_profit = DEFAULT_TAKE_PROFIT
    position_size = DEFAULT_POSITION_SIZE

    def init(self):
        close = pd.Series(self.data.Close)
        self.sma = self.I(lambda: close.rolling(self.sma_period).mean().values, name="SMA")
        self.rsi = self.I(lambda: _compute_rsi(close, self.rsi_period).values, name="RSI")

    def next(self):
        if self.rsi[-1] > self.rsi_overbought and self.data.Close[-1] < self.sma[-1]:
            if not self.position.is_short:
                self.position.close()
                self.sell(size=self.position_size,
                          sl=self.data.Close[-1] * (1 + self.stop_loss),
                          tp=self.data.Close[-1] * (1 - self.take_profit))
        elif self.rsi[-1] < self.rsi_oversold:
            if self.position.is_short:
                self.position.close()
