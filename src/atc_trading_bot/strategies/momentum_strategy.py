import pandas as pd
from backtesting import Strategy

from atc_trading_bot.config import DEFAULT_POSITION_SIZE, DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


class MomentumStrategy(Strategy):
    """Momentum strategy that follows strong directional moves.

    Uses Rate of Change (ROC) to detect momentum and RSI to confirm strength.
    Best suited for bull regimes where trends are persistent.

    Entry: ROC > threshold AND RSI between 40-75 (strong but not overbought).
    Exit: ROC turns negative OR RSI > 80 (exhaustion).
    """

    roc_period = 10
    rsi_period = 14
    roc_threshold = 1  # % price change over roc_period
    stop_loss = DEFAULT_STOP_LOSS
    take_profit = DEFAULT_TAKE_PROFIT
    position_size = DEFAULT_POSITION_SIZE

    def init(self):
        close = pd.Series(self.data.Close)
        self.roc = self.I(
            lambda: close.pct_change(self.roc_period).values * 100,
            name="ROC",
        )
        self.rsi = self.I(
            lambda: _compute_rsi(close, self.rsi_period).values,
            name="RSI",
        )

    def next(self):
        if (self.roc[-1] > self.roc_threshold
                and 40 < self.rsi[-1] < 75
                and not self.position):
            self.buy(size=self.position_size,
                     sl=self.data.Close[-1] * (1 - self.stop_loss),
                     tp=self.data.Close[-1] * (1 + self.take_profit))
        elif self.position.is_long:
            if self.roc[-1] < 0 or self.rsi[-1] > 80:
                self.position.close()
