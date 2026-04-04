import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

from atc_trading_bot.config import DEFAULT_POSITION_SIZE, DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT


class BullStrategy(Strategy):
    """Trend following strategy for bull markets using SMA crossovers."""

    sma_fast = 20
    sma_slow = 50
    stop_loss = DEFAULT_STOP_LOSS
    take_profit = DEFAULT_TAKE_PROFIT
    position_size = DEFAULT_POSITION_SIZE

    def init(self):
        close = pd.Series(self.data.Close)
        self.fast_sma = self.I(lambda: close.rolling(self.sma_fast).mean().values, name="SMA_fast")
        self.slow_sma = self.I(lambda: close.rolling(self.sma_slow).mean().values, name="SMA_slow")

    def next(self):
        if crossover(self.fast_sma, self.slow_sma):
            self.buy(size=self.position_size, sl=self.data.Close[-1] * (1 - self.stop_loss),
                     tp=self.data.Close[-1] * (1 + self.take_profit))
        elif crossover(self.slow_sma, self.fast_sma):
            self.position.close()
