import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover


class BullStrategy(Strategy):
    """Trend following strategy for bull markets using SMA crossovers."""

    sma_fast = 20
    sma_slow = 50

    def init(self):
        close = pd.Series(self.data.Close)
        self.fast_sma = self.I(lambda: close.rolling(self.sma_fast).mean().values, name="SMA_fast")
        self.slow_sma = self.I(lambda: close.rolling(self.sma_slow).mean().values, name="SMA_slow")

    def next(self):
        if crossover(self.fast_sma, self.slow_sma):
            self.buy()
        elif crossover(self.slow_sma, self.fast_sma):
            self.position.close()
