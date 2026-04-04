import pandas as pd
from backtesting import Strategy

from atc_trading_bot.config import DEFAULT_POSITION_SIZE, DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT


class VolatilityStrategy(Strategy):
    """Volatility mean reversion strategy using ATR.

    Exploits the tendency of volatility to revert to its mean. When
    current ATR is far above its moving average, volatility is likely
    to contract (good for selling). When ATR is far below, volatility
    is likely to expand (prepare for a move).

    Long entry: ATR drops below its MA (low vol → expect expansion, buy dips).
    Short entry: ATR spikes above 1.5x its MA (high vol → expect contraction).
    Exit: ATR returns near its MA (vol normalized).
    """

    atr_period = 14
    atr_ma_period = 50
    atr_spike_multiplier = 1.5
    stop_loss = DEFAULT_STOP_LOSS
    take_profit = DEFAULT_TAKE_PROFIT
    position_size = DEFAULT_POSITION_SIZE

    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        close = pd.Series(self.data.Close)

        # True Range
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(self.atr_period).mean()
        atr_ma = atr.rolling(self.atr_ma_period).mean()

        self.atr = self.I(lambda: atr.values, name="ATR")
        self.atr_ma = self.I(lambda: atr_ma.values, name="ATR_MA")

    def next(self):
        if self.atr_ma[-1] == 0:
            return

        atr_ratio = self.atr[-1] / self.atr_ma[-1]

        # Low volatility regime: buy anticipating expansion
        price = self.data.Close[-1]
        if atr_ratio < 0.8 and not self.position:
            self.buy(size=self.position_size,
                     sl=price * (1 - self.stop_loss),
                     tp=price * (1 + self.take_profit))
        # High volatility spike: sell anticipating contraction
        elif atr_ratio > self.atr_spike_multiplier and not self.position:
            self.sell(size=self.position_size,
                      sl=price * (1 + self.stop_loss),
                      tp=price * (1 - self.take_profit))
        # Exit when volatility normalizes
        elif self.position and 0.9 < atr_ratio < 1.2:
            self.position.close()
