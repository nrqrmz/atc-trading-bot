import pandas as pd
from backtesting import Strategy

from atc_trading_bot.config import DEFAULT_POSITION_SIZE, DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT


class BreakoutStrategy(Strategy):
    """Breakout strategy using Donchian Channels and volume confirmation.

    Detects price breaking out of a consolidation range. Donchian channels
    define the range as the highest high / lowest low over a period.
    Volume must be above average to confirm the breakout is real.

    Long entry: Close breaks above upper channel with above-average volume.
    Short entry: Close breaks below lower channel with above-average volume.
    Exit: Price crosses the midline (mean reversion of the breakout).
    """

    channel_period = 20
    volume_ma_period = 20
    stop_loss = DEFAULT_STOP_LOSS
    take_profit = DEFAULT_TAKE_PROFIT
    position_size = DEFAULT_POSITION_SIZE

    def init(self):
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self.upper = self.I(
            lambda: high.rolling(self.channel_period).max().values,
            name="Donchian_upper",
        )
        self.lower = self.I(
            lambda: low.rolling(self.channel_period).min().values,
            name="Donchian_lower",
        )
        self.mid = self.I(
            lambda: ((high.rolling(self.channel_period).max()
                       + low.rolling(self.channel_period).min()) / 2).values,
            name="Donchian_mid",
        )
        self.vol_ma = self.I(
            lambda: volume.rolling(self.volume_ma_period).mean().values,
            name="Volume_MA",
        )

    def next(self):
        price = self.data.Close[-1]
        volume = self.data.Volume[-1]
        above_avg_volume = volume > self.vol_ma[-1]

        # Long breakout: price breaks above upper channel with volume
        if price > self.upper[-2] and above_avg_volume and not self.position:
            self.buy(size=self.position_size,
                     sl=price * (1 - self.stop_loss),
                     tp=price * (1 + self.take_profit))
        # Short breakout: price breaks below lower channel with volume
        elif price < self.lower[-2] and above_avg_volume and not self.position:
            self.sell(size=self.position_size,
                      sl=price * (1 + self.stop_loss),
                      tp=price * (1 - self.take_profit))
        # Exit long at midline
        elif self.position.is_long and price < self.mid[-1]:
            self.position.close()
        # Exit short at midline
        elif self.position.is_short and price > self.mid[-1]:
            self.position.close()
