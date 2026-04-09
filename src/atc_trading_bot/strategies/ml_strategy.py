import numpy as np
from backtesting import Strategy

from atc_trading_bot.config import DEFAULT_POSITION_SIZE, DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT


class MLStrategy(Strategy):
    """Strategy that executes trades from pre-computed ML predictions.

    Instead of computing indicators from price data, this strategy reads
    a pre-computed predictions array where each value maps to a trade action:

    - 1 → buy (go long)
    - -1 → sell (go short)
    - 0 → hold (close any open position)

    The predictions array is injected via dynamic subclassing before
    backtesting, following the same pattern as ``_apply_risk_params()``.

    Example::

        strat = type("MLStrategy", (MLStrategy,), {"predictions": preds_array})
    """

    predictions = np.array([])
    stop_loss = DEFAULT_STOP_LOSS
    take_profit = DEFAULT_TAKE_PROFIT
    position_size = DEFAULT_POSITION_SIZE

    def init(self):
        self.signal = self.I(lambda: self.predictions, name="ML_Signal")

    def next(self):
        pred = self.signal[-1]

        if pred == 1 and not self.position.is_long:
            self.position.close()
            self.buy(size=self.position_size,
                     sl=self.data.Close[-1] * (1 - self.stop_loss),
                     tp=self.data.Close[-1] * (1 + self.take_profit))

        elif pred == -1 and not self.position.is_short:
            self.position.close()
            self.sell(size=self.position_size,
                      sl=self.data.Close[-1] * (1 + self.stop_loss),
                      tp=self.data.Close[-1] * (1 - self.take_profit))

        elif pred == 0 and self.position:
            self.position.close()
