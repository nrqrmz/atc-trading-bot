"""Centralized configuration for the ATC Trading Bot.

All tunable parameters live here. Each mixin imports what it needs,
so students only need to modify this file to experiment with settings.
"""

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------
DEFAULT_EXCHANGE_ID = "binanceus"
DEFAULT_TIMEFRAME = "1d"

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
DEFAULT_N_COMPONENTS = 10

# Features to exclude from regime detection.
# Binary (0/1), price-level (scale-dependent), and accumulative (non-stationary).
EXCLUDE_COLS = {
    # Raw OHLCV
    "Open", "High", "Low", "Close", "Volume",
    # Binary (0/1 only — not continuous)
    "volatility_bbhi", "volatility_bbli",
    "volatility_kchi", "volatility_kcli",
    "trend_psar_up_indicator", "trend_psar_down_indicator",
    # Price-level (scale-dependent, redundant with %B/width)
    "trend_ema_fast", "trend_ema_slow",
    "trend_sma_fast", "trend_sma_slow",
    "volatility_bbh", "volatility_bbl", "volatility_bbm",
    "volatility_kch", "volatility_kcl", "volatility_kcc",
    # Accumulative (no ceiling, non-stationary)
    "volume_adi", "volume_obv", "volume_vpt",
    "volume_nvi", "others_cr", "volume_vwap",
}

CORRELATION_THRESHOLD = 0.95
NAN_COLUMN_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Regime Detection (HMM)
# ---------------------------------------------------------------------------
DEFAULT_N_REGIMES = 3
DEFAULT_HMM_N_ITER = 100
STICKY_TRANSITION_PROB = 0.95
MIN_REGIME_DURATION = 5

REGIME_COLORS = {
    "bull": "#2ecc71",
    "bear": "#e74c3c",
    "sideways": "#f1c40f",
}

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
DEFAULT_CASH = 100_000
DEFAULT_COMMISSION = 0.001
DEFAULT_TEST_RATIO = 0.3
DEFAULT_LEVERAGE = 1

# CPCV (Combinatorial Purged Cross-Validation)
DEFAULT_CPCV_SPLITS = 5
DEFAULT_PURGE_GAP = 5
DEFAULT_EMBARGO_PCT = 0.01
MIN_TRAIN_BARS = 50
MIN_TEST_BARS = 30

# ---------------------------------------------------------------------------
# Risk Management
# ---------------------------------------------------------------------------
DEFAULT_STOP_LOSS = 0.05       # 5% stop-loss
DEFAULT_TAKE_PROFIT = 0.10     # 10% take-profit
DEFAULT_POSITION_SIZE = 0.05   # 5% of equity per trade (risk management rule)

# ---------------------------------------------------------------------------
# Overfitting Detection
# ---------------------------------------------------------------------------
# Warn when out-of-sample metric is below this fraction of in-sample metric.
# E.g. 0.5 means warn if test Sharpe < 50% of train Sharpe.
OVERFIT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------
SIGNAL_LOOKBACK = 100
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Metric Descriptions (used in backtest result DataFrames)
# ---------------------------------------------------------------------------
METRIC_DESCRIPTIONS = {
    "sharpe_ratio": "Risk-adjusted return (annualized)",
    "sortino_ratio": "Downside risk-adjusted return",
    "max_drawdown": "Worst peak-to-trough decline",
    "calmar_ratio": "Annual return / Max drawdown",
    "win_rate": "Percentage of winning trades",
    "profit_factor": "Gross profit / Gross loss",
    "total_return": "Total strategy return",
    "buy_and_hold_return": "Buy & hold benchmark return",
    "num_trades": "Number of trades executed",
}
