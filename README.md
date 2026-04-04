# ATC Trading Bot

Algorithmic trading bot for crypto markets that detects market regimes using Hidden Markov Models (HMM) and applies differentiated strategies per regime.

## Features

- **Regime Detection**: HMM-based classifier (bull / bear / sideways) with one model per asset
- **Technical Features**: 80+ indicators via `ta` library, reduced with PCA
- **Adaptive Strategies**: Automatically selects the optimal strategy based on detected regime
- **Robust Validation**: Combinatorial Purged Cross-Validation (CPCV) to avoid overfitting
- **Performance Metrics**: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor
- **Exchange Integration**: Unified access to crypto exchanges via CCXT
- **Backtesting**: Full backtesting engine powered by `backtesting.py`

## Architecture

OOP design with mixins — each responsibility is encapsulated and testeable independently:

```
Bot(SignalMixin, BacktestMixin, StrategyMixin, RegimeMixin, FeatureMixin, DataMixin)
```

**Pipeline:**

```
fetch_data → compute_features → detect_regime → select_strategy → backtest → generate_signals
```

### Strategies by Regime

| Regime | Strategy | Logic |
|---|---|---|
| Bull | `BullStrategy` | Trend following — SMA 20/50 crossovers |
| Bear | `BearStrategy` | Defensive — short mean reversion on RSI overbought + resistance |
| Sideways | `SidewaysStrategy` | Mean reversion — Bollinger Bands + RSI extremes |

## Installation

```python
!pip install git+https://github.com/your-username/atc-trading-bot.git
```

## Quick Start

```python
from atc_trading_bot import Bot

# Initialize the bot (defaults: exchange=binanceus, timeframe=1d)
bot = Bot(symbols=["BTC", "ETH", "SOL"])

# Run the full pipeline (defaults to BTC)
signals = bot.run_pipeline()
print(signals)
# {'regime': 'bull', 'strategy': 'BullStrategy', 'signal': 'buy'}
```

## Step-by-Step Usage

For more control over each step of the pipeline:

```python
from atc_trading_bot import Bot

bot = Bot(symbols=["BTC"])

# 1. Fetch OHLCV data
bot.fetch_data("BTC")

# Access the raw DataFrame at any time
print(bot.df)

# 2. Compute technical indicators + PCA
bot.compute_features(n_components=10)

# Inspect what PCA did
bot.features_summary()

# 3. Detect market regime
bot.detect_regime(n_regimes=3)
print(f"Current regime: {bot.current_regime}")
print(f"Regime metrics: {bot.regime_metrics}")

# Visualize regimes on price chart
bot.plot_regimes()

# 4. Select strategy based on regime
strategy = bot.select_strategy()
print(f"Active strategy: {strategy.__name__}")

# 5. Run backtest (out-of-sample, 70/30 train/test split)
results = bot.backtest(cash=100_000, commission=0.001)
print(results)
#          metric       value                          description
# 0  backtest_start  2023-08-11             Start of backtest period
# 1  backtest_end    2023-10-18             End of backtest period
# 2  sharpe_ratio    1.23                   Risk-adjusted return (annualized)
# 3  max_drawdown   -0.15                   Worst peak-to-trough decline
# ...

# 6. Generate trading signals
signals = bot.generate_signals()
print(f"Signal: {signals['signal']}")
```

## Cross-Validation (CPCV)

Validate your strategy with Combinatorial Purged Cross-Validation to avoid overfitting:

```python
cv_results = bot.cross_validate_cpcv(
    n_splits=5,
    purge_gap=5,
    embargo_pct=0.01,
    n_components=10,
    n_regimes=3,
)

# Distribution of Sharpe ratios across folds
for fold in cv_results:
    print(f"Fold {fold['fold']}: Sharpe={fold['sharpe_ratio']:.2f}, "
          f"MaxDD={fold['max_drawdown']:.2%}")
```

## Multi-Asset Analysis

Run the pipeline independently for each asset:

```python
bot = Bot(symbols=["BTC", "ETH", "SOL"])

for symbol in bot.symbols:
    signals = bot.run_pipeline(symbol)
    print(f"{symbol}: regime={signals['regime']}, signal={signals['signal']}")
```

## Data Caching

OHLCV data is cached locally as CSV to avoid repeated API calls:

```python
# First call fetches from exchange and caches
bot.fetch_data("BTC")

# Subsequent calls can use cache
bot.fetch_data("BTC", use_cache=True)
```

## Available Metrics

| Metric | Description |
|---|---|
| `backtest_start` | Start of backtest period |
| `backtest_end` | End of backtest period |
| `sharpe_ratio` | Risk-adjusted return (annualized) |
| `sortino_ratio` | Downside risk-adjusted return |
| `max_drawdown` | Worst peak-to-trough decline |
| `calmar_ratio` | Annual return / Max drawdown |
| `win_rate` | Percentage of winning trades |
| `profit_factor` | Gross profit / Gross loss |
| `total_return` | Total strategy return |
| `buy_and_hold_return` | Buy & hold benchmark return |
| `num_trades` | Number of trades executed |

## API Reference

### `Bot(exchange_id, symbols, timeframe, api_key, secret)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `exchange_id` | `str` | `"binanceus"` | CCXT exchange identifier |
| `symbols` | `list[str]` | `[]` | Trading pairs — short (`"BTC"`) or full (`"BTC/USDT"`) |
| `timeframe` | `str` | `"1d"` | Candle timeframe |
| `api_key` | `str` | `""` | Exchange API key |
| `secret` | `str` | `""` | Exchange API secret |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `df` | `DataFrame \| None` | OHLCV DataFrame after calling `fetch_data()` |
| `features` | `DataFrame \| None` | TA indicator features after `compute_features()` |
| `features_pca` | `ndarray \| None` | PCA-reduced features after `compute_features()` |
| `regimes` | `list[str] \| None` | Regime labels per candle after `detect_regime()` |
| `current_regime` | `str \| None` | Latest detected regime (bull/bear/sideways) |

### Methods

| Method | Description |
|---|---|
| `fetch_data(symbol, timeframe, since, use_cache)` | Fetch OHLCV data |
| `compute_features(n_components)` | Compute TA indicators + PCA |
| `detect_regime(n_regimes)` | Train HMM and classify regime |
| `select_strategy()` | Pick strategy for current regime |
| `backtest(strategy, cash, commission, test_ratio)` | Run out-of-sample backtest (returns DataFrame) |
| `cross_validate_cpcv(n_splits, purge_gap, embargo_pct)` | CPCV validation |
| `generate_signals()` | Generate buy/sell/hold signals |
| `run_pipeline(symbol="BTC")` | Execute full pipeline end-to-end |
| `features_summary(top_n)` | Print PCA reduction summary (features, variance, top weights) |
| `plot_regimes()` | Interactive Plotly chart of price colored by regime |

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=atc_trading_bot

# Specific module
pytest tests/test_regime_mixin.py -v
```

## Project Structure

```
atc-trading-bot/
├── pyproject.toml
├── src/
│   └── atc_trading_bot/
│       ├── bot.py                     # Bot class (all mixins composed)
│       ├── mixins/
│       │   ├── data_mixin.py          # CCXT fetch + CSV cache
│       │   ├── feature_mixin.py       # TA indicators + PCA
│       │   ├── regime_mixin.py        # HMM regime detection
│       │   ├── strategy_mixin.py      # Strategy selection
│       │   ├── backtest_mixin.py      # Backtesting + CPCV
│       │   └── signal_mixin.py        # Signal generation
│       └── strategies/
│           ├── bull_strategy.py       # SMA crossover (trend following)
│           ├── bear_strategy.py       # RSI + SMA (defensive)
│           └── sideways_strategy.py   # Bollinger + RSI (mean reversion)
└── tests/
    ├── conftest.py
    ├── test_data_mixin.py
    ├── test_feature_mixin.py
    ├── test_regime_mixin.py
    ├── test_strategy_mixin.py
    ├── test_backtest_mixin.py
    ├── test_signal_mixin.py
    └── test_bot.py
```

## Tech Stack

- **[CCXT](https://github.com/ccxt/ccxt)** — Unified crypto exchange API
- **[backtesting.py](https://github.com/kernc/backtesting.py)** — Backtesting framework
- **[hmmlearn](https://github.com/hmmlearn/hmmlearn)** — Hidden Markov Models
- **[ta](https://github.com/bukosabino/ta)** — Technical analysis indicators
- **[scikit-learn](https://scikit-learn.org)** — PCA, StandardScaler
- **[pytest](https://pytest.org)** — Testing framework

## License

MIT
