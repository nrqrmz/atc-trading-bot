# ATC Trading Bot

Algorithmic trading bot for crypto markets that detects market regimes using Hidden Markov Models (HMM) and applies differentiated strategies per regime. Built as an educational project — clean, tested, and designed for a course on algorithmic trading.

## Features

- **HMM Regime Detection** — Bull/bear/sideways classification with sticky transitions and regime smoothing
- **6 Trading Strategies** — Trend following, mean reversion, momentum, breakout, volatility, defensive
- **Strategy Registry** — Declarative metadata per strategy (best/worst regimes, descriptions)
- **80+ Technical Indicators** — Via `ta` library, reduced with PCA and rigorous feature curation
- **Risk Management** — Stop-loss, take-profit, and position sizing on every strategy
- **CPCV Cross-Validation** — Combinatorial Purged CV with embargo to avoid overfitting
- **Overfitting Detection** — Automatic in-sample vs out-of-sample comparison with warnings
- **Confidence Thresholding** — HMM posterior probabilities filter low-confidence signals
- **Interactive Visualization** — Equity curves, signal charts, CPCV folds, feature importance (Plotly)
- **Model Persistence** — Save/load trained models with joblib
- **Multi-Symbol Scanner** — Quick regime scan across all configured assets
- **Sentiment Integration** — Crypto Fear & Greed Index merged into features
- **Leverage Support** — Margin trading in backtests
- **Paper Trading** — Testnet order execution via CCXT
- **Method Chaining** — Fluent API: `bot.fetch_data("BTC").compute_features().detect_regime()`
- **Automatic Pagination** — Handles exchange API limits for large date ranges
- **95% Test Coverage** — 197 tests across 15 test modules

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from atc_trading_bot import Bot

bot = Bot(symbols=["BTC", "ETH", "SOL"])

# Run the full pipeline for BTC
signals = bot.run_pipeline("BTC")
print(signals)
# {'regime': 'bull', 'strategy': 'BullStrategy', 'signal': 'buy', 'confidence': 0.87}
```

Or using method chaining for more control:

```python
bot = Bot(symbols=["BTC"])

bot.fetch_data("BTC") \
   .compute_features() \
   .detect_regime() \
   .select_strategy()

results = bot.backtest()
print(results)

signals = bot.generate_signals()
print(f"Signal: {signals['signal']} (confidence: {signals['confidence']:.0%})")
```

## Pipeline Overview

```
fetch_data → compute_features → detect_regime → select_strategy → backtest → generate_signals
    |              |                  |                |               |              |
  CCXT +      80+ TA indicators   HMM with       6 strategies    Out-of-sample   Confidence
  pagination  + PCA reduction     sticky          + registry      + CPCV          thresholding
  + cache                         transitions     + risk mgmt     + overfit       + HMM posteriors
                                  + smoothing                     detection
```

## Detailed Guide

### 1. Data Fetching

The `DataMixin` handles OHLCV data via CCXT with automatic pagination and CSV caching.

```python
from atc_trading_bot import Bot

bot = Bot(symbols=["BTC", "ETH"])

# Fetch from exchange (paginated automatically for large date ranges)
bot.fetch_data("BTC", since="2022-01-01T00:00:00Z")

# Use cached data to avoid repeated API calls
bot.fetch_data("BTC", use_cache=True)

# Access the raw DataFrame
print(bot.df.tail())
#                              Open      High       Low     Close      Volume
# 2024-12-27 00:00:00+00:00  94500.0  95200.0  93800.0  94800.0  1200000.0
# ...
```

### 2. Feature Engineering

The `FeatureMixin` computes 80+ technical indicators, applies rigorous feature curation, and reduces dimensionality with PCA.

```python
bot.compute_features(n_components=10)

# See what PCA did
bot.features_summary()
# Total features: 42
# PCA components: 10 (87.3% variance explained)
#
#   PC1 (23.1%): momentum_rsi, momentum_stoch_rsi, trend_adx, ...
#   PC2 (15.4%): volatility_atr, volatility_bbw, ...

# Visualize feature importance
fig = bot.plot_feature_importance(top_n=15)
fig.show()
```

**Feature curation pipeline:**
1. Generate 80+ indicators via `ta` library
2. Exclude 26 problematic features (binary, price-level, accumulative)
3. Forward-fill NaNs, drop columns with >50% missing
4. Remove constant features (zero variance)
5. Remove highly correlated features (|r| > 0.95)
6. Standardize with `StandardScaler`
7. Reduce with PCA (capped to available features)

### 3. Regime Detection

The `RegimeMixin` trains a Gaussian HMM with sticky transitions to classify market regimes.

```python
bot.detect_regime(n_regimes=3)

print(f"Current regime: {bot.current_regime}")   # 'bull', 'bear', or 'sideways'
print(f"Metrics: {bot.regime_metrics}")
# {'log_likelihood': -1234.5, 'bic': 2567.8, 'avg_duration': 18.3}

# Interactive Plotly chart with colored regime spans
fig = bot.plot_regimes()
fig.show()
```

**Why HMM over GMM:** HMM captures regime persistence — the fact that markets tend to stay in a regime for extended periods. The sticky transition matrix (95% self-transition probability) encodes this as a prior, resulting in ~20-bar average regime duration.

**Regime smoothing:** Short-lived regimes (< 5 bars) are replaced with the previous regime to reduce noise and avoid whipsaw strategy switching.

### 4. Strategy Selection

The `StrategyMixin` uses a declarative registry to map regimes to strategies.

```python
bot.select_strategy()
print(f"Active: {bot.active_strategy.__name__}")

# Browse all registered strategies
bot.strategies_summary()
#          strategy                              description best_regimes    worst_regimes
# 0    BullStrategy    Trend following with SMA crossovers         bull         sideways
# 1    BearStrategy    Defensive short mean reversion...           bear             bull
# 2  SidewaysStrategy  Bollinger Bands + RSI mean reversion    sideways       bull, bear
# 3  MomentumStrategy  ROC + RSI momentum following              bull         sideways
# ...

# Filter by regime
bot.strategies_summary(regime="bull")
```

#### 6 Strategies

| Strategy | Regime | Logic | Entry | Exit |
|---|---|---|---|---|
| `BullStrategy` | Bull | Trend following | SMA 20 crosses above SMA 50 | SMA 50 crosses above SMA 20 |
| `BearStrategy` | Bear | Defensive short | RSI > 70 AND Close < SMA | RSI < 30 |
| `SidewaysStrategy` | Sideways | Mean reversion | Close <= BB lower AND RSI < 30 | Close >= BB upper AND RSI > 70 |
| `MomentumStrategy` | Bull | Momentum following | ROC > 1% AND 40 < RSI < 75 | ROC < 0 OR RSI > 80 |
| `BreakoutStrategy` | Bull, Bear | Channel breakout | Price > Donchian upper + volume | Price crosses midline |
| `VolatilityStrategy` | Sideways, Bear | Vol mean reversion | ATR < 0.8x MA (low vol) | ATR normalizes (0.9-1.2x MA) |

#### Risk Management

All strategies include stop-loss, take-profit, and position sizing:

```python
# Defaults from config.py — override per strategy
BullStrategy.stop_loss = 0.05       # 5% SL
BullStrategy.take_profit = 0.10     # 10% TP
BullStrategy.position_size = 0.95   # 95% of equity
```

### 5. Backtesting

The `BacktestMixin` provides out-of-sample backtesting with train/test split and CPCV cross-validation.

```python
# Out-of-sample backtest (70/30 train/test split)
results = bot.backtest(cash=100_000, commission=0.001)
print(results)

# With leverage
results = bot.backtest(leverage=2)  # 2x margin

# Equity curve visualization
fig = bot.plot_equity_curve()
fig.show()
```

#### Overfitting Detection

The backtest automatically compares in-sample (train) vs out-of-sample (test) performance:

```
Warning: Possible overfitting: sharpe_ratio degraded 75% from train (2.40) to test (0.60).
```

This fires when the test metric drops below 50% of the train metric (configurable via `OVERFIT_THRESHOLD` in `config.py`).

#### CPCV Cross-Validation

Combinatorial Purged Cross-Validation — the gold standard in financial ML (Lopez de Prado):

```python
cv_results = bot.cross_validate_cpcv(
    n_splits=5,
    purge_gap=5,       # bars removed near train/test boundary
    embargo_pct=0.01,  # additional gap after test set
)

for fold in cv_results:
    print(f"Fold {fold['fold']}: Sharpe={fold['sharpe_ratio']:.2f}, "
          f"MaxDD={fold['max_drawdown']:.2%}")

# Visualize fold comparison
fig = bot.plot_cpcv_results()
fig.show()
```

#### Available Metrics

| Metric | Description |
|---|---|
| `sharpe_ratio` | Risk-adjusted return (annualized) |
| `sortino_ratio` | Downside risk-adjusted return |
| `max_drawdown` | Worst peak-to-trough decline |
| `calmar_ratio` | Annual return / Max drawdown |
| `win_rate` | Percentage of winning trades |
| `profit_factor` | Gross profit / Gross loss |
| `total_return` | Total strategy return |
| `buy_and_hold_return` | Buy & hold benchmark return |
| `num_trades` | Number of trades executed |

### 6. Signal Generation

The `SignalMixin` generates paper trading signals with HMM confidence thresholding.

```python
signals = bot.generate_signals(confidence_threshold=0.6)
print(signals)
# {'regime': 'bull', 'strategy': 'BullStrategy', 'signal': 'buy', 'confidence': 0.87}

# Low confidence → forced to 'hold'
# {'regime': 'sideways', 'strategy': 'SidewaysStrategy', 'signal': 'hold', 'confidence': 0.42}

# Visualize signal on price chart
fig = bot.plot_signals()
fig.show()
```

**How confidence works:** The HMM provides posterior state probabilities via `predict_proba()`. If the posterior for the detected regime is below the threshold (default 0.6), the signal is overridden to `hold` regardless of what the strategy says.

### 7. Visualization

The `VisualizationMixin` provides 4 interactive chart types, all using Plotly with `plotly_dark` template:

```python
# Equity curve + drawdown (after backtest)
bot.plot_equity_curve()

# Buy/sell/hold signal marker on price (after generate_signals)
bot.plot_signals()

# CPCV fold comparison — Sharpe, return, drawdown, win rate (after cross_validate_cpcv)
bot.plot_cpcv_results()

# PCA feature importance — weighted loadings (after compute_features)
bot.plot_feature_importance(top_n=15)

# Regime spans on price chart (after detect_regime)
bot.plot_regimes()
```

### 8. Model Persistence

The `PersistenceMixin` saves and loads trained models with joblib.

```python
# Save after training
bot.save_model("models/btc_bull_v1.joblib")

# Load in a new session — skip straight to signals
bot2 = Bot(symbols=["BTC"])
metadata = bot2.load_model("models/btc_bull_v1.joblib")
print(metadata)  # {'saved_at': '...', 'exchange_id': 'binanceus', ...}

# Now bot2 has the HMM, PCA, scaler, regime — ready to select_strategy
bot2.fetch_data("BTC", use_cache=True)
bot2.select_strategy()
signals = bot2.generate_signals()
```

### 9. Multi-Symbol Scanner

The `ScannerMixin` provides a quick regime scan across all configured assets.

```python
bot = Bot(symbols=["BTC", "ETH", "SOL"])
scan = bot.scan_regimes()
print(scan)
#   symbol   regime  confidence  last_price  pct_change_24h
# 0  BTC/USDT   bull       0.87    94800.0           0.023
# 1  ETH/USDT   sideways   0.65     3400.0          -0.008
# 2  SOL/USDT   bear       0.72      185.0          -0.041
```

### 10. Sentiment Integration

The `SentimentMixin` fetches the Crypto Fear & Greed Index and merges it into features.

```python
# Fetch last 30 days of sentiment
bot.fetch_sentiment(days=30)
print(bot.sentiment_df.tail())
#         date  value classification
# 25  2024-12-23     72        Greed
# 26  2024-12-24     68        Greed
# ...

# Merge into OHLCV DataFrame (backward-looking, no look-ahead bias)
bot.merge_sentiment()
print(bot.df[["Close", "sentiment_value", "sentiment_classification"]].tail())
```

### 11. Paper Trading

The `TradingMixin` enables testnet order execution.

```python
# Connect to exchange testnet
bot.connect_testnet(api_key="your_key", secret="your_secret")

# Check balance
balance = bot.get_balance()
print(balance)

# Execute the current signal as a market order
bot.execute_signal(symbol="BTC/USDT", amount=0.001)

# Check open positions
positions = bot.get_open_positions()
```

## Configuration

All tunable parameters live in `src/atc_trading_bot/config.py`. Modify this single file to experiment:

| Parameter | Default | Description |
|---|---|---|
| `DEFAULT_N_COMPONENTS` | `10` | PCA components to retain |
| `CORRELATION_THRESHOLD` | `0.95` | Drop features with \|r\| above this |
| `DEFAULT_N_REGIMES` | `3` | HMM hidden states |
| `STICKY_TRANSITION_PROB` | `0.95` | HMM self-transition probability |
| `MIN_REGIME_DURATION` | `5` | Minimum bars for regime smoothing |
| `DEFAULT_CASH` | `100,000` | Starting capital for backtests |
| `DEFAULT_COMMISSION` | `0.001` | Trading commission (0.1%) |
| `DEFAULT_TEST_RATIO` | `0.3` | Train/test split ratio |
| `DEFAULT_LEVERAGE` | `1` | Leverage multiplier |
| `DEFAULT_STOP_LOSS` | `0.05` | 5% stop-loss |
| `DEFAULT_TAKE_PROFIT` | `0.10` | 10% take-profit |
| `DEFAULT_POSITION_SIZE` | `0.95` | 95% of equity per trade |
| `OVERFIT_THRESHOLD` | `0.5` | Warn if test < 50% of train |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.6` | Min regime confidence to trade |
| `SIGNAL_LOOKBACK` | `100` | Bars for signal generation |

## Architecture

OOP design with 11 mixins — each responsibility is encapsulated and testable independently:

```python
class Bot(
    TradingMixin,       # Paper trading execution
    ScannerMixin,       # Multi-symbol regime scan
    SentimentMixin,     # Fear & Greed Index
    VisualizationMixin, # Interactive Plotly charts
    PersistenceMixin,   # Save/load models
    SignalMixin,        # Signal generation + confidence
    BacktestMixin,      # Backtesting + CPCV + overfitting
    StrategyMixin,      # Strategy selection + registry
    RegimeMixin,        # HMM regime detection
    FeatureMixin,       # TA indicators + PCA
    DataMixin,          # CCXT fetch + CSV cache
):
```

### Project Structure

```
atc-trading-bot/
├── pyproject.toml
├── src/atc_trading_bot/
│   ├── bot.py                              # Bot class composing all mixins
│   ├── config.py                           # All tunable parameters
│   ├── pipeline_warning.py                 # Custom warning class
│   ├── mixins/
│   │   ├── data_mixin.py                   # CCXT fetch + pagination + CSV cache
│   │   ├── feature_mixin.py                # TA indicators + PCA
│   │   ├── regime_mixin.py                 # HMM regime detection
│   │   ├── strategy_mixin.py               # Strategy registry + selection
│   │   ├── backtest_mixin.py               # Backtesting + CPCV + overfitting
│   │   ├── signal_mixin.py                 # Signal generation + confidence
│   │   ├── visualization_mixin.py          # Plotly charts (4 types)
│   │   ├── persistence_mixin.py            # Save/load with joblib
│   │   ├── scanner_mixin.py                # Multi-symbol regime scan
│   │   ├── sentiment_mixin.py              # Fear & Greed Index
│   │   └── trading_mixin.py                # Paper trading execution
│   └── strategies/
│       ├── bull_strategy.py                # SMA crossover (trend following)
│       ├── bear_strategy.py                # RSI + SMA (defensive)
│       ├── sideways_strategy.py            # Bollinger + RSI (mean reversion)
│       ├── momentum_strategy.py            # ROC + RSI (momentum)
│       ├── breakout_strategy.py            # Donchian + volume (breakout)
│       └── volatility_strategy.py          # ATR mean reversion
└── tests/                                  # 197 tests, 95% coverage
    ├── conftest.py
    ├── test_bot.py
    ├── test_config.py
    ├── test_data_mixin.py
    ├── test_feature_mixin.py
    ├── test_regime_mixin.py
    ├── test_strategy_mixin.py
    ├── test_backtest_mixin.py
    ├── test_signal_mixin.py
    ├── test_new_strategies.py
    ├── test_visualization_mixin.py
    ├── test_persistence_mixin.py
    ├── test_scanner_mixin.py
    ├── test_sentiment_mixin.py
    ├── test_leverage.py
    └── test_trading_mixin.py
```

## API Reference

### Constructor

```python
Bot(exchange_id="binanceus", symbols=[], timeframe="1d", api_key="", secret="", data_dir=None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `exchange_id` | `str` | `"binanceus"` | CCXT exchange identifier (any exchange supported by CCXT) |
| `symbols` | `list[str]` | `[]` | Trading pairs — short (`"BTC"`) or full (`"BTC/USDT"`) |
| `timeframe` | `str` | `"1d"` | Candlestick timeframe (`"1m"`, `"5m"`, `"1h"`, `"1d"`, etc.) |
| `api_key` | `str` | `""` | Exchange API key (optional — public endpoints work without) |
| `secret` | `str` | `""` | Exchange API secret |
| `data_dir` | `str \| None` | `None` | Directory for CSV cache. Defaults to `data/` in project root |

### Pipeline Methods

These methods form the core pipeline. All return `self` for method chaining (except `backtest`, `cross_validate_cpcv`, and `generate_signals` which return results).

| Method | Returns | Description |
|---|---|---|
| `fetch_data(symbol, timeframe, since, use_cache)` | `self` | Fetch OHLCV data with automatic pagination. `symbol` accepts short (`"BTC"`) or full (`"BTC/USDT"`) format. `since` is ISO 8601 (`"2024-01-01T00:00:00Z"`). |
| `compute_features(n_components=10)` | `self` | Compute 80+ TA indicators, curate, standardize, and reduce with PCA |
| `detect_regime(n_regimes=3, n_iter=100)` | `self` | Train Gaussian HMM with sticky transitions and classify regimes |
| `select_strategy()` | `self` | Pick the default strategy for the current regime from the registry |
| `backtest(strategy, cash, commission, test_ratio, n_components, n_regimes, leverage)` | `DataFrame` | Out-of-sample backtest with train/test split and automatic overfitting detection |
| `cross_validate_cpcv(n_splits, purge_gap, embargo_pct, n_components, n_regimes, cash, commission)` | `list[dict]` | Combinatorial Purged Cross-Validation with embargo |
| `generate_signals(confidence_threshold=0.6)` | `dict` | Generate buy/sell/hold signals with HMM confidence filtering |
| `run_pipeline(symbol="BTC", n_components=10, n_regimes=3)` | `dict` | Execute full pipeline end-to-end (fetch → features → regime → strategy → backtest → signals) |

### Strategy Methods

| Method | Returns | Description |
|---|---|---|
| `strategies_summary(regime=None)` | `DataFrame` | Summary of all strategies with columns: `strategy`, `description`, `best_regimes`, `worst_regimes`. Pass `regime="bull"` to filter |
| `get_strategy_for_regime(regime)` | `type[Strategy]` | Get the default strategy class for a specific regime (`"bull"`, `"bear"`, `"sideways"`) |
| `get_strategies_for_regime(regime)` | `list[StrategyMeta]` | Get all strategies whose `best_regimes` includes the given regime (static method) |
| `list_strategies()` | `list[StrategyMeta]` | List all registered strategies with their metadata (static method) |

### Analysis Methods

| Method | Returns | Description |
|---|---|---|
| `scan_regimes(n_components=10, n_regimes=3)` | `DataFrame` | Quick regime scan across all configured `symbols`. Returns columns: `symbol`, `regime`, `confidence`, `last_price`, `pct_change_24h` |
| `fetch_sentiment(days=30)` | `DataFrame` | Fetch Crypto Fear & Greed Index. Returns columns: `date`, `value` (0-100), `classification` |
| `merge_sentiment()` | `DataFrame` | Merge sentiment into `self.df` via `merge_asof` (backward-looking, no look-ahead bias). Adds `sentiment_value` and `sentiment_classification` columns |
| `features_summary(top_n=5)` | `None` | Print PCA reduction summary — total features, variance explained, top contributing features per component |

### Visualization Methods

All charts use Plotly with the `plotly_dark` template.

| Method | Returns | Description |
|---|---|---|
| `plot_regimes()` | `Figure` | Price chart with background colored by detected regime (bull/bear/sideways) |
| `plot_equity_curve()` | `Figure` | Equity curve (top) + drawdown (bottom) from backtest results |
| `plot_signals()` | `Figure` | Price with buy/sell/hold signal marker on the last bar, showing confidence |
| `plot_cpcv_results()` | `Figure` | 2x2 grid comparing Sharpe, return, drawdown, and win rate across CPCV folds |
| `plot_feature_importance(top_n=15)` | `Figure` | Horizontal bar chart of PCA-weighted feature loadings |

### Persistence Methods

| Method | Returns | Description |
|---|---|---|
| `save_model(path)` | `str` | Save trained pipeline state (HMM, PCA, scaler, regimes, metadata) to disk via joblib |
| `load_model(path)` | `dict` | Load a saved model. Returns metadata dict (`saved_at`, `exchange_id`, `symbols`, `timeframe`). Restores `hmm_model`, `pca`, `scaler`, `current_regime`, `regimes`, `regime_metrics` |

### Trading Methods

Paper trading via exchange testnet. Uses a separate `testnet_exchange` connection to avoid interfering with the data-fetching `exchange`.

| Method | Returns | Description |
|---|---|---|
| `connect_testnet(api_key, secret, exchange_id="binanceus")` | `self` | Connect to an exchange's testnet (sandbox mode) for paper trading |
| `execute_signal(symbol="BTC/USDT", amount=0.001)` | `self` | Execute the current signal (`self.signals`) as a market order on the testnet |
| `get_balance()` | `dict \| None` | Fetch testnet account balance |
| `get_open_positions()` | `list \| None` | List open positions on the testnet |

### Instance Attributes

Set by the pipeline as each step executes. All start as `None` until their corresponding method is called.

#### Data (set by `DataMixin`)

| Attribute | Type | Description |
|---|---|---|
| `exchange_id` | `str` | CCXT exchange identifier (e.g. `"binanceus"`) |
| `symbols` | `list[str]` | Normalized trading pairs (e.g. `["BTC/USDT", "ETH/USDT"]`) |
| `timeframe` | `str` | Candlestick timeframe (e.g. `"1d"`) |
| `exchange` | `ccxt.Exchange` | CCXT exchange instance used for fetching market data |
| `data_dir` | `str` | Directory path for CSV cache files |
| `df` | `DataFrame \| None` | OHLCV DataFrame with columns `Open`, `High`, `Low`, `Close`, `Volume` and a `DatetimeIndex` |

#### Features (set by `FeatureMixin` via `compute_features()`)

| Attribute | Type | Description |
|---|---|---|
| `features` | `DataFrame \| None` | Curated TA indicator DataFrame after exclusion, cleaning, and correlation filtering. Columns are the surviving indicator names |
| `features_scaled` | `ndarray \| None` | StandardScaler-transformed features (mean ~0, std ~1). Shape: `(n_samples, n_features)` |
| `features_pca` | `ndarray \| None` | PCA-reduced features. Shape: `(n_samples, n_components)` |
| `features_index` | `Index \| None` | DatetimeIndex of valid rows after NaN removal — used to align features with `self.df` in downstream steps |
| `scaler` | `StandardScaler \| None` | Fitted scaler instance (for transforming new data with the same parameters) |
| `pca` | `PCA \| None` | Fitted PCA instance. Access `pca.explained_variance_ratio_` for variance per component, `pca.components_` for loadings |

#### Regimes (set by `RegimeMixin` via `detect_regime()`)

| Attribute | Type | Description |
|---|---|---|
| `regimes` | `list[str] \| None` | Regime label per observation after smoothing: `"bull"`, `"bear"`, or `"sideways"`. Length matches `features_pca` |
| `current_regime` | `str \| None` | Regime of the last observation (e.g. `"bull"`) |
| `hmm_model` | `GaussianHMM \| None` | Fitted HMM model. Use `hmm_model.predict_proba(X)` for posterior probabilities |
| `regime_metrics` | `dict \| None` | Classification quality: `{"log_likelihood": float, "bic": float, "avg_duration": float}` |

#### Strategy (set by `StrategyMixin` via `select_strategy()`)

| Attribute | Type | Description |
|---|---|---|
| `active_strategy` | `type[Strategy] \| None` | The selected `backtesting.py` Strategy class for the current regime |

#### Backtest (set by `BacktestMixin`)

| Attribute | Type | Description |
|---|---|---|
| `results` | `DataFrame \| None` | Backtest results with columns `metric`, `value`, `description`. Set by `backtest()` |
| `cv_results` | `list[dict] \| None` | List of metric dicts per CPCV fold. Each dict has `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `calmar_ratio`, `win_rate`, `profit_factor`, `total_return`, `buy_and_hold_return`, `num_trades`, `fold`. Set by `cross_validate_cpcv()` |

#### Signals (set by `SignalMixin` via `generate_signals()`)

| Attribute | Type | Description |
|---|---|---|
| `signals` | `dict \| None` | `{"regime": str, "strategy": str, "signal": "buy"\|"sell"\|"hold", "confidence": float}` |

#### Sentiment (set by `SentimentMixin`)

| Attribute | Type | Description |
|---|---|---|
| `sentiment_df` | `DataFrame \| None` | Fear & Greed data with columns `date`, `value` (0-100), `classification`. Set by `fetch_sentiment()` |

#### Trading (set by `TradingMixin`)

| Attribute | Type | Description |
|---|---|---|
| `testnet_exchange` | `ccxt.Exchange \| None` | Separate CCXT exchange instance in sandbox mode for paper trading. Set by `connect_testnet()` |
| `last_order` | `dict \| None` | The most recent order response from the testnet. Set by `execute_signal()` |

### Class Attributes

| Attribute | Type | Description |
|---|---|---|
| `REGIME_LABELS` | `list[str]` | `["bull", "bear", "sideways"]` — valid regime names |
| `REGIME_STRATEGY_MAP` | `dict[str, type[Strategy]]` | Regime → default strategy mapping, built automatically from the strategy registry |

## Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=atc_trading_bot

# Specific module
pytest tests/test_regime_mixin.py -v
```

## Tech Stack

- **[CCXT](https://github.com/ccxt/ccxt)** — Unified crypto exchange API
- **[backtesting.py](https://github.com/kernc/backtesting.py)** — Backtesting framework
- **[hmmlearn](https://github.com/hmmlearn/hmmlearn)** — Hidden Markov Models
- **[ta](https://github.com/bukosabino/ta)** — Technical analysis indicators (80+)
- **[scikit-learn](https://scikit-learn.org)** — PCA, StandardScaler
- **[joblib](https://joblib.readthedocs.io)** — Model serialization
- **[Plotly](https://plotly.com/python/)** — Interactive charts (`plotly_dark` template)
- **[pytest](https://pytest.org)** — Testing framework (197 tests, 95% coverage)

## License

MIT
