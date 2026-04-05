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
- **Triple-Barrier Labeling** — ATR-calibrated take-profit/stop-loss/timeout labels (Lopez de Prado)
- **3 Gradient Boosting Models** — LightGBM, CatBoost, XGBoost with TimeSeriesSplit
- **Ensemble Methods** — VotingClassifier + StackingClassifier (meta-learner: LogisticRegression)
- **Bayesian Optimization** — Optuna with TPE sampler and pruning (replaces GridSearchCV)
- **Walk-Forward Validation** — Sliding window retraining for production-grade evaluation
- **SHAP Explainability** — Per-prediction explanations and global feature importance
- **Permutation Importance** — Model-agnostic feature importance as sanity check
- **Method Chaining** — Fluent API: `bot.fetch_data("BTC").compute_features().detect_regime()`
- **Automatic Pagination** — Handles exchange API limits for large date ranges
- **95% Test Coverage** — 263 tests across 19 test modules

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

Or use the ML pipeline — same regime detection, but ML decides buy/sell/hold:

```python
result = bot.run_pipeline_ml("BTC")
print(result)
# {'regime': 'bull', 'model': 'StackingClassifier', 'signal': 'buy'}
```

Using method chaining for more control:

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

### Rule-Based Pipeline

```
fetch_data → compute_features → detect_regime → select_strategy → backtest → generate_signals
    |              |                  |                |               |              |
  CCXT +      80+ TA indicators   HMM with       6 strategies    Out-of-sample   Confidence
  pagination  + PCA reduction     sticky          + registry      + CPCV          thresholding
  + cache                         transitions     + risk mgmt     + overfit       + HMM posteriors
                                  + smoothing                     detection
```

### ML Pipeline

```
fetch_data → compute_features → detect_regime → compute_labels → train_models → predict
    |              |                  |                |               |             |
  (same)       (same)            (same)       Triple-barrier    LightGBM +      Ensemble
                                              ATR-calibrated    CatBoost +      prediction
                                              TP/SL/timeout     XGBoost +       (buy/sell/hold)
                                                                Voting/Stacking
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
```

| strategy | description | best_regimes | worst_regimes |
|---|---|---|---|
| BullStrategy | Trend following with SMA crossovers | bull | sideways |
| BearStrategy | Defensive short mean reversion on resistance | bear | bull |
| SidewaysStrategy | Bollinger Bands + RSI mean reversion | sideways | bull, bear |
| MomentumStrategy | ROC + RSI momentum following | bull | sideways |
| BreakoutStrategy | Donchian channel breakout with volume confirmation | bull, bear | sideways |
| VolatilityStrategy | ATR mean reversion for volatility cycles | sideways, bear | bull |

```python
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

All strategies include stop-loss, take-profit, and position sizing. Control them directly from `backtest()`:

```python
# Custom risk parameters per backtest
results = bot.backtest(
    stop_loss=0.03,       # 3% SL
    take_profit=0.08,     # 8% TP
    position_size=0.10,   # 10% of equity per trade
)

# Defaults from config.py: SL=5%, TP=10%, size=5%
results = bot.backtest()
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
print(cv_results)
```

| fold | sharpe_ratio | sortino_ratio | max_drawdown | win_rate | profit_factor | total_return | num_trades |
|---|---|---|---|---|---|---|---|
| Fold 0 | 1.20 | 1.50 | -0.05 | 0.60 | 1.80 | 0.10 | 12 |
| Fold 1 | 0.80 | 1.10 | -0.08 | 0.55 | 1.40 | 0.05 | 8 |
| Fold 2 | 1.50 | 1.90 | -0.03 | 0.65 | 2.10 | 0.12 | 15 |
| Fold 3 | 0.95 | 1.30 | -0.06 | 0.58 | 1.60 | 0.07 | 10 |
| Fold 4 | 1.10 | 1.40 | -0.04 | 0.62 | 1.75 | 0.09 | 11 |
| Mean | 1.11 | 1.44 | -0.05 | 0.60 | 1.73 | 0.09 | 11.2 |

```python
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
```

| symbol | regime | confidence | last_price | pct_change_24h |
|---|---|---|---|---|
| BTC/USDT | bull | 0.87 | 94800.0 | 0.023 |
| ETH/USDT | sideways | 0.65 | 3400.0 | -0.008 |
| SOL/USDT | bear | 0.72 | 185.0 | -0.041 |

### 10. Sentiment Integration

The `SentimentMixin` fetches the Crypto Fear & Greed Index and merges it into features.

```python
# Fetch last 30 days of sentiment
bot.fetch_sentiment(days=30)
print(bot.sentiment_df.tail())
```

| date | value | classification |
|---|---|---|
| 2024-12-23 | 72 | Greed |
| 2024-12-24 | 68 | Greed |
| 2024-12-25 | 73 | Greed |

```python

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

### 12. Triple-Barrier Labeling

The `LabelingMixin` replaces naive "price went up → buy" labels with a realistic triple-barrier method (Lopez de Prado). For each bar, three exit conditions race against each other:

- **Upper barrier** (take-profit): price rises by `tp_factor * ATR` → label = buy (+1)
- **Lower barrier** (stop-loss): price falls by `sl_factor * ATR` → label = sell (-1)
- **Vertical barrier** (timeout): `max_holding` bars pass without hitting either → label = hold (0)

```python
bot.compute_labels(tp_factor=2.0, sl_factor=2.0, max_holding=10)
print(bot.labels_summary())
```

| label | count | percentage |
|---|---|---|
| buy | 72 | 36.0 |
| sell | 58 | 29.0 |
| hold | 70 | 35.0 |

**Why this matters:** Fixed-horizon labels (e.g. "did price go up tomorrow?") ignore how you would actually trade. Triple-barrier labels mirror real trading with stops and targets, producing more balanced classes and more realistic model training.

### 13. ML Models

The `ModelMixin` trains three gradient boosting classifiers and two ensemble methods, then selects the best by out-of-sample F1 score.

```python
bot.train_models(n_estimators=200, learning_rate=0.05, max_depth=6)
print(bot.models_summary())
```

| model | accuracy | f1 | precision | recall | cv_f1_mean |
|---|---|---|---|---|---|
| LightGBM | 0.58 | 0.55 | 0.57 | 0.58 | 0.52 |
| CatBoost | 0.60 | 0.57 | 0.59 | 0.60 | 0.54 |
| XGBoost | 0.56 | 0.53 | 0.55 | 0.56 | 0.51 |
| Voting | 0.61 | 0.58 | 0.60 | 0.61 | NaN |
| Stacking | 0.62 | 0.59 | 0.61 | 0.62 | NaN |

```python
# Predict with the best model
predictions = bot.predict()
print(f"Last signal: {predictions[-1]}")  # -1 (sell), 0 (hold), or 1 (buy)
```

**Models used:**
- **LightGBM** — Fastest GBM, leaf-wise growth, excellent with PCA features
- **CatBoost** — Best defaults, ordered boosting provides inherent overfitting protection
- **XGBoost** — Strong baseline, adds diversity to the ensemble
- **VotingClassifier** — Soft voting (averaged probabilities) across all three
- **StackingClassifier** — Meta-learner (LogisticRegression) trained on base model predictions

**Overfitting detection:** Automatically compares cross-validation F1 vs out-of-sample F1. Warns if OOS drops below 70% of CV.

### 14. Hyperparameter Optimization

The `OptimizationMixin` uses Optuna (Bayesian optimization with TPE) to find optimal hyperparameters — 10-50x more efficient than GridSearchCV.

```python
best_params = bot.optimize_model(n_trials=100, model="lightgbm")
print(best_params)
```

| param | value |
|---|---|
| n_estimators | 342 |
| learning_rate | 0.028 |
| max_depth | 7 |
| min_child_samples | 23 |
| subsample | 0.82 |
| colsample_bytree | 0.74 |
| reg_alpha | 0.0015 |
| reg_lambda | 2.41 |

**Walk-Forward Validation** — sliding window retraining that simulates production deployment:

```python
wf_results = bot.walk_forward(window_size=252, step_size=21)
print(wf_results)
```

| window | accuracy | f1 | precision | recall |
|---|---|---|---|---|
| Window 0 | 0.57 | 0.54 | 0.56 | 0.57 |
| Window 1 | 0.61 | 0.58 | 0.60 | 0.61 |
| Window 2 | 0.55 | 0.52 | 0.54 | 0.55 |
| Mean | 0.58 | 0.55 | 0.57 | 0.58 |

**CPCV vs Walk-Forward:** CPCV is for research (model selection across many paths). Walk-Forward is for production (always testing on future data with sliding windows).

### 15. Model Explainability

The `ExplainabilityMixin` uses SHAP to explain why the model makes each prediction. Trains a separate LightGBM on raw features (not PCA) so feature names are human-readable.

```python
# Why did the model predict "buy" on the last bar?
fig = bot.explain_prediction(index=-1)
fig.show()

# Which features matter most globally?
fig = bot.feature_importance_shap(top_n=20)
fig.show()

# Sanity check with permutation importance
fig = bot.feature_importance_permutation(top_n=20)
fig.show()
```

**Why SHAP over basic feature importance:** `model.feature_importances_` only shows which features the tree splits on most often — it's biased toward high-cardinality features and doesn't detect data leakage. SHAP provides per-prediction explanations showing exactly how each indicator pushes the prediction toward buy or sell.

### 16. Rules vs ML Comparison

The bot supports both pipelines so students can compare directly:

```python
# Rule-based pipeline
bot.fetch_data("BTC", use_cache=True) \
   .compute_features() \
   .detect_regime() \
   .select_strategy()
rules_backtest = bot.backtest()

# ML pipeline (same data, same regime)
bot.compute_labels() \
   .train_models()
ml_predictions = bot.predict()

print(f"Rule-based strategy: {bot.active_strategy.__name__}")
print(f"ML model: {type(bot.active_model).__name__}")
print(bot.models_summary())
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
| `DEFAULT_POSITION_SIZE` | `0.05` | 5% of equity per trade |
| `OVERFIT_THRESHOLD` | `0.5` | Warn if test < 50% of train |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.6` | Min regime confidence to trade |
| `SIGNAL_LOOKBACK` | `100` | Bars for signal generation |
| `DEFAULT_TP_FACTOR` | `2.0` | Triple-barrier take-profit (ATR multiplier) |
| `DEFAULT_SL_FACTOR` | `2.0` | Triple-barrier stop-loss (ATR multiplier) |
| `DEFAULT_MAX_HOLDING` | `10` | Triple-barrier timeout (bars) |
| `DEFAULT_N_ESTIMATORS` | `200` | Trees per GBM model |
| `DEFAULT_LEARNING_RATE` | `0.05` | GBM learning rate |
| `DEFAULT_MAX_DEPTH` | `6` | GBM max tree depth |
| `DEFAULT_ML_TEST_SIZE` | `0.2` | ML train/test split ratio |
| `DEFAULT_ML_CV_SPLITS` | `5` | TimeSeriesSplit folds |
| `ML_OVERFIT_THRESHOLD` | `0.7` | Warn if OOS F1 < 70% of CV F1 |
| `DEFAULT_OPTUNA_TRIALS` | `100` | Bayesian optimization trials |
| `DEFAULT_WF_WINDOW_SIZE` | `252` | Walk-forward training window (bars) |
| `DEFAULT_WF_STEP_SIZE` | `21` | Walk-forward step size (bars) |

## Architecture

OOP design with 15 mixins — each responsibility is encapsulated and testable independently:

```python
class Bot(
    # ML Layer
    ExplainabilityMixin,  # SHAP + permutation importance
    OptimizationMixin,    # Optuna + walk-forward
    ModelMixin,           # LightGBM, CatBoost, XGBoost, ensembles
    LabelingMixin,        # Triple-barrier labeling
    # Trading Layer
    TradingMixin,         # Paper trading execution
    ScannerMixin,         # Multi-symbol regime scan
    SentimentMixin,       # Fear & Greed Index
    VisualizationMixin,   # Interactive Plotly charts
    PersistenceMixin,     # Save/load models
    # Core Pipeline
    SignalMixin,          # Signal generation + confidence
    BacktestMixin,        # Backtesting + CPCV + overfitting
    StrategyMixin,        # Strategy selection + registry
    RegimeMixin,          # HMM regime detection
    FeatureMixin,         # TA indicators + PCA
    DataMixin,            # CCXT fetch + CSV cache
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
│   │   ├── labeling_mixin.py               # Triple-barrier labeling
│   │   ├── model_mixin.py                  # LightGBM + CatBoost + XGBoost + ensembles
│   │   ├── optimization_mixin.py           # Optuna + walk-forward
│   │   ├── explainability_mixin.py         # SHAP + permutation importance
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
└── tests/                                  # 263 tests, 95% coverage
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
    ├── test_labeling_mixin.py
    ├── test_model_mixin.py
    ├── test_optimization_mixin.py
    ├── test_explainability_mixin.py
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
| `backtest(strategy, cash, commission, test_ratio, n_components, n_regimes, leverage, stop_loss, take_profit, position_size)` | `DataFrame` | Out-of-sample backtest with train/test split, risk management, and overfitting detection |
| `cross_validate_cpcv(n_splits, purge_gap, embargo_pct, n_components, n_regimes, cash, commission)` | `DataFrame` | CPCV with per-fold metrics and Mean summary row. Index: `Fold 0`, `Fold 1`, ..., `Mean` |
| `generate_signals(confidence_threshold=0.6)` | `dict` | Generate buy/sell/hold signals with HMM confidence filtering |
| `run_pipeline(symbol="BTC", n_components=10, n_regimes=3)` | `dict` | Execute rule-based pipeline end-to-end (fetch → features → regime → strategy → backtest → signals) |
| `run_pipeline_ml(symbol="BTC", n_components=10, n_regimes=3)` | `dict` | Execute ML pipeline end-to-end (fetch → features → regime → labels → train → predict) |

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

### ML Methods

| Method | Returns | Description |
|---|---|---|
| `compute_labels(tp_factor=2.0, sl_factor=2.0, max_holding=10)` | `self` | Triple-barrier labeling using ATR-calibrated barriers |
| `labels_summary()` | `DataFrame` | Label distribution (buy/sell/hold counts and percentages) |
| `train_models(test_size, cv_splits, n_estimators, learning_rate, max_depth)` | `self` | Train LightGBM + CatBoost + XGBoost + Voting + Stacking. Selects best by OOS F1 |
| `models_summary()` | `DataFrame` | Comparison of all models: accuracy, F1, precision, recall, CV F1. Indexed by model name |
| `predict(X=None)` | `ndarray` | Predict labels (-1/0/1) using the best model. Defaults to `self.features_pca` |
| `optimize_model(n_trials=100, model="lightgbm")` | `DataFrame` | Optuna Bayesian optimization. Returns best params indexed by param name |
| `walk_forward(window_size=252, step_size=21)` | `DataFrame` | Sliding window walk-forward validation with per-window metrics + Mean row |

### Explainability Methods

| Method | Returns | Description |
|---|---|---|
| `explain_prediction(index=-1)` | `Figure` | SHAP waterfall chart for a specific prediction (green=buy, red=sell) |
| `feature_importance_shap(top_n=20)` | `Figure` | Global feature importance via mean \|SHAP\| values |
| `feature_importance_permutation(top_n=20)` | `Figure` | Permutation importance as model-agnostic sanity check |

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
| `cv_results` | `DataFrame \| None` | CPCV results with one row per fold plus a Mean summary row. Columns: `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `win_rate`, `profit_factor`, `total_return`, `num_trades`. Index: fold labels. Set by `cross_validate_cpcv()` |

#### Signals (set by `SignalMixin` via `generate_signals()`)

| Attribute | Type | Description |
|---|---|---|
| `signals` | `dict \| None` | `{"regime": str, "strategy": str, "signal": "buy"\|"sell"\|"hold", "confidence": float}` |

#### Labels (set by `LabelingMixin` via `compute_labels()`)

| Attribute | Type | Description |
|---|---|---|
| `labels` | `Series \| None` | Triple-barrier labels per bar: `1` (buy), `-1` (sell), `0` (hold). Also added as `label` column in `self.df` |

#### ML Models (set by `ModelMixin` via `train_models()`)

| Attribute | Type | Description |
|---|---|---|
| `trained_models` | `dict \| None` | Dict of trained models: `{"LightGBM": ..., "CatBoost": ..., "XGBoost": ..., "Voting": ..., "Stacking": ...}` |
| `active_model` | `estimator \| None` | The best model selected by out-of-sample F1 score |

#### Optimization (set by `OptimizationMixin`)

| Attribute | Type | Description |
|---|---|---|
| `optuna_study` | `Study \| None` | Optuna study object with full trial history. Set by `optimize_model()` |

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
- **[scikit-learn](https://scikit-learn.org)** — PCA, StandardScaler, ensembles, TimeSeriesSplit
- **[LightGBM](https://lightgbm.readthedocs.io)** — Fast gradient boosting (primary ML model)
- **[CatBoost](https://catboost.ai)** — Robust gradient boosting with ordered boosting
- **[XGBoost](https://xgboost.readthedocs.io)** — Gradient boosting (ensemble diversity)
- **[Optuna](https://optuna.org)** — Bayesian hyperparameter optimization (TPE + pruning)
- **[SHAP](https://shap.readthedocs.io)** — Model explainability (per-prediction and global)
- **[joblib](https://joblib.readthedocs.io)** — Model serialization
- **[Plotly](https://plotly.com/python/)** — Interactive charts (`plotly_dark` template)
- **[pytest](https://pytest.org)** — Testing framework (263 tests, 95% coverage)

## License

MIT
