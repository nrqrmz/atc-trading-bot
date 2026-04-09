import itertools
import warnings

import numpy as np
from atc_trading_bot.config import (
    CORRELATION_THRESHOLD,
    DEFAULT_ATR_PERIOD,
    DEFAULT_CASH,
    DEFAULT_COMMISSION,
    DEFAULT_CPCV_SPLITS,
    DEFAULT_EMBARGO_PCT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LEVERAGE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_HOLDING,
    DEFAULT_N_COMPONENTS,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_N_REGIMES,
    DEFAULT_POSITION_SIZE,
    DEFAULT_PURGE_GAP,
    DEFAULT_SL_FACTOR,
    DEFAULT_STOP_LOSS,
    DEFAULT_TAKE_PROFIT,
    DEFAULT_TEST_RATIO,
    DEFAULT_TP_FACTOR,
    EXCLUDE_COLS,
    METRIC_DESCRIPTIONS,
    MIN_TEST_BARS,
    MIN_TRAIN_BARS,
    NAN_COLUMN_THRESHOLD,
    OVERFIT_THRESHOLD,
)
from atc_trading_bot.pipeline_warning import PipelineWarning
import pandas as pd
from backtesting import Strategy
from backtesting.lib import FractionalBacktest as Backtest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import ta as ta_lib


class BacktestMixin:
    """Mixin for backtesting strategies and CPCV cross-validation."""

    def _require_data(self) -> bool:
        """Check that OHLCV data has been loaded."""
        if getattr(self, "df", None) is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
            return False
        return True

    def _require_strategy(self) -> bool:
        """Check that a strategy has been selected."""
        if getattr(self, "active_strategy", None) is None:
            warnings.warn("No strategy selected. Call select_strategy first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results: pd.DataFrame | None = None
        self.ml_results: pd.DataFrame | None = None
        self.cv_results: list[dict] | None = None

    @staticmethod
    def _apply_risk_params(strategy_cls: type[Strategy], stop_loss: float,
                           take_profit: float, position_size: float) -> type[Strategy]:
        """Create a strategy subclass with the given risk parameters.

        Uses dynamic subclassing so the original class is never mutated.
        This is the same pattern backtesting.py uses for bt.optimize().
        """
        return type(strategy_cls.__name__, (strategy_cls,), {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size,
        })

    def backtest(self, strategy: type[Strategy] | None = None,
                 cash: float = DEFAULT_CASH, commission: float = DEFAULT_COMMISSION,
                 test_ratio: float = DEFAULT_TEST_RATIO, n_components: int | None = None,
                 n_regimes: int | None = None,
                 leverage: int = DEFAULT_LEVERAGE,
                 stop_loss: float = DEFAULT_STOP_LOSS,
                 take_profit: float = DEFAULT_TAKE_PROFIT,
                 position_size: float = DEFAULT_POSITION_SIZE) -> pd.DataFrame:
        """Run an out-of-sample backtest with train/test split.

        Splits data into train/test using test_ratio. When no explicit strategy
        is given, refits the full pipeline (features + PCA + HMM) on training
        data only to avoid look-ahead bias.

        Args:
            strategy: Strategy class to backtest. If None, refits pipeline on
                training data and selects strategy for the test period.
            cash: Starting capital. Default: 100,000.
            commission: Trading commission as a fraction. Default: 0.001 (0.1%).
            test_ratio: Fraction of data to use as test set. Default: 0.3.
            n_components: PCA components for refit. Defaults to fitted value.
            n_regimes: HMM regimes for refit. Defaults to fitted value.
            leverage: Leverage multiplier. When > 1, sets margin=1/leverage
                on the Backtest constructor (e.g. leverage=2 → margin=0.5).
                Default: 1 (no leverage).
            stop_loss: Stop-loss as a fraction (e.g. 0.05 = 5%). Default: 0.05.
            take_profit: Take-profit as a fraction (e.g. 0.10 = 10%). Default: 0.10.
            position_size: Fraction of equity per trade (e.g. 0.05 = 5%). Default: 0.05.

        Returns:
            DataFrame with columns (metric, value, description) including
            backtest date range, or None if prerequisites are missing.
        """
        if not self._require_data():
            return

        if strategy is None and not self._require_strategy():
            return

        if n_components is None:
            n_components = getattr(self.pca, "n_components_", DEFAULT_N_COMPONENTS) if hasattr(self, "pca") and self.pca is not None else DEFAULT_N_COMPONENTS
        if n_regimes is None:
            n_regimes = getattr(self.hmm_model, "n_components", DEFAULT_N_REGIMES) if hasattr(self, "hmm_model") and self.hmm_model is not None else DEFAULT_N_REGIMES

        train_df, test_df = train_test_split(self.df, test_size=test_ratio, shuffle=False)

        if strategy is not None:
            strat = strategy
        else:
            strat = self._fit_and_select_strategy(
                train_df, n_components=n_components, n_regimes=n_regimes,
                test_df=test_df,
            )

        # Inject risk parameters into the strategy via dynamic subclass
        strat = self._apply_risk_params(strat, stop_loss, take_profit, position_size)

        # Build extra kwargs for leverage (margin trading)
        bt_kwargs = {}
        if leverage > 1:
            bt_kwargs["margin"] = 1 / leverage

        # Run backtest on test data (out-of-sample)
        bt = Backtest(test_df, strat, cash=cash, commission=commission, finalize_trades=True, **bt_kwargs)
        stats = bt.run()
        metrics_dict = self._extract_metrics(stats, test_df=test_df)

        # Run backtest on train data (in-sample) for overfitting comparison
        try:
            bt_train = Backtest(train_df, strat, cash=cash, commission=commission, finalize_trades=True, **bt_kwargs)
            train_stats = bt_train.run()
            train_metrics = self._extract_metrics(train_stats, test_df=train_df)
            self._check_overfit(train_metrics, metrics_dict)
        except Exception:
            pass

        self.results = self._metrics_to_dataframe(
            metrics_dict,
            start_date=test_df.index[0],
            end_date=test_df.index[-1],
        )
        return self.results

    def backtest_ml(self, cash: float = DEFAULT_CASH,
                    commission: float = DEFAULT_COMMISSION,
                    test_ratio: float = DEFAULT_TEST_RATIO,
                    n_components: int | None = None,
                    model: str = "lightgbm",
                    leverage: int = DEFAULT_LEVERAGE,
                    stop_loss: float = DEFAULT_STOP_LOSS,
                    take_profit: float = DEFAULT_TAKE_PROFIT,
                    position_size: float = DEFAULT_POSITION_SIZE) -> pd.DataFrame:
        """Run an out-of-sample backtest using ML predictions.

        Splits data into train/test. On the training set, refits the entire
        ML pipeline (features → PCA → triple-barrier labels → model) to avoid
        look-ahead bias. Predicts on the test set and backtests through
        :class:`~atc_trading_bot.strategies.ml_strategy.MLStrategy`.

        Args:
            cash: Starting capital. Default: 100,000.
            commission: Commission fraction. Default: 0.001.
            test_ratio: Fraction of data for test set. Default: 0.3.
            n_components: PCA components. Defaults to fitted value.
            model: ML model type (``"lightgbm"``, ``"catboost"``, ``"xgboost"``).
                Default: ``"lightgbm"``.
            leverage: Leverage multiplier. Default: 1.
            stop_loss: Stop-loss fraction. Default: 0.05.
            take_profit: Take-profit fraction. Default: 0.10.
            position_size: Equity fraction per trade. Default: 0.05.

        Returns:
            DataFrame with columns (metric, value, description), same format
            as :meth:`backtest`. Returns None if prerequisites are missing.
        """
        if not self._require_data():
            return

        if n_components is None:
            n_components = (
                getattr(self.pca, "n_components_", DEFAULT_N_COMPONENTS)
                if hasattr(self, "pca") and self.pca is not None
                else DEFAULT_N_COMPONENTS
            )

        train_df, test_df = train_test_split(self.df, test_size=test_ratio, shuffle=False)

        predictions = self._fit_and_predict_ml(train_df, test_df, n_components, model)
        if predictions is None:
            return

        from atc_trading_bot.strategies.ml_strategy import MLStrategy

        # Inject predictions via dynamic subclass (same pattern as _apply_risk_params)
        strat = type("MLStrategy", (MLStrategy,), {"predictions": predictions})
        strat = self._apply_risk_params(strat, stop_loss, take_profit, position_size)

        bt_kwargs = {}
        if leverage > 1:
            bt_kwargs["margin"] = 1 / leverage

        # Out-of-sample backtest
        bt = Backtest(test_df, strat, cash=cash, commission=commission,
                      finalize_trades=True, **bt_kwargs)
        stats = bt.run()
        metrics_dict = self._extract_metrics(stats, test_df=test_df)

        # In-sample backtest for overfitting comparison
        try:
            train_preds = self._fit_and_predict_ml(
                train_df, train_df, n_components, model,
            )
            if train_preds is not None:
                train_strat = type("MLStrategy", (MLStrategy,), {"predictions": train_preds})
                train_strat = self._apply_risk_params(train_strat, stop_loss, take_profit, position_size)
                bt_train = Backtest(train_df, train_strat, cash=cash, commission=commission,
                                    finalize_trades=True, **bt_kwargs)
                train_stats = bt_train.run()
                train_metrics = self._extract_metrics(train_stats, test_df=train_df)
                self._check_overfit(train_metrics, metrics_dict)
        except Exception:
            pass

        self.ml_results = self._metrics_to_dataframe(
            metrics_dict,
            start_date=test_df.index[0],
            end_date=test_df.index[-1],
        )
        return self.ml_results

    def _fit_and_predict_ml(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                            n_components: int, model: str = "lightgbm") -> np.ndarray | None:
        """Fit features + PCA + labels + ML model on train, predict on test.

        This is the ML equivalent of :meth:`_fit_and_select_strategy`.
        Everything is re-derived from raw OHLCV on the training set to
        prevent look-ahead bias.

        Args:
            train_df: Training OHLCV DataFrame.
            test_df: Test OHLCV DataFrame.
            n_components: Number of PCA components.
            model: Model type for ``_create_model()``. Default: ``"lightgbm"``.

        Returns:
            Array of predictions aligned with ``len(test_df)``, values in
            {-1, 0, 1}. Returns None if training fails.
        """
        # --- Compute TA features on training data ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="ta")
            df_ta = ta_lib.add_all_ta_features(
                train_df.copy(), open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=False,
            )
        feature_cols = [c for c in df_ta.columns if c not in EXCLUDE_COLS]
        train_features = df_ta[feature_cols].copy()

        # Clean: forward fill, replace inf, drop NaN
        train_features.ffill(inplace=True)
        train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        thresh = int(len(train_features) * NAN_COLUMN_THRESHOLD)
        train_features.dropna(axis=1, thresh=thresh, inplace=True)
        train_features.dropna(inplace=True)

        valid_index = train_features.index

        # Remove constant and highly correlated features
        non_const = train_features.columns[train_features.std() > 0]
        train_features = train_features[non_const]

        corr = train_features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        corr_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
        train_features = train_features.drop(columns=corr_drop)
        kept_cols = train_features.columns

        # --- Fit scaler + PCA ---
        scaler = StandardScaler()
        scaled = scaler.fit_transform(train_features)

        actual_components = min(n_components, scaled.shape[1])
        pca = PCA(n_components=actual_components)
        train_pca = pca.fit_transform(scaled)

        # --- Triple-barrier labeling on training data ---
        close = train_df["Close"].values
        high = train_df["High"].values
        low = train_df["Low"].values

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        tr = np.concatenate([[high[0] - low[0]], tr])
        atr = pd.Series(tr).rolling(DEFAULT_ATR_PERIOD).mean().values

        n = len(close)
        labels = np.zeros(n, dtype=int)
        for i in range(n):
            if np.isnan(atr[i]) or atr[i] == 0:
                continue
            upper_barrier = close[i] + DEFAULT_TP_FACTOR * atr[i]
            lower_barrier = close[i] - DEFAULT_SL_FACTOR * atr[i]
            end = min(i + DEFAULT_MAX_HOLDING, n)
            for j in range(i + 1, end):
                if close[j] >= upper_barrier:
                    labels[i] = 1
                    break
                if close[j] <= lower_barrier:
                    labels[i] = -1
                    break

        train_labels = pd.Series(labels, index=train_df.index, name="label")

        # --- Align features and labels ---
        common_idx = valid_index.intersection(train_labels.index)
        aligned_labels = train_labels.loc[common_idx].values
        # train_pca rows correspond to valid_index
        idx_positions = [valid_index.get_loc(idx) for idx in common_idx]
        aligned_pca = train_pca[idx_positions]

        # Map labels to sequential ints starting from 0 for XGBoost compatibility.
        # Dynamic mapping handles cases where not all 3 classes are present.
        unique_labels = sorted(np.unique(aligned_labels))
        label_map = {v: i for i, v in enumerate(unique_labels)}
        label_unmap = {i: v for v, i in label_map.items()}
        y = np.array([label_map[v] for v in aligned_labels])

        if len(np.unique(y)) < 2:
            warnings.warn(
                "Training labels have fewer than 2 classes. Returning hold predictions.",
                PipelineWarning,
            )
            return np.zeros(len(test_df), dtype=int)

        # --- Train ML model ---
        params = {
            "n_estimators": DEFAULT_N_ESTIMATORS,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "max_depth": DEFAULT_MAX_DEPTH,
        }
        clf = self._create_model(model, params)
        clf.fit(aligned_pca, y)

        # --- Transform test data through the same pipeline ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="ta")
            df_ta_test = ta_lib.add_all_ta_features(
                test_df.copy(), open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=False,
            )
        test_features = df_ta_test[kept_cols].copy()
        test_features.ffill(inplace=True)
        test_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_features.dropna(inplace=True)

        test_scaled = scaler.transform(test_features)
        test_pca = pca.transform(test_scaled)

        # --- Predict and align ---
        raw_preds = clf.predict(test_pca)
        unmapped = np.array([label_unmap.get(int(v), 0) for v in raw_preds])

        # Pad front with 0 (hold) if TA warmup reduced test rows
        n_test = len(test_df)
        if len(unmapped) < n_test:
            pad = np.zeros(n_test - len(unmapped), dtype=int)
            unmapped = np.concatenate([pad, unmapped])

        return unmapped

    def cross_validate_cpcv(self, n_splits: int = DEFAULT_CPCV_SPLITS,
                            purge_gap: int = DEFAULT_PURGE_GAP,
                            embargo_pct: float = DEFAULT_EMBARGO_PCT,
                            n_components: int = DEFAULT_N_COMPONENTS,
                            n_regimes: int = DEFAULT_N_REGIMES,
                            cash: float = DEFAULT_CASH,
                            commission: float = DEFAULT_COMMISSION) -> pd.DataFrame:
        """Combinatorial Purged Cross-Validation.

        Generates all C(n_splits, 2) train/test combinations where each
        combination uses 1 group as test and the rest as training.
        Purging removes observations near the train/test boundary.
        Embargo adds an additional gap after each test set.

        Returns:
            DataFrame with one row per fold plus a Mean summary row.
            Index is fold label (Fold 0, Fold 1, ..., Mean).
        """
        if not self._require_data():
            return

        n = len(self.df)
        group_size = n // n_splits
        embargo_size = max(1, int(n * embargo_pct))

        # Create group indices
        groups = []
        for i in range(n_splits):
            start = i * group_size
            end = start + group_size if i < n_splits - 1 else n
            groups.append((start, end))

        # Generate all combinations: each group takes a turn as test
        self.cv_results = []
        for test_idx in range(n_splits):
            test_start, test_end = groups[test_idx]

            # Build training indices with purging and embargo
            train_indices = []
            for train_idx in range(n_splits):
                if train_idx == test_idx:
                    continue
                t_start, t_end = groups[train_idx]

                # Purge: remove observations within purge_gap of test boundaries
                purged_start = max(t_start, 0)
                purged_end = min(t_end, n)

                # If training group is adjacent to test, apply purge
                if t_end > test_start - purge_gap and t_end <= test_start:
                    purged_end = max(t_start, test_start - purge_gap)
                if t_start < test_end + embargo_size and t_start >= test_end:
                    purged_start = min(t_end, test_end + embargo_size)

                if purged_start < purged_end:
                    train_indices.extend(range(purged_start, purged_end))

            if len(train_indices) < MIN_TRAIN_BARS:
                continue

            train_df = self.df.iloc[train_indices]
            test_df = self.df.iloc[test_start:test_end]

            if len(test_df) < MIN_TEST_BARS:
                continue

            # Refit pipeline on training data only
            try:
                strategy = self._fit_and_select_strategy(
                    train_df, n_components=n_components, n_regimes=n_regimes,
                    test_df=test_df,
                )
            except Exception:
                continue

            # Backtest on test data
            try:
                bt = Backtest(test_df, strategy, cash=cash, commission=commission, finalize_trades=True)
                stats = bt.run()
                metrics = self._extract_metrics(stats, test_df=test_df)
                metrics["fold"] = test_idx
                self.cv_results.append(metrics)
            except Exception:
                continue

        # Keep raw list for visualization mixin
        self._cv_results_raw = list(self.cv_results)

        # Build DataFrame with fold as index
        cv_cols = ["sharpe_ratio", "sortino_ratio", "max_drawdown",
                   "win_rate", "profit_factor", "total_return", "num_trades"]
        rows = []
        for r in self.cv_results:
            rows.append({col: r.get(col, 0) for col in cv_cols})

        df = pd.DataFrame(rows, columns=cv_cols)
        df.index = [f"Fold {r.get('fold', i)}" for i, r in enumerate(self.cv_results)]
        df.index.name = "fold"

        # Append mean summary row
        mean_row = df.mean()
        mean_row.name = "Mean"
        df = pd.concat([df, mean_row.to_frame().T])
        df.index.name = "fold"

        self.cv_results = df
        return self.cv_results

    @staticmethod
    def _check_overfit(train_metrics: dict, test_metrics: dict,
                       threshold: float = OVERFIT_THRESHOLD) -> list[str]:
        """Compare in-sample vs out-of-sample metrics and warn on overfitting.

        Returns list of warning messages (empty if no overfitting detected).
        """
        overfit_warnings = []
        metrics_to_check = ["sharpe_ratio", "total_return"]

        for metric in metrics_to_check:
            train_val = train_metrics.get(metric, 0)
            test_val = test_metrics.get(metric, 0)

            # Only compare when train metric is meaningfully positive
            if train_val > 0.01:
                ratio = test_val / train_val
                if ratio < threshold:
                    pct = (1 - ratio) * 100
                    msg = (
                        f"Possible overfitting: {metric} degraded {pct:.0f}% "
                        f"from train ({train_val:.3f}) to test ({test_val:.3f})."
                    )
                    overfit_warnings.append(msg)
                    warnings.warn(msg, PipelineWarning)

        return overfit_warnings

    def _fit_and_select_strategy(self, train_df: pd.DataFrame, n_components: int,
                                 n_regimes: int, test_df: pd.DataFrame) -> type[Strategy]:
        """Fit PCA + HMM on training data and select strategy for test data's regime."""
        # Config values already imported at module level

        # Compute features on training data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="ta")
            df_ta = ta_lib.add_all_ta_features(
                train_df.copy(), open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=False,
            )
        feature_cols = [c for c in df_ta.columns if c not in EXCLUDE_COLS]
        train_features = df_ta[feature_cols].copy()

        # Clean: forward fill, replace inf, drop NaN
        train_features.ffill(inplace=True)
        train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        thresh = int(len(train_features) * NAN_COLUMN_THRESHOLD)
        train_features.dropna(axis=1, thresh=thresh, inplace=True)
        train_features.dropna(inplace=True)

        valid_index = train_features.index

        # Remove constant and highly correlated features
        non_const = train_features.columns[train_features.std() > 0]
        train_features = train_features[non_const]

        corr = train_features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        corr_drop = [col for col in upper.columns if any(upper[col] > CORRELATION_THRESHOLD)]
        train_features = train_features.drop(columns=corr_drop)
        kept_cols = train_features.columns

        scaler = StandardScaler()
        scaled = scaler.fit_transform(train_features)

        actual_components = min(n_components, scaled.shape[1])
        pca = PCA(n_components=actual_components)
        pca_features = pca.fit_transform(scaled)

        # Fit HMM
        hmm = GaussianHMM(n_components=n_regimes, covariance_type="full",
                          n_iter=100, random_state=42)
        hmm.fit(pca_features)

        # Transform test data using the same pipeline
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="ta")
            df_ta_test = ta_lib.add_all_ta_features(
                test_df.copy(), open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=False,
            )
        test_features = df_ta_test[kept_cols].copy()
        test_features.ffill(inplace=True)
        test_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        test_features.dropna(inplace=True)

        test_scaled = scaler.transform(test_features)
        test_pca = pca.transform(test_scaled)

        # Predict regime for last observation of test data
        test_states = hmm.predict(test_pca)
        last_state = test_states[-1]

        # Map states to labels using aligned training data returns
        close = train_df.loc[valid_index, "Close"].values
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0.0], returns])
        train_states = hmm.predict(pca_features)

        state_ids = sorted(set(train_states))
        mean_returns = {s: returns[train_states == s].mean() for s in state_ids}
        sorted_states = sorted(state_ids, key=lambda s: mean_returns[s])
        label_map = {sorted_states[-1]: "bull", sorted_states[0]: "bear"}
        for s in sorted_states:
            if s not in label_map:
                label_map[s] = "sideways"

        regime = label_map.get(last_state, "sideways")
        return self.get_strategy_for_regime(regime)

    def _extract_metrics(self, stats, test_df: pd.DataFrame | None = None) -> dict:
        """Extract key performance metrics from backtesting stats."""
        equity = stats._equity_curve["Equity"]
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        total_return = float(stats["Return [%]"]) / 100
        df_for_duration = test_df if test_df is not None else self.df
        duration_days = (df_for_duration.index[-1] - df_for_duration.index[0]).days if df_for_duration is not None else 365
        years = max(duration_days / 365.25, 1 / 365.25)
        annual_return = (1 + total_return) ** (1 / years) - 1

        return {
            "sharpe_ratio": float(stats["Sharpe Ratio"]) if not np.isnan(stats["Sharpe Ratio"]) else 0.0,
            "sortino_ratio": float(stats["Sortino Ratio"]) if not np.isnan(stats["Sortino Ratio"]) else 0.0,
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(annual_return / abs(max_dd)) if max_dd != 0 else 0.0,
            "win_rate": float(stats["Win Rate [%]"]) / 100 if not np.isnan(stats["Win Rate [%]"]) else 0.0,
            "profit_factor": float(stats["Profit Factor"]) if not np.isnan(stats["Profit Factor"]) else 0.0,
            "total_return": total_return,
            "buy_and_hold_return": float(stats["Buy & Hold Return [%]"]) / 100,
            "num_trades": int(stats["# Trades"]),
        }

    @staticmethod
    def _metrics_to_dataframe(metrics: dict, start_date, end_date) -> pd.DataFrame:
        """Convert metrics dict to a DataFrame with metric, value, description."""
        rows = [
            {"metric": "backtest_start", "value": str(start_date.date()), "description": "Start of backtest period"},
            {"metric": "backtest_end", "value": str(end_date.date()), "description": "End of backtest period"},
        ]
        for key, value in metrics.items():
            rows.append({
                "metric": key,
                "value": value,
                "description": METRIC_DESCRIPTIONS.get(key, ""),
            })
        return pd.DataFrame(rows)
