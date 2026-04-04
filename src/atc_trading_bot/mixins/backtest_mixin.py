import itertools
import warnings

import numpy as np
from atc_trading_bot.pipeline_warning import PipelineWarning
import pandas as pd
from backtesting import Strategy
from backtesting.lib import FractionalBacktest as Backtest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import ta as ta_lib

_METRIC_DESCRIPTIONS = {
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


class BacktestMixin:
    """Mixin for backtesting strategies and CPCV cross-validation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results: pd.DataFrame | None = None
        self.cv_results: list[dict] | None = None

    def backtest(self, strategy: type[Strategy] | None = None,
                 cash: float = 100_000, commission: float = 0.001,
                 test_ratio: float = 0.3, n_components: int | None = None,
                 n_regimes: int | None = None) -> pd.DataFrame:
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

        Returns:
            DataFrame with columns (metric, value, description) including
            backtest date range, or None if prerequisites are missing.
        """
        if self.df is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
            return

        if strategy is None and self.active_strategy is None:
            warnings.warn("No strategy selected. Call select_strategy first.", PipelineWarning)
            return

        if n_components is None:
            n_components = getattr(self.pca, "n_components_", 10) if hasattr(self, "pca") and self.pca is not None else 10
        if n_regimes is None:
            n_regimes = getattr(self.hmm_model, "n_components", 3) if hasattr(self, "hmm_model") and self.hmm_model is not None else 3

        train_df, test_df = train_test_split(self.df, test_size=test_ratio, shuffle=False)

        if strategy is not None:
            strat = strategy
        else:
            strat = self._fit_and_select_strategy(
                train_df, n_components=n_components, n_regimes=n_regimes,
                test_df=test_df,
            )

        bt = Backtest(test_df, strat, cash=cash, commission=commission, finalize_trades=True)
        stats = bt.run()
        metrics_dict = self._extract_metrics(stats, test_df=test_df)
        self.results = self._metrics_to_dataframe(
            metrics_dict,
            start_date=test_df.index[0],
            end_date=test_df.index[-1],
        )
        return self.results

    def cross_validate_cpcv(self, n_splits: int = 5, purge_gap: int = 5,
                            embargo_pct: float = 0.01, n_components: int = 10,
                            n_regimes: int = 3, cash: float = 100_000,
                            commission: float = 0.001) -> list[dict]:
        """Combinatorial Purged Cross-Validation.

        Generates all C(n_splits, 2) train/test combinations where each
        combination uses 1 group as test and the rest as training.
        Purging removes observations near the train/test boundary.
        Embargo adds an additional gap after each test set.
        """
        if self.df is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
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

            if len(train_indices) < 50:
                continue

            train_df = self.df.iloc[train_indices]
            test_df = self.df.iloc[test_start:test_end]

            if len(test_df) < 30:
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

        return self.cv_results

    def _fit_and_select_strategy(self, train_df: pd.DataFrame, n_components: int,
                                 n_regimes: int, test_df: pd.DataFrame) -> type[Strategy]:
        """Fit PCA + HMM on training data and select strategy for test data's regime."""
        from atc_trading_bot.mixins.feature_mixin import _EXCLUDE_COLS, _CORRELATION_THRESHOLD

        # Compute features on training data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="ta")
            df_ta = ta_lib.add_all_ta_features(
                train_df.copy(), open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=False,
            )
        feature_cols = [c for c in df_ta.columns if c not in _EXCLUDE_COLS]
        train_features = df_ta[feature_cols].copy()

        # Clean: forward fill, replace inf, drop NaN
        train_features.ffill(inplace=True)
        train_features.replace([np.inf, -np.inf], np.nan, inplace=True)
        thresh = int(len(train_features) * 0.5)
        train_features.dropna(axis=1, thresh=thresh, inplace=True)
        train_features.dropna(inplace=True)

        valid_index = train_features.index

        # Remove constant and highly correlated features
        non_const = train_features.columns[train_features.std() > 0]
        train_features = train_features[non_const]

        corr = train_features.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        corr_drop = [col for col in upper.columns if any(upper[col] > _CORRELATION_THRESHOLD)]
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
                "description": _METRIC_DESCRIPTIONS.get(key, ""),
            })
        return pd.DataFrame(rows)
