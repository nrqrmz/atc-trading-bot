"""Visualization mixin — interactive Plotly charts for the trading pipeline.

All charts use the plotly_dark template for consistency.
"""
import warnings

import numpy as np
from atc_trading_bot.config import REGIME_COLORS
from atc_trading_bot.pipeline_warning import PipelineWarning
import pandas as pd


class VisualizationMixin:
    """Mixin providing interactive charts for pipeline analysis."""

    def _require_data(self) -> bool:
        if getattr(self, "df", None) is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
            return False
        return True

    def plot_equity_curve(self):
        """Plot the backtest equity curve and drawdown.

        Returns:
            plotly Figure, or None if no backtest results.
        """
        results = getattr(self, "results", None)
        if results is None:
            warnings.warn("No backtest results. Call backtest first.", PipelineWarning)
            return

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Extract metrics for the title
        total_ret = results.loc[results["metric"] == "total_return", "value"]
        sharpe = results.loc[results["metric"] == "sharpe_ratio", "value"]
        title_parts = ["Equity Curve"]
        if not total_ret.empty:
            title_parts.append(f"Return: {float(total_ret.iloc[0]):.2%}")
        if not sharpe.empty:
            title_parts.append(f"Sharpe: {float(sharpe.iloc[0]):.2f}")

        # Build a simple equity curve from total return over the test period
        start = results.loc[results["metric"] == "backtest_start", "value"].iloc[0]
        end = results.loc[results["metric"] == "backtest_end", "value"].iloc[0]
        test_df = self.df.loc[start:end]

        if test_df.empty:
            return

        ret = test_df["Close"].pct_change().fillna(0)
        equity = (1 + ret).cumprod()
        drawdown = equity / equity.cummax() - 1

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.05)

        fig.add_trace(go.Scatter(
            x=test_df.index, y=equity, mode="lines",
            name="Equity", line=dict(color="#2ecc71", width=2),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=test_df.index, y=drawdown, mode="lines",
            name="Drawdown", fill="tozeroy",
            line=dict(color="#e74c3c", width=1),
        ), row=2, col=1)

        fig.update_layout(
            title=" | ".join(title_parts),
            template="plotly_dark",
            hovermode="x unified",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".1%", row=2, col=1)

        return fig

    def plot_signals(self):
        """Plot price with buy/sell signal markers.

        Returns:
            plotly Figure, or None if no signals generated.
        """
        if not self._require_data():
            return

        signals = getattr(self, "signals", None)
        if signals is None:
            warnings.warn("No signals generated. Call generate_signals first.", PipelineWarning)
            return

        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["Close"],
            mode="lines", name="Price",
            line=dict(color="white", width=1),
        ))

        # Mark the signal on the last bar
        signal = signals["signal"]
        last_idx = self.df.index[-1]
        last_price = self.df["Close"].iloc[-1]

        marker_map = {
            "buy": ("triangle-up", "#2ecc71", "BUY"),
            "sell": ("triangle-down", "#e74c3c", "SELL"),
            "hold": ("circle", "#f1c40f", "HOLD"),
        }
        symbol, color, label = marker_map.get(signal, ("circle", "#ccc", "?"))

        fig.add_trace(go.Scatter(
            x=[last_idx], y=[last_price],
            mode="markers+text", name=label,
            marker=dict(symbol=symbol, size=16, color=color),
            text=[label], textposition="top center",
            textfont=dict(color=color, size=12),
        ))

        confidence = signals.get("confidence", None)
        conf_str = f" (conf: {confidence:.0%})" if confidence is not None else ""
        fig.update_layout(
            title=f"Signal: {label} — {signals['regime'].capitalize()} regime{conf_str}",
            yaxis_title="Price",
            template="plotly_dark",
            hovermode="x unified",
        )
        return fig

    def plot_cpcv_results(self):
        """Plot CPCV cross-validation results across folds.

        Returns:
            plotly Figure, or None if no CPCV results.
        """
        cv_raw = getattr(self, "_cv_results_raw", None)
        if not cv_raw:
            warnings.warn("No CPCV results. Call cross_validate_cpcv first.", PipelineWarning)
            return

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        metrics = ["sharpe_ratio", "total_return", "max_drawdown", "win_rate"]
        fig = make_subplots(rows=2, cols=2, subplot_titles=[
            "Sharpe Ratio", "Total Return", "Max Drawdown", "Win Rate",
        ])

        folds = [r.get("fold", i) for i, r in enumerate(cv_raw)]

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"]

        for metric, (row, col), color in zip(metrics, positions, colors):
            values = [r.get(metric, 0) for r in cv_raw]
            fig.add_trace(go.Bar(
                x=[f"Fold {f}" for f in folds], y=values,
                marker_color=color, name=metric,
                showlegend=False,
            ), row=row, col=col)

            # Add mean line
            mean_val = np.mean(values)
            fig.add_hline(y=mean_val, line_dash="dash", line_color="white",
                          opacity=0.5, row=row, col=col,
                          annotation_text=f"avg: {mean_val:.3f}")

        fig.update_layout(
            title="CPCV Cross-Validation Results",
            template="plotly_dark",
            height=600,
        )
        return fig

    def plot_feature_importance(self, top_n: int = 15):
        """Plot PCA component loadings showing most important features.

        Args:
            top_n: Number of top features to display. Default: 15.

        Returns:
            plotly Figure, or None if no PCA computed.
        """
        pca = getattr(self, "pca", None)
        features = getattr(self, "features", None)
        if pca is None or features is None:
            warnings.warn("No features computed. Call compute_features first.", PipelineWarning)
            return

        import plotly.graph_objects as go

        # Squared loadings weighted by explained variance (standard PCA importance)
        loadings_sq = pca.components_ ** 2
        weighted = loadings_sq * pca.explained_variance_ratio_[:, np.newaxis]
        importance = weighted.sum(axis=0)
        total_var = pca.explained_variance_ratio_.sum()
        importance_pct = (importance / total_var) * 100

        # Sort and take top_n
        col_names = features.columns.tolist()
        sorted_idx = np.argsort(importance_pct)[::-1][:top_n]
        top_names = [col_names[i] for i in sorted_idx]
        top_values = importance_pct[sorted_idx]

        fig = go.Figure(go.Bar(
            x=top_values[::-1],
            y=top_names[::-1],
            orientation="h",
            marker_color="#3498db",
        ))

        fig.update_layout(
            title=f"Feature Importance (PCA, {pca.n_components_} components, {total_var * 100:.1f}% variance)",
            xaxis_title="Contribution (%)",
            template="plotly_dark",
            height=max(400, top_n * 25),
        )
        return fig
