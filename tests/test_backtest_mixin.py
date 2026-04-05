import numpy as np
import pandas as pd
import pytest

from atc_trading_bot.mixins.backtest_mixin import BacktestMixin
from atc_trading_bot.pipeline_warning import PipelineWarning
from atc_trading_bot.mixins.strategy_mixin import StrategyMixin
from atc_trading_bot.strategies.bull_strategy import BullStrategy


class BacktestBot(BacktestMixin, StrategyMixin):
    """Minimal class using BacktestMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        self.current_regime = kwargs.pop("current_regime", None)
        self.features_pca = kwargs.pop("features_pca", None)
        super().__init__(**kwargs)


@pytest.fixture
def long_ohlcv_data():
    """Generate longer OHLCV data suitable for backtesting (300 days with trend)."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    # Create a trending market so SMA crossovers generate trades
    trend = np.linspace(0, 5000, n)
    noise = np.cumsum(np.random.randn(n) * 100)
    close = 30000 + trend + noise
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestBacktestMixin:
    def test_backtest_returns_metrics(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest()

        assert isinstance(results, pd.DataFrame)
        assert list(results.columns) == ["metric", "value", "description"]
        metrics = results["metric"].tolist()
        expected_keys = [
            "backtest_start", "backtest_end",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "win_rate", "profit_factor", "total_return", "buy_and_hold_return",
            "num_trades",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_backtest_with_explicit_strategy(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data)
        results = bot.backtest(strategy=BullStrategy)

        assert isinstance(results, pd.DataFrame)
        num_trades = results.loc[results["metric"] == "num_trades", "value"].iloc[0]
        assert num_trades >= 0

    def test_backtest_stores_results(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        bot.backtest()

        assert bot.results is not None

    def test_backtest_warns_without_data(self):
        bot = BacktestBot(current_regime="bull")
        bot.select_strategy()
        with pytest.warns(PipelineWarning, match="fetch_data"):
            result = bot.backtest()
        assert result is None

    def test_backtest_warns_without_strategy(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data)
        with pytest.warns(PipelineWarning, match="select_strategy"):
            result = bot.backtest()
        assert result is None

    def test_max_drawdown_is_negative_or_zero(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest()

        max_dd = results.loc[results["metric"] == "max_drawdown", "value"].iloc[0]
        assert max_dd <= 0

    def test_backtest_includes_date_range(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest(test_ratio=0.3)

        metrics = results["metric"].tolist()
        assert "backtest_start" in metrics
        assert "backtest_end" in metrics

        start = results.loc[results["metric"] == "backtest_start", "value"].iloc[0]
        end = results.loc[results["metric"] == "backtest_end", "value"].iloc[0]
        assert start < end

    def test_cpcv_returns_list_of_metrics(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        cv_results = bot.cross_validate_cpcv(n_splits=3)

        assert isinstance(cv_results, list)
        assert len(cv_results) > 0
        for result in cv_results:
            assert "sharpe_ratio" in result
            assert "fold" in result

    def test_cpcv_stores_results(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        bot.cross_validate_cpcv(n_splits=3)

        assert bot.cv_results is not None


class TestOverfitDetection:
    def test_check_overfit_warns_on_degraded_sharpe(self):
        train = {"sharpe_ratio": 2.0, "total_return": 0.5}
        test = {"sharpe_ratio": 0.3, "total_return": 0.4}
        bot = BacktestBot()

        with pytest.warns(PipelineWarning, match="overfitting.*sharpe_ratio"):
            warns = bot._check_overfit(train, test)

        assert len(warns) >= 1
        assert "sharpe_ratio" in warns[0]

    def test_check_overfit_warns_on_degraded_return(self):
        train = {"sharpe_ratio": 1.0, "total_return": 0.5}
        test = {"sharpe_ratio": 0.8, "total_return": 0.1}

        with pytest.warns(PipelineWarning, match="overfitting.*total_return"):
            warns = BacktestMixin._check_overfit(train, test)

        assert any("total_return" in w for w in warns)

    def test_check_overfit_no_warning_when_healthy(self):
        train = {"sharpe_ratio": 1.5, "total_return": 0.3}
        test = {"sharpe_ratio": 1.2, "total_return": 0.25}

        warns = BacktestMixin._check_overfit(train, test)
        assert warns == []

    def test_check_overfit_skips_negative_train(self):
        train = {"sharpe_ratio": -0.5, "total_return": -0.1}
        test = {"sharpe_ratio": 0.1, "total_return": 0.05}

        warns = BacktestMixin._check_overfit(train, test)
        assert warns == []

    def test_check_overfit_custom_threshold(self):
        train = {"sharpe_ratio": 1.0, "total_return": 0.3}
        test = {"sharpe_ratio": 0.8, "total_return": 0.25}

        # With strict threshold (0.9), 0.8/1.0 = 0.8 < 0.9 should warn
        with pytest.warns(PipelineWarning):
            warns = BacktestMixin._check_overfit(train, test, threshold=0.9)
        assert len(warns) >= 1

    def test_backtest_runs_overfit_check(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest()

        # Should run without error; overfitting check happens internally
        assert isinstance(results, pd.DataFrame)


class TestRiskParams:
    def test_backtest_with_custom_risk_params(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        results = bot.backtest(stop_loss=0.03, take_profit=0.08, position_size=0.10)

        assert isinstance(results, pd.DataFrame)
        assert "sharpe_ratio" in results["metric"].values

    def test_risk_params_dont_mutate_original_strategy(self, long_ohlcv_data):
        bot = BacktestBot(df=long_ohlcv_data, current_regime="bull")
        bot.select_strategy()
        original_sl = bot.active_strategy.stop_loss

        bot.backtest(stop_loss=0.99)

        assert bot.active_strategy.stop_loss == original_sl

    def test_apply_risk_params_creates_subclass(self):
        from atc_trading_bot.strategies.bull_strategy import BullStrategy

        modified = BacktestMixin._apply_risk_params(BullStrategy, 0.03, 0.08, 0.10)

        assert modified.stop_loss == 0.03
        assert modified.take_profit == 0.08
        assert modified.position_size == 0.10
        assert issubclass(modified, BullStrategy)
        # Original unchanged
        assert BullStrategy.stop_loss != 0.03
