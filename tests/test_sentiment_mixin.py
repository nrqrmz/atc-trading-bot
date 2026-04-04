import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from atc_trading_bot.mixins.sentiment_mixin import SentimentMixin
from atc_trading_bot.pipeline_warning import PipelineWarning


SAMPLE_API_RESPONSE = {
    "data": [
        {"value": "25", "value_classification": "Extreme Fear", "timestamp": "1672531200"},
        {"value": "40", "value_classification": "Fear", "timestamp": "1672617600"},
        {"value": "55", "value_classification": "Neutral", "timestamp": "1672704000"},
        {"value": "72", "value_classification": "Greed", "timestamp": "1672790400"},
        {"value": "80", "value_classification": "Extreme Greed", "timestamp": "1672876800"},
    ]
}


class SentimentBot(SentimentMixin):
    """Minimal class using SentimentMixin for testing."""

    def __init__(self, **kwargs):
        self.df = kwargs.pop("df", None)
        super().__init__(**kwargs)


@pytest.fixture
def ohlcv_data():
    """OHLCV data spanning the same dates as the sample API response."""
    dates = pd.to_datetime([
        "2023-01-01", "2023-01-02", "2023-01-03",
        "2023-01-04", "2023-01-05",
    ], utc=True)
    np.random.seed(42)
    n = len(dates)
    close = 30000 + np.random.randn(n) * 100
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(n) * 50,
            "High": close + np.abs(np.random.randn(n) * 200),
            "Low": close - np.abs(np.random.randn(n) * 200),
            "Close": close,
            "Volume": np.abs(np.random.randn(n) * 1_000_000) + 500_000,
        },
        index=dates,
    )


@pytest.fixture
def mock_response():
    """Create a mock requests.Response returning sample data."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = SAMPLE_API_RESPONSE
    mock.raise_for_status.return_value = None
    return mock


class TestFetchSentiment:
    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_returns_dataframe_with_expected_columns(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        bot = SentimentBot()
        result = bot.fetch_sentiment(days=5)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["date", "value", "classification"]

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_values_are_integers_0_to_100(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        bot = SentimentBot()
        result = bot.fetch_sentiment(days=5)

        assert result["value"].dtype in (np.int64, int)
        assert (result["value"] >= 0).all()
        assert (result["value"] <= 100).all()

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_dates_are_sorted(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        bot = SentimentBot()
        result = bot.fetch_sentiment(days=5)

        assert result["date"].is_monotonic_increasing

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_stores_sentiment_df(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        bot = SentimentBot()
        bot.fetch_sentiment(days=5)

        assert bot.sentiment_df is not None
        assert len(bot.sentiment_df) == 5

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_passes_days_to_url(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        bot = SentimentBot()
        bot.fetch_sentiment(days=14)

        call_url = mock_get.call_args[0][0]
        assert "limit=14" in call_url

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_classifications_are_strings(self, mock_get, mock_response):
        mock_get.return_value = mock_response
        bot = SentimentBot()
        result = bot.fetch_sentiment(days=5)

        for cls in result["classification"]:
            assert isinstance(cls, str)
            assert len(cls) > 0


class TestFetchSentimentFailures:
    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_warns_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("Network error")
        bot = SentimentBot()

        with pytest.warns(PipelineWarning, match="Failed to fetch sentiment"):
            result = bot.fetch_sentiment()
        assert result is None

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_warns_on_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Server Error")
        mock_get.return_value = mock_resp
        bot = SentimentBot()

        with pytest.warns(PipelineWarning, match="Failed to fetch sentiment"):
            result = bot.fetch_sentiment()
        assert result is None

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_warns_on_empty_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp
        bot = SentimentBot()

        with pytest.warns(PipelineWarning, match="no data"):
            result = bot.fetch_sentiment()
        assert result is None


class TestMergeSentiment:
    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_adds_sentiment_columns_to_df(self, mock_get, mock_response, ohlcv_data):
        mock_get.return_value = mock_response
        bot = SentimentBot(df=ohlcv_data)
        bot.fetch_sentiment(days=5)
        result = bot.merge_sentiment()

        assert "sentiment_value" in result.columns
        assert "sentiment_classification" in result.columns

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_preserves_original_columns(self, mock_get, mock_response, ohlcv_data):
        mock_get.return_value = mock_response
        bot = SentimentBot(df=ohlcv_data)
        bot.fetch_sentiment(days=5)
        bot.merge_sentiment()

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in bot.df.columns

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_preserves_row_count(self, mock_get, mock_response, ohlcv_data):
        mock_get.return_value = mock_response
        original_len = len(ohlcv_data)
        bot = SentimentBot(df=ohlcv_data)
        bot.fetch_sentiment(days=5)
        bot.merge_sentiment()

        assert len(bot.df) == original_len

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_updates_self_df(self, mock_get, mock_response, ohlcv_data):
        mock_get.return_value = mock_response
        bot = SentimentBot(df=ohlcv_data)
        bot.fetch_sentiment(days=5)
        bot.merge_sentiment()

        assert "sentiment_value" in bot.df.columns

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_no_lookahead_bias(self, mock_get, mock_response, ohlcv_data):
        """Sentiment values should only use data available at or before each bar."""
        mock_get.return_value = mock_response
        bot = SentimentBot(df=ohlcv_data)
        bot.fetch_sentiment(days=5)
        bot.merge_sentiment()

        # The first row (2023-01-01) should have the sentiment from
        # 2023-01-01 or earlier, never from 2023-01-02
        first_sentiment = bot.df["sentiment_value"].iloc[0]
        assert first_sentiment == 25  # Value for 2023-01-01

    def test_warns_without_data(self):
        bot = SentimentBot()
        with pytest.warns(PipelineWarning, match="fetch_data"):
            result = bot.merge_sentiment()
        assert result is None

    def test_warns_without_sentiment(self, ohlcv_data):
        bot = SentimentBot(df=ohlcv_data)
        with pytest.warns(PipelineWarning, match="fetch_sentiment"):
            result = bot.merge_sentiment()
        assert result is None

    @patch("atc_trading_bot.mixins.sentiment_mixin.requests.get")
    def test_remerge_does_not_duplicate_columns(self, mock_get, mock_response, ohlcv_data):
        """Calling merge_sentiment twice should not create duplicate columns."""
        mock_get.return_value = mock_response
        bot = SentimentBot(df=ohlcv_data)
        bot.fetch_sentiment(days=5)
        bot.merge_sentiment()
        bot.merge_sentiment()

        sentiment_cols = [c for c in bot.df.columns if c.startswith("sentiment_")]
        assert len(sentiment_cols) == 2  # sentiment_value + sentiment_classification
