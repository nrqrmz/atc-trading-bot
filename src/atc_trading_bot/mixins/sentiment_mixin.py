import warnings

import pandas as pd
import requests
from atc_trading_bot.pipeline_warning import PipelineWarning

FEAR_GREED_URL = "https://api.alternative.me/fgi/?limit={days}&format=json"


class SentimentMixin:
    """Mixin for fetching the Crypto Fear & Greed Index.

    Provides two methods:
    - fetch_sentiment(): Fetches daily Fear & Greed data from the API.
    - merge_sentiment(): Merges sentiment data into self.df using
      pd.merge_asof to align daily sentiment to the bot's timeframe
      without look-ahead bias.
    """

    def _require_data(self) -> bool:
        """Check that OHLCV data has been loaded."""
        if getattr(self, "df", None) is None:
            warnings.warn("No data available. Call fetch_data first.", PipelineWarning)
            return False
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sentiment_df: pd.DataFrame | None = None

    def fetch_sentiment(self, days: int = 30) -> pd.DataFrame | None:
        """Fetch Crypto Fear & Greed Index data from the API.

        Args:
            days: Number of days of history to fetch. Default: 30.

        Returns:
            DataFrame with columns: date, value, classification.
            Returns None if the API call fails.
        """
        url = FEAR_GREED_URL.format(days=days)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            warnings.warn(
                f"Failed to fetch sentiment data: {exc}",
                PipelineWarning,
            )
            return None

        records = data.get("data", [])
        if not records:
            warnings.warn(
                "Sentiment API returned no data.",
                PipelineWarning,
            )
            return None

        rows = []
        for entry in records:
            rows.append({
                "date": pd.to_datetime(int(entry["timestamp"]), unit="s", utc=True),
                "value": int(entry["value"]),
                "classification": entry["value_classification"],
            })

        self.sentiment_df = (
            pd.DataFrame(rows)
            .sort_values("date")
            .reset_index(drop=True)
        )
        return self.sentiment_df

    def merge_sentiment(self) -> pd.DataFrame | None:
        """Merge sentiment data into the OHLCV DataFrame.

        Uses pd.merge_asof to align daily sentiment values to the
        bot's DataFrame index. The merge direction is 'backward' so
        each bar receives the most recent sentiment value that is at
        or before its timestamp, avoiding look-ahead bias.

        Returns:
            The merged DataFrame (also stored in self.df), or None
            if prerequisites are missing.
        """
        if not self._require_data():
            return None

        if self.sentiment_df is None or self.sentiment_df.empty:
            warnings.warn(
                "No sentiment data available. Call fetch_sentiment first.",
                PipelineWarning,
            )
            return None

        # Prepare sentiment for merge: set date as index, sorted
        sentiment = self.sentiment_df.copy()
        sentiment["date"] = pd.to_datetime(sentiment["date"], utc=True)
        sentiment = sentiment.sort_values("date")

        # Ensure the main df index is datetime and sorted
        df = self.df.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()

        # Drop existing sentiment columns to avoid duplicates on re-merge
        for col in ["sentiment_value", "sentiment_classification"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # merge_asof requires a column (not index) for the left key
        df = df.reset_index()
        left_col = df.columns[0]  # the timestamp column name

        merged = pd.merge_asof(
            df,
            sentiment.rename(columns={
                "value": "sentiment_value",
                "classification": "sentiment_classification",
            }),
            left_on=left_col,
            right_on="date",
            direction="backward",
        )

        # Restore the original index
        merged = merged.set_index(left_col)
        merged = merged.drop(columns=["date"], errors="ignore")

        self.df = merged
        return self.df
