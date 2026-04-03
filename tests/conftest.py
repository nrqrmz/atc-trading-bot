import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    close = 30000 + np.cumsum(np.random.randn(n) * 500)
    high = close + np.abs(np.random.randn(n) * 200)
    low = close - np.abs(np.random.randn(n) * 200)
    open_ = close + np.random.randn(n) * 100
    volume = np.abs(np.random.randn(n) * 1000000) + 500000

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


@pytest.fixture
def sample_ohlcv_raw():
    """Raw OHLCV data as returned by CCXT (list of lists)."""
    np.random.seed(42)
    n = 200
    base_ts = 1672531200000  # 2023-01-01 UTC
    data = []
    close = 30000.0
    for i in range(n):
        close += np.random.randn() * 500
        high = close + abs(np.random.randn() * 200)
        low = close - abs(np.random.randn() * 200)
        open_ = close + np.random.randn() * 100
        volume = abs(np.random.randn() * 1000000) + 500000
        ts = base_ts + i * 86400000  # 1 day in ms
        data.append([ts, open_, high, low, close, volume])
    return data
