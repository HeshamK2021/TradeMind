from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).astype(float)
    loss = (-delta.where(delta < 0, 0.0)).astype(float)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace({0.0: np.nan})
    rsi = 100 - (100 / (1 + rs))
    return rsi

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def bollinger_mid(series: pd.Series, period: int = 20) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = bollinger_mid(series, period=period)
    std = series.rolling(window=period, min_periods=period).std(ddof=0)
    up = mid + num_std * std
    dn = mid - num_std * std
    return mid, up, dn
