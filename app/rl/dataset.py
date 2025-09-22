from __future__ import annotations


from dataclasses import dataclass
from typing import List, Tuple
import warnings
import numpy as np
import pandas as pd

from app.data.session import session_scope

from app.data.models import FeatureRow
try:
    from app.data.models import Candle  
    _HAS_CANDLE = True
except Exception:
    Candle = None  
    _HAS_CANDLE = False



FEATURE_COLUMNS = [
    "ema_5",
    "ema_20",
    "rsi_14",
    "atr_14",
    "bb_mid",
    "bb_up",
    "bb_dn",
]

@dataclass(frozen=True)
class Split:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def load_features(symbol: str, tf: str) -> pd.DataFrame:

    with session_scope() as s:
        q = (
            s.query(
                FeatureRow.ts,
                FeatureRow.ema_5,
                FeatureRow.ema_20,
                FeatureRow.rsi_14,
                FeatureRow.atr_14,
                FeatureRow.bb_mid,
                FeatureRow.bb_up,
                FeatureRow.bb_dn,
            )
            .filter(FeatureRow.symbol == symbol)
            .filter(FeatureRow.tf == tf)
            .filter(FeatureRow.shifted == True)  
            .order_by(FeatureRow.ts.asc())
        )
        rows = q.all()

    if not rows:
        raise ValueError(f"No shifted features found for {symbol} {tf}.")

    df = pd.DataFrame(rows, columns=["ts"] + FEATURE_COLUMNS)
    # ensure tz-aware UTC index
    ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])
    df.index = ts
    df = df.sort_index()
    # drop duplicate indices if any (defensive)
    df = df[~df.index.duplicated(keep="last")]
    return df


def load_prices(symbol: str, tf: str) -> pd.Series:

    if _HAS_CANDLE:
        with session_scope() as s:
            q = (
                s.query(Candle.ts, Candle.close)
                .filter(Candle.symbol == symbol)
                .filter(Candle.tf == tf)
                .order_by(Candle.ts.asc())
            )
            rows = q.all()
        if not rows:
            warnings.warn(
                "Candle table returned no rows systm falling back to FeatureRow.close if present."
            )
        else:
            df = pd.DataFrame(rows, columns=["ts", "close"])
            ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
            s_close = pd.Series(df["close"].astype(float).values, index=ts, name="close")
            s_close = s_close[~s_close.index.duplicated(keep="last")].sort_index()
            return s_close

    if hasattr(FeatureRow, "close"):
        with session_scope() as s:
            q = (
                s.query(FeatureRow.ts, FeatureRow.close)
                .filter(FeatureRow.symbol == symbol)
                .filter(FeatureRow.tf == tf)
                .order_by(FeatureRow.ts.asc())
            )
            rows = q.all()
        if rows:
            df = pd.DataFrame(rows, columns=["ts", "close"])
            ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
            s_close = pd.Series(df["close"].astype(float).values, index=ts, name="close")
            s_close = s_close[~s_close.index.duplicated(keep="last")].sort_index()
            return s_close

    raise ValueError(
        "Unable to load close pricesplease Ensure a Candle model/table exists OR "
        "add a `close` column to the FeatureRow in database to enable fallback."
    )


def align(features_df: pd.DataFrame, close_s: pd.Series) -> tuple[pd.DataFrame, pd.Series]:

    idx = features_df.index.intersection(close_s.index)
    X = features_df.loc[idx].sort_index()
    c = close_s.loc[idx].sort_index()

    if len(X) < 10:
        raise ValueError(f"Too few aligned rows after join: {len(X)}")

    if not X.index.is_monotonic_increasing:
        X = X.sort_index()
    if not c.index.is_monotonic_increasing:
        c = c.sort_index()

    if X.index.tz is None:
        X.index = X.index.tz_localize("UTC")
    if c.index.tz is None:
        c.index = c.index.tz_localize("UTC")

    return X, c


def make_windows(
    ts_index: pd.DatetimeIndex,
    train_span: int,
    test_span: int,
    stride: int,
) -> List[Split]:

    if not isinstance(ts_index, pd.DatetimeIndex):
        ts_index = pd.DatetimeIndex(ts_index)

    if ts_index.tz is None:
        ts_index = ts_index.tz_localize("UTC")

    n = len(ts_index)
    if n < (train_span + test_span):
        return []

    splits: List[Split] = []
    train_start_i = 0
    while True:
        train_end_i = train_start_i + train_span - 1
        test_start_i = train_end_i + 1
        test_end_i = test_start_i + test_span - 1

        if test_end_i >= n:
            break

        split = Split(
            train_start=ts_index[train_start_i],
            train_end=ts_index[train_end_i],
            test_start=ts_index[test_start_i],
            test_end=ts_index[test_end_i],
        )
        splits.append(split)

        train_start_i += stride
        if train_start_i + train_span + test_span > n:
            break

    return splits



def load_aligned(symbol: str, tf: str) -> tuple[pd.DataFrame, pd.Series]:

    feats = load_features(symbol, tf)
    close = load_prices(symbol, tf)
    X, c = align(feats, close)
    return X, c
