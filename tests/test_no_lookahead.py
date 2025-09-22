import math
import time
import numpy as np
import pandas as pd
from pathlib import Path

import pytest

from app.data.session import init_db, session_scope
from app.data.models import Candle, FeatureRow
from app.features.builder import build_features_df, persist_features
from app.features.indicators import ema, rsi, atr, bollinger_bands

def _temp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path}/features.db"

def _make_synth_ohlcv(n: int, start_ts_ms: int, tf_ms: int):
    prices = 100 + np.cumsum(np.random.RandomState(42).normal(0, 1, size=n))
    high = prices + np.random.RandomState(1).rand(n) * 1.5
    low  = prices - np.random.RandomState(2).rand(n) * 1.5
    open_ = prices + np.random.RandomState(3).randn(n) * 0.3
    close = prices
    vol   = 100 + np.random.RandomState(4).rand(n) * 10
    ts = np.arange(start_ts_ms, start_ts_ms + n * tf_ms, tf_ms, dtype=np.int64)
    df = pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=ts)
    df.index.name = "ts"
    return df

@pytest.mark.parametrize("n", [200])
def test_no_lookahead_and_alignment(tmp_path, monkeypatch, n):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))
    init_db(reset=True)

    symbol, tf = "BTC/USDT", "1h"
    tf_ms = 3_600_000
    t0 = int(time.time() * 1000) // tf_ms * tf_ms - n * tf_ms

    df = _make_synth_ohlcv(n, t0, tf_ms)
    with session_scope() as s:
        for ts, row in df.iterrows():
            s.add(Candle(symbol=symbol, tf=tf, ts=int(ts),
                         open=float(row["open"]), high=float(row["high"]),
                         low=float(row["low"]), close=float(row["close"]), volume=float(row["volume"])))

    feats = build_features_df(df, symbol=symbol, tf=tf)
    assert feats.index.is_monotonic_increasing
    assert len(feats) < len(df)
    with session_scope() as s:
        upserts = persist_features(s, feats)
    assert upserts == len(feats)

    ema5  = ema(df["close"], span=5).shift(1)
    ema20 = ema(df["close"], span=20).shift(1)
    rsi14 = rsi(df["close"], period=14).shift(1)
    atr14 = atr(df["high"], df["low"], df["close"], period=14).shift(1)
    bb_mid, bb_up, bb_dn = bollinger_bands(df["close"], period=20, num_std=2.0)
    bb_mid = bb_mid.shift(1); bb_up = bb_up.shift(1); bb_dn = bb_dn.shift(1)

    check_df = pd.DataFrame({
        "ema_5": ema5, "ema_20": ema20, "rsi_14": rsi14, "atr_14": atr14,
        "bb_mid": bb_mid, "bb_up": bb_up, "bb_dn": bb_dn
    }).dropna()

    with session_scope() as s:
        rows = (s.query(FeatureRow)
                  .filter(FeatureRow.symbol==symbol, FeatureRow.tf==tf)
                  .order_by(FeatureRow.ts.asc()).all())
    assert rows, "Features not persisted"

    compare = rows[-min(50, len(rows)):]
    for r in compare:
        ts = int(r.ts)
        assert ts in check_df.index
        exp = check_df.loc[ts]
        assert math.isclose(r.ema_5,  float(exp["ema_5"]),  rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(r.ema_20, float(exp["ema_20"]), rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(r.rsi_14, float(exp["rsi_14"]), rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(r.atr_14, float(exp["atr_14"]), rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(r.bb_mid, float(exp["bb_mid"]), rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(r.bb_up,  float(exp["bb_up"]),  rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(r.bb_dn,  float(exp["bb_dn"]),  rel_tol=1e-6, abs_tol=1e-6)

    assert rows[-1].ts <= int(df.index[-1])
