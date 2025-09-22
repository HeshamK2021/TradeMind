import time
import numpy as np
import pandas as pd
from pathlib import Path

from app.data.session import init_db, session_scope
from app.data.models import Candle, FeatureRow
from app.backtest.runner import run_backtest, run_backtest_db, BtParams

def _temp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path}/bt.db"

def _make_candles(n, t0, tf_ms, start_px=100.0, drift=0.05, vol=0.5):
    rng = np.random.RandomState(7)
    rets = rng.normal(loc=drift/ (24*365), scale=vol/100, size=n)
    prices = [start_px]
    for r in rets:
        prices.append(prices[-1] * (1 + r))
    prices = prices[1:]
    highs = np.array(prices) * 1.003
    lows = np.array(prices) * 0.997
    opens = np.array(prices) * 0.999
    vols = np.full(n, 10.0)
    ts = np.arange(t0, t0 + n * tf_ms, tf_ms, dtype=np.int64)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": prices, "volume": vols}, index=ts)
    df.index.name = "ts"
    return df

def _features_from_pattern(ts_index, pattern="toggle"):
    rows = []
    for i, ts in enumerate(ts_index):
        if i < 5:
            ema5, ema20, rsi, atr = 100, 100.1, 49, 1.0
        else:
            if pattern == "toggle":
                if i % 3 == 0:
                    ema5, ema20, rsi = 102, 100, 55
                else:
                    ema5, ema20, rsi = 99, 100, 45
                atr = 1.0
            else:
                ema5, ema20, rsi, atr = 102, 100, 60, 1.0
        rows.append({
            "ts": int(ts),
            "ema_5": float(ema5), "ema_20": float(ema20), "rsi_14": float(rsi), "atr_14": float(atr),
            "bb_mid": 100.0, "bb_up": 102.0, "bb_dn": 98.0, "shifted": True
        })
    f = pd.DataFrame(rows).set_index("ts")
    return f

def test_run_backtest_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))
    init_db(reset=True)

    symbol, tf = "SYNTH/USD", "1h"
    tf_ms = 3_600_000
    n = 300
    t0 = int(time.time() * 1000) // tf_ms * tf_ms - n * tf_ms

    candles = _make_candles(n, t0, tf_ms)
    feats = _features_from_pattern(candles.index, pattern="toggle")

    metrics, eq, trades_df = run_backtest(
        candles, feats, symbol=symbol, tf=tf,
        fees_bps=10.0, slippage_bps=1.0, equity0=10_000.0,
    )
    assert "trades" in metrics and metrics["trades"] >= 1
    assert eq.index.is_monotonic_increasing
    assert eq.iat[-1] > 0
    assert "sharpe" in metrics and isinstance(metrics["sharpe"], float)

def test_run_backtest_db_and_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))
    init_db(reset=True)

    symbol, tf = "SYNTH/USD", "1h"
    tf_ms = 3_600_000
    n = 200
    t0 = int(time.time() * 1000) // tf_ms * tf_ms - n * tf_ms

    from app.data.session import session_scope
    with session_scope() as s:
        for ts, row in _make_candles(n, t0, tf_ms).iterrows():
            s.add(Candle(symbol=symbol, tf=tf, ts=int(ts),
                         open=float(row["open"]), high=float(row["high"]),
                         low=float(row["low"]), close=float(row["close"]), volume=float(row["volume"])))
        for ts, row in _features_from_pattern(np.array([t0 + i*tf_ms for i in range(n)])).iterrows():
            s.add(FeatureRow(symbol=symbol, tf=tf, ts=int(ts),
                             ema_5=float(row["ema_5"]), ema_20=float(row["ema_20"]), rsi_14=float(row["rsi_14"]),
                             atr_14=float(row["atr_14"]), bb_mid=float(row["bb_mid"]), bb_up=float(row["bb_up"]),
                             bb_dn=float(row["bb_dn"]), shifted=True))

    with session_scope() as s:
        run_id, metrics, png = run_backtest_db(
            s,
            BtParams(symbol=symbol, tf=tf, fees_bps=10.0, slippage_bps=1.0, equity0=10_000.0,
                     start=t0, end=t0 + n*tf_ms, out_dir=str(tmp_path / "artifacts" / "backtests"))
        )
        assert isinstance(run_id, str) and len(run_id) > 0
        assert "pnl_pct" in metrics
        assert Path(png).exists()
