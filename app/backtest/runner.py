from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session

from app.data.models import Candle, FeatureRow, BacktestRun
from app.backtest.strategy_rules import signal_baseline, BaselineParams
from app.backtest.metrics import compute_metrics
def _ensure_dir(p: str | Path) -> str:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

@dataclass(frozen=True)
class BtParams:
    symbol: str
    tf: str
    fees_bps: float = 10.0
    slippage_bps: float = 1.0
    equity0: float = 10_000.0
    start: Optional[int] = None   
    end: Optional[int] = None     
    out_dir: str = "artifacts/backtests"

def _load_candles_df(session: Session, symbol: str, tf: str, start: Optional[int], end: Optional[int]) -> pd.DataFrame:
    q = session.query(Candle).filter(Candle.symbol == symbol, Candle.tf == tf)
    if start is not None:
        q = q.filter(Candle.ts >= int(start))
    if end is not None:
        q = q.filter(Candle.ts < int(end))
    rows = q.order_by(Candle.ts.asc()).all()
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df = pd.DataFrame({
        "ts": [int(r.ts) for r in rows],
        "open": [float(r.open) for r in rows],
        "high": [float(r.high) for r in rows],
        "low": [float(r.low) for r in rows],
        "close": [float(r.close) for r in rows],
        "volume": [float(r.volume) for r in rows],
    }).set_index("ts")
    df.index.name = "ts"
    return df

def _load_features_df(session: Session, symbol: str, tf: str, start: Optional[int], end: Optional[int]) -> pd.DataFrame:
    q = session.query(FeatureRow).filter(FeatureRow.symbol == symbol, FeatureRow.tf == tf)
    if start is not None:
        q = q.filter(FeatureRow.ts >= int(start))
    if end is not None:
        q = q.filter(FeatureRow.ts < int(end))
    rows = q.order_by(FeatureRow.ts.asc()).all()
    if not rows:
        return pd.DataFrame(columns=["ema_5","ema_20","rsi_14","atr_14","bb_mid","bb_up","bb_dn","shifted"])
    df = pd.DataFrame({
        "ts": [int(r.ts) for r in rows],
        "ema_5": [float(r.ema_5) for r in rows],
        "ema_20": [float(r.ema_20) for r in rows],
        "rsi_14": [float(r.rsi_14) for r in rows],
        "atr_14": [float(r.atr_14) for r in rows],
        "bb_mid": [float(r.bb_mid) for r in rows],
        "bb_up": [float(r.bb_up) for r in rows],
        "bb_dn": [float(r.bb_dn) for r in rows],
        "shifted": [bool(r.shifted) for r in rows],
    }).set_index("ts")
    df.index.name = "ts"
    return df



def run_backtest(
    candles_df: pd.DataFrame,
    feats_df: pd.DataFrame,
    symbol: str,
    tf: str,
    fees_bps: float,
    slippage_bps: float,
    equity0: float,
    params: Optional[BaselineParams] = None,
) -> Tuple[Dict[str, Any], pd.Series, pd.DataFrame]:

    if candles_df.empty or feats_df.empty:
        raise ValueError("No candles or features available for backtest window.")

    close = candles_df["close"].copy()
    feats = feats_df.loc[feats_df.index.intersection(close.index)].copy()
    close = close.loc[feats.index]

    if len(feats) < 5:
        raise ValueError("Not enough feature rows in the selected window.")

    bl = params or BaselineParams()
    actions = []
    for ts, row in feats.iterrows():
        payload = signal_baseline(
            {
                "ema_5": float(row["ema_5"]),
                "ema_20": float(row["ema_20"]),
                "rsi_14": float(row["rsi_14"]),
                "atr_14": float(row["atr_14"]),
                "bb_mid": float(row["bb_mid"]),
                "bb_up": float(row["bb_up"]),
                "bb_dn": float(row["bb_dn"]),
            },
            bl,
        )
        actions.append(str(payload["action"]))

    actions = pd.Series(actions, index=feats.index, name="action")

    pos = (actions == "BUY").astype(int)

    pos_next = pos.shift(1).fillna(0).astype(int)

    rets = close.pct_change().fillna(0.0)

    strat_rets = (pos_next * rets).astype(float)

    trans = pos_next.diff().fillna(pos_next).astype(int) 
    cost_per_side = (fees_bps + slippage_bps) / 10_000.0
    costs = (trans.abs() > 0).astype(float) * cost_per_side  

    equity_vals = [float(equity0)]
    for r, c in zip(strat_rets.values, costs.values):
        equity_vals.append(equity_vals[-1] * (1.0 + float(r)) * (1.0 - float(c)))
    eq = pd.Series(equity_vals[1:], index=feats.index, name="equity")

    trades = []
    in_pos = False
    entry_ts = entry_px = entry_eq = None
    for i in range(len(pos_next)):
        ts = feats.index[i]
        if not in_pos and trans.iat[i] == 1:
            in_pos = True
            entry_ts = ts
            entry_px = float(close.iat[i])
            entry_eq = float(eq.iat[i])
        elif in_pos and trans.iat[i] == -1:
            exit_ts = ts
            exit_px = float(close.iat[i])
            exit_eq = float(eq.iat[i])
            ret_pct = (exit_eq / entry_eq) - 1.0 if entry_eq else 0.0
            trades.append({
                "entry_ts": int(entry_ts),
                "exit_ts": int(exit_ts),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "ret_pct": float(ret_pct),
            })
            in_pos = False
            entry_ts = entry_px = entry_eq = None
    trades_df = pd.DataFrame(trades)

    metrics = compute_metrics(
        tf=tf,
        equity=eq,
        strat_returns=strat_rets,
        pos_next=pos_next,
        trades_df=trades_df,
        fees_bps=fees_bps,
        slippage_bps=slippage_bps,
    )

    return metrics, eq, trades_df

def run_backtest_db(session: Session, params: BtParams) -> Tuple[str, Dict[str, Any], str]:
    candles = _load_candles_df(session, params.symbol, params.tf, params.start, params.end)
    feats = _load_features_df(session, params.symbol, params.tf, params.start, params.end)
    if candles.empty or feats.empty:
        raise ValueError("No candles/features in DB for the requested window.")

    metrics, eq, _trades = run_backtest(
        candles, feats, symbol=params.symbol, tf=params.tf,
        fees_bps=params.fees_bps, slippage_bps=params.slippage_bps, equity0=params.equity0,
    )

    out_dir = _ensure_dir(params.out_dir)
    fname = f"BT_{params.symbol.replace('/','')}_{params.tf}_{eq.index[0]}_{eq.index[-1]}.png"
    png_path = str(Path(out_dir) / fname)
    plt.figure(figsize=(8, 3.2))
    plt.plot(eq.index.values, eq.values)
    plt.title(f"Equity — {params.symbol} {params.tf}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()

    run = BacktestRun(
        params_json={
            "symbol": params.symbol, "tf": params.tf,
            "fees_bps": params.fees_bps, "slippage_bps": params.slippage_bps,
            "equity": params.equity0, "start": params.start, "end": params.end,
        },
        metrics_json=metrics,
        equity_curve_path=png_path,
    )
    session.add(run)
    session.flush()
    return run.id, metrics, png_path
