# app/backtest/runner.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Backtrader imports
try:
    import backtrader as bt
except ImportError as e:
    bt = None

# Project imports (TM-03 / TM-04)
try:
    from app.features.builder import build_features_df
except Exception:
    build_features_df = None

try:
    from app.backtest.strategy_rules import (
        BaselineParams,
        RiskParams,
        signal_baseline,
        apply_risk_layer,
    )
except Exception as e:
    raise RuntimeError(
        "TM-04 strategy_rules not found. Ensure app/backtest/strategy_rules.py exists."
    ) from e

from app.backtest.metrics import summarize_metrics, DEFAULT_PERIODS_PER_YEAR

# ---------- Helpers: load candles ----------

def load_candles_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns: ts, open, high, low, close, volume (ts in ISO or epoch ms)
    if "ts" not in df.columns:
        raise ValueError("CSV must include a 'ts' column (UTC).")
    if np.issubdtype(df["ts"].dtype, np.number):
        idx = pd.to_datetime(df["ts"], unit="ms", utc=True)
    else:
        idx = pd.to_datetime(df["ts"], utc=True)
    df.index = idx
    return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()

def try_load_from_db(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    """
    Optional: if TM-02 models exist, pull from DB.
    Otherwise return None (caller must provide CSV).
    """
    try:
        from app.data.session import SessionLocal
        from app.data.models import Candle
    except Exception:
        return None

    with SessionLocal() as s:
        q = (
            s.query(Candle)
            .filter(Candle.symbol == symbol, Candle.tf == tf)
            .order_by(Candle.ts.asc())
        )
        rows = q.all()
        if not rows:
            return None
        df = pd.DataFrame(
            [
                dict(
                    ts=r.ts,
                    open=r.open,
                    high=r.high,
                    low=r.low,
                    close=r.close,
                    volume=r.volume,
                )
                for r in rows
            ]
        )
        df.index = pd.to_datetime(df["ts"], utc=True)
        return df[["open", "high", "low", "close", "volume"]].astype(float).sort_index()

# ---------- Backtrader Strategy using precomputed signals ----------

class ValueRecorder(bt.Observer):
    lines = ("value",)

    def next(self):
        self.lines.value[0] = self._owner.broker.getvalue()

class PrecomputedSignalStrategy(bt.Strategy):
    params = dict(
        target_pos=None,   # pd.Series of {0,1} indexed by datetime
        risk_df=None,      # pd.DataFrame with 'qty' and 'stop_price' at entry bars
        fees_perc=0.0,     # decimal, e.g. 0.001
        slippage_perc=0.0, # decimal
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self._value_series = []
        self._pos_series = []
        self._dt_index = []
        self._open_entry_order = None
        self._open_stop_order = None

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        self._dt_index.append(dt)
        self._value_series.append(self.broker.getvalue())
        self._pos_series.append(1 if self.position.size > 0 else 0)

        target = 0
        if self.p.target_pos is not None:
            # align by timestamp if present
            idx = pd.Timestamp(dt, tz="UTC")
            if idx in self.p.target_pos.index:
                target = int(self.p.target_pos.loc[idx])
            else:
                # fallback: use last known
                target = int(self.p.target_pos.iloc[self.datas[0].buflen()-1]) if len(self.p.target_pos) else 0

        is_long = self.position.size > 0

        # EXIT if target==0 and we have position
        if is_long and target == 0:
            self.close()
            if self._open_stop_order:
                self.cancel(self._open_stop_order)
            self._open_entry_order = None
            self._open_stop_order = None
            return

        # ENTRY if target==1 and we are flat
        if (not is_long) and target == 1:
            idx = pd.Timestamp(dt, tz="UTC")
            qty = None
            stop_price = None
            if self.p.risk_df is not None and idx in self.p.risk_df.index:
                row = self.p.risk_df.loc[idx]
                qty = float(row.get("qty", 0.0))
                stop_price = row.get("stop_price", None)

            if qty is None or not np.isfinite(qty) or qty <= 0:
                # fallback: 100% of cash at market (shouldn't happen if risk layer present)
                cash = self.broker.getcash()
                close_px = float(self.data_close[0])
                qty = max(0.0, (cash / close_px) * 0.99)

            o = self.buy(size=qty)  # market buy
            self._open_entry_order = o

            # place stop if available
            if stop_price is not None and np.isfinite(stop_price):
                self._open_stop_order = self.sell(
                    exectype=bt.Order.Stop, price=float(stop_price), size=qty
                )

    # expose recorded series
    @property
    def results_dict(self):
        idx = pd.to_datetime(pd.Series(self._dt_index), utc=True)
        eq = pd.Series(self._value_series, index=idx, name="equity")
        pos = pd.Series(self._pos_series, index=idx, name="position")
        return {"equity": eq, "position": pos}

# ---------- Run function ----------

def run_backtest(
    candles: pd.DataFrame,
    symbol: str,
    tf: str,
    fees_bps: float = 10.0,
    slippage_bps: float = 1.0,
    equity: float = 10_000.0,
    baseline_params: Optional[BaselineParams] = None,
    risk_params: Optional[RiskParams] = None,
    out_dir: str = "artifacts/backtests",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Tuple[str, str]:
    """
    Returns (metrics_json_path, equity_curve_png_path)
    """
    assert bt is not None, "backtrader must be installed to run the backtest."

    baseline_params = baseline_params or BaselineParams()
    risk_params = risk_params or RiskParams(fees_bps=fees_bps, slippage_bps=slippage_bps)

    # Slice by date if provided
    df = candles.sort_index()
    if start is not None:
        df = df[df.index >= pd.to_datetime(start, utc=True)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end, utc=True)]
    # Ensure TM-03 builder contract: candles_df with 'ts' column + OHLCV (ASC)
    df = df.copy()
    if 'ts' not in df.columns:
        df['ts'] = pd.to_datetime(df.index, utc=True)



    # Build features (TM-03)
    if build_features_df is None:
        raise RuntimeError("build_features_df not available. Ensure TM-03 features are implemented.")
    feats = build_features_df(df, symbol=symbol, tf=tf)  # must be shift(1)

    # Compute signal & risk (TM-04)
    pos = signal_baseline(feats, baseline_params)
    ann = apply_risk_layer(df, feats, pos, equity=equity, risk=risk_params)

    # Backtrader setup
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(equity)
    cerebro.broker.setcommission(commission=(fees_bps / 10_000.0))
    try:
        cerebro.broker.set_slippage_perc(perc=(slippage_bps / 10_000.0))
    except Exception:
        pass  # older versions may not support this method

    # Data feed
    data = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, compression=60)

    cerebro.adddata(data)
    cerebro.addobserver(ValueRecorder)

    strat = cerebro.addstrategy(
        PrecomputedSignalStrategy,
        target_pos=pos,
        risk_df=ann,
        fees_perc=(fees_bps / 10_000.0),
        slippage_perc=(slippage_bps / 10_000.0),
    )

    cerebro.run()

    # Collect results
    res = strat[0].results_dict
    equity = res["equity"]
    position = res["position"]

    # Metrics
    metrics = summarize_metrics(
        equity, position=position, periods_per_year=DEFAULT_PERIODS_PER_YEAR
    )
    metrics.update(
        {
            "symbol": symbol,
            "tf": tf,
            "fees_bps": fees_bps,
            "slippage_bps": slippage_bps,
            "start": equity.index.min().isoformat() if len(equity) else None,
            "end": equity.index.max().isoformat() if len(equity) else None,
        }
    )

    # Exports
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{symbol.replace('/','-')}_{tf}_{ts}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save metrics JSON
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot equity curve
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    equity.plot(ax=ax)
    ax.set_title(f"Equity Curve — {symbol} {tf}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    png_path = os.path.join(run_dir, "equity_curve.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    return metrics_path, png_path

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="TM-05 Backtest Runner (Backtrader)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="Path to OHLCV CSV with columns: ts, open, high, low, close, volume")
    src.add_argument("--symbol", type=str, help="Symbol in DB, e.g., BTC/USDT")
    p.add_argument("--tf", type=str, default="1h", help="Timeframe (default: 1h)")
    p.add_argument("--start", type=str, help="Start datetime (e.g., 2023-01-01)")
    p.add_argument("--end", type=str, help="End datetime (e.g., 2025-01-01)")
    p.add_argument("--fees-bps", type=float, default=10.0, help="Commission in bps (default 10)")
    p.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage in bps (default 1)")
    p.add_argument("--equity", type=float, default=10_000.0, help="Starting equity (default 10k)")
    p.add_argument("--out-dir", type=str, default="artifacts/backtests", help="Output folder for artifacts")
    return p.parse_args()

def main():
    args = parse_args()

    if args.csv:
        candles = load_candles_from_csv(args.csv)
        symbol = os.path.splitext(os.path.basename(args.csv))[0]
        tf = args.tf
    else:
        tf = args.tf
        symbol = args.symbol
        candles = try_load_from_db(symbol, tf)
        if candles is None:
            raise SystemExit(
                "No DB data found. Provide --csv <path> or seed the DB for the given symbol/tf."
            )

    start = pd.to_datetime(args.start, utc=True) if args.start else None
    end = pd.to_datetime(args.end, utc=True) if args.end else None

    metrics_path, png_path = run_backtest(
        candles,
        symbol=symbol,
        tf=tf,
        fees_bps=args.fees_bps,
        slippage_bps=args.slippage_bps,
        equity=args.equity,
        out_dir=args.out_dir,
        start=start,
        end=end,
    )

    print(json.dumps({"metrics_path": metrics_path, "equity_curve_path": png_path}, indent=2))

if __name__ == "__main__":
    main()
