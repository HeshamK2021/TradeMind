from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from app.rl.dataset import load_aligned, make_windows
from app.backtest.runner import run_backtest_db, BtParams
from app.data.session import session_scope

FEES_BPS = 10.0
SLIPPAGE_BPS = 1.0

def ts_ms(pd_ts) -> int: 
    return int(pd_ts.timestamp() * 1000)

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def main():
    ap = argparse.ArgumentParser(description="Compute baseline window metrics and a generate OOS PNG.")
    ap.add_argument("--symbol", required=True, help="e.g., ETH/USDT")
    ap.add_argument("--tf", required=True, help="e.g., 4h")
    ap.add_argument("--out", default="artifacts/rl/runs", help="directory for baseline_windows.csv (default: artifacts/rl/runs)")
    ap.add_argument("--fees_bps", type=float, default=FEES_BPS)
    ap.add_argument("--slippage_bps", type=float, default=SLIPPAGE_BPS)
    ap.add_argument("--train_span", type=int, default=3000)
    ap.add_argument("--test_span", type=int, default=500)
    ap.add_argument("--stride", type=int, default=250)
    args = ap.parse_args()

    symbol, tf = args.symbol, args.tf
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, _ = load_aligned(symbol, tf)
    if X.empty:
        print(f"[error] No aligned features for {symbol} {tf}. Build features first.")
        sys.exit(2)

    splits = make_windows(X.index, args.train_span, args.test_span, args.stride)
    if not splits:
        print("[error] No windows produced (check spans/stride and data length).")
        sys.exit(2)

    rows = []
    oos_starts = []
    oos_ends = []

    with session_scope() as s:
        for i, sp in enumerate(splits, 1):
            params = BtParams(
                symbol=symbol, tf=tf,
                fees_bps=args.fees_bps, slippage_bps=args.slippage_bps,
                start=ts_ms(sp.test_start),
                end=ts_ms(sp.test_end)  
            )
            _run_id, metrics, _png = run_backtest_db(s, params)
            rows.append({
                "window": i,
                "test_start": sp.test_start.isoformat(),
                "test_end": sp.test_end.isoformat(),
                "baseline_equity_final": float(1.0 + metrics["pnl_pct"]),
                "baseline_sharpe": float(metrics["sharpe"]),
                "baseline_win_rate": float(metrics["win_rate"]),
                "baseline_profit_factor": float(metrics["profit_factor"]),
                "baseline_trades": int(metrics.get("trades", 0)),
                "baseline_max_dd_pct": float(metrics.get("max_dd_pct", 0.0)),
            })
            oos_starts.append(sp.test_start)
            oos_ends.append(sp.test_end)

    out_csv = out_dir / "baseline_windows.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    min_oos = min(oos_starts)
    max_oos = max(oos_ends)

    backtests_dir = Path("artifacts") / "backtests"
    backtests_dir.mkdir(parents=True, exist_ok=True)

    with session_scope() as s:
        params_full = BtParams(
            symbol=symbol, tf=tf,
            fees_bps=args.fees_bps, slippage_bps=args.slippage_bps,
            start=ts_ms(min_oos), end=ts_ms(max_oos)
        )
        _run_id, _metrics, png_path = run_backtest_db(s, params_full)

    stamped = f"baseline_{symbol.replace('/','')}_{tf}_{_stamp()}.png"
    dst = backtests_dir / stamped
    try:
        src = Path(png_path)
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
    except Exception as e:
        print(f"[warn] Could not copy equity PNG ({e}); source path: {png_path}")
    else:
        print("Baseline OOS PNG:", dst)

    rl_csv = out_dir / "rl_windows.csv"
    if rl_csv.exists():
        rl = pd.read_csv(rl_csv)
        bl = pd.read_csv(out_csv)
        key = ["window", "test_start", "test_end"]
        df = rl.merge(bl, on=key, how="inner")
        df["rl_beats_baseline"] = df["equity_final"] > df["baseline_equity_final"]
        print(df[["window","equity_final","baseline_equity_final","rl_beats_baseline"]].to_string(index=False))
        print(f"Wins: {int(df['rl_beats_baseline'].sum())} / {len(df)}")
    else:
        print(f"(Note) RL CSV not found at {rl_csv}. Generate it first, then re-run to compare.")

if __name__ == "__main__":
    main()
