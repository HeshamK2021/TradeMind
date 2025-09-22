from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import sys as _sys

from app.data.session import session_scope
from app.backtest.runner import BtParams, run_backtest_db

def main():
    p = argparse.ArgumentParser(description="Run backtest from DB candles + features.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True)
    p.add_argument("--fees_bps", type=float, default=10.0)
    p.add_argument("--slippage_bps", type=float, default=1.0)
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--out", default="artifacts/backtests")
    args = p.parse_args()

    with session_scope() as s:
        run_id, metrics, png_path = run_backtest_db(
            s,
            BtParams(
                symbol=args.symbol, tf=args.tf,
                fees_bps=args.fees_bps, slippage_bps=args.slippage_bps,
                equity0=args.equity, start=args.start, end=args.end,
                out_dir=args.out,
            ),
        )
    print(json.dumps({"run_id": run_id, "metrics": metrics, "png": png_path}, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}", file=_sys.stderr)
        _sys.exit(1)
