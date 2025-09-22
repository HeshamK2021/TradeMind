from __future__ import annotations
import csv, os, time
from pathlib import Path
from typing import Iterable
from typing import Dict, Any

from app.core.config import settings
from app.data.session import session_scope
from app.backtest.runner import BtParams, run_backtest_db
from app.backtest.strategy_rules import BaselineParams

OUT = Path("artifacts/eval")
OUT.mkdir(parents=True, exist_ok=True)


def run_case(symbol: str, tf: str, use_rsi: bool, use_bb: bool) -> Dict[str, Any]:

    p = BtParams(
        symbol=symbol,
        tf=tf,
        fees_bps=10.0,
        slippage_bps=1.0,
        equity0=10_000.0,
        out_dir="artifacts/backtests",
    )
    bp = BaselineParams(use_rsi=use_rsi, use_bb=use_bb)

    with session_scope() as s:
        run_id, metrics, png_path = run_backtest_db(s, p, params=bp)

    return {"run_id": run_id, "metrics": metrics, "equity_curve_png": png_path}

def main():
    sym = os.getenv("ABL_SYMBOL", settings.SYMBOLS.split(",")[0].strip())
    tf  = os.getenv("ABL_TF", settings.TIMEFRAME)
    grid = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = OUT / f"ablate_{sym.replace('/','')}_{tf}_{stamp}.csv"

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol","tf","use_rsi","use_bb","pnl_pct","sharpe","max_dd_pct","trades","run_id","equity_png"])

    for use_rsi, use_bb in grid:
        res = run_case(sym, tf, use_rsi, use_bb)
        m = res["metrics"]
        row = [sym, tf, use_rsi, use_bb, m["pnl_pct"], m["sharpe"], m["max_dd_pct"], m["trades"], res["run_id"], res["equity_curve_png"]]
        with out_csv.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[ablate] use_rsi={use_rsi} use_bb={use_bb} -> PnL {m['pnl_pct']:.3f} Sharpe {m['sharpe']:.3f} MDD {m['max_dd_pct']:.3f}")

    print(f"[ablate] wrote {out_csv}")

if __name__ == "__main__":
    main()
