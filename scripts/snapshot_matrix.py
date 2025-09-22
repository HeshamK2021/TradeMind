from __future__ import annotations
import os, json, time
from pathlib import Path

from app.core.config import settings
from app.tasks.task_funcs import task_seed, task_build_features, task_backtest

def main():
    symbols = [s.strip() for s in (os.getenv("MATRIX_SYMBOLS") or settings.SYMBOLS or "BTC/USDT").split(",") if s.strip()]
    tfs     = [t.strip() for t in (os.getenv("MATRIX_TFS")     or settings.TIMEFRAME or "1h").split(",") if t.strip()]
    limit   = int(os.getenv("MATRIX_LIMIT","1000"))
    fees    = float(os.getenv("MATRIX_FEES_BPS","10"))
    slip    = float(os.getenv("MATRIX_SLIPPAGE_BPS","1"))
    eq0     = float(os.getenv("MATRIX_EQUITY","10000"))

    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for sym in symbols:
        for tf in tfs:
            print(f"[matrix] {sym} {tf}: seed/build/backtest…")
            r1 = task_seed(sym, tf, limit, settings.EXCHANGE)
            r2 = task_build_features(sym, tf)
            r3 = task_backtest({"symbol": sym, "tf": tf, "fees_bps": fees, "slippage_bps": slip, "equity": eq0})
            out = {"ts": int(time.time()*1000), "symbol": sym, "tf": tf, "seed": r1, "build": r2, "backtest": r3}
            path = out_dir / f"snapshot_{sym.replace('/','')}_{tf}_{stamp}.json"
            path.write_text(json.dumps(out, indent=2))
            m = r3["metrics"]
            print(f"[matrix] {sym} {tf}: PnL {m['pnl_pct']:.3f} | Sharpe {m['sharpe']:.3f} | MDD {m['max_dd_pct']:.3f} | Trades {m['trades']} -> {path}")

if __name__ == "__main__":
    main()
