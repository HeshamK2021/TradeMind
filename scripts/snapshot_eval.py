from __future__ import annotations
import json, os, time
from pathlib import Path

from app.core.config import settings
from app.tasks.task_funcs import task_seed, task_build_features, task_backtest

def main():
    sym = os.getenv("SNAP_SYMBOL", settings.SYMBOLS.split(",")[0].strip())
    tf  = os.getenv("SNAP_TF", settings.TIMEFRAME)
    limit = int(os.getenv("SNAP_LIMIT", "1000"))
    fees = float(os.getenv("SNAP_FEES_BPS", "10"))
    slip = float(os.getenv("SNAP_SLIPPAGE_BPS", "1"))
    eq0  = float(os.getenv("SNAP_EQUITY", "10000"))

    print(f"[snapshot] seeding {sym} {tf} limit={limit}")
    r1 = task_seed(sym, tf, limit, settings.EXCHANGE)
    print(f"[snapshot] build features")
    r2 = task_build_features(sym, tf)
    print(f"[snapshot] backtest")
    r3 = task_backtest({"symbol":sym,"tf":tf,"fees_bps":fees,"slippage_bps":slip,"equity":eq0})

    out_dir = Path("artifacts/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    out = {
        "ts": int(time.time()*1000),
        "symbol": sym, "tf": tf,
        "seed": r1, "build": r2, "backtest": r3,
    }
    path = out_dir / f"snapshot_{sym.replace('/','')}_{tf}_{stamp}.json"
    path.write_text(json.dumps(out, indent=2))
    m = r3["metrics"]
    print(f"[snapshot] PnL {m['pnl_pct']:.3f} | Sharpe {m['sharpe']:.3f} | MDD {m['max_dd_pct']:.3f} | Trades {m['trades']} -> {path}")

if __name__ == "__main__":
    main()
