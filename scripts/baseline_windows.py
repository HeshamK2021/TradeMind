from pathlib import Path
import pandas as pd
from app.rl.utils import json_load
from app.rl.dataset import load_aligned, make_windows
from app.rl import LATEST_JSON

FEES_BPS = 10.0
SLIPPAGE_BPS = 1.0
FLIP_UNIT = (FEES_BPS + SLIPPAGE_BPS) / 10_000.0

def equity_from_pos(pos: pd.Series, close: pd.Series) -> pd.Series:
    ret = close.pct_change().dropna()
    pos = pos.reindex(ret.index).fillna(0).astype(int)
    flip = (pos != pos.shift(1).fillna(0)).astype(int) * FLIP_UNIT
    eq = [1.0]
    for p, r, f in zip(pos.values, ret.values, flip.values):
        eq.append(eq[-1] * (1.0 + p*r - f))
    return pd.Series(eq, index=[ret.index[0]] + list(ret.index), name="equity")

def baseline_positions(symbol: str, tf: str, test_index: pd.DatetimeIndex) -> pd.Series:

    return pd.Series(0, index=test_index, dtype=int)

def main():
    latest = json_load(LATEST_JSON)
    symbol, tf = latest["symbol"], latest["tf"]
    spec = latest.get("train_spec", {})
    train_span = spec.get("train_span", 3000)
    test_span = spec.get("test_span", 500)
    stride = spec.get("stride", 250)

    X, c = load_aligned(symbol, tf)
    splits = make_windows(X.index, train_span, test_span, stride)

    rows = []
    for i, sp in enumerate(splits, 1):
        test_idx = X.loc[sp.test_start:sp.test_end].index
        close_test = c.loc[test_idx]
        pos = baseline_positions(symbol, tf, test_idx)

        eq = equity_from_pos(pos, close_test)
        equity_final = float(eq.iloc[-1])
        pnl = float((close_test.pct_change().reindex(pos.index).fillna(0) * pos).sum())

        rows.append(dict(
            window=i,
            test_start=sp.test_start.isoformat(),
            test_end=sp.test_end.isoformat(),
            baseline_equity_final=equity_final,
            baseline_pnl=pnl,
        ))

    out_dir = Path(latest.get("run_dir", "artifacts/rl/runs"))
    out = out_dir / "baseline_windows.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("Wrote:", out)

if __name__ == "__main__":
    main()
