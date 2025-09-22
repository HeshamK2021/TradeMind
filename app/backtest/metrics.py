from __future__ import annotations
import math
from typing import Dict, Any
import pandas as pd

def bars_per_year(tf: str) -> int:
    mapping = {
        "1m": 60*24*365,
        "5m": 12*24*365,
        "15m": 4*24*365,
        "30m": 2*24*365,
        "1h": 24*365,
        "2h": 12*365,
        "4h": 6*365,
        "6h": 4*365,
        "8h": 3*365,
        "12h": 2*365,
        "1d": 365,
    }
    return mapping.get(tf, 24*365)

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min())

def compute_metrics(
    tf: str,
    equity: pd.Series,
    strat_returns: pd.Series,
    pos_next: pd.Series,
    trades_df: pd.DataFrame,
    fees_bps: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    equity0 = float(equity.iloc[0]) if len(equity) else 1.0
    total_ret = float(equity.iloc[-1] / equity0 - 1.0) if len(equity) else 0.0

    bpy = bars_per_year(tf)
    mu = float(strat_returns.mean()) if len(strat_returns) else 0.0
    sigma = float(strat_returns.std(ddof=0)) if len(strat_returns) else 0.0
    sharpe = (mu / sigma * math.sqrt(bpy)) if sigma > 0 else 0.0

    downside = strat_returns[strat_returns < 0].std(ddof=0) if len(strat_returns) else 0.0
    sortino = (mu / downside * math.sqrt(bpy)) if downside is not None and downside > 0 else 0.0

    mdd = max_drawdown(equity)
    exposure = float(pos_next.mean()) if len(pos_next) else 0.0

    if trades_df is not None and not trades_df.empty:
        win_rate = float((trades_df["ret_pct"] > 0).mean())
        gains = trades_df.loc[trades_df["ret_pct"] > 0, "ret_pct"].sum()
        losses = -trades_df.loc[trades_df["ret_pct"] < 0, "ret_pct"].sum()
        profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0)
        trades_count = int(len(trades_df))
    else:
        win_rate = 0.0
        profit_factor = 0.0
        trades_count = 0

    return {
        "trades": trades_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "pnl_pct": total_ret,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd_pct": float(mdd),
        "exposure_pct": float(exposure),
        "fees_bps": float(fees_bps),
        "slippage_bps": float(slippage_bps),
    }
