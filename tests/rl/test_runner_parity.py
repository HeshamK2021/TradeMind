from __future__ import annotations


import numpy as np
import pandas as pd

from app.rl.env import TradingEnv, EnvConfig


def _make_tiny_prices_and_feats():
    ts = pd.date_range("2025-02-01", periods=7, freq="H", tz="UTC")
    close = pd.Series([100, 101, 99, 100, 105, 104, 106], index=ts, name="close", dtype=float)
    X = pd.DataFrame(
        {
            "ema_5": np.linspace(1, 2, len(ts)),
            "ema_20": np.linspace(0.5, 1.5, len(ts)),
            "rsi_14": np.linspace(30, 70, len(ts)),
            "atr_14": np.linspace(0.01, 0.02, len(ts)),
            "bb_mid": np.linspace(1.0, 1.1, len(ts)),
            "bb_up": np.linspace(1.1, 1.2, len(ts)),
            "bb_dn": np.linspace(0.9, 1.0, len(ts)),
        },
        index=ts,
    )
    return X, close


def _reference_equity(close: pd.Series, actions: list[int], fees_bps=10.0, slippage_bps=1.0):

    assert len(actions) == len(close) - 1, "Actions must have len == len(close)-1"
    pos_prev = 0
    flip_cost_unit = (fees_bps + slippage_bps) / 10_000.0
    eq = [1.0]
    for t in range(len(actions)):
        pos_t = int(actions[t])
        flip_cost = flip_cost_unit if pos_t != pos_prev else 0.0
        ret = (close.iloc[t + 1] / close.iloc[t]) - 1.0
        eq.append(eq[-1] * (1.0 + (pos_t * ret) - flip_cost))
        pos_prev = pos_t
    return pd.Series(eq, index=close.index[: len(eq)], name="equity")


