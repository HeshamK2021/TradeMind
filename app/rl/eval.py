from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import argparse, json
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .dataset import load_aligned, make_windows
from .env import TradingEnv, EnvConfig
from .utils import json_dump, json_dumps, json_load, now_stamp, run_dir, plot_equity_png
from . import latest_path_for  

def _load_model_and_scaler(ckpt_path: Path) -> Tuple[PPO, Dict[str, Any]]:
    model = PPO.load(str(ckpt_path), device="cpu")
    scaler_state = json_load(ckpt_path.with_suffix(".scaler.json"))
    return model, scaler_state

def _make_env(
    X: pd.DataFrame,
    close: pd.Series,
    scaler_state: Dict[str, Any],
    fees_bps: float,
    slippage_bps: float,
    use_short: bool,
    use_atr_stop: bool,
    atr_k: float,
    atr_penalty: float,
    short_fee_bps: float,
) -> TradingEnv:
    from .utils import SimpleScaler, FEATURE_ORDER
    sc = SimpleScaler(feature_order=FEATURE_ORDER)
    sc.load_state_dict(scaler_state)
    env = TradingEnv(
        X, close,
        config=EnvConfig(
            fees_bps=fees_bps, slippage_bps=slippage_bps,
            normalize=True, fit_scaler_on_reset=False,
            use_short=use_short, use_atr_stop=use_atr_stop, atr_k=atr_k,
            atr_penalty_lambda=atr_penalty, short_fee_bps=short_fee_bps
        ),
        scaler=sc
    )
    return env

def _roll_equity(model: PPO, env: TradingEnv) -> Tuple[pd.Series, Dict[str, Any]]:
    obs = env.reset()
    equity = [1.0]
    rewards = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = env.step(int(action))
        rewards.append(float(r))
        equity.append(equity[-1] * (1.0 + float(r)))
        if done:
            break
    eq_index = env._feats_df.index[env._i0:env._i1+1]
    eq = pd.Series(equity, index=[eq_index[0]] + list(eq_index[1:]))
    r = np.array(rewards, dtype=np.float64)
    metrics = {
        "sum_reward": float(r.sum()),
        "avg_reward": float(r.mean() if r.size else 0.0),
        "std_reward": float(r.std() if r.size else 0.0),
        "sharpe_like": float((r.mean() / (r.std() + 1e-12)) if r.size else 0.0),
        "n_steps": int(r.size),
    }
    return eq, metrics

def evaluate_oos(
    symbol: str,
    tf: str,
    ckpt_path: Optional[str] = None,
    fees_bps: float = 10.0,
    slippage_bps: float = 1.0,
    use_short: Optional[bool] = None,
    use_atr_stop: Optional[bool] = None,
    atr_k: Optional[float] = None,
    atr_penalty: Optional[float] = None,
    short_fee_bps: Optional[float] = None,
) -> Dict[str, Any]:

    pair_latest = json_load(latest_path_for(symbol, tf))

    if ckpt_path is None:
        ckpt_path = pair_latest["best_checkpoint"]
    model, scaler = _load_model_and_scaler(Path(ckpt_path))

    if use_short is None:
        use_short = (pair_latest.get("action_space", "discrete2") == "discrete3")
    atr_meta = pair_latest.get("atr", {"mode": "off", "atr_k": 2.0, "lambda": 0.0})
    if use_atr_stop is None:
        use_atr_stop = (atr_meta.get("mode") == "stop")
    if atr_k is None:
        atr_k = float(atr_meta.get("atr_k", 2.0))
    if atr_penalty is None:
        atr_penalty = float(atr_meta.get("lambda", 0.0))
    if short_fee_bps is None:
        short_fee_bps = float(pair_latest.get("short_fee_bps", 0.0))

    X, close = load_aligned(symbol, tf)
    spec = pair_latest.get("train_spec", {})
    train_span = int(spec.get("train_span", 3000))
    test_span = int(spec.get("test_span", 500))
    stride = int(spec.get("stride", 250))
    splits = make_windows(X.index, train_span, test_span, stride)
    if not splits:
        return {
            "csv": "",
            "metrics_json": "",
            "png": "",
            "note": "No walk-forward windows (insufficient data)."
        }
    rows: List[Dict[str, Any]] = []
    equities: List[pd.Series] = []
    for i, sp in enumerate(splits, 1):
        test_X = X.loc[sp.test_start:sp.test_end]
        test_c = close.loc[sp.test_start:sp.test_end]
        env = _make_env(
            test_X, test_c, scaler,
            fees_bps, slippage_bps,
            bool(use_short), bool(use_atr_stop), float(atr_k), float(atr_penalty), float(short_fee_bps),
        )
        eq, m = _roll_equity(model, env)
        equities.append(eq)
        rows.append({
            "window": i,
            "train_start": sp.train_start.isoformat(),
            "train_end": sp.train_end.isoformat(),
            "test_start": sp.test_start.isoformat(),
            "test_end": sp.test_end.isoformat(),
            "equity_final": float(eq.iloc[-1]),
            **m
        })

    run_dir = Path(pair_latest.get("run_dir", "artifacts/rl/runs"))
    rl_csv = run_dir / "rl_windows.csv"
    pd.DataFrame(rows).to_csv(rl_csv, index=False)

    finals = np.array([r["equity_final"] for r in rows], dtype=np.float64)
    agg = {
        "oos_equity_final_mean": float(finals.mean()) if finals.size else 0.0,
        "oos_equity_final_median": float(np.median(finals)) if finals.size else 0.0,
        "n_windows": len(rows),
    }
    json_dump(agg, run_dir / "metrics.json")

    eq_pieces = []
    carry = 1.0
    for eq in equities:
        scaled = eq * carry
        carry = float(scaled.iloc[-1])
        eq_pieces.append(scaled)
        eq_pieces.append(pd.Series([np.nan], index=[scaled.index[-1] + pd.Timedelta(seconds=1)]))
    eq_all = pd.concat(eq_pieces)
    if not eq_pieces:
        raise RuntimeError(
            f"No equity pieces produced for {symbol} {tf}. "
            "Likely no valid test slices or the checkpoint/run_dir is stale."
        )
    png = plot_equity_png(
        eq_all,
        title=f"RL OOS — {symbol} {tf}",
        prefix=f"rl_{symbol.replace('/','')}_{tf}"
    )
    return {"csv": str(rl_csv), "metrics_json": str(run_dir / "metrics.json"), "png": png}

def parse_args():
    ap = argparse.ArgumentParser(description="Deterministic OOS evaluation for a (symbol, tf) RL checkpoint.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--fees_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=1.0)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--use-short", type=lambda s: s.lower() in {"1","true","yes"}, default=None)
    ap.add_argument("--use-atr-stop", type=lambda s: s.lower() in {"1","true","yes"}, default=None)
    ap.add_argument("--atr-k", type=float, default=None)
    ap.add_argument("--atr-penalty", type=float, default=None)
    ap.add_argument("--short-fee-bps", type=float, default=None)
    return ap.parse_args()

if __name__ == "__main__":
    a = parse_args()
    out = evaluate_oos(
        symbol=a.symbol, tf=a.tf, ckpt_path=a.checkpoint,
        fees_bps=a.fees_bps, slippage_bps=a.slippage_bps,
        use_short=a.use_short, use_atr_stop=a.use_atr_stop, atr_k=a.atr_k, atr_penalty=a.atr_penalty,
        short_fee_bps=a.short_fee_bps
    )
    print(json.dumps(out, indent=2))
