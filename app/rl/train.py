from __future__ import annotations
from dataclasses import asdict, dataclass
import argparse, json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

from .dataset import load_aligned, make_windows, Split
from .env import TradingEnv, EnvConfig
from .utils import (
    FEATURE_ORDER, SimpleScaler, checkpoint_path, json_dump, json_dumps,
    json_load, now_stamp, run_dir, set_global_seed
)
from . import LATEST_JSON, LATEST_DIR, LATEST_INDEX, latest_path_for

class SB3EnvAdapter(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, core_env: TradingEnv):
        super().__init__()
        self.core = core_env
        self.action_space = spaces.Discrete(self.core.action_space_n())
        low = np.full(self.core.observation_space_shape(), -np.inf, dtype=np.float32)
        high = np.full(self.core.observation_space_shape(), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            set_global_seed(seed)
        options = options or {}
        obs = self.core.reset(
            start_idx=options.get("start_idx"),
            end_idx=options.get("end_idx"),
            start_ts=options.get("start_ts"),
            end_ts=options.get("end_ts"),
        )
        return obs, {}

    def step(self, action: int):
        obs, reward, done, info = self.core.step(int(action))
        return obs, float(reward), bool(done), False, info

@dataclass
class TrainSpec:
    symbol: str
    tf: str
    fees_bps: float = 10.0
    slippage_bps: float = 1.0
    steps: int = 200_000
    train_span: int = 3000
    test_span: int = 500
    stride: int = 250
    seed: int = 42
    use_short: bool = True
    use_atr_stop: bool = False
    atr_k: float = 2.0
    atr_penalty: float = 0.0
    short_fee_bps: float = 0.0

def _slice_indices(index, start_ts, end_ts) -> Tuple[int, int]:
    si = int(index.get_indexer_for([start_ts])[0])
    ei = int(index.get_indexer_for([end_ts])[0])
    if si == -1 or ei == -1 or si >= ei:
        raise ValueError("Window timestamps not found or invalid order.")
    return si, ei

def _oos_score_with_model(model: PPO, env_test: TradingEnv) -> Dict[str, float]:
    adapter = SB3EnvAdapter(env_test)
    obs, _ = adapter.reset()
    rets = []
    s = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = adapter.step(int(action))
        rets.append(float(r)); s += float(r)
        if term or trunc:
            break
    r = np.array(rets, dtype=np.float64)
    mean = float(np.mean(r)) if r.size else 0.0
    std = float(np.std(r) + 1e-12)
    sharpe_like = mean / std if std > 0 else 0.0
    return {"sum_reward": s, "sharpe_like": sharpe_like}

def train_walk_forward(spec: TrainSpec) -> Dict[str, Any]:
    set_global_seed(spec.seed)

    X_df, close_s = load_aligned(spec.symbol, spec.tf)
    idx = X_df.index
    splits = make_windows(idx, spec.train_span, spec.test_span, spec.stride)
    if not splits:
        raise RuntimeError("Insufficient data for windows.")

    run_stamp = now_stamp()
    logs_dir = run_dir(run_stamp)
    run_meta = {
        "algo": "ppo",
        "version": "2",
        "action_space": "discrete3" if spec.use_short else "discrete2",
        "atr": {
            "mode": ("stop" if spec.use_atr_stop else ("proxy" if spec.atr_penalty > 0 else "off")),
            "atr_k": spec.atr_k,
            "lambda": spec.atr_penalty,
        },
        "short_fee_bps": spec.short_fee_bps,
        "symbol": spec.symbol, "tf": spec.tf,
        "train_spec": asdict(spec),
        "windows": [asdict(s) for s in splits],
    }
    json_dump(run_meta, logs_dir / "params.json")

    best_ckpt = None
    best_score_tuple = (-1e18, -1e18)
    best_score_dict: Dict[str, float] = {}

    for w_i, sp in enumerate(splits, 1):
        train_si, train_ei = _slice_indices(idx, sp.train_start, sp.train_end)
        test_si, test_ei = _slice_indices(idx, sp.test_start, sp.test_end)

        scaler = SimpleScaler(feature_order=FEATURE_ORDER)
        core_train = TradingEnv(
            X_df.iloc[train_si:train_ei+1],
            close_s.iloc[train_si:train_ei+1],
            config=EnvConfig(
                fees_bps=spec.fees_bps, slippage_bps=spec.slippage_bps,
                normalize=True, fit_scaler_on_reset=True,
                use_short=spec.use_short,
                use_atr_stop=spec.use_atr_stop,
                atr_k=spec.atr_k, atr_penalty_lambda=spec.atr_penalty,
                short_fee_bps=spec.short_fee_bps
            ),
            scaler=scaler,
        )
        env_train = SB3EnvAdapter(core_train)
        env_train.reset()

        ent_coef = 0.02 if spec.use_short else 0.01
        model = PPO(
            "MlpPolicy", env_train, verbose=0, seed=spec.seed,
            learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=ent_coef, device="cpu",
        )
        model.learn(total_timesteps=int(spec.steps), progress_bar=True)

        core_test = TradingEnv(
            X_df.iloc[test_si:test_ei+1],
            close_s.iloc[test_si:test_ei+1],
            config=EnvConfig(
                fees_bps=spec.fees_bps, slippage_bps=spec.slippage_bps,
                normalize=True, fit_scaler_on_reset=False,
                use_short=spec.use_short,
                use_atr_stop=spec.use_atr_stop,
                atr_k=spec.atr_k, atr_penalty_lambda=spec.atr_penalty,
                short_fee_bps=spec.short_fee_bps
            ),
            scaler=scaler,
        )
        score = _oos_score_with_model(model, core_test)

        ckpt_path_zip = checkpoint_path(spec.symbol, spec.tf, stamp=now_stamp()).with_suffix(".zip")
        model.save(str(ckpt_path_zip))
        sidecar = ckpt_path_zip.with_suffix(".scaler.json")
        json_dump(scaler.state_dict(), sidecar)

        line = {
            "window_index": w_i, "train": asdict(sp), "score": score,
            "checkpoint": str(ckpt_path_zip), "scaler": str(sidecar)
        }
        with (logs_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json_dumps(line) + "\n")

        tup = (score.get("sharpe_like", -1e18), score.get("sum_reward", -1e18))
        if tup > best_score_tuple:
            best_score_tuple = tup
            best_score_dict = score
            best_ckpt = ckpt_path_zip

    if best_ckpt is None:
        raise RuntimeError("Training completed without a checkpoint.")

    latest = {
        "best_checkpoint": str(best_ckpt),
        "algo": "ppo", "version": "2",
        "action_space": "discrete3" if spec.use_short else "discrete2",
        "atr": {
            "mode": ("stop" if spec.use_atr_stop else ("proxy" if spec.atr_penalty > 0 else "off")),
            "atr_k": spec.atr_k, "lambda": spec.atr_penalty
        },
        "short_fee_bps": spec.short_fee_bps,
        "symbol": spec.symbol, "tf": spec.tf,
        "train_spec": asdict(spec),
        "scores": best_score_dict,
        "run_dir": str(logs_dir),
    }

    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    pair_latest_path = latest_path_for(spec.symbol, spec.tf)
    json_dump(latest, pair_latest_path)

    index = {}
    if LATEST_INDEX.exists():
        try:
            index = json_load(LATEST_INDEX)
        except Exception:
            index = {}
    key = f"{spec.symbol}|{spec.tf}"
    index[key] = {
        "latest_path": str(pair_latest_path),
        "best_checkpoint": str(best_ckpt),
        "scores": best_score_dict,
        "updated_at": now_stamp(),
    }
    json_dump(index, LATEST_INDEX)

    json_dump(latest, LATEST_JSON)

    return latest

def parse_args() -> TrainSpec:
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True); p.add_argument("--tf", required=True)
    p.add_argument("--fees_bps", type=float, default=10.0)
    p.add_argument("--slippage_bps", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--train-span", dest="train_span", type=int, default=3000)
    p.add_argument("--test-span", dest="test_span", type=int, default=500)
    p.add_argument("--stride", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-short", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    p.add_argument("--use-atr-stop", type=lambda s: s.lower() in {"1","true","yes"}, default=False)
    p.add_argument("--atr-k", type=float, default=2.0)
    p.add_argument("--atr-penalty", type=float, default=0.0)
    p.add_argument("--short-fee-bps", type=float, default=0.0)
    a = p.parse_args()
    return TrainSpec(
        symbol=a.symbol, tf=a.tf, fees_bps=a.fees_bps, slippage_bps=a.slippage_bps,
        steps=a.steps, train_span=a.train_span, test_span=a.test_span, stride=a.stride,
        seed=a.seed, use_short=a.use_short, use_atr_stop=a.use_atr_stop,
        atr_k=a.atr_k, atr_penalty=a.atr_penalty, short_fee_bps=a.short_fee_bps
    )

if __name__ == "__main__":
    spec = parse_args()
    print(json.dumps(train_walk_forward(spec), indent=2))
