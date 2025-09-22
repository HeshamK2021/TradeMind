from __future__ import annotations



import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from app.rl.env import TradingEnv, EnvConfig
from app.rl.train import SB3EnvAdapter
from app.rl.utils import set_global_seed, SimpleScaler


def _tiny_env():
    n = 64
    ts = pd.date_range("2025-03-01", periods=n, freq="H", tz="UTC")
    X = pd.DataFrame(
        {
            "ema_5": np.linspace(1, 2, n),
            "ema_20": np.linspace(0.5, 1.5, n),
            "rsi_14": np.linspace(30, 70, n),
            "atr_14": np.linspace(0.01, 0.02, n),
            "bb_mid": np.linspace(1.0, 1.1, n),
            "bb_up": np.linspace(1.1, 1.2, n),
            "bb_dn": np.linspace(0.9, 1.0, n),
        },
        index=ts,
    )
    close = pd.Series(np.linspace(100, 102, n), index=ts, name="close", dtype=float)
    scaler = SimpleScaler()
    core = TradingEnv(
        X_df=X, close_s=close,
        config=EnvConfig(fees_bps=10.0, slippage_bps=1.0, normalize=True, fit_scaler_on_reset=True),
        scaler=scaler,
    )
    env = SB3EnvAdapter(core)
    env.reset()
    return env


def _train_once(seed: int, steps: int = 2048):
    set_global_seed(seed)
    env = _tiny_env()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=3e-4,
        n_steps=256,     
        batch_size=128,
        n_epochs=3,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=steps, progress_bar=False)

    obs, _ = env.reset()
    logits = []
    for _ in range(10):
        dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
        logits.append(dist.distribution.logits.detach().cpu().numpy().copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(int(action))
        if term or trunc:
            break
    return np.array(logits, dtype=np.float32)


