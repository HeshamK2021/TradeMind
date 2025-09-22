from __future__ import annotations
import os, json, datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as _np

FEATURE_ORDER = [
    "ema_5", "ema_20", "ema_spread", "rsi_14",
    "atr_14", "bb_width", "ema5_minus_bbmid",
]

class SimpleScaler:

    def __init__(self, feature_order: List[str]):
        self.feature_order = list(feature_order)
        self.mean: Optional[_np.ndarray] = None
        self.std: Optional[_np.ndarray] = None

    def fit(self, X: _np.ndarray) -> None:
        if X.ndim == 1:
            X = X[None, :]
        self.mean = _np.mean(X, axis=0)
        self.std = _np.std(X, axis=0) + 1e-12

    def transform(self, X: _np.ndarray) -> _np.ndarray:
        if self.mean is None or self.std is None:
            return X
        return (X - self.mean) / self.std

    def state_dict(self) -> Dict[str, Any]:
        return {
            "feature_order": self.feature_order,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
        }

    def load_state_dict(self, st: Dict[str, Any]) -> None:
        self.feature_order = list(st.get("feature_order", self.feature_order))
        m = st.get("mean"); s = st.get("std")
        self.mean = _np.array(m, dtype=_np.float32) if m is not None else None
        self.std = _np.array(s, dtype=_np.float32) if s is not None else None


ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
RL_DIR = ARTIFACTS_DIR / "rl"
CHECKPOINTS_DIR = RL_DIR / "checkpoints"
RUNS_DIR = RL_DIR / "runs"

LATEST_JSON = RL_DIR / "latest.json"

LATEST_DIR = RL_DIR / "latest"
LATEST_INDEX = LATEST_DIR / "index.json"


def ensure_dirs() -> None:

    for d in (ARTIFACTS_DIR, RL_DIR, CHECKPOINTS_DIR, RUNS_DIR, LATEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run_dir(stamp: str) -> Path:
    ensure_dirs()
    d = RUNS_DIR / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(symbol: str, tf: str, stamp: str) -> Path:
    ensure_dirs()
    sym = symbol.replace("/", "")
    return CHECKPOINTS_DIR / f"ckpt_{sym}_{tf}_{stamp}"


def now_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _json_default(o):
    if isinstance(o, (_dt.datetime, _dt.date, _dt.time)):
        return o.isoformat()
    if isinstance(o, (_np.integer, _np.floating, _np.bool_)):
        return _np.asarray(o).item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    return str(o)


def json_dump(obj: Any, path: os.PathLike | str, indent: int = 2) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)


def json_dumps(obj: Any, indent: int | None = None) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=_json_default)


def json_load(path: os.PathLike | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_key(symbol: str, tf: str) -> str:
    return f"{symbol.replace('/','')}_{tf}"


def latest_path_for(symbol: str, tf: str) -> Path:

    ensure_dirs()
    return LATEST_DIR / f"{latest_key(symbol, tf)}.json"


def write_pair_latest(latest_meta: Dict[str, Any]) -> Path:

    ensure_dirs()
    symbol = latest_meta.get("symbol")
    tf = latest_meta.get("tf")
    if not symbol or not tf:
        raise ValueError("write_pair_latest: latest_meta must include 'symbol' and 'tf'.")

    p = latest_path_for(symbol, tf)
    json_dump(latest_meta, p)

    idx: Dict[str, Any] = {}
    if LATEST_INDEX.exists():
        try:
            idx = json_load(LATEST_INDEX)
        except Exception:
            idx = {}

    k = latest_key(symbol, tf)
    idx[k] = {
        "path": str(p),
        "symbol": symbol,
        "tf": tf,
        "mtime": _dt.datetime.utcnow().isoformat() + "Z",
    }
    json_dump(idx, LATEST_INDEX)
    return p


def plot_equity_png(equity: "pd.Series", title: str, prefix: str) -> str:
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        if matplotlib.get_backend().lower() != "agg":
            matplotlib.use("Agg")
    except Exception:
        pass

    bdir = ARTIFACTS_DIR / "backtests"
    bdir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{now_stamp()}.png"
    out = bdir / fname

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index, equity.values, drawstyle="steps-post")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)



def set_global_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
