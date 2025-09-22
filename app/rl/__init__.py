
from __future__ import annotations
import os
from pathlib import Path

__all__ = [
    "ARTIFACTS_DIR",
    "RL_DIR",
    "CHECKPOINTS_DIR",
    "RUNS_DIR",
    "LATEST_DIR",
    "LATEST_INDEX",
    "LATEST_JSON",          
    "latest_path_for",      
    "PACKAGE_VERSION",
]

PACKAGE_VERSION: str = "1.1.0"

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts")).resolve()

RL_DIR = ARTIFACTS_DIR / "rl"
CHECKPOINTS_DIR = RL_DIR / "checkpoints"
RUNS_DIR = RL_DIR / "runs"

LATEST_DIR = RL_DIR / "latest"                   
LATEST_INDEX = RL_DIR / "latest_index.json"     

LATEST_JSON = RL_DIR / "latest.json"

for _p in (CHECKPOINTS_DIR, RUNS_DIR, LATEST_DIR):
    _p.mkdir(parents=True, exist_ok=True)

def latest_path_for(symbol: str, tf: str) -> Path:

    key = f"{symbol.replace('/','')}_{tf}"
    return LATEST_DIR / f"{key}.json"
