from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import os, sys, json, subprocess

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rl import ARTIFACTS_DIR

router = APIRouter(prefix="/backtest", tags=["backtest"])

class BacktestRequest(BaseModel):
    symbol: str
    tf: str
    fees_bps: float = 10.0
    slippage_bps: float = 1.0
    equity: float = 10_000  

class BacktestResponse(BaseModel):
    csv: Optional[str] = None
    png: Optional[str] = None
    note: Optional[str] = None

def _run(cmd: list[str], cwd: Path) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", str(ROOT))
    out = subprocess.run(
        cmd, cwd=str(cwd), env=env, check=True, capture_output=True, text=True
    )
    return out.stdout

@router.post("", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest) -> BacktestResponse:

    try:
        stdout = _run(
            [
                sys.executable, "-m", "scripts.baseline_windows_from_db",
                "--symbol", req.symbol, "--tf", req.tf, "--out", str(ARTIFACTS_DIR / "rl" / "runs")
            ],
            cwd=ROOT
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"baseline_windows_from_db failed: {(e.stderr or e.stdout)[-500:]}")

    runs_dir = ARTIFACTS_DIR / "rl" / "runs"
    candidates = sorted(runs_dir.glob("**/baseline_windows.csv"))
    csv_path = str(candidates[-1]) if candidates else None

    bdir = ARTIFACTS_DIR / "backtests"
    pair_prefix = f"baseline_{req.symbol.replace('/','')}_{req.tf}_"
    pngs = sorted(bdir.glob(pair_prefix + "*.png"))
    png_path = str(pngs[-1]) if pngs else None

    note = None
    if not png_path:
        note = "Baseline PNG not found; generate it with your plotting job if needed."

    return BacktestResponse(csv=csv_path, png=png_path, note=note)
