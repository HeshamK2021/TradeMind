from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/baseline", tags=["baseline"])

class BaselineEvalRequest(BaseModel):
    symbol: str
    tf: str
    script_path: Optional[str] = None

class BaselineEvalResponse(BaseModel):
    csv: str
    wins: int
    total: int
    stdout_tail: str

@router.post("/eval", response_model=BaselineEvalResponse)
def baseline_eval(req: BaselineEvalRequest) -> BaselineEvalResponse:

    root = Path(__file__).resolve().parents[2]
    script = Path(req.script_path) if req.script_path else (root / "scripts" / "baseline_windows_from_db.py")
    if not script.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script}")

    env = dict(**{**dict(**{}), **dict(**{})}) 
    env["PYTHONPATH"] = env.get("PYTHONPATH", str(root))

    cmd = [
        sys.executable, str(script),
        "--symbol", req.symbol, "--tf", req.tf,
    ]
    try:
        out = subprocess.run(cmd, cwd=str(root), env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=(e.stderr or e.stdout or "")[-800:])

    csv_path: Optional[str] = None
    wins = 0
    total = 0
    for line in out.stdout.splitlines():
        line = line.strip()
        if line.startswith("Wrote: ") and line.endswith(".csv"):
            csv_path = line.replace("Wrote:", "", 1).strip()
        elif line.startswith("Wins:"):
            try:
                frag = line.split("Wins:", 1)[1].strip()
                wins_s, total_s = frag.split("/", 1)
                wins = int(wins_s.strip())
                total = int(total_s.strip())
            except Exception:
                pass

    if not csv_path:
        raise HTTPException(status_code=500, detail="Could not parse baseline CSV path from script output")

    return BaselineEvalResponse(
        csv=csv_path,
        wins=wins,
        total=total,
        stdout_tail=out.stdout[-800:],
    )
