from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from .utils import json_load, json_dump
from . import latest_path_for

def _score_tuple(js: Dict[str, Any]) -> tuple[float, float]:
    s = js.get("scores", {})
    return float(s.get("sharpe_like", -1e18)), float(s.get("sum_reward", -1e18))

def promote_if_better(symbol: str, tf: str, candidate_latest_path: Path) -> Dict[str, Any]:

    pair_latest_path = latest_path_for(symbol, tf)
    cand = json_load(candidate_latest_path)
    if not cand:
        return {"promoted": False, "reason": "candidate_latest_missing", "current": str(pair_latest_path), "candidate": str(candidate_latest_path)}

    cand_score = _score_tuple(cand)

    if pair_latest_path.exists():
        cur = json_load(pair_latest_path)
        cur_score = _score_tuple(cur)
        if cand_score > cur_score:
            json_dump(cand, pair_latest_path)
            return {"promoted": True, "reason": f"better_score {cand_score} > {cur_score}", "current": str(pair_latest_path), "candidate": str(candidate_latest_path)}
        else:
            return {"promoted": False, "reason": f"not_better {cand_score} <= {cur_score}", "current": str(pair_latest_path), "candidate": str(candidate_latest_path)}
    else:
        json_dump(cand, pair_latest_path)
        return {"promoted": True, "reason": "first_pair_latest", "current": str(pair_latest_path), "candidate": str(candidate_latest_path)}
