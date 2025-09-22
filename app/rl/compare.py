from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import pandas as pd

def _read_json_safe(p: Path) -> Dict[str, Any]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _read_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists() and p.stat().st_size > 0:
            return pd.read_csv(p)
    except Exception:
        pass
    return None

def _metrics_from_run_dir(run_dir: Path) -> Dict[str, Any]:

    out: Dict[str, Any] = {}

    mj = run_dir / "metrics.json"
    m = _read_json_safe(mj)
    if m:
        for k in ("sharpe", "max_drawdown", "cagr", "equity_final", "final_equity", "win_rate"):
            if k in m:
                out[k] = m[k]

    rl_csv = run_dir / "rl_windows.csv"
    df = _read_csv_safe(rl_csv)
    if df is not None and not df.empty and "equity_final" in df.columns:
        out.setdefault("mean_equity_final", float(pd.to_numeric(df["equity_final"], errors="coerce").dropna().mean()))
        out.setdefault("num_windows", int(len(df)))
    return out

def _overall_score(metrics: Dict[str, Any]) -> float:

    for key in ("sharpe", "mean_equity_final", "equity_final", "final_equity"):
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except Exception:
                continue
    return float("-inf")

def compare_run_dir_vs_latest(new_run_dir: Path | str, latest_meta_path: Path | str | None) -> Tuple[bool, Dict[str, Any]]:

    new_run_dir = Path(new_run_dir)
    old_meta: Dict[str, Any] = {}
    if latest_meta_path:
        latest_meta_path = Path(latest_meta_path)
        old_meta = _read_json_safe(latest_meta_path)

    old_run_dir = Path(old_meta.get("run_dir", "")) if old_meta else None
    new_m = _metrics_from_run_dir(new_run_dir)
    new_m["run_dir"] = str(new_run_dir)

    if not old_meta or not old_run_dir or not old_run_dir.exists():
        return True, {
            "new": new_m,
            "old": {"run_dir": None},
            "decision": "promote",
            "reason": "no_old_latest",
        }

    old_m = _metrics_from_run_dir(old_run_dir)
    old_m["run_dir"] = str(old_run_dir)

    s_new = _overall_score(new_m)
    s_old = _overall_score(old_m)

    if s_new > s_old:
        decision, reason = "promote", f"score_improved: {s_old:.4f} → {s_new:.4f}"
        better = True
    elif s_new < s_old:
        decision, reason = "hold", f"score_worse: {s_old:.4f} vs {s_new:.4f}"
        better = False
    else:
        md_new = float(new_m.get("max_drawdown", 0.0))
        md_old = float(old_m.get("max_drawdown", 0.0))
        if md_new > md_old:
            decision, reason = "promote", f"score_tie; better_drawdown: {md_old} → {md_new}"
            better = True
        else:
            decision, reason = "hold", "score_tie; drawdown_not_better"
            better = False

    details = {
        "new": new_m,
        "old": old_m,
        "decision": decision,
        "reason": reason,
    }
    return better, details
