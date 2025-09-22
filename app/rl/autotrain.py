from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any

from .train import TrainSpec, train_walk_forward
from .eval import evaluate_oos
from .utils import json_load, json_dump
from . import LATEST_JSON

def _load_oos_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    return json_load(p) if p.exists() else {}

def _is_better(new_m: Dict[str, Any], old_m: Dict[str, Any]) -> bool:
    n = float(new_m.get("oos_equity_final_mean", 0.0))
    o = float(old_m.get("oos_equity_final_mean", 0.0))
    if n > o + 1e-6: 
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True); ap.add_argument("--tf", required=True)
    ap.add_argument("--fees_bps", type=float, default=10.0)
    ap.add_argument("--slippage_bps", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--train-span", dest="train_span", type=int, default=3000)
    ap.add_argument("--test-span", dest="test_span", type=int, default=500)
    ap.add_argument("--stride", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-short", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    ap.add_argument("--use-atr-stop", type=lambda s: s.lower() in {"1","true","yes"}, default=False)
    ap.add_argument("--atr-k", type=float, default=2.0)
    ap.add_argument("--atr-penalty", type=float, default=0.0)
    ap.add_argument("--short-fee-bps", type=float, default=0.0)
    ap.add_argument("--promote-if-better", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    a = ap.parse_args()

    old_latest = json_load(LATEST_JSON) if Path(LATEST_JSON).exists() else {}
    old_metrics = _load_oos_metrics(Path(old_latest.get("run_dir","."))) if old_latest else {}

    spec = TrainSpec(
        symbol=a.symbol, tf=a.tf, fees_bps=a.fees_bps, slippage_bps=a.slippage_bps,
        steps=a.steps, train_span=a.train_span, test_span=a.test_span, stride=a.stride,
        seed=a.seed, use_short=a.use_short, use_atr_stop=a.use_atr_stop,
        atr_k=a.atr_k, atr_penalty=a.atr_penalty, short_fee_bps=a.short_fee_bps
    )
    new_latest = train_walk_forward(spec)

    _ = evaluate_oos(
        symbol=a.symbol, tf=a.tf,
        fees_bps=a.fees_bps, slippage_bps=a.slippage_bps,
        use_short=a.use_short, use_atr_stop=a.use_atr_stop, atr_k=a.atr_k, atr_penalty=a.atr_penalty,
        short_fee_bps=a.short_fee_bps
    )
    new_metrics = _load_oos_metrics(Path(new_latest["run_dir"]))

    decided_latest = old_latest
    did_promote = False
    if a.promote_if_better and (not old_latest or _is_better(new_metrics, old_metrics)):
        json_dump(new_latest, LATEST_JSON)
        decided_latest = new_latest
        did_promote = True

    print(json.dumps({
        "promoted": did_promote,
        "latest_json": decided_latest,
        "new_metrics": new_metrics,
        "old_metrics": old_metrics,
    }, indent=2))

if __name__ == "__main__":
    main()
