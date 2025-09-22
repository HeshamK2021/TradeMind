from __future__ import annotations
import sys, json, subprocess
from pathlib import Path
from typing import Optional

from app.rl.utils import json_load
from app.rl import latest_path_for
from app.rl.promote import promote_if_better

def run(cmd: list[str]):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip())
    return p.stdout

def main(symbol: str, tf: str, quick: bool = True, use_short: bool = True, short_fee_bps: float = 2.0):
    train_cmd = [
        sys.executable, "-m", "app.rl.train",
        "--symbol", symbol, "--tf", tf,
        "--fees_bps", "10", "--slippage_bps", "1",
        "--seed", "42",
        "--use-short", "true" if use_short else "false",
        "--short-fee-bps", str(short_fee_bps),
    ]
    if quick:
        train_cmd += ["--steps", "30000", "--train-span", "1500", "--test-span", "300", "--stride", "300"]
    else:
        train_cmd += ["--steps", "300000", "--train-span", "3000", "--test-span", "500", "--stride", "250"]

    print(">>> TRAIN:", " ".join(train_cmd))
    train_out = run(train_cmd)
    print(train_out)

    eval_cmd = [sys.executable, "-m", "app.rl.eval", "--symbol", symbol, "--tf", tf]
    print(">>> EVAL:", " ".join(eval_cmd))
    eval_out = run(eval_cmd)
    print(eval_out)

    bl_cmd = [sys.executable, "scripts/baseline_windows_from_db.py", "--symbol", symbol, "--tf", tf]
    print(">>> BASELINE:", " ".join(bl_cmd))
    bl_out = run(bl_cmd)
    print(bl_out)

    pair_latest = latest_path_for(symbol, tf)
    res = promote_if_better(symbol, tf, pair_latest)
    print(">>> PROMOTE:", json.dumps(res, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-quick", action="store_true")
    ap.add_argument("--use-short", action="store_true")
    ap.add_argument("--no-short", action="store_true")
    ap.add_argument("--short-fee-bps", type=float, default=2.0)
    a = ap.parse_args()
    quick = True if a.quick and not a.no-quick else (False if a.no-quick else True)
    use_short = True if a.use_short and not a.no_short else (False if a.no_short else True)
    main(a.symbol, a.tf, quick=quick, use_short=use_short, short_fee_bps=a.short_fee_bps)
