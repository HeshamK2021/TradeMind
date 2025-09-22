
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
import pandas as pd

from app.rl.train import TrainSpec, train_walk_forward
from app.rl.eval import evaluate_oos
from app.rl import latest_path_for, write_latest_for
from app.rl.utils import json_load, json_dump, RUNS_DIR, CHECKPOINTS_DIR

def _pair_key(symbol: str, tf: str) -> str:
    return f"{symbol.replace('/','')}_{tf}"

def _read_windows_equity_mean(rl_windows_csv: Path) -> Optional[float]:
    if not rl_windows_csv.exists() or rl_windows_csv.stat().st_size == 0:
        return None
    df = pd.read_csv(rl_windows_csv)
    if "equity_final" in df.columns and len(df):
        return float(df["equity_final"].mean())
    return None

def _read_sharpe_like_from_latest(latest_json: Path) -> Optional[float]:
    try:
        j = json_load(latest_json)
        s = j.get("scores") or {}
        v = s.get("sharpe_like")
        return float(v) if v is not None else None
    except Exception:
        return None

def _compare_new_vs_old(
    new_run_dir: Path,
    old_latest_json: Optional[Path],
) -> Tuple[bool, Dict[str, Any]]:

    details: Dict[str, Any] = {}

    new_csv = new_run_dir / "rl_windows.csv"
    new_mean_eq = _read_windows_equity_mean(new_csv)
    details["new_windows_csv"] = str(new_csv)
    details["new_mean_equity_final"] = new_mean_eq

    old_mean_eq: Optional[float] = None
    old_sharpe: Optional[float] = None

    if old_latest_json and old_latest_json.exists():
        old_latest = json_load(old_latest_json)
        details["old_latest"] = str(old_latest_json)
        old_run_dir = Path(old_latest.get("run_dir", ""))
        if old_run_dir:
            old_csv = old_run_dir and Path(old_run_dir) / "rl_windows.csv"
            if old_csv and old_csv.exists():
                old_mean_eq = _read_windows_equity_mean(old_csv)
                details["old_windows_csv"] = str(old_csv)
                details["old_mean_equity_final"] = old_mean_eq
        old_sharpe = _read_sharpe_like_from_latest(old_latest_json)
        details["old_sharpe_like"] = old_sharpe

    if (new_mean_eq is not None) and (old_mean_eq is not None):
        better = new_mean_eq > old_mean_eq
        details["decision"] = "compare_equity_final_mean"
        details["result"] = "promote" if better else "hold"
        return better, details

    new_metrics_json = new_run_dir / "metrics.json"
    new_sharpe: Optional[float] = None
    if new_metrics_json.exists():
        try:
            new_m = json_load(new_metrics_json)
            new_sharpe = float(new_m.get("sharpe_like")) if "sharpe_like" in new_m else None
        except Exception:
            new_sharpe = None
    details["new_sharpe_like"] = new_sharpe

    if (new_sharpe is not None) and (old_sharpe is not None):
        better = new_sharpe > old_sharpe
        details["decision"] = "compare_sharpe_like"
        details["result"] = "promote" if better else "hold"
        return better, details

    details["decision"] = "no_baseline; promote-by-default"
    return True, details

def _retention_prune(symbol: str, tf: str, keep: int) -> Dict[str, Any]:

    prefix = f"ckpt_{symbol.replace('/','')}_{tf}_"
    files = sorted([p for p in CHECKPOINTS_DIR.glob(f"{prefix}*.zip")], key=lambda p: p.stat().st_mtime, reverse=True)
    deleted = []
    if len(files) > keep:
        for p in files[keep:]:
            try:
                side = p.with_suffix(".scaler.json")
                if side.exists():
                    side.unlink()
                p.unlink()
                deleted.append(str(p))
            except Exception:
                pass
    return {"kept": len(files[:keep]), "deleted": deleted}

def _load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def run_for_pairs(
    cfg: Dict[str, Any],
    only: Optional[List[Tuple[str, str]]] = None,
    promote_if_better: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    pairs = cfg.get("pairs", [])
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    retention_cfg = cfg.get("retention", {})
    keep_n = int(retention_cfg.get("max_checkpoints_per_pair", 10))

    results: List[Dict[str, Any]] = []

    for item in pairs:
        symbol = item["symbol"]; tf = item["tf"]
        if only and (symbol, tf) not in only:
            continue

        pair_key = _pair_key(symbol, tf)
        t0 = time.time()
        out_row: Dict[str, Any] = {"pair": pair_key}

        spec = TrainSpec(
            symbol=symbol,
            tf=tf,
            fees_bps=float(train_cfg.get("fees_bps", 10)),
            slippage_bps=float(train_cfg.get("slippage_bps", 1)),
            steps=int(train_cfg.get("steps", 120_000)),
            train_span=int(train_cfg.get("train_span", 3000)),
            test_span=int(train_cfg.get("test_span", 500)),
            stride=int(train_cfg.get("stride", 250)),
            seed=int(train_cfg.get("seed", 42)),
            use_short=bool(train_cfg.get("use_short", False)),
            use_atr_stop=bool(train_cfg.get("use_atr_stop", False)),
            atr_k=float(train_cfg.get("atr_k", 2.0)),
            atr_penalty=float(train_cfg.get("atr_penalty", 0.0)),
            short_fee_bps=float(train_cfg.get("short_fee_bps", 0.0)),
        )

        try:
            train_meta = train_walk_forward(spec)
        except Exception as e:
            out_row.update(status="train_error", error=str(e))
            results.append(out_row); continue

        try:
            eval_out = evaluate_oos(
                symbol=symbol,
                tf=tf,
                fees_bps=float(eval_cfg.get("fees_bps", train_cfg.get("fees_bps", 10))),
                slippage_bps=float(eval_cfg.get("slippage_bps", train_cfg.get("slippage_bps", 1))),
                use_short=bool(eval_cfg.get("use_short", train_cfg.get("use_short", False))),
                use_atr_stop=bool(eval_cfg.get("use_atr_stop", train_cfg.get("use_atr_stop", False))),
                atr_k=float(eval_cfg.get("atr_k", train_cfg.get("atr_k", 2.0))),
                atr_penalty=float(eval_cfg.get("atr_penalty", train_cfg.get("atr_penalty", 0.0))),
                symbol_tf_latest=None,   
            )
        except Exception as e:
            out_row.update(status="eval_error", error=str(e))
            results.append(out_row); continue

        new_run_dir = Path(train_meta.get("run_dir", ""))
        latest_json = latest_path_for(symbol, tf)  
        old_latest = latest_json if latest_json.exists() else None

        better, cmp_details = _compare_new_vs_old(new_run_dir, old_latest)
        out_row.update({"compare": cmp_details})

        if promote_if_better and better and not dry_run:
            write_latest_for(symbol, tf, train_meta)
            out_row["status"] = "promoted"
        else:
            out_row["status"] = "held" if not better else ("dry_run_promote" if dry_run else "promote_disabled")

        if not dry_run and keep_n > 0:
            out_row["retention"] = _retention_prune(symbol, tf, keep_n)

        out_row["elapsed_sec"] = round(time.time() - t0, 2)
        results.append(out_row)

    return {"results": results}

def main():
    p = argparse.ArgumentParser(description="Auto retrain & promote best checkpoints per pair.")
    p.add_argument("--config", required=True, type=str, help="Path to config/pairs.yaml")
    p.add_argument("--only", action="append", help="Limit to specific pair(s) like ETH/USDT:4h (can repeat)")
    p.add_argument("--promote-if-better", action="store_true", help="Promote new checkpoint if strictly better")
    p.add_argument("--dry-run", action="store_true", help="Run train/eval but do not write latest or prune")
    args = p.parse_args()

    cfg = _load_yaml_config(Path(args.config))
    only_pairs = None
    if args.only:
        only_pairs = []
        for s in args.only:
            if ":" not in s:
                print(f"--only expects SYMBOL:TF, got {s}", file=sys.stderr); sys.exit(2)
            sym, tf = s.split(":", 1)
            only_pairs.append((sym, tf))

    out = run_for_pairs(cfg, only=only_pairs, promote_if_better=args.promote_if_better, dry_run=args.dry_run)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
