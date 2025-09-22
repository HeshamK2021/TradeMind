from __future__ import annotations
import sys, subprocess, json, time
from pathlib import Path

from app.data.session import session_scope
from app.data.models import Candle

TF_MS = {
    "15m": 15*60*1000,
    "1h": 60*60*1000,
    "4h": 4*60*60*1000,
    "1d": 24*60*60*1000,
}

def candle_count(symbol: str, tf: str) -> int:
    with session_scope() as s:
        return s.query(Candle).filter_by(symbol=symbol, tf=tf).count()

def oldest_ts(symbol: str, tf: str):
    with session_scope() as s:
        r = s.query(Candle.ts).filter_by(symbol=symbol, tf=tf).order_by(Candle.ts.asc()).first()
        return int(r[0]) if r else None

def run_seed(symbol: str, tf: str, limit: int, since: int | None):
    cmd = [sys.executable, "-m", "scripts.seed_data", "--symbol", symbol, "--tf", tf, "--limit", str(limit)]
    if since is not None:
        cmd += ["--since", str(since)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip())
    try:
        return json.loads(p.stdout.strip())
    except Exception:
        print(p.stdout) 
        raise

def backfill(symbol: str, tf: str, target_count: int, page_limit: int = 1000, safety_pages: int = 20):
    tf_ms = TF_MS.get(tf)
    if not tf_ms:
        raise ValueError(f"Unsupported tf={tf}")
    print(f"[backfill] target={target_count}, page_limit={page_limit}")

    n0 = candle_count(symbol, tf)
    if n0 == 0:
        print("[backfill] table empty → pulling latest page first")
        run_seed(symbol, tf, page_limit, since=None)

    pages = 0
    while candle_count(symbol, tf) < target_count and pages < safety_pages:
        oldest = oldest_ts(symbol, tf)
        if oldest is None:
            print("[backfill] no candles found after initial seed; retrying latest…")
            run_seed(symbol, tf, page_limit, since=None)
            time.sleep(0.2)
            continue
        since = oldest - (page_limit * tf_ms)
        print(f"[backfill] page={pages+1} since={since} (ms) oldest_now={oldest}")
        out = run_seed(symbol, tf, page_limit, since=since)
        print(f"  fetched={out.get('fetched')} inserted={out.get('inserted')}")
        pages += 1
        time.sleep(0.2)

    n_final = candle_count(symbol, tf)
    print(f"[backfill] done. candles={n_final}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--page-limit", type=int, default=1000)
    args = ap.parse_args()
    backfill(args.symbol, args.tf, target_count=args.target, page_limit=args.page_limit)
