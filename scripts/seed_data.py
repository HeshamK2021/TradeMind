from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import logging
import sys
from typing import Optional

from app.core.config import settings
from app.data.session import init_db, session_scope
from app.ingest.fetch import fetch_latest_ohlcv
from app.ingest.loader import load_ohlcv_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("scripts.seed_data")

def main():
    p = argparse.ArgumentParser(description="Seed candles via CCXT → DB (no synthetic).")
    p.add_argument("--symbol", required=True, help="e.g., BTC/USDT")
    p.add_argument("--tf", "--timeframe", dest="tf", required=True, help="e.g., 1h")
    p.add_argument("--limit", type=int, default=1000, help="number of bars to fetch")
    p.add_argument("--since", type=int, default=None, help="epoch-ms to start from")
    p.add_argument("--exchange", default=None, help="override EXCHANGE env (optional)")
    args = p.parse_args()

    if settings.DATA_MODE != "real":
        raise RuntimeError("Synthetic data is disabled. Set DATA_MODE=real.")

    init_db()  

    log.info("seed_start", extra={
        "symbol": args.symbol, "tf": args.tf, "limit": args.limit,
        "exchange": args.exchange or settings.EXCHANGE, "since": args.since
    })

    rows = fetch_latest_ohlcv(
        symbol=args.symbol,
        tf=args.tf,
        limit=args.limit,
        since=args.since,
        exchange_name=args.exchange,
    )

    with session_scope() as s:
        inserted = load_ohlcv_batch(s, args.symbol, args.tf, rows)

    print(json.dumps({
        "status": "ok",
        "symbol": args.symbol,
        "tf": args.tf,
        "requested": args.limit,
        "fetched": len(rows),
        "inserted": inserted,
        "exchange": (args.exchange or settings.EXCHANGE),
    }, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("seed_failed")
        sys.exit(1)
