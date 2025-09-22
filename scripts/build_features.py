from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import logging
import sys as _sys

from app.core.config import settings
from app.data.session import init_db, session_scope
from app.features.builder import build_and_persist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("scripts.build_features")

def main():
    p = argparse.ArgumentParser(description="Build shift(1) features from DB candles and persist.")
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", "--timeframe", dest="tf", required=True)
    args = p.parse_args()

    if settings.DATA_MODE != "real":
        raise RuntimeError("Synthetic data is disabled. Set DATA_MODE=real.")

    init_db()  
    with session_scope() as s:
        n = build_and_persist(s, args.symbol, args.tf)

    print(json.dumps({"status": "ok", "symbol": args.symbol, "tf": args.tf, "upserts": n}, indent=2))

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("build_features_failed")
        _sys.exit(1)
