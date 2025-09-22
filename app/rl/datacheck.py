from __future__ import annotations


import argparse
from datetime import timezone
from typing import Optional

import pandas as pd

try:
    from app.data.session import session_scope
    from app.data.models import FeatureRow
except Exception as e:
    raise RuntimeError(
        "Could not import session_scope/FeatureRow. "
    ) from e


def check(symbol: str, tf: str, train_span: int = 3000, test_span: int = 500) -> dict:
    with session_scope() as s:
        q = (
            s.query(FeatureRow.ts)
            .filter(FeatureRow.symbol == symbol)
            .filter(FeatureRow.tf == tf)
            .filter(FeatureRow.shifted == True)
            .order_by(FeatureRow.ts.asc())
        )
        rows = q.all()

    n = len(rows)
    if n == 0:
        return {
            "symbol": symbol,
            "tf": tf,
            "count": 0,
            "ok_for_walk_forward": False,
            "reason": "No shifted feature rows found.",
        }

    ts = pd.to_datetime([r[0] for r in rows], utc=True)
    first_ts = ts[0]
    last_ts = ts[-1]
    need = train_span + test_span
    ok = n >= need

    return {
        "symbol": symbol,
        "tf": tf,
        "count": n,
        "first_ts": first_ts.isoformat(),
        "last_ts": last_ts.isoformat(),
        "required_min_rows": need,
        "ok_for_walk_forward": ok,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Check shifted feature availability for RL windows.")
    p.add_argument("--symbol", required=True, type=str)
    p.add_argument("--tf", required=True, type=str)
    p.add_argument("--train-span", dest="train_span", default=3000, type=int)
    p.add_argument("--test-span", dest="test_span", default=500, type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = check(args.symbol, args.tf, train_span=args.train_span, test_span=args.test_span)
    import json
    print(json.dumps(out, indent=2))
