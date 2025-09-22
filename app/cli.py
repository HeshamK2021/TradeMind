from __future__ import annotations

import argparse
from typing import Optional
from datetime import datetime, timezone, timedelta

import pandas as pd

from app.data.session import SessionLocal, init_db
from app.ingest.fetch import fetch_ohlcv_all, utc_ms, utc_now
from app.ingest.loader import (
    save_candles_or_ignore,
    coverage as candles_coverage,
    get_last_n as candles_get_last_n,
)

def _save_and_maybe_cache(db, symbol: str, tf: str, rows):
    try:
        from app.ingest.loader import save_and_cache  
        return save_and_cache(db, symbol, tf, rows)
    except Exception:
        n = save_candles_or_ignore(db, rows)
        try:
            from app.ingest.cache import refresh_cache_last_bars 
            refresh_cache_last_bars(db, symbol, tf, n=500)
        except Exception:
            pass
        return n



def cmd_seed(args: argparse.Namespace) -> None:
    init_db()
    with SessionLocal() as s:
        months = args.months
        end_utc = utc_now()
        start_utc = end_utc - timedelta(days=int(months * 30))
        rows = fetch_ohlcv_all(
            exchange=args.exchange,
            symbol=args.symbol,
            tf=args.tf,
            since_ms=utc_ms(start_utc),
            until_ms=utc_ms(end_utc),
        )
        inserted = _save_and_maybe_cache(s, args.symbol, args.tf, rows)
        print(f"[seed] exchange={args.exchange} symbol={args.symbol} tf={args.tf} months={months} inserted={inserted}")


def cmd_fetch(args: argparse.Namespace) -> None:
    init_db()
    since_iso = args.since
    until_iso = args.until
    since_dt = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    until_dt = datetime.fromisoformat(until_iso.replace("Z", "+00:00"))
    with SessionLocal() as s:
        rows = fetch_ohlcv_all(
            exchange=args.exchange,
            symbol=args.symbol,
            tf=args.tf,
            since_ms=utc_ms(since_dt),
            until_ms=utc_ms(until_dt),
        )
        inserted = _save_and_maybe_cache(s, args.symbol, args.tf, rows)
        print(f"[fetch] {since_iso} → {until_iso} inserted={inserted}")


def cmd_coverage(args: argparse.Namespace) -> None:
    init_db()
    with SessionLocal() as s:
        info = candles_coverage(s, args.symbol, args.tf)
        print(info)


def cmd_last(args: argparse.Namespace) -> None:
    init_db()
    with SessionLocal() as s:
        rows = candles_get_last_n(s, args.symbol, args.tf, args.n)
        for r in rows:
            print(f"{r.ts.isoformat()}  O:{r.open:.2f} H:{r.high:.2f} L:{r.low:.2f} C:{r.close:.2f} V:{r.volume:.4f}")



def cmd_features_build(args: argparse.Namespace) -> None:

    from app.data.models import FeatureRow
    from app.features.builder import build_features_df, persist_features

    init_db()
    with SessionLocal() as s:
        rows = candles_get_last_n(s, args.symbol, args.tf, args.n)
        if not rows:
            print("No candles found. Seed some data first .")
            return

        df = pd.DataFrame(
            [{"ts": r.ts, "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": r.volume} for r in rows]
        ).sort_values("ts").set_index("ts")

        feats = build_features_df(df, args.symbol, args.tf)
        inserted = persist_features(s, FeatureRow, feats)
        print(f"[features.build] {args.symbol} {args.tf} n_in={len(df)} n_feats={len(feats)} inserted/updated={inserted}")


def cmd_features_coverage(args: argparse.Namespace) -> None:

    from sqlalchemy import select, func
    from app.data.models import FeatureRow

    init_db()
    with SessionLocal() as s:
        q = select(
            func.count(FeatureRow.id),
            func.min(FeatureRow.ts),
            func.max(FeatureRow.ts),
        ).where(
            FeatureRow.symbol == args.symbol,
            FeatureRow.tf == args.tf,
        )
        count, first_ts, last_ts = s.execute(q).one()
        print({
            "symbol": args.symbol,
            "tf": args.tf,
            "count": int(count or 0),
            "first_ts": None if first_ts is None else first_ts.isoformat(),
            "last_ts": None if last_ts is None else last_ts.isoformat(),
        })



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trademind", description="TradeMind CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("seed", help="Seed N months of data")
    sp.add_argument("--exchange", default="binance")
    sp.add_argument("--symbol", required=True, help="e.g., BTC/USDT")
    sp.add_argument("--tf", default="1h")
    sp.add_argument("--months", type=int, default=6)
    sp.set_defaults(func=cmd_seed)

    fp = sub.add_parser("fetch", help="Fetch custom time range (ISO8601)")
    fp.add_argument("--exchange", default="binance")
    fp.add_argument("--symbol", required=True)
    fp.add_argument("--tf", default="1h")
    fp.add_argument("--since", required=True, help="ISO8601, e.g., 2024-01-01T00:00:00Z")
    fp.add_argument("--until", required=True, help="ISO8601, e.g., 2024-09-01T00:00:00Z")
    fp.set_defaults(func=cmd_fetch)

    cp = sub.add_parser("coverage", help="Show DB coverage for candles (pair/tf)")
    cp.add_argument("--symbol", required=True)
    cp.add_argument("--tf", default="1h")
    cp.set_defaults(func=cmd_coverage)

    lp = sub.add_parser("last", help="Show last N candle rows")
    lp.add_argument("--symbol", required=True)
    lp.add_argument("--tf", default="1h")
    lp.add_argument("-n", type=int, default=10)
    lp.set_defaults(func=cmd_last)

    fb = sub.add_parser("features.build", help="Build & persist features from existing candles")
    fb.add_argument("--symbol", required=True)
    fb.add_argument("--tf", default="1h")
    fb.add_argument("-n", type=int, default=5000, help="How many recent candles to use")
    fb.set_defaults(func=cmd_features_build)

    fc = sub.add_parser("features.coverage", help="Show FeatureRow coverage (count/first/last)")
    fc.add_argument("--symbol", required=True)
    fc.add_argument("--tf", default="1h")
    fc.set_defaults(func=cmd_features_coverage)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
