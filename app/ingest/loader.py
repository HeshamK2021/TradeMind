from __future__ import annotations
import logging
from typing import Iterable, List, Optional, Tuple, Dict, Any, Sequence

from sqlalchemy import text, select
from sqlalchemy.orm import Session

from app.data.models import Candle
from app.core.config import settings

log = logging.getLogger("ingest.loader")

def _validate_row(symbol: str, tf: str, row: Sequence[float]) -> Tuple[int, float, float, float, float, float]:

    if len(row) < 6:
        raise ValueError(f"Bad OHLCV row (len<6): {row}")
    ts, o, h, l, c, v = row[:6]

    try:
        ts = int(ts)
        o = float(o); h = float(h); l = float(l); c = float(c); v = float(v)
    except Exception as e:
        raise ValueError(f"Type error in OHLCV row: {row}") from e

    if any(x < 0 for x in (o, h, l, c, v)):
        raise ValueError(f"Negative OHLCV not allowed: {row}")

    hi_ref = max(o, c)
    lo_ref = min(o, c)
    if h < hi_ref or l > lo_ref:
        log.warning(
            "ohlc_integrity_warning",
            extra={"symbol": symbol, "tf": tf, "ts": ts, "o": o, "h": h, "l": l, "c": c}
        )

    return ts, o, h, l, c, v

def _dict_to_row(d: Dict[str, Any]) -> Sequence[float]:

    ts = d.get("ts", d.get("timestamp", d.get("time", d.get("t"))))
    o  = d.get("open",  d.get("o"))
    h  = d.get("high",  d.get("h"))
    l  = d.get("low",   d.get("l"))
    c  = d.get("close", d.get("c"))
    v  = d.get("volume", d.get("vol", d.get("baseVolume", 0.0)))
    return [ts, o, h, l, c, v]

def normalize_ohlcv(symbol: str, tf: str, rows: Iterable[Sequence[float] | Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    last_ts: Optional[int] = None
    for r in rows:
        if isinstance(r, dict):
            r = _dict_to_row(r)
        ts, o, h, l, c, v = _validate_row(symbol, tf, r)
        if last_ts is not None and ts <= last_ts:
            if ts == last_ts:
                continue
            log.warning("ts error", extra={"symbol": symbol, "tf": tf, "prev": last_ts, "cur": ts})
        out.append(
            {"symbol": symbol, "tf": tf, "ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}
        )
        last_ts = ts
    return out

def insert_candles_dedupe(session: Session, rows: List[Dict[str, Any]]) -> int:

    if not rows:
        return 0

    sql = text("""
        INSERT OR IGNORE INTO candle (symbol, tf, ts, open, high, low, close, volume)
        VALUES (:symbol, :tf, :ts, :open, :high, :low, :close, :volume)
    """)
    inserted = 0
    for r in rows:
        res = session.execute(sql, r)
        inserted += int(res.rowcount or 0)
    return inserted

def load_ohlcv_batch(session: Session, symbol: str, tf: str, raw_rows: Iterable[Sequence[float] | Dict[str, Any]]) -> int:

    rows = normalize_ohlcv(symbol, tf, raw_rows)
    n = insert_candles_dedupe(session, rows)
    if n:
        log.info("ingest_write", extra={"stage": "ingest", "symbol": symbol, "tf": tf, "inserted": n})
    return n


def insert_candles(session: Session, symbol: str, tf: str, candles: Iterable[Sequence[float] | Dict[str, Any]]) -> int:

    rows = normalize_ohlcv(symbol, tf, candles)
    n = insert_candles_dedupe(session, rows)
    if n:
        log.info("ingest_write", extra={"stage": "ingest", "symbol": symbol, "tf": tf, "inserted": n})
    return n


def candles_range(
    session: Session,
    symbol: str,
    tf: str,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> List[Candle]:
    q = session.query(Candle).filter(Candle.symbol == symbol, Candle.tf == tf)
    if start_ts is not None:
        q = q.filter(Candle.ts >= start_ts)
    if end_ts is not None:
        q = q.filter(Candle.ts < end_ts)
    return q.order_by(Candle.ts.asc()).all()

def minmax_ts(session: Session, symbol: str, tf: str) -> Optional[Tuple[int, int]]:
    q = session.query(Candle.ts).filter(Candle.symbol == symbol, Candle.tf == tf).order_by(Candle.ts.asc())
    first = q.first()
    if not first:
        return None
    last = session.query(Candle.ts).filter(Candle.symbol == symbol, Candle.tf == tf).order_by(Candle.ts.desc()).first()
    return int(first[0]), int(last[0])
