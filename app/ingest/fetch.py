from __future__ import annotations
import logging
from typing import Iterable, List, Tuple, Optional, Dict, Any

import ccxt  
from app.core.config import settings

log = logging.getLogger("ingest.fetch")


_TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}
def timeframe_ms(tf: str) -> int:
    try:
        return _TIMEFRAME_MS[tf]
    except KeyError:
        raise ValueError(f"Unsupported timeframe '{tf}'")


def _make_exchange(name: str | None = None) -> Any:
    ex = (name or settings.EXCHANGE).lower()
    if ex == "binance":
        return ccxt.binance({"enableRateLimit": True})
    elif ex == "binanceusdm":
        return ccxt.binanceusdm({"enableRateLimit": True})
    elif ex == "bybit":
        return ccxt.bybit({"enableRateLimit": True})
    else:
        raise ValueError(f"Unsupported exchange '{ex}'")


def fetch_latest_ohlcv(
    symbol: str,
    tf: str,
    limit: int = 1000,
    since: Optional[int] = None,
    exchange_name: Optional[str] = None,
) -> List[List[float]]:

    ex = _make_exchange(exchange_name)
    tfms = timeframe_ms(tf)

    out: List[List[float]] = []
    cursor = since
    remaining = int(limit)

    per_call = min(remaining, 1000)

    while remaining > 0:
        rows: List[List[float]] = ex.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=per_call)
        if not rows:
            break

        out.extend(rows)
        remaining -= len(rows)
        if len(rows) < per_call:
            break

        last_ts = rows[-1][0]
        cursor = last_ts + tfms 

    if len(out) > limit:
        out = out[-limit:]

    log.info(
        "fetch", extra={
            "stage": "ingest",
            "source": ex.id,
            "symbol": symbol, "tf": tf,
            "since": since, "limit": limit, "count": len(out),
        }
    )
    return out
