import time
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.data.session import init_db, session_scope
from app.ingest.loader import (
    normalize_ohlcv,
    insert_candles_dedupe,
    load_ohlcv_batch,
    candles_range,
    minmax_ts,
)

def _temp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path}/ingest.db"

def _row(ts, o, h, l, c, v):
    return [ts, o, h, l, c, v]

def test_dedupe_and_integrity(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))
    init_db(reset=True)

    symbol, tf = "BTC/USDT", "1h"
    t0 = int(time.time() * 1000) // 3_600_000 * 3_600_000  

    batch = [
        _row(t0,     100, 105,  95, 102, 10),
        _row(t0+3600000, 102, 110, 101, 109, 12),
        _row(t0+2*3600000, 109, 111, 108, 110,  9),
        _row(t0+2*3600000, 109, 111, 108, 110,  9),  
    ]

    with session_scope() as s:
        inserted = load_ohlcv_batch(s, symbol, tf, batch)
        assert inserted == 3  

        inserted_again = load_ohlcv_batch(s, symbol, tf, batch)
        assert inserted_again == 0

        rows = candles_range(s, symbol, tf)
        assert len(rows) == 3
        assert rows[0].ts == t0
        assert rows[-1].ts == t0 + 2*3600000

        mnmx = minmax_ts(s, symbol, tf)
        assert mnmx == (t0, t0 + 2*3600000)

def test_negative_values_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))
    init_db(reset=True)

    symbol, tf = "ETH/USDT", "1h"
    t0 = int(time.time() * 1000) // 3_600_000 * 3_600_000

    bad = [
        _row(t0, 100, 105, 95, -1, 10),   
    ]
    with pytest.raises(ValueError):
        normalize_ohlcv(symbol, tf, bad)

def test_non_monotonic_ts_drops_exact_dupe(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))
    init_db(reset=True)

    symbol, tf = "BTC/USDT", "1h"
    t0 = int(time.time() * 1000) // 3_600_000 * 3_600_000

    batch = [
        _row(t0, 100, 105, 95, 102, 10),
        _row(t0, 100, 105, 95, 102, 10),   
        _row(t0+3600000, 102, 110, 101, 109, 12),
    ]

    with session_scope() as s:
        inserted = load_ohlcv_batch(s, symbol, tf, batch)
        assert inserted == 2  
