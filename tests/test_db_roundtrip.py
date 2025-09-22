import time
import pytest
from pathlib import Path
from sqlalchemy.exc import IntegrityError

from app.data.session import init_db, session_scope
from app.data.models import Candle

def _temp_db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path}/tm02.db"

def test_candle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))

    init_db(reset=True)

    ts_now_ms = int(time.time() * 1000)
    with session_scope() as s:
        c = Candle(
            symbol="BTC/USDT",
            tf="1h",
            ts=ts_now_ms,
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=123.45,
        )
        s.add(c)

    with session_scope() as s:
        row = s.query(Candle).filter_by(symbol="BTC/USDT", tf="1h", ts=ts_now_ms).one()
        assert row.close == 50200.0
        assert row.volume == 123.45

def test_candle_uniqueness(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_MODE", "real")
    monkeypatch.setenv("DATABASE_URL", _temp_db_url(tmp_path))
    init_db(reset=True)

    ts_ms = int(time.time() * 1000) // 3_600_000 * 3_600_000
    with session_scope() as s:
        s.add(Candle(symbol="ETH/USDT", tf="1h", ts=ts_ms, open=1, high=2, low=0.5, close=1.5, volume=10))

    with pytest.raises(IntegrityError):
        with session_scope() as s:
            s.add(Candle(symbol="ETH/USDT", tf="1h", ts=ts_ms, open=1, high=2, low=0.5, close=1.5, volume=11))
