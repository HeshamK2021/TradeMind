from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}

def _normalize_sqlite_url(url: str) -> str:
    if not url.startswith("sqlite"):
        return url
    if url.startswith("sqlite:///"):
        p = url[len("sqlite:///") :]
        if p.startswith("/"):
            return url  
        abs_path = str((Path(__file__).resolve().parents[2] / p).resolve())
        return f"sqlite:///{abs_path}"
    return url

@dataclass(frozen=True)
class Settings:
    DATA_MODE: str = os.getenv("DATA_MODE", "real")
    USE_REDIS: bool = _bool("USE_REDIS", False)
    DATABASE_URL: str = _normalize_sqlite_url(os.getenv("DATABASE_URL", "sqlite:///./db/trademind.db"))
    EXCHANGE: str = os.getenv("EXCHANGE", "binance")
    SYMBOLS: str = os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT")
    TIMEFRAME: str = os.getenv("TIMEFRAME", "1h")

def get_settings() -> Settings:
    s = Settings()
    if s.DATA_MODE != "real":
        raise RuntimeError("Synthetic data is disabled. Set DATA_MODE=real.")
    return s

settings = get_settings()
