from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.data.models import Candle, FeatureRow
from app.features.indicators import ema, rsi, atr, bollinger_bands

log = logging.getLogger("features.builder")

def _candles_df(session: Session, symbol: str, tf: str) -> pd.DataFrame:
    rows: List[Candle] = (
        session.query(Candle)
        .filter(Candle.symbol == symbol, Candle.tf == tf)
        .order_by(Candle.ts.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        {
            "ts": [int(r.ts) for r in rows],
            "open": [float(r.open) for r in rows],
            "high": [float(r.high) for r in rows],
            "low": [float(r.low) for r in rows],
            "close": [float(r.close) for r in rows],
            "volume": [float(r.volume) for r in rows],
        }
    ).set_index("ts")
    df.index.name = "ts"
    return df

def build_features_df(candles_df: pd.DataFrame, symbol: str, tf: str) -> pd.DataFrame:
    if candles_df.empty:
        return pd.DataFrame(
            columns=["ema_5","ema_20","rsi_14","atr_14","bb_mid","bb_up","bb_dn","shifted"]
        )
    df = candles_df.copy()
    ema5  = ema(df["close"], span=5)
    ema20 = ema(df["close"], span=20)
    rsi14 = rsi(df["close"], period=14)
    atr14 = atr(df["high"], df["low"], df["close"], period=14)
    bb_mid, bb_up, bb_dn = bollinger_bands(df["close"], period=20, num_std=2.0)

    feats = pd.DataFrame(
        {
            "ema_5": ema5,
            "ema_20": ema20,
            "rsi_14": rsi14,
            "atr_14": atr14,
            "bb_mid": bb_mid,
            "bb_up": bb_up,
            "bb_dn": bb_dn,
        },
        index=df.index,
    )

    feats = feats.shift(1)
    feats["shifted"] = True

    feats = feats.dropna()

    feats["symbol"] = symbol
    feats["tf"] = tf
    return feats

def persist_features(session: Session, feats_df: pd.DataFrame) -> int:

    if feats_df.empty:
        return 0

    sql = text("""
        INSERT OR REPLACE INTO feature_row
        (symbol, tf, ts, ema_5, ema_20, rsi_14, atr_14, bb_mid, bb_up, bb_dn, regime_flag, quality, shifted, id)
        VALUES (:symbol, :tf, :ts, :ema_5, :ema_20, :rsi_14, :atr_14, :bb_mid, :bb_up, :bb_dn, :regime_flag, :quality, :shifted,
            COALESCE((SELECT id FROM feature_row WHERE symbol=:symbol AND tf=:tf AND ts=:ts), NULL)
        )
    """)

    upserts = 0
    for ts, row in feats_df.iterrows():
        params = {
            "symbol": row["symbol"],
            "tf": row["tf"],
            "ts": int(ts),
            "ema_5": float(row["ema_5"]),
            "ema_20": float(row["ema_20"]),
            "rsi_14": float(row["rsi_14"]),
            "atr_14": float(row["atr_14"]),
            "bb_mid": float(row["bb_mid"]),
            "bb_up": float(row["bb_up"]),
            "bb_dn": float(row["bb_dn"]),
            "regime_flag": None,
            "quality": None,
            "shifted": True,
        }
        res = session.execute(sql, params)
        upserts += int(res.rowcount or 0)

    if upserts:
        log.info("features_persist", extra={"stage":"features","inserted_or_replaced": upserts})
    return upserts

def build_and_persist(session: Session, symbol: str, tf: str) -> int:

    candles = _candles_df(session, symbol, tf)
    feats = build_features_df(candles, symbol=symbol, tf=tf)
    return persist_features(session, feats)
