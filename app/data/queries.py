from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.data.models import Candle, FeatureRow

def distinct_symbol_tf(session: Session) -> List[Tuple[str, str]]:
    c_pairs = session.query(Candle.symbol, Candle.tf).distinct().all()
    f_pairs = session.query(FeatureRow.symbol, FeatureRow.tf).distinct().all()
    seen, out = set(), []
    for sym, tf in c_pairs + f_pairs:
        key = (sym, tf)
        if key not in seen:
            seen.add(key)
            out.append(key)
    out.sort(key=lambda x: (x[0], x[1]))
    return out

def _range_stats(session: Session, model, symbol: str, tf: str):
    q = session.query(func.count(model.ts), func.min(model.ts), func.max(model.ts))\
               .filter(model.symbol == symbol, model.tf == tf)
    cnt, mn, mx = q.one()
    return int(cnt or 0), (int(mn) if mn is not None else None), (int(mx) if mx is not None else None)

def candle_range_stats(session: Session, symbol: str, tf: str):
    return _range_stats(session, Candle, symbol, tf)

def feature_range_stats(session: Session, symbol: str, tf: str):
    return _range_stats(session, FeatureRow, symbol, tf)

def last_candles(session: Session, symbol: str, tf: str, n: int) -> List[Dict[str, float]]:
    rows = (session.query(Candle)
            .filter(Candle.symbol == symbol, Candle.tf == tf)
            .order_by(Candle.ts.desc())
            .limit(n).all())
    rows = list(reversed(rows))
    return [{"ts": int(r.ts), "open": float(r.open), "high": float(r.high),
             "low": float(r.low), "close": float(r.close), "volume": float(r.volume)} for r in rows]

def last_features(session: Session, symbol: str, tf: str, n: int) -> List[Dict[str, float]]:
    rows = (session.query(FeatureRow)
            .filter(FeatureRow.symbol == symbol, FeatureRow.tf == tf)
            .order_by(FeatureRow.ts.desc())
            .limit(n).all())
    rows = list(reversed(rows))
    return [{"ts": int(r.ts), "ema_5": float(r.ema_5), "ema_20": float(r.ema_20), "rsi_14": float(r.rsi_14),
             "atr_14": float(r.atr_14), "bb_mid": float(r.bb_mid), "bb_up": float(r.bb_up),
             "bb_dn": float(r.bb_dn), "shifted": bool(r.shifted)} for r in rows]

def latest_feature_row(session: Session, symbol: str, tf: str) -> Optional[FeatureRow]:
    return (session.query(FeatureRow)
            .filter(FeatureRow.symbol == symbol, FeatureRow.tf == tf)
            .order_by(FeatureRow.ts.desc())
            .first())

def latest_candle_ts(session: Session, symbol: str, tf: str) -> Optional[int]:
    row = (session.query(Candle.ts)
           .filter(Candle.symbol == symbol, Candle.tf == tf)
           .order_by(Candle.ts.desc())
           .first())
    return int(row[0]) if row else None
