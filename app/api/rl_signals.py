from __future__ import annotations
from typing import List, Dict, Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rl.policy import RLPolicy

try:
    from app.data.session import session_scope
    from app.data.models import FeatureRow, Candle  
    _HAS_DB = True
except Exception:
    _HAS_DB = False
    session_scope = None
    FeatureRow = None
    Candle = None

router = APIRouter(prefix="/rl", tags=["rl"])

class RecentSignalsRequest(BaseModel):
    symbol: str
    tf: str
    limit: int = Field(20, ge=1, le=500)

class RecentSignal(BaseModel):
    ts: int
    action: str
    confidence: float
    close: float | None = None
    probs: Dict[str, float] | None = None

class RecentSignalsResponse(BaseModel):
    symbol: str
    tf: str
    count: int
    items: List[RecentSignal]

@router.post("/recent", response_model=RecentSignalsResponse)
def recent_signals(req: RecentSignalsRequest) -> RecentSignalsResponse:
    if not _HAS_DB:
        raise HTTPException(status_code=500, detail="DB not available")

    policy = RLPolicy.latest(req.symbol, req.tf)

    with session_scope() as s:
        rows = (
            s.query(
                FeatureRow.ts,
                FeatureRow.ema_5, FeatureRow.ema_20, FeatureRow.rsi_14,
                FeatureRow.atr_14, FeatureRow.bb_mid, FeatureRow.bb_up, FeatureRow.bb_dn,
            )
            .filter_by(symbol=req.symbol, tf=req.tf, shifted=True)
            .order_by(FeatureRow.ts.desc())
            .limit(req.limit)
            .all()
        )

        closes_by_ts: dict[int, float] = {}
        if rows and Candle is not None:
            ts_list = [int(r[0]) for r in rows]
            try:
                crows = (
                    s.query(Candle.ts, Candle.close)
                    .filter(Candle.symbol == req.symbol)
                    .filter(Candle.tf == req.tf)
                    .filter(Candle.ts.in_(ts_list))
                    .all()
                )
                closes_by_ts = {int(ts): float(c) for ts, c in crows}
            except Exception:
                pass

    items: List[RecentSignal] = []
    for r in rows:
        ts_ms = int(r[0])
        feat = dict(
            ema_5=float(r[1]), ema_20=float(r[2]), rsi_14=float(r[3]),
            atr_14=float(r[4]), bb_mid=float(r[5]), bb_up=float(r[6]), bb_dn=float(r[7]),
        )
        pred = policy.predict(feat)
        items.append(
            RecentSignal(
                ts=ts_ms,
                action=pred["action"],
                confidence=float(pred.get("confidence", 0.5)),
                close=closes_by_ts.get(ts_ms),
                probs=pred.get("probs"),
            )
        )

    return RecentSignalsResponse(symbol=req.symbol, tf=req.tf, count=len(items), items=items)
