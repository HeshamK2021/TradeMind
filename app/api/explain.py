from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rl.policy import RLPolicy
from app.rl.utils import latest_path_for
from app.data.session import session_scope
from app.data.models import FeatureRow, Candle

router = APIRouter(prefix="/explain", tags=["explain"])


class ExplainResponse(BaseModel):
    symbol: str
    tf: str
    ts: Optional[int] = None
    ts_iso: Optional[str] = None
    action: str
    confidence: float
    probs: Dict[str, float]
    features: Dict[str, float]
    reasons: List[Dict[str, Any]] = Field(default_factory=list)
    text: str = ""


def _build_rationale(feat: Dict[str, float], close: Optional[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        ema5 = float(feat["ema_5"])
        ema20 = float(feat["ema_20"])
        rsi = float(feat["rsi_14"])
        atr = float(feat["atr_14"])
        bb_mid = float(feat["bb_mid"])
        bb_up = float(feat["bb_up"])
        bb_dn = float(feat["bb_dn"])
    except Exception:
        return out

    ema_gap_pct = 100.0 * (ema5 - ema20) / abs(ema20) if ema20 else 0.0
    if ema_gap_pct > 0.05:
        out.append({"name": "EMA fast > EMA slow", "detail": f"gap {ema_gap_pct:.2f}%", "weight": +0.8})
    elif ema_gap_pct < -0.05:
        out.append({"name": "EMA fast < EMA slow", "detail": f"gap {ema_gap_pct:.2f}%", "weight": -0.8})
    else:
        out.append({"name": "EMAs near", "detail": f"gap {ema_gap_pct:.2f}%", "weight": 0.1})

    if rsi >= 60:
        out.append({"name": "RSI bullish", "detail": f"RSI {rsi:.1f}", "weight": +0.5})
    elif rsi <= 40:
        out.append({"name": "RSI bearish", "detail": f"RSI {rsi:.1f}", "weight": -0.5})
    else:
        out.append({"name": "RSI neutral", "detail": f"RSI {rsi:.1f}", "weight": 0.1})

    if close and close != 0:
        atr_pct = 100.0 * max(0.0, atr) / abs(close)
        if atr_pct > 2.0:
            out.append({"name": "Volatility high", "detail": f"ATR {atr_pct:.2f}%", "weight": -0.35})
        elif atr_pct < 0.7:
            out.append({"name": "Volatility low", "detail": f"ATR {atr_pct:.2f}%", "weight": +0.2})
        else:
            out.append({"name": "Volatility moderate", "detail": f"ATR {atr_pct:.2f}%", "weight": 0.1})

    if close is not None:
        band_w = max(1e-9, bb_up - bb_dn)
        z = (close - bb_mid) / (band_w / 2.0)
        if z > 0.4:
            out.append({"name": "Above middle band", "detail": f"z≈{z:.2f}", "weight": +0.25})
        elif z < -0.4:
            out.append({"name": "Below middle band", "detail": f"z≈{z:.2f}", "weight": -0.25})
        else:
            out.append({"name": "Near middle band", "detail": f"z≈{z:.2f}", "weight": 0.05})

    return out


def _reasons_to_text(action: str, reasons: List[Dict[str, Any]]) -> str:
    if not reasons:
        return "No clear signals detected."
    top = sorted(reasons, key=lambda r: abs(r.get("weight", 0)), reverse=True)[:3]
    bits = [f"{r['name']} ({r['detail']})" for r in top]
    if action.upper() == "BUY":
        return "Buying pressure led by " + ", ".join(bits) + "."
    if action.upper() == "SELL":
        return "Selling pressure led by " + ", ".join(bits) + "."
    return "Signals are mixed: " + ", ".join(bits) + "."


@router.get("/latest", response_model=ExplainResponse)
def explain_latest(symbol: str, tf: str) -> ExplainResponse:
    lp = latest_path_for(symbol, tf)
    if not lp.exists():
        raise HTTPException(status_code=404, detail=f"Latest checkpoint not found for {symbol} {tf}")

    with session_scope() as s:
        row = (
            s.query(
                FeatureRow.ts,
                FeatureRow.ema_5, FeatureRow.ema_20, FeatureRow.rsi_14,
                FeatureRow.atr_14, FeatureRow.bb_mid, FeatureRow.bb_up, FeatureRow.bb_dn,
            )
            .filter_by(symbol=symbol, tf=tf, shifted=True)
            .order_by(FeatureRow.ts.desc())
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail="No shifted feature row found")

        ts_ms = int(row[0])
        feat = dict(
            ema_5=float(row[1]), ema_20=float(row[2]), rsi_14=float(row[3]),
            atr_14=float(row[4]), bb_mid=float(row[5]), bb_up=float(row[6]), bb_dn=float(row[7]),
        )

        close = None
        try:
            c = (
                s.query(Candle.close)
                .filter(Candle.symbol == symbol, Candle.tf == tf, Candle.ts == ts_ms)
                .first()
            )
            if c is not None:
                close = float(c[0])
        except Exception:
            close = None

    try:
        policy = RLPolicy.latest(symbol, tf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load policy: {e}")

    out = policy.predict(feat)
    action = str(out.get("action", "HOLD")).upper()
    confidence = float(out.get("confidence", 0.0))
    probs = {k.upper(): float(v) for k, v in (out.get("probs") or {}).items()}

    reasons = _build_rationale(feat, close)
    text = _reasons_to_text(action, reasons)

    return ExplainResponse(
        symbol=symbol,
        tf=tf,
        ts=ts_ms,
        ts_iso=pd.to_datetime(ts_ms, unit="ms", utc=True).isoformat(),
        action=action,
        confidence=confidence,
        probs=probs,
        features=feat,
        reasons=reasons,
        text=text,
    )
