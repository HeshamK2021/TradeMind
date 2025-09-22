from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any
from pydantic import BaseModel

Action = Literal["BUY", "SELL", "HOLD"]

@dataclass(frozen=True)
class BaselineParams:
    rsi_buy_min: float = 50.0      
    rsi_sell_max: float = 50.0    
    use_short: bool = False       
    atr_floor: float = 1e-9        
    bb_floor: float = 1e-9
    use_rsi: bool = True       
    use_bb: bool = True   

@dataclass(frozen=True)
class RiskParams:
    risk_per_trade_pct: float = 1.0

def signal_baseline(feat: Dict[str, float], params: BaselineParams = BaselineParams()) -> Dict[str, Any]:
    ema5  = float(feat["ema_5"])
    ema20 = float(feat["ema_20"])
    rsi14 = float(feat["rsi_14"])
    atr14 = max(float(feat["atr_14"]), getattr(params, "atr_floor", 1e-9))
    bb_up = float(feat["bb_up"])
    bb_dn = float(feat["bb_dn"])
    bb_width = max(bb_up - bb_dn, getattr(params, "bb_floor", 1e-9))

    use_rsi: bool   = getattr(params, "use_rsi", True)
    use_bb:  bool   = getattr(params, "use_bb", True)
    use_short: bool = getattr(params, "use_short", False)
    rsi_buy_min: float  = getattr(params, "rsi_buy_min", 50.0)
    rsi_sell_max: float = getattr(params, "rsi_sell_max", 50.0)

    trend = ema5 - ema20
    norm_strength = abs(trend) / max(atr14, bb_width) 
    want_long = trend > 0
    want_short = trend < 0 and use_short

    rsi_buy_ok  = (rsi14 >= rsi_buy_min) if use_rsi else True
    rsi_sell_ok = (rsi14 <= rsi_sell_max) if (use_rsi and use_short) else True

    if want_long and rsi_buy_ok:
        action: Action = "BUY"
        conf = min(1.0, 0.5 + 0.5 * norm_strength) 
    elif want_short and rsi_sell_ok:
        action = "SELL"
        conf = min(1.0, 0.5 + 0.5 * norm_strength)
    else:
        action = "HOLD"
        conf = max(0.0, 0.5 - 0.5 * norm_strength)  

    if use_bb:
        if action == "BUY" and ema5 > bb_up:
            conf *= 0.8
        if action == "SELL" and ema5 < bb_dn:
            conf *= 0.8

    conf = float(round(max(0.0, min(1.0, conf)), 6))

    rationale = {
        "ema5": ema5, "ema20": ema20, "rsi14": rsi14,
        "atr14": atr14, "bb_up": bb_up, "bb_dn": bb_dn,
        "trend": trend, "norm_strength": norm_strength,
        "toggles": {"use_rsi": use_rsi, "use_bb": use_bb, "use_short": use_short},
        "thresholds": {"rsi_buy_min": rsi_buy_min, "rsi_sell_max": rsi_sell_max},
        "rules": {
            "ema_cross": "ema5>ema20 → BUY; ema5<ema20 → SELL (if enabled); else HOLD",
            "rsi_filter": (
                "enabled → BUY if RSI≥min; SELL if RSI≤max" if use_rsi
                else "disabled"
            ),
            "bb_damping": "enabled → dampen if price beyond band in trade direction" if use_bb else "disabled",
        },
    }
    return {"action": action, "confidence": conf, "rationale": rationale}

def apply_risk_layer(action_payload: Dict[str, Any], risk: RiskParams = RiskParams()) -> Dict[str, Any]:

    out = dict(action_payload)
    out["risk"] = {"risk_per_trade_pct": risk.risk_per_trade_pct}
    return out
