from __future__ import annotations
from typing import Literal, Optional, Dict, Any, List

from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import json

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.rl.policy import RLPolicy
from app.rl.eval import evaluate_oos
from app.rl.train import TrainSpec, train_walk_forward
from app.rl.compare import compare_run_dir_vs_latest

from app.rl.utils import json_load, write_pair_latest
from app.rl import latest_path_for

from json import JSONDecodeError
import logging

_BASELINE_OK = False
try:
    from app.backtest.strategy_rules import signal_baseline  
    _BASELINE_OK = True
except Exception:
    signal_baseline = None  

_HAS_DB = False
try:
    from app.data.session import session_scope
    from app.data.models import FeatureRow
    _HAS_DB = True
except Exception:
    pass

import pandas as pd
from pandas.errors import EmptyDataError


router = APIRouter(prefix="/rl", tags=["rl"])



class FeaturePayload(BaseModel):
    ema_5: float
    ema_20: float
    rsi_14: float
    atr_14: float
    bb_mid: float
    bb_up: float
    bb_dn: float


class PredictResponse(BaseModel):
    engine: Literal["rl"]
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    meta: Dict[str, Any] = Field(default_factory=dict)


class EvalRequest(BaseModel):
    symbol: str
    tf: str
    fees_bps: float = 10.0
    slippage_bps: float = 1.0
    use_short: bool = False
    use_atr_stop: bool = False
    atr_k: float = 2.0
    atr_penalty: float = 0.0


class EvalResponse(BaseModel):
    csv: str
    metrics_json: str
    png: str


class TrainRequest(BaseModel):
    symbol: str
    tf: str
    steps: int = 120_000
    train_span: int = 3000
    test_span: int = 500
    stride: int = 250
    seed: int = 42
    fees_bps: float = 10.0
    slippage_bps: float = 1.0
    use_short: bool = False
    short_fee_bps: float = 0.0
    use_atr_stop: bool = False
    atr_k: float = 2.0
    atr_penalty: float = 0.0


class TrainResponse(BaseModel):
    latest_path: str
    meta: Dict[str, Any]


class RecommendationResponse(BaseModel):
    engine: Literal["rl", "both"]
    rl: PredictResponse
    baseline: Optional[Dict[str, Any]] = None


class WindowsRow(BaseModel):
    window: int
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    equity_final: Optional[float] = None
    baseline_equity_final: Optional[float] = None
    rl_beats_baseline: Optional[bool] = None


class WindowsResponse(BaseModel):
    rows: List[WindowsRow]
    wins: Optional[int] = None
    total: Optional[int] = None
    run_dir: Optional[str] = None


class HistoryRow(BaseModel):
    ts: int
    ts_iso: str
    action: str
    confidence: Optional[float] = None
    close: Optional[float] = None


class HistoryResponse(BaseModel):
    symbol: str
    tf: str
    limit: int
    rows: List[HistoryRow]


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except EmptyDataError:
        return None
    except Exception:
        return None
    return None



@router.post("/predict", response_model=PredictResponse)
def rl_predict(payload: FeaturePayload) -> PredictResponse:
    policy = RLPolicy.latest()  
    out = policy.predict(payload.model_dump())
    return PredictResponse(
        engine="rl", action=out["action"], confidence=float(out["confidence"]),
        meta=dict(
            algo=policy.meta.algo,
            version=policy.meta.version,
            symbol=policy.meta.symbol,
            tf=policy.meta.tf,
            checkpoint=policy.meta.best_checkpoint,
            action_space=policy.meta.action_space,
        ),
    )


@router.get("/predict/latest", response_model=PredictResponse)
def rl_predict_latest(symbol: str, tf: str) -> PredictResponse:
    policy = RLPolicy.latest(symbol, tf)
    out = policy.predict_from_db_latest(symbol, tf)
    return PredictResponse(
        engine="rl", action=out["action"], confidence=float(out["confidence"]),
        meta=dict(
            algo=policy.meta.algo,
            version=policy.meta.version,
            symbol=symbol, tf=tf,
            checkpoint=policy.meta.best_checkpoint,
            action_space=policy.meta.action_space,
        ),
    )



@router.get("/recommendation", response_model=RecommendationResponse)
def recommendation(
    symbol: str,
    tf: str,
    engine: Literal["rl", "both"] = "rl",
) -> RecommendationResponse:
    rl = rl_predict_latest(symbol=symbol, tf=tf)
    baseline_block = None

    if engine == "both" and _BASELINE_OK and _HAS_DB:
        with session_scope() as s:
            row = (
                s.query(
                    FeatureRow.ema_5, FeatureRow.ema_20, FeatureRow.rsi_14,
                    FeatureRow.atr_14, FeatureRow.bb_mid, FeatureRow.bb_up, FeatureRow.bb_dn,
                )
                .filter_by(symbol=symbol, tf=tf, shifted=True)
                .order_by(FeatureRow.ts.desc())
                .first()
            )
        if row:
            feats = dict(
                ema_5=float(row[0]), ema_20=float(row[1]), rsi_14=float(row[2]),
                atr_14=float(row[3]), bb_mid=float(row[4]), bb_up=float(row[5]), bb_dn=float(row[6]),
            )
            try:
                res = signal_baseline(feats)  
                if isinstance(res, dict):
                    b_action = str(res.get("action", "HOLD")).upper()
                    if b_action not in ("BUY", "HOLD", "SELL"):
                        b_action = "HOLD"
                    baseline_block = {
                        "action": b_action,
                        "confidence": float(res.get("confidence", 0.0)),
                        "source": "baseline_rules",
                    }
                else:
                    a = int(res)
                    b_action = "BUY" if a == 1 else ("SELL" if a == 2 else "HOLD")
                    baseline_block = {"action": b_action, "source": "baseline_rules"}
            except Exception as e:
                baseline_block = {"error": f"baseline_rules_failed: {e.__class__.__name__}"}
        else:
            baseline_block = {"error": "no_shifted_row"}

    return RecommendationResponse(engine=engine, rl=rl, baseline=baseline_block)



@router.post("/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest) -> EvalResponse:
    latest_path = latest_path_for(req.symbol, req.tf)
    if not latest_path.exists():
        raise HTTPException(status_code=404, detail=f"Latest checkpoint not found for {req.symbol} {req.tf}")
    try:
        out = evaluate_oos(
            symbol=req.symbol, tf=req.tf,
            fees_bps=req.fees_bps, slippage_bps=req.slippage_bps,
            use_short=req.use_short,
            use_atr_stop=req.use_atr_stop, atr_k=req.atr_k, atr_penalty=req.atr_penalty,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    for k in ("csv", "metrics_json", "png"):
        if k not in out:
            raise HTTPException(status_code=400, detail="Evaluation produced no outputs (no windows?)")

    try:
        src_csv = Path(out.get("csv", ""))
        if src_csv.exists() and src_csv.is_file():
            runs_root = Path("artifacts/rl/runs")
            runs_root.mkdir(parents=True, exist_ok=True)
            df = pd.read_csv(src_csv)
            df.to_csv(runs_root / "rl_windows.csv", index=False)
    except Exception:
        pass

    return EvalResponse(**out)



@router.post("/train", response_model=TrainResponse)
def train_endpoint(req: TrainRequest) -> TrainResponse:
    spec = TrainSpec(
        symbol=req.symbol, tf=req.tf,
        fees_bps=req.fees_bps, slippage_bps=req.slippage_bps,
        steps=req.steps, train_span=req.train_span, test_span=req.test_span, stride=req.stride,
        seed=req.seed,
        use_short=req.use_short, short_fee_bps=req.short_fee_bps,
        use_atr_stop=req.use_atr_stop, atr_k=req.atr_k, atr_penalty=req.atr_penalty,
    )
    meta = train_walk_forward(spec)

    if meta.get("symbol") and meta.get("tf"):
        write_pair_latest(meta)

    lp = latest_path_for(req.symbol, req.tf)
    return TrainResponse(latest_path=str(lp), meta=meta)



class RetrainPayload(BaseModel):
    symbol: str
    tf: str
    steps: int = 80_000
    fees_bps: float = 10
    slippage_bps: float = 1
    use_short: bool = True
    short_fee_bps: float = 2.0
    use_atr_stop: bool = False
    atr_k: float = 2.0
    atr_penalty: float = 0.0
    promote_if_better: bool = True
    dry_run: bool = False


@router.post("/retrain")
def retrain_endpoint(p: RetrainPayload):
    spec = TrainSpec(
        symbol=p.symbol, tf=p.tf,
        fees_bps=p.fees_bps, slippage_bps=p.slippage_bps,
        steps=p.steps,
    )
    for k, v in dict(
        use_short=p.use_short,
        short_fee_bps=p.short_fee_bps,
        use_atr_stop=p.use_atr_stop,
        atr_k=p.atr_k,
        atr_penalty=p.atr_penalty,
    ).items():
        if hasattr(spec, k):
            setattr(spec, k, v)

    train_meta = train_walk_forward(spec)
    new_run_dir = Path(train_meta.get("run_dir", ""))

    if not new_run_dir.exists():
        raise HTTPException(status_code=500, detail="Train finished but run_dir missing")

    try:
        eval_out = evaluate_oos(
            symbol=p.symbol, tf=p.tf,
            fees_bps=p.fees_bps, slippage_bps=p.slippage_bps,
            use_short=p.use_short,
            use_atr_stop=p.use_atr_stop, atr_k=p.atr_k, atr_penalty=p.atr_penalty,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Evaluation failed: {e}")

    old_latest_path = latest_path_for(p.symbol, p.tf)
    old_latest_meta_path = old_latest_path if old_latest_path.exists() else None

    better, cmp_details = compare_run_dir_vs_latest(new_run_dir, old_latest_meta_path)

    status = "held"
    promotion_reason = cmp_details.get("reason", "")
    promoted_path = None

    if better:
        if p.promote_if_better and not p.dry_run:
            meta_to_write = dict(train_meta)
            meta_to_write["symbol"] = p.symbol
            meta_to_write["tf"] = p.tf
            promoted_path = str(write_pair_latest(meta_to_write))
            status = "promoted"
        elif p.dry_run:
            status = "dry_run_promote"
        else:
            status = "better_but_not_promoted"
    else:
        status = "worse_or_equal"

    return {
        "pair": f"{p.symbol}_{p.tf}",
        "status": status,
        "promotion_reason": promotion_reason,
        "promoted_latest_path": promoted_path,
        "compare": cmp_details,
        "train": train_meta,
        "eval": eval_out,
    }



@router.get("/latest/index")
def list_latest_index():
    from app.rl import RL_DIR
    from app.rl.utils import json_load

    idx_paths = [
        RL_DIR / "latest" / "index.json",
        RL_DIR / "latest_index.json",
    ]
    idx = None
    chosen = None
    for p in idx_paths:
        if p.exists():
            try:
                idx = json_load(p)
                chosen = p
                break
            except Exception:
                pass

    if not idx:
        return {"pairs": [], "note": "no latest index yet; call /rl/latest/rebuild"}

    out = []
    for k, v in idx.items():
        out.append({
            "key": k,
            "symbol": v.get("symbol"),
            "tf": v.get("tf"),
            "path": v.get("path"),
            "mtime": v.get("mtime"),
        })
    return {"pairs": out, "source": str(chosen)}


def _rebuild_latest_index() -> dict:
    from app.rl import RL_DIR
    from app.rl.utils import json_load, json_dump

    latest_dir = RL_DIR / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    idx_path = latest_dir / "index.json"
    idx: dict = {}

    for p in latest_dir.glob("*.json"):
        if p.name == "index.json":
            continue
        try:
            j = json_load(p)
        except Exception:
            continue
        symbol = j.get("symbol")
        tf = j.get("tf")
        if not symbol or not tf:
            continue
        key = f"{symbol.replace('/','')}_{tf}"
        idx[key] = {
            "path": str(p.resolve()),
            "symbol": symbol,
            "tf": tf,
            "mtime": j.get("mtime") or j.get("updated_at") or None,
        }

    json_dump(idx, idx_path)
    return {"rebuilt": True, "pairs": len(idx), "index_path": str(idx_path)}


@router.post("/latest/rebuild")
def latest_rebuild():
    return _rebuild_latest_index()



@router.get("/windows", response_model=WindowsResponse)
def windows_endpoint(symbol: str, tf: str) -> WindowsResponse:
 
    lp = latest_path_for(symbol, tf)
    if not lp.exists():
        raise HTTPException(status_code=404, detail=f"Latest checkpoint not found for {symbol} {tf}")

    meta = json_load(lp)
    run_dir = Path(meta.get("run_dir", "artifacts/rl/runs"))

    rl_csv = run_dir / "rl_windows.csv"
    bl_csv = run_dir / "baseline_windows.csv"

    rl_df = _read_csv_safe(rl_csv)
    if rl_df is None or rl_df.empty:
        return WindowsResponse(rows=[], wins=0, total=0, run_dir=str(run_dir))

    bl_df = _read_csv_safe(bl_csv)

    keep = [c for c in [
        "window", "train_start", "train_end", "test_start", "test_end", "equity_final"
    ] if c in rl_df.columns]
    rl_df = rl_df[keep].copy()

    rows: List[WindowsRow] = []

    if bl_df is not None and not bl_df.empty:
        key_cols = [c for c in ["window", "test_start", "test_end"] if c in rl_df.columns and c in bl_df.columns]
        merged = rl_df.merge(bl_df, on=key_cols, how="left", suffixes=("", "_baseline"))

        if "baseline_equity_final" not in merged.columns and "equity_final_baseline" in merged.columns:
            merged.rename(columns={"equity_final_baseline": "baseline_equity_final"}, inplace=True)

        if "baseline_equity_final" in merged.columns:
            merged["rl_beats_baseline"] = merged["equity_final"] > merged["baseline_equity_final"]

        for _, r in merged.iterrows():
            rows.append(WindowsRow(
                window=int(r.get("window")),
                train_start=r.get("train_start"),
                train_end=r.get("train_end"),
                test_start=r.get("test_start"),
                test_end=r.get("test_end"),
                equity_final=float(r.get("equity_final")) if pd.notna(r.get("equity_final")) else None,
                baseline_equity_final=float(r.get("baseline_equity_final")) if pd.notna(r.get("baseline_equity_final")) else None,
                rl_beats_baseline=bool(r.get("rl_beats_baseline")) if "rl_beats_baseline" in r else None,
            ))
        wins = int(merged["rl_beats_baseline"].sum()) if "rl_beats_baseline" in merged.columns else None
        total = int(len(merged))
        return WindowsResponse(rows=rows, wins=wins, total=total, run_dir=str(run_dir))

    for _, r in rl_df.iterrows():
        rows.append(WindowsRow(
            window=int(r.get("window")),
            train_start=r.get("train_start"),
            train_end=r.get("train_end"),
            test_start=r.get("test_start"),
            test_end=r.get("test_end"),
            equity_final=float(r.get("equity_final")) if pd.notna(r.get("equity_final")) else None,
        ))
    return WindowsResponse(rows=rows, wins=None, total=len(rows), run_dir=str(run_dir))



@router.get("/history", response_model=HistoryResponse)
def history_endpoint(symbol: str, tf: str, limit: int = 50) -> HistoryResponse:
  
    if not _HAS_DB:
        raise HTTPException(status_code=503, detail="DB access not available")

    log = logging.getLogger(__name__)

    policy = None
    try:
        policy = RLPolicy.latest(symbol, tf)
    except (FileNotFoundError, JSONDecodeError, ValueError) as e:
        log.warning("RLPolicy.latest unavailable for %s %s: %s", symbol, tf, e)
        policy = None
    except Exception as e:
        log.exception("RLPolicy.latest hard failure for %s %s", symbol, tf)
        policy = None

    rows_out: List[HistoryRow] = []

    _has_candle = False
    try:
        from app.data.models import Candle  
        _has_candle = True
    except Exception:
        pass

    with session_scope() as s:
        q = (
            s.query(
                FeatureRow.ts,
                FeatureRow.ema_5, FeatureRow.ema_20, FeatureRow.rsi_14,
                FeatureRow.atr_14, FeatureRow.bb_mid, FeatureRow.bb_up, FeatureRow.bb_dn,
            )
            .filter_by(symbol=symbol, tf=tf, shifted=True)
            .order_by(FeatureRow.ts.desc())
            .limit(max(1, int(limit)))
        )
        recs = q.all()

        closes_by_ts: Dict[int, float] = {}
        if _has_candle and recs:
            try:
                ts_list = [int(r[0]) for r in recs]
                crows = (
                    s.query(Candle.ts, Candle.close)
                    .filter(Candle.symbol == symbol)
                    .filter(Candle.tf == tf)
                    .filter(Candle.ts.in_(ts_list))
                    .all()
                )
                closes_by_ts = {int(ts): float(c) for ts, c in crows}
            except Exception:
                closes_by_ts = {}

    for r in recs:
        ts_ms = int(r[0])
        feat = dict(
            ema_5=float(r[1]), ema_20=float(r[2]), rsi_14=float(r[3]),
            atr_14=float(r[4]), bb_mid=float(r[5]), bb_up=float(r[6]), bb_dn=float(r[7]),
        )

        action = "N/A"
        confidence: Optional[float] = None

        if policy is not None:
            try:
                pred = policy.predict(feat) or {}
                action = str(pred.get("action", "N/A"))
                if "confidence" in pred and pred["confidence"] is not None:
                    confidence = float(pred["confidence"])
            except Exception as e:
                log.warning("policy.predict failed for %s %s ts=%s: %s", symbol, tf, ts_ms, e)
                action = "N/A"
                confidence = None

        rows_out.append(HistoryRow(
            ts=ts_ms,
            ts_iso=pd.to_datetime(ts_ms, unit="ms", utc=True).isoformat(),
            action=action,
            confidence=confidence,
            close=closes_by_ts.get(ts_ms),
        ))

    return HistoryResponse(symbol=symbol, tf=tf, limit=limit, rows=rows_out)



class ExplainResponse(BaseModel):
    symbol: str
    tf: str
    ts: Optional[int] = None
    ts_iso: Optional[str] = None
    action: str
    confidence: float
    probs: Dict[str, float]
    features: Dict[str, float]
    reasons: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of human-readable reasons with weights (positive boosts, negative penalties).",
    )
    text: str = Field("", description="Concise natural language rationale.")


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

    if ema20 != 0:
        ema_gap_pct = 100.0 * (ema5 - ema20) / abs(ema20)
    else:
        ema_gap_pct = 0.0

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

    band_w = max(1e-9, bb_up - bb_dn)
    if close is not None:
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


@router.get("/explain", response_model=ExplainResponse)
def explain_endpoint(symbol: str, tf: str) -> ExplainResponse:
   
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
