from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
from torch import Tensor
from stable_baselines3 import PPO

try:
    from stable_baselines3.common.torch_utils import obs_as_tensor
except Exception:  
    import torch as _torch, numpy as _np
    def obs_as_tensor(obs, device):
        if isinstance(obs, _np.ndarray):
            return _torch.as_tensor(obs, device=device)
        return obs.to(device) if hasattr(obs, "to") else _torch.as_tensor(obs, device=device)

from .utils import json_load, FEATURE_ORDER, SimpleScaler
from . import LATEST_JSON, LATEST_DIR, LATEST_INDEX, latest_path_for

try:
    from app.data.session import session_scope
    from app.data.models import FeatureRow
    _HAS_DB = True
except Exception:
    _HAS_DB = False


@dataclass
class PolicyMeta:
    best_checkpoint: str
    algo: str
    version: str
    symbol: str
    tf: str
    action_space: str  


class RLPolicy:

    def __init__(self, model: PPO, scaler_state: Dict[str, Any], meta: PolicyMeta):
        self.model = model
        self.model.policy.eval()
        self.scaler = SimpleScaler(feature_order=FEATURE_ORDER)
        self.scaler.load_state_dict(scaler_state)
        self.meta = meta


        self._vec_names = [
            "ema_5", "ema_20", "ema_gap", "rsi_14", "atr_14", "bb_width", "ema_dev_mid"
        ]

    @classmethod
    def _load_meta(cls, symbol: Optional[str], tf: Optional[str]) -> Dict[str, Any]:

        if symbol and tf:
            p = latest_path_for(symbol, tf)
            return json_load(p)

        if LATEST_INDEX.exists():
            try:
                idx = json_load(LATEST_INDEX)
                if idx:
                    def keyfun(kv):
                        meta = kv[1]
                        return meta.get("updated_at", "") + "|" + kv[0]
                    key, meta = sorted(idx.items(), key=keyfun)[-1]
                    return json_load(meta["latest_path"])
            except Exception:
                pass

        return json_load(LATEST_JSON)

    @classmethod
    def latest(cls, symbol: Optional[str] = None, tf: Optional[str] = None) -> "RLPolicy":
        j = cls._load_meta(symbol, tf)
        ckpt = Path(j["best_checkpoint"])
        model = PPO.load(str(ckpt), device="cpu")
        scaler = json_load(ckpt.with_suffix(".scaler.json"))
        meta = PolicyMeta(
            best_checkpoint=str(ckpt),
            algo=str(j.get("algo", "ppo")),
            version=str(j.get("version", "2")),
            symbol=str(j.get("symbol")),
            tf=str(j.get("tf")),
            action_space=str(j.get("action_space", "discrete2")),
        )
        return cls(model, scaler, meta)

    def _vectorize(self, feat: Dict[str, float]) -> np.ndarray:

        ema5 = float(feat["ema_5"]); ema20 = float(feat["ema_20"])
        rsi = float(feat["rsi_14"]); atr = float(feat["atr_14"])
        bb_mid = float(feat["bb_mid"]); bb_up = float(feat["bb_up"]); bb_dn = float(feat["bb_dn"])

        x_raw = np.array(
            [
                ema5,
                ema20,
                ema5 - ema20,          
                rsi,
                atr,
                bb_up - bb_dn,        
                (ema5 - bb_mid),       
            ],
            dtype=np.float32,
        )[None, :]  

        x_scaled = self.scaler.transform(x_raw)  
        return x_scaled

    def _labels(self) -> List[str]:
        return ["HOLD", "BUY", "SELL"] if self.meta.action_space == "discrete3" else ["HOLD", "BUY"]

    def _logits(self, obs_t: Tensor) -> Tensor:

        latent_pi, _ = self.model.policy._get_latent(obs_t)  
        logits: Tensor = self.model.policy.action_net(latent_pi)  
        return logits

    def predict(self, feat: Dict[str, float]) -> Dict[str, Any]:

        x = self._vectorize(feat) 
        obs_t = obs_as_tensor(x, self.model.policy.device)

        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            if hasattr(dist, "distribution") and hasattr(dist.distribution, "probs"):
                probs = dist.distribution.probs.squeeze(0)  
            else:
                logits = self._logits(obs_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0)

            probs_np = probs.detach().cpu().numpy().astype(np.float64)
            action = int(probs_np.argmax())

        labels = self._labels()
        action = min(action, len(labels) - 1)
        label = labels[action]
        confidence = float(probs_np[action])

        prob_map = {labels[i].lower(): float(probs_np[i]) for i in range(len(labels))}
        return {"action": label, "confidence": confidence, "probs": prob_map}

    def explain(
        self,
        feat: Dict[str, float],
        method: str = "ig",
        steps: int = 32,
        target_action: Optional[int] = None,
    ) -> Dict[str, Any]:
       
        pred = self.predict(feat)
        labels = self._labels()
        action_idx = labels.index(pred["action"]) if target_action is None else int(target_action)

        x_scaled: np.ndarray = self._vectorize(feat)  
        x_t = torch.tensor(x_scaled, dtype=torch.float32, device=self.model.policy.device, requires_grad=True)

        if method.lower() == "grad":
            logits = self._logits(x_t)          
            out = logits[0, action_idx]          
            self.model.policy.zero_grad(set_to_none=True)
            if x_t.grad is not None: x_t.grad.zero_()
            out.backward()
            grad = x_t.grad.detach().cpu().numpy()[0]   
            attr = (x_scaled[0] * grad).astype(np.float64)  
        else:
            baseline = torch.zeros_like(x_t)
            total_grad = torch.zeros_like(x_t)
            for alpha in torch.linspace(0, 1, steps, device=x_t.device):
                x_interp = baseline + alpha * (x_t - baseline)
                x_interp.requires_grad_(True)
                logits = self._logits(x_interp)
                out = logits[0, action_idx]
                self.model.policy.zero_grad(set_to_none=True)
                if x_interp.grad is not None: x_interp.grad.zero_()
                out.backward(retain_graph=True)
                total_grad += x_interp.grad
            avg_grad = total_grad / float(steps)
            attr_t = (x_t - baseline) * avg_grad
            attr = attr_t.detach().cpu().numpy()[0].astype(np.float64)

        feature_vals_raw = self._reconstruct_raw_vector(feat)  
        recs = []
        for i, name in enumerate(self._vec_names):
            v = float(attr[i])
            fv = float(feature_vals_raw[i])
            direction = "pro-buy" if v > 0 else ("pro-sell" if v < 0 else "neutral")
            recs.append({
                "name": name,
                "value": v,
                "abs": abs(v),
                "direction": direction,
                "feature_value": fv,
            })
        recs.sort(key=lambda r: r["abs"], reverse=True)
        top_features = [r["name"] for r in recs[:3]]

        return {
            "action": pred["action"],
            "confidence": pred["confidence"],
            "probs": pred["probs"],
            "attributions": recs,
            "top_features": top_features,
        }

    def predict_with_explanation(
        self,
        feat: Dict[str, float],
        method: str = "ig",
        steps: int = 32,
        include_rationale: bool = True,
    ) -> Dict[str, Any]:

        pred = self.predict(feat)
        exp = self.explain(feat, method=method, steps=steps, target_action=None)

        out: Dict[str, Any] = {**pred, "explanation": exp}

        if include_rationale:
            out["rationale"] = self._human_rationale(feat, exp)
        return out

    def predict_from_db_latest(self, symbol: str, tf: str) -> Dict[str, Any]:
        if not _HAS_DB:
            raise RuntimeError("No DB access in this context")
        feat = self._fetch_latest_feature_row(symbol, tf)
        return self.predict(feat)

    def explain_from_db_latest(
        self, symbol: str, tf: str, method: str = "ig", steps: int = 32, include_rationale: bool = True
    ) -> Dict[str, Any]:
        if not _HAS_DB:
            raise RuntimeError("No DB access in this context")
        feat = self._fetch_latest_feature_row(symbol, tf)
        return self.predict_with_explanation(feat, method=method, steps=steps, include_rationale=include_rationale)

    def _fetch_latest_feature_row(self, symbol: str, tf: str) -> Dict[str, float]:
        with session_scope() as s:
            row = (
                s.query(
                    FeatureRow.ema_5, FeatureRow.ema_20, FeatureRow.rsi_14,
                    FeatureRow.atr_14, FeatureRow.bb_mid, FeatureRow.bb_up, FeatureRow.bb_dn
                )
                .filter_by(symbol=symbol, tf=tf, shifted=True)
                .order_by(FeatureRow.ts.desc())
                .first()
            )
        if row is None:
            raise RuntimeError("No shifted feature row found")
        feat = dict(
            ema_5=float(row[0]), ema_20=float(row[1]), rsi_14=float(row[2]),
            atr_14=float(row[3]), bb_mid=float(row[4]), bb_up=float(row[5]), bb_dn=float(row[6]),
        )
        return feat

    def _reconstruct_raw_vector(self, feat: Dict[str, float]) -> List[float]:

        ema5 = float(feat["ema_5"]); ema20 = float(feat["ema_20"])
        rsi = float(feat["rsi_14"]); atr = float(feat["atr_14"])
        bb_mid = float(feat["bb_mid"]); bb_up = float(feat["bb_up"]); bb_dn = float(feat["bb_dn"])
        return [
            ema5,
            ema20,
            ema5 - ema20,
            rsi,
            atr,
            bb_up - bb_dn,
            (ema5 - bb_mid),
        ]

    def _human_rationale(self, feat: Dict[str, float], exp: Dict[str, Any]) -> Dict[str, Any]:
       
        ema5 = float(feat["ema_5"]); ema20 = float(feat["ema_20"])
        rsi = float(feat["rsi_14"]); atr = float(feat["atr_14"])
        bb_mid = float(feat["bb_mid"]); bb_up = float(feat["bb_up"]); bb_dn = float(feat["bb_dn"])

        ema_gap = ema5 - ema20
        bb_width = bb_up - bb_dn
        ema_dev_mid = ema5 - bb_mid

        facts: List[str] = []
        if ema_gap > 0:
            pct = (ema_gap / max(abs(ema20), 1e-9)) * 100.0
            facts.append(f"Fast EMA above slow by {pct:.2f}% (bullish bias).")
        elif ema_gap < 0:
            pct = (-ema_gap / max(abs(ema20), 1e-9)) * 100.0
            facts.append(f"Fast EMA below slow by {pct:.2f}% (bearish bias).")
        else:
            facts.append("EMAs roughly equal (neutral trend).")

        if rsi > 60:
            facts.append(f"RSI14 at {rsi:.1f} (strength).")
        elif rsi < 40:
            facts.append(f"RSI14 at {rsi:.1f} (weakness).")
        else:
            facts.append(f"RSI14 at {rsi:.1f} (balanced).")

        facts.append(f"BB width {bb_width:.4f}; ATR {atr:.4f} (volatility context).")

        if ema_dev_mid > 0:
            facts.append("Fast EMA above BB mid (upward tilt).")
        elif ema_dev_mid < 0:
            facts.append("Fast EMA below BB mid (downward tilt).")

        attr_msgs: List[str] = []
        for r in (exp.get("attributions") or [])[:3]:
            direction = r.get("direction")
            name = r.get("name")
            fv = r.get("feature_value")
            if direction == "pro-buy":
                attr_msgs.append(f"{name} (+) supports BUY (value {fv:.4f}).")
            elif direction == "pro-sell":
                attr_msgs.append(f"{name} (−) supports SELL (value {fv:.4f}).")
            else:
                attr_msgs.append(f"{name} neutral (value {fv:.4f}).")

        return {
            "summary": facts[:3],         
            "attribution_top": attr_msgs,  
        }
