from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np


@dataclass
class EnvConfig:
    fees_bps: float
    slippage_bps: float
    normalize: bool = True
    fit_scaler_on_reset: bool = False
    use_short: bool = True                
    use_atr_stop: bool = False             
    atr_k: float = 2.0                   
    atr_penalty_lambda: float = 0.0        
    short_fee_bps: float = 0.0             

class SimpleScaler:
    def __init__(self, feature_order): ...
    def fit(self, x: np.ndarray): ...
    def transform(self, x: np.ndarray) -> np.ndarray: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, st: Dict[str, Any]): ...

FEATURE_ORDER = [
    "ema_5", "ema_20", "ema_spread", "rsi_14",
    "atr_14", "bb_width", "ema5_minus_bbmid"
]

def vectorize_features(feat_row: Dict[str, float]) -> np.ndarray:
    ema5 = float(feat_row["ema_5"])
    ema20 = float(feat_row["ema_20"])
    rsi = float(feat_row["rsi_14"])
    atr = float(feat_row["atr_14"])
    bb_mid = float(feat_row["bb_mid"])
    bb_up = float(feat_row["bb_up"])
    bb_dn = float(feat_row["bb_dn"])
    return np.array([
        ema5,
        ema20,
        ema5 - ema20,          
        rsi,
        atr,                 
        bb_up - bb_dn,          
        (ema5 - bb_mid),       
    ], dtype=np.float32)

class TradingEnv:

    def __init__(
        self,
        feats_df,                      
        close_s,                      
        config: EnvConfig,
        scaler: Optional[SimpleScaler] = None,
        ohlc_df=None                  
    ):
        self._feats_df = feats_df
        self._close = close_s
        self._ohlc = ohlc_df
        self.cfg = config
        self._scaler = scaler
        self._fit_done = False

        self._i0 = 0
        self._i1 = len(self._feats_df) - 1  
        self.t = None
        self.prev_pos = 0  
        self.pos = 0

        self._X = np.vstack([vectorize_features(self._feats_df.iloc[i].to_dict())
                             for i in range(len(self._feats_df))]).astype(np.float32)

        if self.cfg.normalize and self._scaler is not None and self.cfg.fit_scaler_on_reset:
            pass
        elif self.cfg.normalize and self._scaler is None:
            raise ValueError("normalize=True requires a scaler instance")

        if self.cfg.use_atr_stop and self._ohlc is None:
            pass

    def action_space_n(self) -> int:
        return 3 if self.cfg.use_short else 2

    def observation_space_shape(self) -> Tuple[int]:
        return (self._X.shape[1],)

    def _slice_indices(self, start_idx=None, end_idx=None, start_ts=None, end_ts=None):
        if start_idx is not None and end_idx is not None:
            si, ei = int(start_idx), int(end_idx)
        else:
            idx = self._feats_df.index
            def _loc(ts):
                loc = idx.get_indexer([ts])[0]
                if loc == -1:
                    loc = idx.get_indexer([ts], method="ffill")[0]
                return int(loc)
            si = _loc(start_ts) if start_ts is not None else 0
            ei = _loc(end_ts) if end_ts is not None else len(idx) - 1
        if si >= ei:
            raise ValueError("Invalid slice: start >= end")
        return si, ei

    def reset(self, start_idx=None, end_idx=None, start_ts=None, end_ts=None):
        self._i0, self._i1 = self._slice_indices(start_idx, end_idx, start_ts, end_ts)
        self._i1 = min(self._i1, len(self._X) - 2)

        if self.cfg.normalize and self._scaler is not None and self.cfg.fit_scaler_on_reset:
            self._scaler.fit(self._X[self._i0:self._i1+1])
            self._fit_done = True

        self.t = self._i0
        self.prev_pos = 0
        self.pos = 0
        return self._obs()

    def _obs(self):
        x = self._X[self.t]
        if self.cfg.normalize and self._scaler is not None:
            if not self._fit_done and self.cfg.fit_scaler_on_reset:
                raise RuntimeError("Scaler fit expected before transform")
            x = self._scaler.transform(x[None, :])[0]
        return x.astype(np.float32)

    def _flip_cost(self, new_pos: int, old_pos: int) -> float:
        flip_units = abs(new_pos - old_pos)
        return flip_units * (self.cfg.fees_bps + self.cfg.slippage_bps) / 10_000.0

    def _short_fee(self, pos: int) -> float:
        if pos < 0 and self.cfg.short_fee_bps > 0.0:
            return abs(pos) * self.cfg.short_fee_bps / 10_000.0
        return 0.0

    def _atr_proxy_penalty(self, pos: int, atr_val: float, ret: float) -> float:
        if self.cfg.atr_penalty_lambda <= 0.0:
            return 0.0
        adverse = -pos * ret  
        thresh = self.cfg.atr_k * float(atr_val)
        overflow = max(0.0, adverse - thresh)
        return self.cfg.atr_penalty_lambda * overflow

    def _apply_atr_stop(self, entry_price: float, next_bar_ohlc: Dict[str, float], pos: int) -> Tuple[float, bool]:
    
        if next_bar_ohlc is None:
            raise RuntimeError("ATR stop enabled but OHLC not provided")
        low = float(next_bar_ohlc["low"])
        high = float(next_bar_ohlc["high"])
        close = float(next_bar_ohlc["close"])
        atr = float(self._feats_df.iloc[self.t]["atr_14"])  

        stop_dist_abs = self.cfg.atr_k * atr
        if pos > 0:
            stop_price = entry_price - stop_dist_abs
            if low <= stop_price:
                r = (stop_price / entry_price) - 1.0
                return r, True
            else:
                return (close / entry_price) - 1.0, False
        elif pos < 0:
            stop_price = entry_price + stop_dist_abs  
            if high >= stop_price:
                r = (entry_price / stop_price) - 1.0
                return r, True
            else:
                return (entry_price / close) - 1.0, False
        else:
            return 0.0, False

    def step(self, action: int):
        if self.t is None:
            raise RuntimeError("Call reset() before step()")

        if self.cfg.use_short:
            pos_t = 0 if action == 0 else (1 if action == 1 else -1)
        else:
            pos_t = 1 if action == 1 else 0

        if self.t >= self._i1:
            done = True
            info = {"reason": "end_of_window"}
            return self._obs(), 0.0, done, info

        c_t = float(self._close.iloc[self.t])
        c_tp1 = float(self._close.iloc[self.t + 1])
        raw_ret = (c_tp1 / c_t) - 1.0

        stopped = False
        eff_ret = raw_ret
        if self.cfg.use_atr_stop:
            if self._ohlc is None:
                raise RuntimeError("use_atr_stop=True requires OHLC dataframe aligned with close")
            next_ohlc = self._ohlc.iloc[self.t + 1][["open", "high", "low", "close"]].to_dict()
            eff_ret, stopped = self._apply_atr_stop(entry_price=c_t, next_bar_ohlc=next_ohlc, pos=pos_t)

        reward = pos_t * (eff_ret if self.cfg.use_atr_stop else raw_ret)

        reward -= self._flip_cost(pos_t, self.prev_pos)
        reward -= self._short_fee(pos_t)

        if not self.cfg.use_atr_stop and self.cfg.atr_penalty_lambda > 0.0:
            atr_val = float(self._feats_df.iloc[self.t]["atr_14"])
            reward -= self._atr_proxy_penalty(pos_t, atr_val, raw_ret)

        self.prev_pos = self.pos
        self.pos = 0 if stopped else pos_t
        self.t += 1
        done = (self.t >= self._i1)

        info = {"ret": raw_ret, "pos": self.pos, "stopped": stopped}
        return self._obs(), float(reward), done, info
