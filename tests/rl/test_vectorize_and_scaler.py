from __future__ import annotations
import numpy as np

from app.rl.utils import SimpleScaler, FEATURE_ORDER

# Recreate the vectorization that RLPolicy uses:
# [ema5, ema20, ema5-ema20, rsi, atr, bb_up-bb_dn, ema5-bb_mid]
def _vectorize_like_policy(feat: dict) -> np.ndarray:
    ema5 = float(feat["ema_5"]); ema20 = float(feat["ema_20"])
    rsi = float(feat["rsi_14"]); atr = float(feat["atr_14"])
    bb_mid = float(feat["bb_mid"]); bb_up = float(feat["bb_up"]); bb_dn = float(feat["bb_dn"])
    arr = np.array(
        [ema5, ema20, ema5 - ema20, rsi, atr, bb_up - bb_dn, (ema5 - bb_mid)],
        dtype=np.float32,
    )
    return arr


def test_vectorize_values_and_dtype():
    feat = dict(
        ema_5=10.0,
        ema_20=9.5,
        rsi_14=55.0,
        atr_14=0.25,
        bb_mid=9.8,
        bb_up=10.5,
        bb_dn=9.0,
    )
    v = _vectorize_like_policy(feat)
    assert v.dtype == np.float32

    expected = np.array(
        [10.0, 9.5, 0.5, 55.0, 0.25, 1.5, 0.2],
        dtype=np.float32,
    )
    assert np.allclose(v, expected)


def test_simple_scaler_roundtrip_and_transform():
    X = np.array(
        [
            [1.0, 2.0,  3.0],
            [2.0, 4.0,  6.0],
            [3.0, 6.0,  9.0],
        ],
        dtype=np.float32,
    )
    sc = SimpleScaler(feature_order=["a", "b", "c"])
    sc.fit(X)
    Z = sc.transform(X)
    assert Z.shape == X.shape
    assert Z.dtype == np.float32
    # standardized
    assert np.allclose(Z.mean(axis=0), np.zeros(3, dtype=np.float32), atol=1e-6)
    assert np.allclose(Z.std(axis=0),  np.ones(3,  dtype=np.float32), atol=1e-6)



    # state dict roundtrip
    state = sc.state_dict()
    sc2 = SimpleScaler(feature_order=["a", "b", "c"])
    sc2.load_state_dict(state)
    Z2 = sc2.transform(X)
    assert np.allclose(Z, Z2, atol=0)
