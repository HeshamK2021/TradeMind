from __future__ import annotations
import numpy as np

from app.rl.utils import SimpleScaler

def test_scaler_state_dict_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 7)).astype(np.float32)

    sc = SimpleScaler(feature_order=["f1","f2","f3","f4","f5","f6","f7"])
    sc.fit(X)
    Z = sc.transform(X)

    # Save/Load
    state = sc.state_dict()
    sc2 = SimpleScaler(feature_order=["f1","f2","f3","f4","f5","f6","f7"])
    sc2.load_state_dict(state)
    Z2 = sc2.transform(X)

    assert Z.shape == Z2.shape
    assert np.allclose(Z, Z2)
