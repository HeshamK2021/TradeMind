from __future__ import annotations


import pandas as pd

from app.rl.dataset import make_windows, Split


def test_make_windows_basic_no_leakage():
    idx = pd.date_range("2024-01-01", periods=5000, freq="H", tz="UTC")

    train_span = 3000
    test_span = 500
    stride = 250

    splits = make_windows(idx, train_span, test_span, stride)
    assert len(splits) > 0, "Expected at least one window."

    for i, sp in enumerate(splits):
        train_len = (idx.get_indexer([sp.train_end])[0] - idx.get_indexer([sp.train_start])[0]) + 1
        test_len = (idx.get_indexer([sp.test_end])[0] - idx.get_indexer([sp.test_start])[0]) + 1
        assert train_len == train_span, f"Train length mismatch on window {i}"
        assert test_len == test_span, f"Test length mismatch on window {i}"

        assert sp.train_end < sp.test_start, f"Leakage: train_end >= test_start on window {i}"

        if i < len(splits) - 1:
            nxt = splits[i + 1]
            cur_start_i = idx.get_indexer([sp.train_start])[0]
            nxt_start_i = idx.get_indexer([nxt.train_start])[0]
            assert (nxt_start_i - cur_start_i) == stride, "Stride not respected."


def test_make_windows_insufficient_data_returns_empty():
    idx = pd.date_range("2024-01-01", periods=100, freq="H", tz="UTC")
    out = make_windows(idx, train_span=80, test_span=30, stride=10)
    assert out == [], "Expected empty list when not enough bars are available."
