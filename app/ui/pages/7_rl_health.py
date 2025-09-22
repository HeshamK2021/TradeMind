from __future__ import annotations

import os
import sys
import json
import time
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rl import LATEST_DIR, LATEST_INDEX
from app.rl.utils import json_load

API_BASE = os.environ.get("API_BASE", "").rstrip("/")  

st.set_page_config(page_title="RL Health", layout="wide")
st.title("RL Health — latest checkpoints by (symbol, timeframe)")


def _read_csv_safe(path: pathlib.Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except EmptyDataError:
        return None
    except Exception:
        return None
    return None

def _fetch_health_from_api() -> Optional[Dict[str, Any]]:
    if not API_BASE:
        return None
    try:
        import requests  
        r = requests.get(f"{API_BASE}/rl/health", timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"Health API error: {r.status_code} {r.text[:200]}")
        return None
    except Exception as e:
        st.error(f"Health API exception: {e}")
        return None

def _load_health_from_fs() -> Dict[str, Any]:

    latest_files: List[Dict[str, Any]] = []
    if LATEST_DIR.exists():
        for p in sorted(LATEST_DIR.glob("*.json")):
            try:
                j = json_load(p)
                latest_files.append({
                    "pair": p.stem,
                    "symbol": j.get("symbol"),
                    "tf": j.get("tf"),
                    "algo": j.get("algo"),
                    "version": j.get("version"),
                    "action_space": j.get("action_space", "discrete2"),
                    "checkpoint": j.get("best_checkpoint"),
                    "run_dir": j.get("run_dir"),
                    "scores": j.get("scores", {}),
                    "short_fee_bps": j.get("short_fee_bps", 0.0),
                    "atr": j.get("atr", {"mode": "off"}),
                    "mtime": pathlib.Path(p).stat().st_mtime,
                })
            except Exception:
                continue
    idx = []
    if LATEST_INDEX.exists():
        try:
            idx = json_load(LATEST_INDEX)
        except Exception:
            idx = []
    return {"latest_files": latest_files, "index": idx}

def _scores_to_cols(row: Dict[str, Any]) -> Dict[str, float]:
    s = row.get("scores", {}) or {}
    return {
        "sum_reward": float(s.get("sum_reward", 0.0)),
        "sharpe_like": float(s.get("sharpe_like", 0.0)),
    }


with st.sidebar:
    st.header("Source")
    st.caption("If API_BASE is set, we’ll call /rl/health; otherwise we read local artifacts.")
    st.text(f"API_BASE = {API_BASE or '• not set •'}")
    btn_refresh = st.button("Refresh")

if btn_refresh:
    time.sleep(0.15)
    st.experimental_rerun()  

health = _fetch_health_from_api() if API_BASE else None
if health is None:
    health = _load_health_from_fs()

latest_files: List[Dict[str, Any]] = health.get("latest_files", [])
index_entries: List[Dict[str, Any]] = health.get("index", [])

if not latest_files:
    st.info("No per-pair latest files found yet. Train a model first.")
    st.stop()


df = pd.DataFrame(latest_files)
score_cols = df.apply(_scores_to_cols, axis=1, result_type="expand")
df = pd.concat([df.drop(columns=["scores"]), score_cols], axis=1)


df["atr_mode"] = df["atr"].apply(lambda a: (a or {}).get("mode", "off"))
df["updated"] = pd.to_datetime(df["mtime"], unit="s")
df["pair_readable"] = df["symbol"].astype(str) + " • " + df["tf"].astype(str)


with st.sidebar:
    st.header("Filters")
    symbols = sorted(df["symbol"].dropna().unique().tolist())
    tfs = sorted(df["tf"].dropna().unique().tolist())
    action_spaces = sorted(df["action_space"].dropna().unique().tolist())
    atr_modes = sorted(df["atr_mode"].dropna().unique().tolist())

    pick_sym = st.multiselect("Symbol", symbols, default=symbols)
    pick_tf = st.multiselect("Timeframe", tfs, default=tfs)
    pick_space = st.multiselect("Action space", action_spaces, default=action_spaces)
    pick_atr = st.multiselect("ATR mode", atr_modes, default=atr_modes)

mask = (
    df["symbol"].isin(pick_sym) &
    df["tf"].isin(pick_tf) &
    df["action_space"].isin(pick_space) &
    df["atr_mode"].isin(pick_atr)
)
dfv = df[mask].sort_values(["symbol", "tf"]).reset_index(drop=True)



colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Tracked pairs", f"{len(dfv)}")
with colB:
    st.metric("Discrete-3 (with SELL)", f"{int((dfv['action_space']=='discrete3').sum())}")
with colC:
    st.metric("ATR stop: on", f"{int((dfv['atr_mode']=='stop').sum())}")
with colD:
    st.metric("Best Sharpe (shown)", f"{dfv['sharpe_like'].max():.3f}" if not dfv.empty else "—")

st.divider()



show_cols = [
    "pair_readable", "action_space", "atr_mode", "sharpe_like", "sum_reward",
    "checkpoint", "run_dir", "updated"
]
st.subheader("Per-pair latest")
st.dataframe(
    dfv[show_cols].rename(columns={
        "pair_readable": "Pair",
        "action_space": "Action space",
        "atr_mode": "ATR",
        "sharpe_like": "Sharpe-like",
        "sum_reward": "Sum reward",
        "checkpoint": "Checkpoint",
        "run_dir": "Run dir",
        "updated": "Updated",
    }),
    width="stretch",
    hide_index=True,
)



st.subheader("Quick links")
bdir = ROOT / "artifacts" / "backtests"

def _pair_pngs(symbol: str, tf: str) -> List[pathlib.Path]:
    prefix = f"rl_{symbol.replace('/','')}_{tf}_"
    return sorted(bdir.glob(prefix + "*.png"))

if dfv.empty:
    st.info("No pairs after filters.")
else:
    for _, row in dfv.iterrows():
        s, t = str(row["symbol"]), str(row["tf"])
        with st.expander(f"{s} • {t}", expanded=False):
            st.write(f"**Action space:** {row['action_space']} | **ATR:** {row['atr_mode']}")
            st.write(f"**Sharpe-like:** {row['sharpe_like']:.4f} | **Sum reward:** {row['sum_reward']:.4f}")
            st.code(str(row["checkpoint"]), language="bash")
            st.code(str(row["run_dir"]), language="bash")

            pngs = _pair_pngs(s, t)
            if pngs:
                st.image(str(pngs[-1]), caption=pngs[-1].name, width="stretch")
            else:
                st.caption("No OOS PNG yet for this pair. Run eval to generate one.")

st.divider()



with st.expander("Raw health payload (debug)"):
    st.json(health)
