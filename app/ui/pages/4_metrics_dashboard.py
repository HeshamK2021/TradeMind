from __future__ import annotations
import os, sys, json
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st
import requests
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Metrics Dashboard", page_icon="📈", layout="wide")
st.markdown("""
<style>
  .grid {display:grid; grid-template-columns: 1.6fr .9fr 1.5fr; gap:14px;}
  .card {border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.045); border-radius:14px; padding:12px;}
  .chip{border-radius:999px; padding:.22rem .5rem; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); font-size:.8rem;}
  .badge { font-weight:800; letter-spacing:.04em; padding:.2rem .6rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); }
  .buy  { background:rgba(30,158,66,.18); border-color:rgba(30,158,66,.35); }
  .hold { background:rgba(200,200,200,.18); border-color:rgba(200,200,200,.35); }
  .sell { background:rgba(255,107,107,.18); border-color:rgba(255,107,107,.35); }
</style>
""", unsafe_allow_html=True)

st.title("📈 Metrics Dashboard")

def _index():
    try:
        r = requests.get(f"{API_BASE}/rl/latest/index", timeout=20)
        if r.status_code==200: return r.json().get("pairs", [])
    except Exception as e:
        st.error(f"Index error: {e}")
    return []

pairs = _index()
if not pairs:
    st.info("No pairs yet.")
    st.stop()

colA, colB, colC = st.columns([1,1,1.2])
with colA:
    symbol = st.selectbox("Symbol", sorted({p["symbol"] for p in pairs}), index=0)
with colB:
    tf = st.selectbox("TF", sorted({p["tf"] for p in pairs if p["symbol"]==symbol}), index=0)
with colC:
    refresh = st.button("↻ Refresh")

meta_path = next((p["path"] for p in pairs if p["symbol"]==symbol and p["tf"]==tf), None)
if not meta_path or not Path(meta_path).exists():
    st.error("Missing latest.json for selected pair.")
    st.stop()

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

run_dir = Path(meta.get("run_dir", "artifacts/rl/runs"))
metrics_path = run_dir / "metrics.json"
metrics = {}
if metrics_path.exists():
    try: metrics = json.loads(metrics_path.read_text()) or {}
    except Exception: metrics = {}

st.subheader("Run Metrics")
if metrics:
    df = pd.DataFrame(list(metrics.items()), columns=["metric","value"])
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("metrics.json not found or empty.")

st.divider()

st.subheader("🔎 Latest Rationale")
def _explain(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/explain", params={"symbol": symbol, "tf": tf}, timeout=30)
        if r.status_code == 200: return r.json()
    except Exception:
        pass
    return None

ex = _explain(symbol, tf)
col1, col2 = st.columns([1,2])
if ex:
    with col1:
        act = (ex.get("action","HOLD") or "").upper()
        st.markdown(f"<span class='badge {('buy' if act=='BUY' else 'sell' if act=='SELL' else 'hold')}'>{act}</span>", unsafe_allow_html=True)
        st.caption(f"Confidence: {float(ex.get('confidence',0.0)):.2%}")
        st.caption(f"As of: {ex.get('ts_iso','—')}")
    with col2:
        probs = ex.get("probs", {})
        if probs:
            dfp = pd.DataFrame([{"label": k.upper(), "p": float(v)} for k,v in probs.items()])
            ch = (
                alt.Chart(dfp)
                .mark_bar()
                .encode(
                    x=alt.X("label:N", title="Action"),
                    y=alt.Y("p:Q", title="Probability", scale=alt.Scale(domain=[0,1])),
                    color=alt.Color("label:N", legend=None, scale=alt.Scale(
                        domain=["BUY","HOLD","SELL"], range=["#1e9e42","#E8E8E8","#ff6b6b"]))
                )
                .properties(height=160)
                .configure_axis(grid=False)
                .configure_view(stroke=None)
                .configure(background="#0E1117")
            )
            st.altair_chart(ch, use_container_width=True)
        rs = ex.get("reasons", [])
        if rs:
            st.markdown("**Top reasons**")
            rs = sorted(rs, key=lambda r: abs(r.get("weight",0.0)), reverse=True)[:5]
            st.dataframe(pd.DataFrame(rs)[["name","detail","weight"]], hide_index=True, use_container_width=True)
        st.caption(ex.get("text",""))
else:
    st.info("No explanation yet.")

st.divider()

st.subheader("Equity PNGs (latest)")
bdir = Path("artifacts") / "backtests"
rl_pngs = sorted(bdir.glob(f"rl_{symbol.replace('/','')}_{tf}_*.png"))
bl_pngs = sorted(bdir.glob(f"baseline_{symbol.replace('/','')}_{tf}_*.png")) or sorted(bdir.glob(f"{symbol.replace('/','')}_{tf}_*.png"))
cL, cR = st.columns(2)
with cL:
    st.write("**RL**")
    if rl_pngs: st.image(str(rl_pngs[-1]), use_container_width=True)
    else: st.info("No RL OOS yet.")
with cR:
    st.write("**Baseline**")
    if bl_pngs: st.image(str(bl_pngs[-1]), use_container_width=True)
    else: st.info("No Baseline OOS yet.")
