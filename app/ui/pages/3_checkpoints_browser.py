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

st.set_page_config(page_title="Checkpoints Browser", page_icon="🧩", layout="wide")
st.markdown("""
<style>
  .card{border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.045); border-radius:14px; padding:12px;}
  .badge { font-weight:800; letter-spacing:.04em; padding:.2rem .6rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); }
  .buy  { background:rgba(30,158,66,.18); border-color:rgba(30,158,66,.35); }
  .hold { background:rgba(200,200,200,.18); border-color:rgba(200,200,200,.35); }
  .sell { background:rgba(255,107,107,.18); border-color:rgba(255,107,107,.35); }
  .chip{border-radius:999px; padding:.22rem .5rem; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); font-size:.8rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧩 Checkpoints Browser")

def _index():
    try:
        r = requests.get(f"{API_BASE}/rl/latest/index", timeout=20)
        if r.status_code==200: return r.json().get("pairs", [])
    except Exception as e:
        st.error(f"Index error: {e}")
    return []

def _explain(s: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/explain", params={"symbol": s, "tf": tf}, timeout=30)
        if r.status_code == 200: return r.json()
    except Exception:
        pass
    return None

def _history(s: str, tf: str, n: int) -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/history", params={"symbol": s, "tf": tf, "limit": n}, timeout=30)
        if r.status_code==200: return r.json().get("rows", [])
    except Exception:
        pass
    return []

def _feature_chart(rows: List[Dict[str, Any]]):
    if not rows: 
        st.caption("No recent closes.")
        return
    df = pd.DataFrame(rows)
    if "ts" not in df.columns or "close" not in df.columns: 
        st.caption("History missing ts/close.")
        return
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("ts").tail(200)
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

    price = alt.Chart(df).mark_line(stroke="#8ca0b3", opacity=0.65).encode(
        x=alt.X("ts:T", axis=None), y=alt.Y("close:Q", title="Price")
    ).properties(height=160)
    ema12 = alt.Chart(df).mark_line(stroke="#4ade80").encode(x="ts:T", y="ema12:Q")
    ema26 = alt.Chart(df).mark_line(stroke="#60a5fa").encode(x="ts:T", y="ema26:Q")
    st.altair_chart((price + ema12 + ema26).configure_axis(grid=False).configure_view(stroke=None).configure(background="#0E1117"),
                    use_container_width=True)

pairs = _index()
if not pairs:
    st.info("No pairs yet.")
    st.stop()

symbols = sorted({p["symbol"] for p in pairs})
colA, colB = st.columns([1.2, 1.2])
with colA:
    symbol = st.selectbox("Symbol", symbols, index=0)
with colB:
    tf = st.selectbox("TF", sorted({p["tf"] for p in pairs if p["symbol"]==symbol}), index=0)

path = next((p["path"] for p in pairs if p["symbol"]==symbol and p["tf"]==tf), None)
if not path or not Path(path).exists():
    st.error("latest.json missing for selection.")
    st.stop()

meta = json.loads(Path(path).read_text())
run_dir = Path(meta.get("run_dir","artifacts/rl/runs"))

st.subheader("Latest JSON")
c1, c2 = st.columns([1.2, 1.2])
with c1:
    st.code(path)
with c2:
    st.code(str(run_dir))

st.subheader("Files in run dir")
if run_dir.exists():
    files = [{"name": p.name, "size": p.stat().st_size, "mtime": p.stat().st_mtime} for p in run_dir.glob("*")]
    df = pd.DataFrame(files).sort_values("mtime", ascending=False)
    st.dataframe(df[["name","size"]], hide_index=True, use_container_width=True)
else:
    st.info("Run dir not found.")

st.divider()

st.subheader("🔎 Explain & Preview")
ex = _explain(symbol, tf)
cols = st.columns([1,1,1])
with cols[0]:
    if ex:
        act = (ex.get("action","HOLD") or "").upper()
        st.markdown(f"<span class='badge {('buy' if act=='BUY' else 'sell' if act=='SELL' else 'hold')}'>{act}</span>", unsafe_allow_html=True)
        st.caption(f"Confidence: {float(ex.get('confidence',0.0)):.2%}")
        st.caption(f"As of: {ex.get('ts_iso','—')}")
    else:
        st.caption("No explanation.")
with cols[1]:
    if ex and ex.get("probs"):
        dfp = pd.DataFrame([{"label": k.upper(), "p": float(v)} for k,v in ex["probs"].items()])
        ch = (
            alt.Chart(dfp)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Action"),
                y=alt.Y("p:Q", title="Probability", scale=alt.Scale(domain=[0,1])),
                color=alt.Color("label:N", legend=None, scale=alt.Scale(
                    domain=["BUY","HOLD","SELL"], range=["#1e9e42","#E8E8E8","#ff6b6b"]))
            )
            .properties(height=140)
            .configure_axis(grid=False)
            .configure_view(stroke=None)
            .configure(background="#0E1117")
        )
        st.altair_chart(ch, use_container_width=True)
with cols[2]:
    rows = _history(symbol, tf, 160)
    _feature_chart(rows)

if ex and ex.get("reasons"):
    st.markdown("**Top reasons**")
    rs = sorted(ex["reasons"], key=lambda r: abs(r.get("weight",0.0)), reverse=True)[:6]
    st.dataframe(pd.DataFrame(rs)[["name","detail","weight"]], hide_index=True, use_container_width=True)
    st.caption(ex.get("text",""))
