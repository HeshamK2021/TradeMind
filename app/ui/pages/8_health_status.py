from __future__ import annotations
import os, sys, json
from pathlib import Path
import streamlit as st
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Health Status", page_icon="💡", layout="wide")
st.title("💡 Health Status")

st.markdown("""
<style>
  .chip{border-radius:999px; padding:.28rem .6rem; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); font-size:.85rem;}
  .ok{background:rgba(39,174,96,.15); border-color:rgba(39,174,96,.3)}
  .err{background:rgba(231,76,60,.15); border-color:rgba(231,76,60,.3)}
</style>
""", unsafe_allow_html=True)

rows = []

try:
    r = requests.get(f"{API_BASE}/rl/latest/index", timeout=10)
    ok = (r.status_code == 200)
    rows.append(("API", "Reachable", ok))
except Exception:
    rows.append(("API", "Reachable", False))

idx = {}
try:
    idx = r.json()
    rows.append(("/rl/latest/index", "Pairs found" , bool(idx.get("pairs"))))
except Exception:
    rows.append(("/rl/latest/index", "Pairs found" , False))

for p in ["artifacts/rl/latest", "artifacts/rl/runs", "artifacts/backtests", "artifacts/rl/checkpoints", "artifacts/backtests"]:
    exists = Path(p).exists()
    rows.append(("FS", p, exists))

for a,b,ok in rows:
    st.markdown(f"<span class='chip {'ok' if ok else 'err'}'>{a} — {b}: {'OK' if ok else 'FAIL'}</span>", unsafe_allow_html=True)

st.divider()
st.subheader("Raw latest index")
st.json(idx or {})
