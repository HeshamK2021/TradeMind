from __future__ import annotations
import os, sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Backtests Gallery", page_icon="🖼️", layout="wide")
st.title("🖼️ Backtests Gallery")

bdir = Path("artifacts") / "backtests"
if not bdir.exists():
    st.info("No backtests directory yet.")
    st.stop()

pngs = sorted(bdir.glob("*.png"))
symbols = sorted({p.name.split("_")[1] for p in pngs if "_" in p.name}) if pngs else []

c1, c2 = st.columns([1.2, 1])
with c1:
    sym = st.selectbox("Symbol (optional filter)", ["All"] + symbols, index=0)
with c2:
    tf = st.selectbox("Timeframe (optional)", ["All","15m","1h","4h","1d"], index=0)

def _match(p):
    name = p.name
    return ((sym=="All" or sym in name) and (tf=="All" or f"_{tf}_" in name))

filtered = [p for p in pngs if _match(p)]
cols = st.columns(3)
if not filtered:
    st.info("No images match the filter.")
else:
    for i, p in enumerate(filtered):
        with cols[i % 3]:
            st.image(str(p), caption=p.name, use_container_width=True)
