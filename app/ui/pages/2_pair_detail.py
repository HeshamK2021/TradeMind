from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st
import requests
import altair as alt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Pair Detail", page_icon="🪙", layout="wide")
st.markdown("""
<style>
  .chip{border-radius:999px; padding:.28rem .6rem; border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.06); font-size:.82rem;}
  .ok{background:rgba(39,174,96,.15); border-color:rgba(39,174,96,.3)}
  .warn{background:rgba(241,196,15,.15); border-color:rgba(241,196,15,.3)}
  .err{background:rgba(231,76,60,.15); border-color:rgba(231,76,60,.3)}
  .section{border:1px solid rgba(255,255,255,.06); background:rgba(255,255,255,.04); border-radius:16px; padding:12px}
  .divider{height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent); margin:8px 0 6px 0}
  .badge { font-weight:800; letter-spacing:.04em; padding:.2rem .6rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); }
  .buy  { background:rgba(30,158,66,.18); border-color:rgba(30,158,66,.35); }
  .hold { background:rgba(200,200,200,.18); border-color:rgba(200,200,200,.35); }
  .sell { background:rgba(255,107,107,.18); border-color:rgba(255,107,107,.35); }
</style>
""", unsafe_allow_html=True)

st.title("🪙 Pair Detail")

def _latest_index():
    try:
        r = requests.get(f"{API_BASE}/rl/latest/index", timeout=20)
        r.raise_for_status()
        return r.json().get("pairs", [])
    except Exception as e:
        st.error(f"Index load failed: {e}")
        return []

def _history(symbol: str, tf: str, n: int) -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/history", params={"symbol": symbol, "tf": tf, "limit": n}, timeout=30)
        if r.status_code!=200: return []
        return r.json().get("rows", [])
    except Exception:
        return []

def _explain(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/explain", params={"symbol": symbol, "tf": tf}, timeout=30)
        if r.status_code == 200: return r.json()
    except Exception:
        pass
    return None

def _probs_bar(probs: Dict[str, float]):
    if not probs: return
    df = pd.DataFrame([{"label": k.upper(), "p": float(v)} for k,v in probs.items()])
    chart = (
        alt.Chart(df)
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
    st.altair_chart(chart, use_container_width=True)

def _feature_chart(rows: List[Dict[str, Any]]):
    if not rows: 
        st.info("No price/feature history yet.")
        return
    df = pd.DataFrame(rows)
    if "ts" not in df.columns: 
        st.info("Missing timestamps in history.")
        return
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    if "close" not in df.columns:
        st.info("Missing close in history.")
        return

    df = df.sort_values("ts").copy()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

    price = alt.Chart(df).mark_line(stroke="#8ca0b3", opacity=0.65).encode(
        x=alt.X("ts:T", axis=None), y=alt.Y("close:Q", title="Price")
    ).properties(height=220)

    ema12 = alt.Chart(df).mark_line(stroke="#4ade80").encode(x="ts:T", y="ema12:Q")
    ema26 = alt.Chart(df).mark_line(stroke="#60a5fa").encode(x="ts:T", y="ema26:Q")

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    n = 14
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi14 = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame({"ts": df["ts"], "rsi14": rsi14.fillna(50.0)})

    rsi = alt.Chart(rsi_df).mark_line().encode(
        x=alt.X("ts:T", axis=None), y=alt.Y("rsi14:Q", title="RSI 14", scale=alt.Scale(domain=[0,100]))
    ).properties(height=120)
    bands = alt.Chart(pd.DataFrame({"y":[30, 70]})).mark_rule(strokeDash=[4,3], color="#999").encode(y="y:Q")

    st.altair_chart((price + ema12 + ema26).configure_axis(grid=False).configure_view(stroke=None).configure(background="#0E1117"),
                    use_container_width=True)
    st.altair_chart((rsi + bands).configure_view(stroke=None).configure(background="#0E1117"),
                    use_container_width=True)

pairs = _latest_index()
if not pairs:
    st.stop()

symbols = sorted({p["symbol"] for p in pairs})
colA, colB, colC, colD = st.columns([1.4, 1, .9, 1.5])
with colA:
    symbol = st.selectbox("Symbol", symbols, index=0)
with colB:
    tfs = sorted({p["tf"] for p in pairs if p["symbol"] == symbol})
    tf = st.selectbox("TF", tfs, index=0)
with colC:
    recent_n = st.slider("Recent rows", 10, 200, 120, step=10)
with colD:
    do_refresh = st.button("↻ Refresh")

pair_meta_path = next((p["path"] for p in pairs if p["symbol"]==symbol and p["tf"]==tf), None)
if not pair_meta_path or not Path(pair_meta_path).exists():
    st.error("latest.json for this pair is missing.")
    st.stop()

with open(pair_meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

run_dir = Path(meta.get("run_dir", "artifacts/rl/runs"))
ckpt_path = meta.get("best_checkpoint", "—")
action_space = meta.get("action_space", "discrete2")
atr_meta = meta.get("atr", {"mode":"off"})

c1, c2, c3, c4 = st.columns(4)
def _post(url, payload):
    try:
        r = requests.post(url, json=payload, timeout=600)
        if r.status_code == 200: return r.json()
        st.error(f"Request failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(str(e))
    return None
with c1:
    if st.button("📈 Eval RL"):
        with st.spinner("Evaluating RL…"):
            _post(f"{API_BASE}/rl/eval", {"symbol": symbol, "tf": tf})
with c2:
    if st.button("📊 Baseline"):
        with st.spinner("Running baseline…"):
            _post(f"{API_BASE}/baseline/eval", {"symbol": symbol, "tf": tf})
with c3:
    if st.button("⚙️ Train"):
        with st.spinner("Training…"):
            _post(f"{API_BASE}/rl/train", {"symbol": symbol, "tf": tf, "steps": 80_000, "use_short": (action_space=="discrete3")})
with c4:
    if st.button("🚀 Retrain→Promote"):
        with st.spinner("Retrain→Promote…"):
            _post(f"{API_BASE}/rl/retrain", {
                "symbol": symbol, "tf": tf, "steps": 60_000,
                "use_short": (action_space=="discrete3"),
                "short_fee_bps": meta.get("short_fee_bps", 0.0),
                "promote_if_better": True
            })

st.divider()

rows = _history(symbol, tf, max(10, recent_n))
latest = rows[0] if rows else {}
act = (latest.get("action","HOLD") or "").upper()
conf = latest.get("confidence")
close = latest.get("close")
emoji = {"BUY":"🟢","HOLD":"⚪️","SELL":"🔴"}.get(act,"⚪️")
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Latest")
    st.metric("Action", f"{emoji} {act}")
    st.metric("Confidence", f"{conf:.2%}" if conf is not None else "—")
    st.caption(f"<span class='chip'>🎮 {action_space}</span> <span class='chip'>📏 ATR: {atr_meta.get('mode','off')}</span> <span class='chip'>💵 {close if close is not None else '—'}</span>", unsafe_allow_html=True)
    st.caption(f"Checkpoint: {ckpt_path}")
with col2:
    st.subheader("Recent predictions")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        st.dataframe(df[["ts","close","action","confidence"]], hide_index=True, use_container_width=True)
    else:
        st.info("No recent rows.")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.subheader("🔎 Explain (rationale)")
ex = _explain(symbol, tf)
if ex:
    ecols = st.columns([1,1,2])
    with ecols[0]:
        a = (ex.get("action","HOLD") or "").upper()
        cls = "buy" if a=="BUY" else ("sell" if a=="SELL" else "hold")
        st.markdown(f"<span class='badge {cls}'>{a}</span>", unsafe_allow_html=True)
        st.caption(f"Confidence: {float(ex.get('confidence',0.0)):.2%}")
        st.caption(f"As of: {ex.get('ts_iso','—')}")
    with ecols[1]:
        st.markdown("**Probabilities**")
        _probs_bar(ex.get("probs", {}))
    with ecols[2]:
        rs = ex.get("reasons", [])
        if rs:
            rs = sorted(rs, key=lambda r: abs(r.get("weight", 0.0)), reverse=True)[:5]
            st.markdown("**Top reasons**")
            st.dataframe(pd.DataFrame(rs)[["name","detail","weight"]], hide_index=True, use_container_width=True)
        st.caption(ex.get("text",""))

    st.markdown("**Features chart (Price + EMAs + RSI)**")
    _feature_chart(rows[:max(120, recent_n)][::-1]) 
else:
    st.info("No explanation available. Train/eval first.")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.subheader("Out-of-sample charts")
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

st.subheader("Walk-forward windows")
try:
    r = requests.get(f"{API_BASE}/rl/windows", params={"symbol": symbol, "tf": tf}, timeout=30)
    if r.status_code == 200:
        win = r.json()
        rows2 = win.get("rows", [])
        if rows2:
            wdf = pd.DataFrame(rows2)
            st.dataframe(wdf, hide_index=True, use_container_width=True)
            if "wins" in win and win["wins"] is not None:
                st.success(f"RL wins {win['wins']} / {win.get('total', len(rows2))} windows.")
        else:
            st.info("No window rows yet.")
    else:
        st.warning(f"/rl/windows → {r.status_code}")
except Exception as e:
    st.warning(f"windows error: {e}")

st.subheader("Metrics")
mp = run_dir / "metrics.json"
if mp.exists():
    try:
        st.json(json.loads(mp.read_text()))
    except Exception:
        st.code(mp.read_text())
else:
    st.info("metrics.json not found")
