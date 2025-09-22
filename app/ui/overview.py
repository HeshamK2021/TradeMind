# app/ui/overview.py
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import requests
import altair as alt

# Ensure repo root in sys.path for local imports if needed
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")

# -------------------------- Page & Styles --------------------------
st.set_page_config(page_title="RL Overview", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .wrap { display:flex; gap:.6rem; flex-wrap: wrap; }
      .card {
        flex: 1 1 420px;
        border-radius: 14px;
        padding: 15px 12px;
        border: 1px solid rgba(255,255,255,.08);
        background: rgba(255,255,255,.045);
        margin-bottom: .5rem;
      }
      .row { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
      .pair { display:flex; align-items:center; gap:.5rem; font-weight:700; font-size:1rem; }
      .pair small { opacity:.6; font-weight:600; }
      .chip {
        border-radius:999px; padding:.18rem .48rem; font-size:.78rem;
        border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.08);
      }
      .chip.ok { background: rgba(46,204,113,.16); border-color: rgba(46,204,113,.28);}
      .chip.warn { background: rgba(241,196,15,.16); border-color: rgba(241,196,15,.28);}
      .chip.err { background: rgba(231,76,60,.16); border-color: rgba(231,76,60,.28);}
      .act { font-weight:700; }
      .act.buy { color: #1e9e42; }
      .act.hold { color: #bbb;}
      .act.sell { color: #ff6b6b;}
      .muted { opacity:.65; font-size:.82rem; }
      .kpis { display:flex; gap:.35rem; flex-wrap:wrap; margin:.25rem 0 .15rem 0; }
      .btnrow { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.4rem; margin-top:.35rem; }
      .btnrow > div { display:flex; }
      .metricmini { display:flex; align-items:center; gap:.25rem; }
      .metricmini b { font-variant-numeric: tabular-nums; }
      .smallcaps { letter-spacing:.06em; text-transform: uppercase; font-weight:700; font-size:.76rem; opacity:.7;}
      .hdr { display:flex; gap:.6rem; align-items:center; margin: 0 0 .2rem 0;}
      .hdr .pill { display:inline-flex; align-items:center; gap:.35rem; padding:.18rem .48rem; border-radius:999px;
                   border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.08); font-size:.78rem;}
      .sparkcap { font-size:.78rem; text-align:right; margin-top:-6px; opacity:.8; }
      .divider { height:1px; width:100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,.22), transparent);
                 border-radius:1px; margin:6px 0 6px 0; }
      .code-sm { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.78rem; }
      /* Explain row inside expander */
      .explain { display:flex; gap:.6rem; flex-wrap:wrap; }
      .badge { font-weight:800; letter-spacing:.04em; padding:.2rem .6rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); }
      .buy  { background:rgba(30,158,66,.18); border-color:rgba(30,158,66,.35); }
      .hold { background:rgba(200,200,200,.18); border-color:rgba(200,200,200,.35); }
      .sell { background:rgba(255,107,107,.18); border-color:rgba(255,107,107,.35); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hdr"><h1 style="margin:0"> TradeMind RL Overview</h1>'
            f'<span class="pill">API: {("connected" if API_BASE else "local-only")}</span></div>',
            unsafe_allow_html=True)

# -------------------------- Controls --------------------------
c1, c2, c3, c4, c5 = st.columns([2,1,1,1.4,1.4])
with c1:
    st.caption("Pairs are discovered from `/rl/latest/index`.")
with c2:
    refresh_btn = st.button("↻ Refresh")
with c3:
    recent_points = st.slider("points", 10, 100, 40, step=5)
with c4:
    tf_filter = st.selectbox("Filter timeframe", ["All", "15m", "1h", "4h", "1d"], index=0)
with c5:
    sort_key = st.selectbox(
        "Sort by",
        ["Updated (newest)", "Confidence", "Price", "Sharpe", "CAGR", "Win rate", "Symbol A→Z"],
        index=0
    )
order_desc = st.toggle("Descending", value=True, help="Toggle sort order")

# -------------------------- Helpers --------------------------
COIN_EMOJI = {"BTC":"🪙","ETH":"🪙","SOL":"🪙","XRP":"🪙","SHIB":"🪙","DOGE":"🪙","ADA":"🪙","BNB":"🪙","LTC":"🪙","AVAX":"🪙"}
def coin_emoji(symbol: str) -> str:
    base = (symbol or "").split("/")[0].upper()
    return COIN_EMOJI.get(base, "💹")

def _rerun():
    try: st.rerun()
    except Exception: getattr(st, "experimental_rerun", lambda: None)()

def _file_age_minutes(p: Path) -> Optional[float]:
    try: return max(0.0, (time.time() - p.stat().st_mtime) / 60.0)
    except Exception: return None

def _mtime_iso(p: Path) -> Optional[str]:
    try: return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception: return None

def _get_latest_index() -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/latest/index", timeout=30)
        r.raise_for_status()
        j = r.json()
        return j.get("pairs", [])
    except Exception as e:
        st.error(f"Failed to load /rl/latest/index: {e}")
        return []

def _get_history(symbol: str, tf: str, limit: int) -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/history", params={"symbol": symbol, "tf": tf, "limit": int(limit)}, timeout=30)
        if r.status_code != 200:
            return []
        return r.json().get("rows", [])
    except Exception:
        return []

def _get_last_action(symbol: str, tf: str, limit: int = 1) -> Dict[str, Any]:
    rows = _get_history(symbol, tf, limit=limit)
    return rows[0] if rows else {}

def _get_recent_prices(symbol: str, tf: str, n: int) -> pd.DataFrame:
    rows = _get_history(symbol, tf, limit=max(1, n))
    if not rows:
        return pd.DataFrame(columns=["ts", "close"])
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    if "close" not in df.columns:
        df["close"] = None
    df = df[["ts", "close"]].dropna(subset=["close"]).sort_values("ts").tail(n)
    return df

def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    mp = run_dir / "metrics.json"
    if mp.exists() and mp.stat().st_size > 0:
        try:
            with mp.open("r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}
    return {}

def _run_eval(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{API_BASE}/rl/eval", json={"symbol": symbol, "tf": tf}, timeout=300)
        if r.status_code == 200:
            return r.json()
        st.error(f"Eval failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"Eval exception: {e}")
    return None

def _run_baseline(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(f"{API_BASE}/baseline/eval", json={"symbol": symbol, "tf": tf}, timeout=600)
        if r.status_code == 200:
            return r.json()
        st.error(f"Baseline failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"Baseline exception: {e}")
    return None

def _quick_train(symbol: str, tf: str, use_short: bool = True) -> Optional[Dict[str, Any]]:
    try:
        payload = {"symbol": symbol, "tf": tf, "steps": 80_000, "use_short": bool(use_short)}
        r = requests.post(f"{API_BASE}/rl/train", json=payload, timeout=1200)
        if r.status_code == 200:
            return r.json()
        st.error(f"Train failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"Train exception: {e}")
    return None

def _retrain_promote(symbol: str, tf: str, action_space: str, latest_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        payload = {
            "symbol": symbol,
            "tf": tf,
            "steps": 60_000,
            "use_short": (action_space == "discrete3"),
            "short_fee_bps": latest_meta.get("short_fee_bps", 0.0),
            "promote_if_better": True,
            "dry_run": False,
        }
        r = requests.post(f"{API_BASE}/rl/retrain", json=payload, timeout=1800)
        if r.status_code == 200:
            return r.json()
        st.error(f"Retrain failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"Retrain exception: {e}")
    return None

def _class_for_action(a: str) -> str:
    a = (a or "").upper()
    return "buy" if a == "BUY" else ("sell" if a == "SELL" else "hold")

def _get_explain(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/rl/explain", params={"symbol": symbol, "tf": tf}, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# -------------------------- Load index --------------------------
pairs = _get_latest_index()
if not pairs:
    st.info("No trained pairs yet. Train at least one pair via /rl/train or open the Workflow Dashboard.")
    st.stop()

# In-memory caches
if "last_rows" not in st.session_state or refresh_btn:
    st.session_state["last_rows"] = {}
if "spark_cache" not in st.session_state or refresh_btn:
    st.session_state["spark_cache"] = {}

# Optional filter by timeframe
if tf_filter != "All":
    pairs = [p for p in pairs if p.get("tf") == tf_filter]

# -------------------------- Enrich & Sort --------------------------
def _enrich_pair(p: Dict[str, Any]) -> Dict[str, Any]:
    symbol, tf, path = p.get("symbol"), p.get("tf"), p.get("path")
    meta = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        meta = {}

    run_dir = Path(meta.get("run_dir", "artifacts/rl/runs"))
    action_space = meta.get("action_space", "discrete2")
    atr_meta = meta.get("atr", {"mode": "off"})
    ckpt_path = meta.get("best_checkpoint", "—")
    lp = Path(path)
    age_min = _file_age_minutes(lp)
    mtime_str = _mtime_iso(lp) or ""

    key = f"{symbol}|{tf}"
    last = st.session_state["last_rows"].get(key) or _get_last_action(symbol, tf, limit=recent_points)
    st.session_state["last_rows"][key] = last
    action = (last or {}).get("action", "N/A")
    conf = (last or {}).get("confidence", None)
    close = (last or {}).get("close", None)
    ts_iso = (last or {}).get("ts_iso", None)

    metrics = _load_metrics(run_dir)
    sharpe = metrics.get("sharpe", metrics.get("sharpe_ratio"))
    cagr = metrics.get("cagr")
    win_rate = metrics.get("win_rate")

    return dict(
        symbol=symbol, tf=tf, path=path, meta=meta,
        run_dir=run_dir, action_space=action_space, atr_meta=atr_meta, ckpt_path=ckpt_path,
        age_min=age_min, mtime_str=mtime_str,
        action=action, confidence=conf, close=close, ts_iso=ts_iso,
        sharpe=sharpe, cagr=cagr, win_rate=win_rate
    )

enriched: List[Dict[str, Any]] = [_enrich_pair(p) for p in pairs]

def _sort_key(row: Dict[str, Any]) -> Tuple:
    if sort_key == "Updated (newest)":
        age = row.get("age_min")
        return (999999 if age is None else age,)
    if sort_key == "Confidence":
        v = row.get("confidence")
        return (-(v if isinstance(v, (int, float)) else -1e9),)
    if sort_key == "Price":
        v = row.get("close")
        return (-(v if isinstance(v, (int, float)) else -1e18),)
    if sort_key == "Sharpe":
        try: v = float(row.get("sharpe")) if row.get("sharpe") is not None else -1e9
        except Exception: v = -1e9
        return (-v,)
    if sort_key == "CAGR":
        v = row.get("cagr"); return (-(v if isinstance(v, (int, float)) else -1e9),)
    if sort_key == "Win rate":
        v = row.get("win_rate"); return (-(v if isinstance(v, (int, float)) else -1e9),)
    if sort_key == "Symbol A→Z":
        return (str(row.get("symbol","")), str(row.get("tf","")))
    return (0,)

enriched.sort(key=_sort_key, reverse=False if sort_key == "Updated (newest)" else False)
if order_desc:
    enriched = list(reversed(enriched))

# -------------------------- Snapshot chips --------------------------
counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
for row in enriched:
    counts[row.get("action", "HOLD")] = counts.get(row.get("action", "HOLD"), 0) + 1

st.markdown(
    f"<div class='smallcaps'>Snapshot</div>"
    f"<div class='kpis'>"
    f"<span class='chip'>Total: {len(enriched)}</span>"
    f"<span class='chip ok'>BUY: {counts.get('BUY',0)}</span>"
    f"<span class='chip'>HOLD: {counts.get('HOLD',0)}</span>"
    f"<span class='chip err'>SELL: {counts.get('SELL',0)}</span>"
    f"</div>",
    unsafe_allow_html=True
)

# -------------------------- Thin Sparkline --------------------------
def _sparkline(symbol: str, tf: str, n: int):
    key = f"{symbol}|{tf}|{n}"
    df = st.session_state["spark_cache"].get(key)
    if df is None:
        df = _get_recent_prices(symbol, tf, n)
        st.session_state["spark_cache"][key] = df

    if df.empty:
        st.caption("No price data")
        return

    base = (
        alt.Chart(df)
        .mark_line(interpolate="monotone")
        .encode(
            x=alt.X("ts:T", axis=None),
            y=alt.Y("close:Q", axis=None),
        )
        .properties(height=36, width="container")
    )

    chart = base.configure_axis(grid=False).configure_view(stroke=None).configure(background="#0E1117")
    st.altair_chart(chart, use_container_width=True)
    st.markdown(f"<div class='sparkcap'>last: <b>{df['close'].iloc[-1]:,.4f}</b></div>", unsafe_allow_html=True)

# -------------------------- Cards Grid --------------------------
st.markdown("<div class='wrap'>", unsafe_allow_html=True)
for row in enriched:
    symbol = row["symbol"]; tf = row["tf"]; path = row["path"]; meta = row["meta"]
    run_dir = row["run_dir"]; action_space = row["action_space"]; atr_meta = row["atr_meta"]
    ckpt_path = row["ckpt_path"]; age_min = row["age_min"]; mtime_str = row["mtime_str"]
    action = row["action"]; conf = row["confidence"]; close = row["close"]; ts_iso = row["ts_iso"]
    sharpe = row["sharpe"]; cagr = row["cagr"]; win_rate = row["win_rate"]

    icon = coin_emoji(symbol)
    acls = _class_for_action(action)
    emoji = {"BUY": "🟢", "HOLD": "⚪️", "SELL": "🔴"}.get(action, "⚪️")

    age_badge = ""
    if age_min is not None:
        if age_min > 24*60: age_badge = f"<span class='chip warn'>⏱ {age_min/60:.1f}h old</span>"
        else: age_badge = f"<span class='chip ok'>⏱ {age_min:.0f}m</span>"

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2.0, 1.6], gap="small")

    # Left side
    with col1:
        st.markdown(
            f"<div class='row'><div class='pair'>{icon} {symbol} <small>• {tf}</small></div>"
            f"<div>{age_badge}</div></div>", unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='row'><div class='act {acls}'>{emoji} {action}</div>"
            f"<div class='muted'>{ts_iso or ''}</div></div>", unsafe_allow_html=True
        )
        st.markdown(
            "<div class='kpis'>"
            f"<span class='chip'>🎮 {action_space}</span>"
            f"<span class='chip'>📏 ATR: {atr_meta.get('mode','off')}</span>"
            f"<span class='chip'>💵 {(f'{close:,.4f}' if isinstance(close,(int,float)) else '—')}</span>"
            f"<span class='chip'>{(f'🎯 {float(conf):.2%}' if isinstance(conf,(int,float)) else '—')}</span>"
            "</div>", unsafe_allow_html=True
        )
        mm = []
        try:
            if sharpe is not None: mm.append(("Sharpe", f"{float(sharpe):.2f}"))
        except Exception: pass
        try:
            if cagr is not None: mm.append(("CAGR", f"{float(cagr):.2%}"))
        except Exception: pass
        try:
            if win_rate is not None: mm.append(("Win", f"{float(win_rate):.1%}"))
        except Exception: pass
        if mm:
            chips = " ".join([f"<span class='chip'><span class='metricmini'><span>{k}:</span> <b>{v}</b></span></span>" for k,v in mm])
            st.markdown(chips, unsafe_allow_html=True)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<span class='code-sm'>Checkpoint:</span> <span class='code-sm'>{ckpt_path}</span>", unsafe_allow_html=True)

    # Right side
    with col2:
        _sparkline(symbol, tf, recent_points)

        # ---- Inline Explain (lightweight) ----
        with st.expander("🔎 Explain (latest)", expanded=False):
            ex = _get_explain(symbol, tf)
            if ex:
                act = (ex.get("action","HOLD") or "").upper()
                cls = "buy" if act=="BUY" else ("sell" if act=="SELL" else "hold")
                st.markdown(f"<div class='explain'><span class='badge {cls}'>{act}</span>"
                            f"<span class='chip'>Conf: {float(ex.get('confidence',0.0)):.2%}</span>"
                            f"<span class='chip'>As of: {ex.get('ts_iso','—')}</span></div>", unsafe_allow_html=True)
                rs = ex.get("reasons", [])
                if rs:
                    rs = sorted(rs, key=lambda r: abs(r.get("weight", 0.0)), reverse=True)[:3]
                    st.markdown("**Top drivers**")
                    st.dataframe(pd.DataFrame(rs)[["name","detail","weight"]], use_container_width=True, hide_index=True)
                st.caption(ex.get("text",""))
            else:
                st.caption("No explanation available.")

        st.markdown("<div class='btnrow'>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if st.button("📈 OOS", key=f"ev_{symbol}_{tf}"):
                with st.spinner(f"Evaluating RL OOS: {symbol} • {tf} …"):
                    _ = _run_eval(symbol, tf); st.session_state["last_rows"].pop(f"{symbol}|{tf}", None)
                    st.session_state["spark_cache"].pop(f"{symbol}|{tf}|{recent_points}", None); _rerun()
        with b2:
            if st.button("📊 Base", key=f"bl_{symbol}_{tf}"):
                with st.spinner(f"Running Baseline: {symbol} • {tf} …"):
                    _ = _run_baseline(symbol, tf); _rerun()
        with b3:
            if st.button("⚙️ Train", key=f"tr_{symbol}_{tf}"):
                with st.spinner(f"Quick training: {symbol} • {tf} …"):
                    _ = _quick_train(symbol, tf, use_short=(action_space == "discrete3"))
                    st.session_state["last_rows"].pop(f"{symbol}|{tf}", None)
                    st.session_state["spark_cache"].pop(f"{symbol}|{tf}|{recent_points}", None); _rerun()
        with b4:
            if st.button("🚀 Promote", key=f"rp_{symbol}_{tf}"):
                with st.spinner(f"Retrain→Promote if better: {symbol} • {tf} …"):
                    _ = _retrain_promote(symbol, tf, action_space, meta)
                    st.session_state["last_rows"].pop(f"{symbol}|{tf}", None)
                    st.session_state["spark_cache"].pop(f"{symbol}|{tf}|{recent_points}", None); _rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Details", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Updated**")
                st.write(mtime_str or "—")
                st.markdown("**Latest JSON**")
                st.code(str(path))
                st.markdown("**Run dir**")
                st.code(str(run_dir))
            with c2:
                st.markdown("**Metrics (subset)**")
                st.json({k: v for k, v in _load_metrics(run_dir).items()
                         if k.lower() in ("sharpe","sharpe_ratio","cagr","win_rate","max_dd","turnover")})

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # wrap

st.caption("TradeMind @ 2025")
