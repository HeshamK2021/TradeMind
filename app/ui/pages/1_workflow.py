from __future__ import annotations
import os, sys, json, subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import streamlit as st
import requests
import yaml
import altair as alt
import plotly.graph_objects as go  

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_BASE = os.environ.get("API_BASE", "http://localhost:8000").rstrip("/")

try:
    from app.data.session import session_scope
    from app.data.models import Candle, FeatureRow
    _HAS_DB = True
except Exception:
    _HAS_DB = False
    session_scope = None  
    Candle = None         
    FeatureRow = None     

st.set_page_config(page_title="TradeMind Workflow", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
      .thin-divider { height:1px; width:100%; background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent); margin:14px 0; }
      .pill { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .55rem; border-radius:999px;
              border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.08); font-size:.8rem;}
      .smallcaps { letter-spacing:.06em; text-transform:uppercase; font-weight:700; font-size:.78rem; opacity:.8;}
      .wrap { display:flex; gap:1rem; flex-wrap:wrap; }
      .card { flex:1 1 520px; border-radius:16px; padding:16px; border:1px solid rgba(255,255,255,.10); background:rgba(255,255,255,.05); }
      .ok   { background:rgba(46,204,113,.12); border-color:rgba(46,204,113,.25);}
      .warn { background:rgba(241,196,15,.12); border-color:rgba(241,196,15,.25);}
      .err  { background:rgba(231,76,60,.12); border-color:rgba(231,76,60,.25);}
      .code-sm { font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.85rem; }
      .kpis { display:flex; gap:.45rem; flex-wrap:wrap; margin:.25rem 0 .25rem 0; }
      .chip { border-radius:999px; padding:.22rem .5rem; font-size:.78rem;
              border:1px solid rgba(255,255,255,.12); background:rgba(255,255,255,.08);}
      .chip.ok   { background: rgba(46,204,113,.18); border-color: rgba(46,204,113,.28);}
      .chip.err  { background: rgba(231,76,60,.18);  border-color: rgba(231,76,60,.28);}
      .chip.warn { background: rgba(241,196,15,.18); border-color: rgba(241,196,15,.28);}
      .stdout { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.78rem; opacity:.9; }
      .section-title { margin: 0 0 8px 2px; }

      /* Decision card */
      .decision { display:flex; flex-direction:column; gap:.4rem; }
      .decision-top { display:flex; align-items:center; justify-content:space-between; gap:1rem; }
      .decision-top .left { display:flex; align-items:center; gap:.9rem; }
      .badge { font-weight:800; letter-spacing:.04em; padding:.35rem .8rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); }
      .buy  { background:rgba(30,158,66,.18); border-color:rgba(30,158,66,.35); }
      .hold { background:rgba(200,200,200,.18); border-color:rgba(200,200,200,.35); }
      .sell { background:rgba(255,107,107,.18); border-color:rgba(255,107,107,.35); }
      .muted { opacity:.7; }
      .big { font-size:1.1rem; font-weight:800; }
      .mono { font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
    f"<h1 style='margin:0'>🧭 TradeMind RL Workflow</h1>"
    f"<span class='pill'>API: {('connected' if API_BASE else 'local-only')}</span>"
    f"</div>",
    unsafe_allow_html=True,
)

DEFAULT_SYMBOLS = ["ETH/USDT","BTC/USDT","SHIB/USDT","SOL/USDT","XRP/USDT"]
DEFAULT_TFS     = ["15m","1h","4h","1d"]

def _safe_index(options: List[str], preferred: Optional[str] = None, fallback_idx: int = 0) -> int:
    if not options: return 0
    if preferred in options: return options.index(preferred)
    return min(max(fallback_idx, 0), len(options)-1)

def _get(url: str, **kw): return requests.get(url, timeout=kw.pop("timeout", 60), **kw)
def _post(url: str, **kw): return requests.post(url, timeout=kw.pop("timeout", 300), **kw)

def list_latest_pairs() -> List[Dict[str, Any]]:
    try:
        r = _get(f"{API_BASE}/rl/latest/index", timeout=30)
        if r.status_code == 200:
            return r.json().get("pairs", [])
    except Exception as e:
        st.error(f"Failed to load /rl/latest/index: {e}")
    return []

def seed_data(symbol: str, tf: str, total_rows: int, page_size: int, exchange: str) -> Optional[Dict[str, Any]]:
    payload = dict(symbol=symbol, tf=tf, total_rows=total_rows, page_size=page_size, exchange=exchange)
    try:
        r = _post(f"{API_BASE}/data/seed", json=payload, timeout=1800)
        if r.status_code == 200:
            return r.json()
        st.error(f"Seed failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"Seed exception: {e}")
    return None

def build_features_local(symbol: str, tf: str) -> Dict[str, Any]:

    import itertools

    here = Path(__file__).resolve()
    ancestors_file = [p for p in here.parents]  
    cwd = Path.cwd().resolve()
    ancestors_cwd = [p for p in cwd.parents] + [cwd]

    seen = set()
    candidate_roots = []
    for p in itertools.chain([here.parent], ancestors_file, ancestors_cwd):
        if p not in seen:
            seen.add(p)
            candidate_roots.append(p)


    root = None
    script_path = None
    for r in candidate_roots:
        sp = r / "scripts" / "build_features.py"
        if sp.exists():
            root, script_path = r, sp
            break

    if root is None or script_path is None:
        return {
            "status": "error",
            "stderr": "Could not locate scripts/build_features.py by walking up from UI file and CWD.",
            "searched": [str(p) for p in candidate_roots[:10]],
        }

    env = os.environ.copy()
    py_paths = [str(root), str(root / "app")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    args = ["--symbol", symbol, "--tf", tf]

    is_pkg = (root / "scripts" / "__init__.py").exists()
    candidates = []
    if is_pkg:
        candidates.append((
            [sys.executable, "-m", "scripts.build_features", *args],
            f"module:scripts.build_features @ {root}"
        ))
    candidates.append((
        [sys.executable, str(script_path), *args],
        f"file:{script_path}"
    ))

    last_err = ""
    for cmd, label in candidates:
        try:
            p = subprocess.run(
                cmd, cwd=str(root), env=env,
                check=True, capture_output=True, text=True, timeout=900
            )
            try:
                out = json.loads(p.stdout.strip())
                out.setdefault("mode", label)
                out.setdefault("cwd", str(root))
                return out
            except Exception:
                return {
                    "status": "ok",
                    "mode": label,
                    "cwd": str(root),
                    "stdout": p.stdout[-1200:],
                }
        except subprocess.CalledProcessError as e:
            last_err = f"[{label}] failed in cwd={root}:\n{(e.stderr or e.stdout or '')[-1200:]}"
        except Exception as e:
            last_err = f"[{label}] exception in cwd={root}: {e!r}"

    return {"status": "error", "stderr": last_err}

def get_history(symbol: str, tf: str, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        r = _get(f"{API_BASE}/rl/history", params={"symbol": symbol, "tf": tf, "limit": int(limit)}, timeout=60)
        if r.status_code == 200:
            return r.json().get("rows", [])
    except Exception:
        pass
    return []

def latest_decision(symbol: str, tf: str) -> Dict[str, Any]:
    rows = get_history(symbol, tf, limit=1)
    return rows[0] if rows else {}

def train_rl(symbol: str, tf: str, steps: int, fees_bps: float, slippage_bps: float,
             use_short: bool, short_fee_bps: float, use_atr_stop: bool, atr_k: float, atr_penalty: float,
             extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    payload = dict(
        symbol=symbol, tf=tf, steps=int(steps),
        fees_bps=float(fees_bps), slippage_bps=float(slippage_bps),
        use_short=bool(use_short), short_fee_bps=float(short_fee_bps),
        use_atr_stop=bool(use_atr_stop), atr_k=float(atr_k), atr_penalty=float(atr_penalty),
    )
    if extra:
        payload.update(extra)
    try:
        r = _post(f"{API_BASE}/rl/train", json=payload, timeout=1800)
        if r.status_code == 200:
            return r.json()
        st.error(f"Train failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"Train exception: {e}")
    return None

def eval_rl(symbol: str, tf: str, fees_bps=10.0, slippage_bps=1.0,
            use_short=False, use_atr_stop=False, atr_k=2.0, atr_penalty=0.0) -> Optional[Dict[str, Any]]:
    payload = dict(
        symbol=symbol, tf=tf, fees_bps=float(fees_bps), slippage_bps=float(slippage_bps),
        use_short=bool(use_short), use_atr_stop=bool(use_atr_stop),
        atr_k=float(atr_k), atr_penalty=float(atr_penalty),
    )
    try:
        r = _post(f"{API_BASE}/rl/eval", json=payload, timeout=900)
        if r.status_code == 200:
            return r.json()
        st.error(f"RL OOS failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"RL OOS exception: {e}")
    return None

def eval_baseline(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = _post(f"{API_BASE}/baseline/eval", json={"symbol": symbol, "tf": tf}, timeout=900)
        if r.status_code == 200:
            return r.json()
        st.error(f"Baseline OOS failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        st.error(f"Baseline OOS exception: {e}")
    return None

def load_windows(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = _get(f"{API_BASE}/rl/windows", params={"symbol": symbol, "tf": tf}, timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def db_counts(symbol: str, tf: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    if not _HAS_DB: return None, None, None
    with session_scope() as s:
        candles = s.query(Candle).filter_by(symbol=symbol, tf=tf).count()
        feats   = s.query(FeatureRow).filter_by(symbol=symbol, tf=tf).count()
        shifted = s.query(FeatureRow).filter_by(symbol=symbol, tf=tf, shifted=True).count()
    return candles, feats, shifted

def recent_candles(symbol: str, tf: str, limit=150) -> Optional[pd.DataFrame]:
    if not _HAS_DB: return None
    with session_scope() as s:
        rows = (s.query(Candle.ts, Candle.open, Candle.high, Candle.low, Candle.close, Candle.volume)
                  .filter(Candle.symbol==symbol, Candle.tf==tf)
                  .order_by(Candle.ts.desc()).limit(limit).all())
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts")

def recent_features(symbol: str, tf: str, limit=150) -> Optional[pd.DataFrame]:
    if not _HAS_DB: return None
    with session_scope() as s:
        rows = (s.query(
                    FeatureRow.ts,
                    FeatureRow.ema_5, FeatureRow.ema_20, FeatureRow.rsi_14,
                    FeatureRow.atr_14, FeatureRow.bb_mid, FeatureRow.bb_up, FeatureRow.bb_dn,
                    FeatureRow.shifted
                )
                .filter(FeatureRow.symbol==symbol, FeatureRow.tf==tf)
                .order_by(FeatureRow.ts.desc()).limit(limit).all())
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["ts","ema_5","ema_20","rsi_14","atr_14","bb_mid","bb_up","bb_dn","shifted"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts")

PAIRS_YAML = ROOT / "config" / "pairs.yaml"
def load_pairs_yaml() -> Dict[str, Any]:
    if not PAIRS_YAML.exists(): return {"pairs": []}
    try: return yaml.safe_load(PAIRS_YAML.read_text()) or {"pairs": []}
    except Exception: return {"pairs": []}
def save_pairs_yaml(cfg: Dict[str, Any]) -> None:
    PAIRS_YAML.parent.mkdir(parents=True, exist_ok=True)
    PAIRS_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

with st.sidebar:
    st.header("Global")
    st.text_input("API Base", value=API_BASE, disabled=True)
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.caption("Add a new pair to the config/pairs.yaml file")
    ns, ntf = st.columns(2)
    with ns: new_symbol = st.text_input("Symbol", value="ETH/USDT")
    with ntf:
        opts_tf = list(DEFAULT_TFS)
        new_tf = st.selectbox("Timeframe", opts_tf, index=_safe_index(opts_tf, "4h", 2))
    if st.button("➕ Add Pair"):
        cfg = load_pairs_yaml()
        pairs = cfg.get("pairs", [])
        if any(p.get("symbol")==new_symbol and p.get("tf")==new_tf for p in pairs):
            st.warning("Pair already exists.")
        else:
            pairs.append({"symbol": new_symbol, "tf": new_tf})
            cfg["pairs"] = pairs
            save_pairs_yaml(cfg)
            st.success("Pair added.")

pairs_index = list_latest_pairs()
known_symbols = sorted({p.get("symbol") for p in pairs_index if p.get("symbol")}) or DEFAULT_SYMBOLS
known_tfs_set = {p.get("tf") for p in pairs_index if p.get("tf")}
known_tfs = list(dict.fromkeys(DEFAULT_TFS + sorted([t for t in known_tfs_set if t])))

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 class='section-title'>Step 0 — Pick Pair</h3>", unsafe_allow_html=True)
c0a, c0b = st.columns(2)
with c0a:
    sel_symbol = st.selectbox("Symbol", known_symbols, index=_safe_index(known_symbols, "ETH/USDT", 0), key="sel_symbol")
with c0b:
    sel_tf     = st.selectbox("Timeframe", known_tfs, index=_safe_index(known_tfs, "4h", 2), key="sel_tf")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
pair_label = f"{sel_symbol} • {sel_tf}"

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 class='section-title'>Step 1 — Seed / Backfill Data</h3>", unsafe_allow_html=True)

if _HAS_DB:
    can0, feat0, shf0 = db_counts(sel_symbol, sel_tf)
    existing_note = f"Already inserted: <b>{can0:,}</b> candles" if isinstance(can0, int) and can0 > 0 else "No candles yet"
    st.markdown(
        f"<div class='kpis'>"
        f"<span class='chip'>Pair: <b>{sel_symbol}</b> • <b>{sel_tf}</b></span>"
        f"<span class='chip {'ok' if (can0 or 0)>0 else 'warn'}'>{existing_note}</span>"
        f"<span class='chip'>Features: <b>{feat0 if feat0 is not None else '—'}</b></span>"
        f"<span class='chip'>Shifted: <b>{shf0 if shf0 is not None else '—'}</b></span>"
        f"</div>", unsafe_allow_html=True
    )

with st.form("seed_form"):
    s3, s4, s5 = st.columns([1,1,1.2])
    with s3:
        seed_total  = st.number_input("Total rows (target)", min_value=1000, max_value=200_000, value=5000, step=1000)
    with s4:
        seed_page   = st.number_input("Page size (page-limit)", min_value=100, max_value=5000, value=1000, step=100)
    with s5:
        seed_exch   = st.selectbox("Exchange", ["binance","bybit","okx"], index=0)
    seed_go = st.form_submit_button(f"⬇️ Fetch & Insert for {pair_label}")

if "seed_last" not in st.session_state:
    st.session_state["seed_last"] = None

if seed_go:
    with st.spinner(f"Seeding {pair_label} via API…"):
        res = seed_data(sel_symbol, sel_tf, int(seed_total), int(seed_page), seed_exch)
        st.session_state["seed_last"] = {
            "request": dict(symbol=sel_symbol, tf=sel_tf, total_rows=int(seed_total), page_size=int(seed_page), exchange=seed_exch),
            "response": res,
        }

if st.session_state["seed_last"]:
    payload = st.session_state["seed_last"]["request"]
    resp = st.session_state["seed_last"]["response"] or {}
    ok = bool(resp) and "totals" in resp

    total_fetched = (resp.get("totals") or {}).get("fetched")
    total_inserted = (resp.get("totals") or {}).get("inserted")
    pages = resp.get("pages") or []

    st.markdown(
        f"<div class='kpis'>"
        f"<span class='chip {'ok' if ok else 'err'}'>{'✅ OK' if ok else '❌ Failed'}</span>"
        f"<span class='chip'>Requested: {payload['total_rows']:,}</span>"
        f"<span class='chip'>Page size: {payload['page_size']:,}</span>"
        f"<span class='chip'>Pages: {len(pages)}</span>"
        f"<span class='chip'>Fetched: <b>{total_fetched if total_fetched is not None else '—'}</b></span>"
        f"<span class='chip'>Inserted: <b>{total_inserted if total_inserted is not None else '—'}</b></span>"
        f"</div>", unsafe_allow_html=True
    )

    if pages:
        dfp = pd.DataFrame(pages)
        cols = [c for c in ["page","requested","fetched","inserted","since","stdout_tail"] if c in dfp.columns]
        st.markdown("**Per-page results**")
        st.dataframe(dfp[cols] if cols else dfp, hide_index=True, use_container_width=True)

    tail = pages[-1].get("stdout_tail") if pages else None
    if tail:
        with st.expander("Script output (tail)", expanded=False):
            st.markdown(f"<div class='stdout'>{tail}</div>", unsafe_allow_html=True)

    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Fresh Data Snapshot (DB)**")
    if _HAS_DB:
        cA, cB, cC = st.columns(3)
        with cA:
            can, feat, shf = db_counts(payload["symbol"], payload["tf"])
            st.markdown(
                f"<div class='kpis'>"
                f"<span class='chip ok'>Candles now in DB: <b>{can if can is not None else '—'}</b></span>"
                f"<span class='chip'>Features: <b>{feat if feat is not None else '—'}</b></span>"
                f"<span class='chip'>Shifted: <b>{shf if shf is not None else '—'}</b></span>"
                f"</div>", unsafe_allow_html=True
            )
        with cB:
            st.markdown("Recent candles")
            d = recent_candles(payload["symbol"], payload["tf"], 150)
            if isinstance(d, pd.DataFrame) and not d.empty:
                st.dataframe(d.tail(60), hide_index=True, use_container_width=True)
            else:
                st.info("—")
        with cC:
            st.markdown("Recent feature rows")
            d = recent_features(payload["symbol"], payload["tf"], 150)
            if isinstance(d, pd.DataFrame) and not d.empty:
                st.dataframe(d.tail(60), hide_index=True, use_container_width=True)
            else:
                st.info("—")
    else:
        st.info("DB not available in this environment; skipping snapshot.")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 class='section-title'>Step 2 — Build Shifted Features</h3>", unsafe_allow_html=True)

build_go = st.button(f"🧪 Build features for {pair_label} (local)")
if build_go:
    with st.spinner(f"Building shift(1) features for {pair_label}…"):
        out = build_features_local(sel_symbol, sel_tf)
        if out.get("status") == "ok":
            st.success(f"Features built. Upserts: {out.get('upserts','?')}")
        else:
            st.error("Feature build failed.")
            if out.get("stderr"): st.code(out["stderr"])
            if out.get("stdout"): st.code(out["stdout"])

if _HAS_DB:
    can, feat, shf = db_counts(sel_symbol, sel_tf)
    st.markdown(
        f"<div class='kpis'>"
        f"<span class='chip'>Candles: <b>{can if can is not None else '—'}</b></span>"
        f"<span class='chip'>Features: <b>{feat if feat is not None else '—'}</b></span>"
        f"<span class='chip'>Shifted: <b>{shf if shf is not None else '—'}</b></span>"
        f"</div>", unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 class='section-title'>Step 3 — Train RL (walk-forward)</h3>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns(4)
with t1:
    tr_steps = st.number_input("Steps", min_value=10_000, max_value=2_000_000, value=120_000, step=10_000)
with t2:
    tr_use_short = st.checkbox("Use short (discrete3)", value=True)
    tr_short_fee = st.number_input("Short fee (bps)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
with t3:
    tr_use_atr_stop = st.checkbox("Use ATR stop", value=False)
    tr_atr_k = st.number_input("ATR k", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
with t4:
    tr_atr_penalty = st.number_input("ATR penalty λ", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

f1, f2 = st.columns(2)
with f1:
    tr_fees = st.number_input("Fees (bps)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
with f2:
    tr_slip = st.number_input("Slippage (bps)", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

c_train1, c_train2 = st.columns([1,1])
with c_train1:
    train_go = st.button(f"⚡️ Train {pair_label} Now")
with c_train2:
    demo_go = st.button(f"🎮 Demo Train {pair_label} (10k steps, n=5)")

if train_go:
    with st.spinner("Training walk-forward…"):
        out = train_rl(sel_symbol, sel_tf, tr_steps, tr_fees, tr_slip,
                       tr_use_short, tr_short_fee, tr_use_atr_stop, tr_atr_k, tr_atr_penalty)
        if out:
            st.success("Training finished.")
            st.json({k: out.get(k) for k in ("latest_path","meta")})

if demo_go:
    with st.spinner("Running demo train (10k steps, n=5 windows)…"):
        out = train_rl(sel_symbol, sel_tf, 10_000, tr_fees, tr_slip,
                       tr_use_short, tr_short_fee, tr_use_atr_stop, tr_atr_k, tr_atr_penalty,
                       extra={"n_windows": 5})
        if out:
            st.success("Demo training finished.")
            st.json({k: out.get(k) for k in ("latest_path","meta")})

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 class='section-title'>Step 4 — Evaluate OOS (RL & Baseline)</h3>", unsafe_allow_html=True)

e1, e2, e3, e4, e5 = st.columns(5)
with e1:
    ev_fees = st.number_input("Fees (bps)", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key="ev_fees")
with e2:
    ev_slip = st.number_input("Slippage (bps)", min_value=0.0, max_value=50.0, value=1.0, step=0.5, key="ev_slip")
with e3:
    ev_use_short = st.checkbox("Use short", value=True, key="ev_short")
with e4:
    ev_use_atr_stop = st.checkbox("ATR stop", value=False, key="ev_atr")
with e5:
    ev_atr_k = st.number_input("ATR k", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="ev_atr_k")

ev2_1, ev2_2 = st.columns(2)
with ev2_1:
    ev_atr_penalty = st.number_input("ATR penalty λ", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="ev_atr_pen")
with ev2_2:
    pass

c_rl, c_bl = st.columns(2)
with c_rl: go_rl = st.button(f"📈 Run RL OOS for {pair_label}")
with c_bl: go_bl = st.button(f"📊 Run Baseline OOS for {pair_label}")

if go_rl:
    with st.spinner("Running RL OOS…"):
        out = eval_rl(sel_symbol, sel_tf, ev_fees, ev_slip, ev_use_short, ev_use_atr_stop, ev_atr_k, ev_atr_penalty)
        if out:
            st.success("RL OOS done.")
            st.session_state["last_oos_rl"] = out
            st.json(out)

if go_bl:
    with st.spinner("Running Baseline OOS…"):
        out = eval_baseline(sel_symbol, sel_tf)
        if out:
            st.success("Baseline OOS done.")
            st.session_state["last_oos_bl"] = out
            st.json(out)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 class='section-title'>Step 5 — Results & Explanation</h3>", unsafe_allow_html=True)

win = load_windows(sel_symbol, sel_tf)
if not win or not isinstance(win, dict):
    st.info("No window tables yet. Run OOS first.")
else:
    rows = win.get("rows", [])
    if rows:
        st.markdown("**Walk-forward windows**")
        wdf = pd.DataFrame(rows)
        st.dataframe(wdf, hide_index=True, use_container_width=True)
        st.download_button("Download windows (CSV)", wdf.to_csv(index=False), file_name="windows.csv", mime="text/csv")

    wins, total = win.get("wins"), win.get("total")
    if wins is not None and total is not None:
        st.markdown(f"<div class='kpis'><span class='chip ok'>RL wins: <b>{wins} / {total}</b> (by equity_final)</span></div>", unsafe_allow_html=True)

    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Equity Charts (latest)**")
    bdir = Path("artifacts") / "backtests"
    rl_prefix = f"rl_{sel_symbol.replace('/','')}_{sel_tf}_"
    bl_prefix = f"baseline_{sel_symbol.replace('/','')}_{sel_tf}_"
    rl_pngs = sorted(bdir.glob(rl_prefix + "*.png"))
    bl_pngs = sorted(bdir.glob(bl_prefix + "*.png")) or sorted(bdir.glob(f"{sel_symbol.replace('/','')}_{sel_tf}_*.png"))
    c1, c2 = st.columns(2)
    with c1:
        if rl_pngs:
            st.image(str(rl_pngs[-1]), caption=rl_pngs[-1].name, use_container_width=True)
        else:
            st.info("No RL PNG yet.")
    with c2:
        if bl_pngs:
            st.image(str(bl_pngs[-1]), caption=bl_pngs[-1].name, use_container_width=True)
        else:
            st.info("No Baseline PNG yet.")

st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
st.markdown("**Price & Indicators (Candles + EMA12/26 + Bollinger Bands)**")

def _plot_price_with_features(symbol: str, tf: str, limit: int = 400):
    df = recent_candles(symbol, tf, limit=limit) if _HAS_DB else None
    if df is None or df.empty:
        st.caption("No price data available from the DB for this pair and/ or the timeframe.")
        return

    d = df.copy()
    d["ema12"] = d["close"].ewm(span=12, adjust=False).mean()
    d["ema26"] = d["close"].ewm(span=26, adjust=False).mean()
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    d["bb_mid"] = mid
    d["bb_up"]  = mid + 2 * std
    d["bb_dn"]  = mid - 2 * std

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d["ts"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="OHLC"
    ))
    fig.add_trace(go.Scatter(x=d["ts"], y=d["ema12"], name="EMA 12", mode="lines"))
    fig.add_trace(go.Scatter(x=d["ts"], y=d["ema26"], name="EMA 26", mode="lines"))
    fig.add_trace(go.Scatter(x=d["ts"], y=d["bb_mid"], name="BB mid", mode="lines"))
    fig.add_trace(go.Scatter(x=d["ts"], y=d["bb_up"],  name="BB up",  mode="lines"))
    fig.add_trace(go.Scatter(x=d["ts"], y=d["bb_dn"],  name="BB dn",  mode="lines"))

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

_plot_price_with_features(sel_symbol, sel_tf, limit=400)

st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
st.markdown("**Latest RL Decision & Rationale**")

def _fetch_explain(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        r = _get(f"{API_BASE}/rl/explain", params={"symbol": symbol, "tf": tf}, timeout=60)
        if r.status_code == 200:
            return r.json()
        st.warning(f"/rl/explain → {r.status_code}")
    except Exception as e:
        st.error(f"Explain error: {e}")
    return None

def _plot_probs_bar(probs: Dict[str, float]):
    df = pd.DataFrame(
        [{"label": k.upper(), "p": float(v)} for k, v in probs.items()]
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("label:N", title="Action"),
            y=alt.Y("p:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("label:N", legend=None,
                            scale=alt.Scale(domain=["BUY","HOLD","SELL"], range=["#1e9e42","#E8E8E8","#ff6b6b"]))
        )
        .properties(height=140)
        .configure_axis(grid=False)
        .configure_view(stroke=None)
        .configure(background="#0E1117")
    )
    st.altair_chart(chart, use_container_width=True)

def _features_illustration(symbol: str, tf: str, limit: int = 160):
    df = recent_candles(symbol, tf, limit=limit) if _HAS_DB else None
    if df is None or df.empty:
        st.caption("There are No candles available for features .")
        return
    df = df.copy()
    df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()

    a = (
        alt.Chart(df)
        .transform_calculate(color="'#8ca0b3'")
        .mark_line()
        .encode(x=alt.X("ts:T", axis=None), y=alt.Y("close:Q", axis=alt.Axis(title="Price")))
        .properties(height=180)
    )
    ema12 = alt.Chart(df).mark_line(stroke="#4ade80").encode(x="ts:T", y="ema12:Q")
    ema26 = alt.Chart(df).mark_line(stroke="#60a5fa").encode(x="ts:T", y="ema26:Q")
    price_block = (a + ema12 + ema26).configure_axis(grid=False).configure_view(stroke=None).configure(background="#0E1117")
    st.altair_chart(price_block, use_container_width=True)

    rsi = pd.Series(df["close"]).diff()
    gain = rsi.clip(lower=0.0)
    loss = -rsi.clip(upper=0.0)
    n = 14
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, pd.NA))
    rsi14 = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame({"ts": df["ts"], "rsi14": rsi14.fillna(50.0)})

    r = alt.Chart(rsi_df).mark_line().encode(
        x=alt.X("ts:T", axis=None), y=alt.Y("rsi14:Q", scale=alt.Scale(domain=[0,100]), title="RSI 14")
    ).properties(height=120)
    bands = alt.Chart(pd.DataFrame({"y":[30, 70]})).mark_rule(strokeDash=[4,3], color="#999").encode(y="y:Q")
    rsi_block = (r + bands).configure_view(stroke=None).configure(background="#0E1117")
    st.altair_chart(rsi_block, use_container_width=True)

exp = _fetch_explain(sel_symbol, sel_tf)
if exp:
    act = str(exp.get("action","HOLD")).upper()
    cls = "buy" if act=="BUY" else ("sell" if act=="SELL" else "hold")
    emoji = {"BUY":"🟢","HOLD":"⚪️","SELL":"🔴"}.get(act,"⚪️")
    conf = exp.get("confidence", 0.0)
    ts_iso = exp.get("ts_iso")

    left_html = (
        f"<div class='left'>"
        f"<span class='badge {cls}'>{emoji} {act}</span>"
        f"<span class='big mono'>Conf: {conf:.2%}</span>"
        f"</div>"
    )
    right_bits = []
    if ts_iso: right_bits.append(f"<span class='chip'><b>As of</b> {ts_iso}</span>")
    st.markdown(f"<div class='decision'><div class='decision-top'><div>{left_html}</div><div>{' '.join(right_bits)}</div></div>", unsafe_allow_html=True)

    cA, cB = st.columns([1,1])
    with cA:
        st.markdown("**Model probabilities**")
        _plot_probs_bar(exp.get("probs", {}))
    with cB:
        st.markdown("**Top reasons**")
        reasons = exp.get("reasons", [])
        if reasons:
            reasons = sorted(reasons, key=lambda r: abs(r.get("weight", 0.0)), reverse=True)[:5]
            df_r = pd.DataFrame(reasons)
            st.dataframe(df_r[["name","detail","weight"]], use_container_width=True, hide_index=True)
        else:
            st.caption("No rationales available yet.")

    st.markdown(f"> {exp.get('text','')}")
    st.markdown("<div class='thin-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Features illustration (EMA & RSI)**")
    _features_illustration(sel_symbol, sel_tf, limit=160)

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No explanation available yet! Please train or seed and build features first.")

st.caption("TradeMind @ 2025")
