# 0 ) System Requirements 
OS: macOS 
Python: 3.11 - !!--- MUST ---!!
Optional: Docker 

# 1 ) First Time Setup 
python3 -m venv .venv
source .venv/bin/activate           
pip install --upgrade pip
pip install -r requirements.txt

# 1.1) Database build using sqlalchemy (CRUCIAL!!!)
python - <<'PY'
from app.data.session import engine
from app.data.models import Base
Base.metadata.create_all(engine)
print("✅ Tables created")
PY

# 2 )  Feed system with some data
# VIP!!!!!! before running anything else system msut be feed with data
python -m scripts.backfill_pair --symbol BTC/USDT --tf 4h --target 5000

# 3 ) Run backend fast api using uvicorn 
source .venv/bin/activate           
uvicorn app.api.server:app --host 0.0.0.0 --port 8000 --reload
# NOTE: if the first uvicorn connection fail close it (ctr+c) and re run again (check entry point and port connection)

# 4 ) Run the Frontend Streamlit UI (in another terminal with venv activated)
source .venv/bin/activate           
streamlit run app/ui/overview.py


# 5 ) Training flow 

# 5.1 )  Feed system with some data
python scripts/backfill_pair.py --symbol BTC/USDT --tf 4h --target 5000

# 5.2 ) Build shifted features
python -m scripts.build_features --symbol BTC/USDT --tf 4h

# 5.3 ) Verify shifted rows
python - <<'PY'
from app.data.session import session_scope
from app.data.models import FeatureRow
with session_scope() as s:
    n = s.query(FeatureRow).filter_by(symbol="BTC/USDT", tf="4h", shifted=True).count()
print("BTC/USDT 4h shifted rows:", n)
PY

# 5.4 ) full run 
python -m app.rl.train --symbol BTC/USDT --tf 4h \
  --fees_bps 10 --slippage_bps 1 \
  --steps 300000 --train-span 3000 --test-span 500 --stride 250 --seed 42 \
  --use-short true --short-fee-bps 2

# 5.5 ) Evaluate OOS (deterministic)
# Reads per-pair latest/<PAIR>.json unless you pass --checkpoint
python -m app.rl.eval --symbol BTC/USDT --tf 4h


# 5.6 ) Baseline comparison (same windows)
python scripts/baseline_windows_from_db.py --symbol BTC/USDT --tf 4h


# 6 ) Tests 
pytest -q
