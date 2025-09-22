from __future__ import annotations
import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[1]  
    metrics_dir = repo_root / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(metrics_dir.glob("api_requests_*.jsonl")) or \
                 sorted(metrics_dir.glob("api_requests.jsonl"))

    if not candidates:
        print("[rollup] no metrics files found in", metrics_dir)
        return

    path = candidates[-1]
    data = {}
    for line in path.read_text().splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        route = obj.get("route","?")
        ms = float(obj.get("latency_ms",0))
        data.setdefault(route, []).append(ms)

    print(f"[rollup] {path.name}")
    print(f"{'route':30}  {'count':>5}  {'p50(ms)':>8}  {'p95(ms)':>8}")
    for rt, arr in sorted(data.items()):
        arr = sorted(arr)
        n = len(arr)
        if n == 0:
            continue
        p50 = arr[int(0.50*(n-1))]
        p95 = arr[int(0.95*(n-1))]
        print(f"{rt:30}  {n:5d}  {p50:8.1f}  {p95:8.1f}")

if __name__ == "__main__":
    main()
