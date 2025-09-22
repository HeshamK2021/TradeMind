from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/data", tags=["data"])
ROOT = Path(__file__).resolve().parents[2] 


class SeedRequest(BaseModel):
    symbol: str = Field(..., description="e.g. 'BTC/USDT'")
    tf: str = Field(..., description="e.g. '4h'")
    total_rows: int = Field(5000, ge=1, description="Desired rows total (target) for backfill")
    page_size: int = Field(1000, ge=1, description="Rows per page (API/exchange max often 1000)")
    exchange: str = Field("binance", description="Ignored by backfill_pair.py (kept for compatibility)")


class PageResult(BaseModel):
    page: int
    requested: int
    fetched: Optional[int] = None
    inserted: Optional[int] = None
    stdout_tail: Optional[str] = None  
    since: Optional[int] = None      


class SeedResponse(BaseModel):
    symbol: str
    tf: str
    requested: int
    page_size: int
    pages: List[PageResult]
    totals: Dict[str, int]



def _run_backfill(symbol: str, tf: str, target: int, page_limit: int, exchange: str) -> str:

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", str(ROOT))

    py = str(Path(env.get("VIRTUAL_ENV", "")) / "bin" / "python") if env.get("VIRTUAL_ENV") else "python"

    script = ROOT / "scripts" / "backfill_pair.py"
    if not script.exists():
        raise HTTPException(status_code=404, detail=f"Script not found: {script}")

    cmd = [
        py, str(script),
        "--symbol", symbol,
        "--tf", tf,
        "--target", str(int(target)),
        "--page-limit", str(int(page_limit)),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        return proc.stdout or ""
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "")[-1200:]
        raise HTTPException(status_code=500, detail=f"backfill_pair failed: {msg}")
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="backfill_pair timed out")


def _try_parse_json_blocks(stdout: str) -> List[Dict[str, Any]]:

    blocks: List[Dict[str, Any]] = []
    for match in re.finditer(r'\{.*?\}', stdout, flags=re.DOTALL):
        frag = match.group(0)
        if len(frag) > 10000:
            continue
        try:
            j = json.loads(frag)
            if isinstance(j, dict):
                blocks.append(j)
        except Exception:
            pass
    return blocks


def _parse_pages_from_stdout(stdout: str) -> Tuple[List[PageResult], int, int]:

    pages: List[PageResult] = []
    total_fetched = 0
    total_inserted = 0

    blocks = _try_parse_json_blocks(stdout)
    page_no = 1
    for b in blocks:
        if any(k in b for k in ("page", "fetched", "inserted", "requested", "since", "limit", "page_size")):
            fetched = b.get("fetched")
            inserted = b.get("inserted")
            requested = b.get("requested") or b.get("limit") or b.get("page_size") or 0
            page = int(b.get("page") or page_no)
            since = b.get("since")
            if isinstance(fetched, (int, float)):
                total_fetched += int(fetched)
            if isinstance(inserted, (int, float)):
                total_inserted += int(inserted)
            pages.append(PageResult(
                page=page,
                requested=int(requested) if isinstance(requested, (int, float)) else 0,
                fetched=int(fetched) if isinstance(fetched, (int, float)) else None,
                inserted=int(inserted) if isinstance(inserted, (int, float)) else None,
                stdout_tail=None,
                since=int(since) if isinstance(since, (int, float)) else None
            ))
            page_no += 1

    if pages:
        return pages, total_fetched, total_inserted

    fetched_matches = [int(m.group(1)) for m in re.finditer(r'"fetched"\s*:\s*(\d+)', stdout)]
    inserted_matches = [int(m.group(1)) for m in re.finditer(r'"inserted"\s*:\s*(\d+)', stdout)]
    requested_matches = [int(m.group(1)) for m in re.finditer(r'"requested"\s*:\s*(\d+)', stdout)]
    n = max(len(fetched_matches), len(inserted_matches), len(requested_matches))
    if n > 0:
        for i in range(n):
            f = fetched_matches[i] if i < len(fetched_matches) else None
            ins = inserted_matches[i] if i < len(inserted_matches) else None
            req = requested_matches[i] if i < len(requested_matches) else 0
            if isinstance(f, int):
                total_fetched += f
            if isinstance(ins, int):
                total_inserted += ins
            pages.append(PageResult(page=i + 1, requested=req, fetched=f, inserted=ins))
        return pages, total_fetched, total_inserted

    return [], 0, 0



@router.post("/seed", response_model=SeedResponse)
def seed_endpoint(p: SeedRequest) -> SeedResponse:

    stdout = _run_backfill(
        symbol=p.symbol,
        tf=p.tf,
        target=int(p.total_rows),
        page_limit=int(p.page_size),
        exchange=p.exchange,  
    )

    pages, total_fetched, total_inserted = _parse_pages_from_stdout(stdout)

    if not pages:
        pages = [
            PageResult(
                page=1,
                requested=p.page_size,
                fetched=None,
                inserted=None,
                stdout_tail=stdout[-800:],
            )
        ]
    else:
        pages[-1].stdout_tail = stdout[-800:]

    return SeedResponse(
        symbol=p.symbol,
        tf=p.tf,
        requested=p.total_rows,
        page_size=p.page_size,
        pages=pages,
        totals={"fetched": int(total_fetched), "inserted": int(total_inserted)},
    )


@router.get("/seed/echo")
def seed_echo(symbol: str, tf: str, rows: int = 1000, exchange: str = "binance"):
    return {
        "ok": True,
        "symbol": symbol,
        "tf": tf,
        "rows": rows,
        "exchange": exchange,
        "root": str(ROOT),
    }
