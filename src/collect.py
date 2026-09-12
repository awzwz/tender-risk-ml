"""
Collect the raw dataset from OWS v3 into data/raw/*.jsonl (+ parquet at the end).

Stages (each resumable via data/raw/_checkpoint.json):
  1. trd_buy   — procurements with publishDate in [date_from, date_to]
  2. lots      — lots of those procurements       (filter trdBuyId, chunks of 150)
  3. contracts — contracts of those procurements  (filter trdBuyId, chunks of 150)
  4. trd_app   — applications (bids). OWS' buyId filter takes a single id and the
                 dateApply filter is broken server-side, so we binary-search the id
                 range covering [date_from - 7d, date_to + 60d] (ids are chronological)
                 and walk it with the unfiltered `after` cursor; rows are then restricted
                 to buyId in trd_buy at parquet time.
  5. rnu       — full register of unreliable suppliers

Usage:
  python -m src.collect --date-from 2024-01-01 --date-to 2024-03-31
  python -m src.collect --stages trd_buy,rnu          # subset of stages
  python -m src.collect --to-parquet                  # only convert existing jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ows_client import (
    OWSClient, GQL_TRD_BUY, GQL_LOTS, GQL_CONTRACT, GQL_TRD_APP, GQL_RNU,
)

log = logging.getLogger("collect")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CKPT = RAW / "_checkpoint.json"
ALL_STAGES = ["trd_buy", "lots", "contracts", "trd_app", "rnu"]


# ── checkpoint helpers ───────────────────────────────────────────────────────

def load_ckpt() -> dict:
    return json.loads(CKPT.read_text()) if CKPT.exists() else {}


def save_ckpt(ck: dict) -> None:
    CKPT.write_text(json.dumps(ck, indent=2))


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_ids(path: Path, key: str = "id") -> list[int]:
    ids = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)[key])
    return sorted(set(ids))


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ── stages ───────────────────────────────────────────────────────────────────

def stage_trd_buy(c: OWSClient, ck: dict, date_from: str, date_to: str) -> None:
    out = RAW / "trd_buy.jsonl"
    st = ck.setdefault("trd_buy", {"after": 0, "rows": 0, "done": False})
    if st["done"]:
        log.info("trd_buy: already done (%d rows)", st["rows"]); return
    # plain YYYY-MM-DD bounds; the microsecond format used by the old ETL now returns a server error
    flt = {"publishDate": [date_from, date_to]}
    total = c.total_count(GQL_TRD_BUY, {"filter": flt})
    log.info("trd_buy: totalCount=%s for %s..%s, resuming after id=%s", total, date_from, date_to, st["after"])
    bar = tqdm(total=total, initial=st["rows"], unit="rows", desc="trd_buy")
    for batch, last_id in c.paginate(GQL_TRD_BUY, "TrdBuy", {"filter": flt}, after=st["after"]):
        append_jsonl(out, batch)
        st["after"] = last_id; st["rows"] += len(batch); save_ckpt(ck)
        bar.update(len(batch))
    bar.close()
    st["done"] = True; save_ckpt(ck)


def stage_by_parent(c: OWSClient, ck: dict, name: str, query: str, data_key: str,
                    filter_key: str, chunk: int) -> None:
    """Fetch child entities for every collected trd_buy id, in id chunks."""
    out = RAW / f"{name}.jsonl"
    st = ck.setdefault(name, {"chunk_idx": 0, "rows": 0, "done": False})
    if st["done"]:
        log.info("%s: already done (%d rows)", name, st["rows"]); return
    parent_ids = read_ids(RAW / "trd_buy.jsonl")
    id_chunks = list(chunks(parent_ids, chunk))
    log.info("%s: %d parent ids -> %d chunks, resuming at chunk %d", name, len(parent_ids), len(id_chunks), st["chunk_idx"])
    bar = tqdm(total=len(id_chunks), initial=st["chunk_idx"], unit="chunk", desc=name)
    for i in range(st["chunk_idx"], len(id_chunks)):
        for batch, _ in c.paginate(query, data_key, {"filter": {filter_key: id_chunks[i]}}):
            append_jsonl(out, batch); st["rows"] += len(batch)
        st["chunk_idx"] = i + 1; save_ckpt(ck)
        bar.update(1)
    bar.close()
    st["done"] = True; save_ckpt(ck)


def stage_rnu(c: OWSClient, ck: dict) -> None:
    out = RAW / "rnu.jsonl"
    st = ck.setdefault("rnu", {"after": 0, "rows": 0, "done": False})
    if st["done"]:
        log.info("rnu: already done (%d rows)", st["rows"]); return
    total = c.total_count(GQL_RNU)
    bar = tqdm(total=total, initial=st["rows"], unit="rows", desc="rnu")
    for batch, last_id in c.paginate(GQL_RNU, "Rnu", after=st["after"]):
        append_jsonl(out, batch)
        st["after"] = last_id; st["rows"] += len(batch); save_ckpt(ck)
        bar.update(len(batch))
    bar.close()
    st["done"] = True; save_ckpt(ck)


_TRD_APP_PROBE = "query($limit:Int!,$after:Int){ TrdApp(limit:$limit,after:$after){ id dateApply } }"


def find_trd_app_id_bound(c: OWSClient, target_date: str, lo: int = 1, hi: int | None = None) -> int:
    """
    Smallest TrdApp id whose dateApply >= target_date ("YYYY-MM-DD ...").
    Uses the fact that `after=x` returns the newest record with id < x and that ids are
    chronological. ~26 requests for the full id space.
    """
    if hi is None:
        newest = c.graphql(_TRD_APP_PROBE, {"limit": 1})["data"]["TrdApp"][0]
        hi = newest["id"] + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        rows = c.graphql(_TRD_APP_PROBE, {"limit": 1, "after": mid})["data"]["TrdApp"]
        if not rows:                        # nothing below mid -> boundary is above
            lo = mid
            continue
        rec = rows[0]                        # newest record with id < mid
        if (rec["dateApply"] or "") >= target_date:
            hi = rec["id"]                   # boundary is at or below rec
        else:
            lo = mid                         # everything below mid is older
        time.sleep(c.delay)
    return hi


def stage_trd_app(c: OWSClient, ck: dict, date_from: str, date_to: str) -> None:
    """Walk the TrdApp id range that covers the period, newest -> oldest, with the unfiltered cursor."""
    from datetime import date, timedelta
    out = RAW / "trd_app.jsonl"
    st = ck.setdefault("trd_app", {"id_lo": None, "id_hi": None, "after": None, "rows": 0, "done": False})
    if st["done"]:
        log.info("trd_app: already done (%d rows)", st["rows"]); return
    if st["id_lo"] is None:
        d_lo = (date.fromisoformat(date_from) - timedelta(days=7)).isoformat()
        d_hi = (date.fromisoformat(date_to) + timedelta(days=60)).isoformat()   # late bids for March procurements
        st["id_lo"] = find_trd_app_id_bound(c, d_lo)
        st["id_hi"] = find_trd_app_id_bound(c, d_hi)
        st["after"] = st["id_hi"]
        save_ckpt(ck)
    log.info("trd_app: id range [%d, %d) (~%d ids), resuming after id=%d",
             st["id_lo"], st["id_hi"], st["id_hi"] - st["id_lo"], st["after"])
    bar = tqdm(total=st["id_hi"] - st["id_lo"], initial=st["id_hi"] - st["after"], unit="ids", desc="trd_app")
    for batch, last_id in c.paginate(GQL_TRD_APP, "TrdApp", after=st["after"]):
        keep = [r for r in batch if r["id"] >= st["id_lo"]]
        append_jsonl(out, keep)
        bar.update(st["after"] - max(last_id, st["id_lo"]))
        st["after"] = last_id; st["rows"] += len(keep); save_ckpt(ck)
        if last_id < st["id_lo"]:
            break
    bar.close()
    st["done"] = True; save_ckpt(ck)


# ── jsonl -> parquet ─────────────────────────────────────────────────────────

def flatten(row: dict) -> dict:
    """Pull nested Ref{nameRu} objects up to flat columns; keep AppLots as JSON string."""
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}_{kk}"] = vv
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        else:
            out[k] = v
    return out


def to_parquet() -> None:
    for name in ALL_STAGES:
        src = RAW / f"{name}.jsonl"
        if not src.exists():
            continue
        rows = [flatten(json.loads(l)) for l in src.open(encoding="utf-8")]
        df = pd.DataFrame(rows)
        if "id" in df:
            df = df.drop_duplicates("id")
        if name == "trd_app" and (RAW / "trd_buy.parquet").exists():
            buy_ids = set(pd.read_parquet(RAW / "trd_buy.parquet", columns=["id"])["id"])
            before = len(df); df = df[df["buyId"].isin(buy_ids)]
            log.info("trd_app: kept %d of %d rows with buyId in trd_buy", len(df), before)
        dst = RAW / f"{name}.parquet"
        df.to_parquet(dst, index=False)
        log.info("%s: %d rows x %d cols -> %s (%.1f MB)", name, len(df), df.shape[1], dst.name, dst.stat().st_size / 1e6)


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--date-from", default="2024-01-01")
    ap.add_argument("--date-to", default="2024-03-31")
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between pages")
    ap.add_argument("--to-parquet", action="store_true", help="only convert jsonl -> parquet")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    RAW.mkdir(parents=True, exist_ok=True)

    if a.to_parquet:
        to_parquet(); return

    c = OWSClient(delay=a.delay)
    ck = load_ckpt()
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    for s in stages:
        if s == "trd_buy":
            stage_trd_buy(c, ck, a.date_from, a.date_to)
        elif s == "lots":
            stage_by_parent(c, ck, "lots", GQL_LOTS, "Lots", "trdBuyId", 150)
        elif s == "contracts":
            stage_by_parent(c, ck, "contracts", GQL_CONTRACT, "Contract", "trdBuyId", 150)
        elif s == "trd_app":
            stage_trd_app(c, ck, a.date_from, a.date_to)
        elif s == "rnu":
            stage_rnu(c, ck)
        else:
            ap.error(f"unknown stage {s}")
    to_parquet()


if __name__ == "__main__":
    main()
