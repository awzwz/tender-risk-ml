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
  6. subjects  — full register of participants (company age, size, legal form, OKED);
                 contacts are stored only as keyed hashes (see `_h`)
  7. contract_bank — supplier bank account / legal address of the collected contracts,
                 hashed the same way (for "shared requisites" features)
  8. lot_plans — plan point version each collected lot is linked to (one row per lot × version;
                 includes superseded versions, which the Plans register no longer serves)
  9. plans     — the whole register of current plan points for --plan-years (plans_<year>)
Stages 8–9 are large, so they are written gzipped and converted to parquet with DuckDB.

Usage:
  python -m src.collect --date-from 2024-01-01 --date-to 2024-03-31
  python -m src.collect --stages trd_buy,rnu          # subset of stages
  python -m src.collect --stages lot_plans,plans --plan-years 2023,2024
  python -m src.collect --to-parquet                  # only convert existing jsonl
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import json
import logging
import re
import secrets
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.ows_client import (
    OWSClient, GQL_TRD_BUY, GQL_LOTS, GQL_CONTRACT, GQL_TRD_APP, GQL_RNU,
    GQL_SUBJECT, GQL_CONTRACT_BANK, GQL_PLANS, GQL_LOT_PLANS,
)

log = logging.getLogger("collect")
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CKPT = RAW / "_checkpoint.json"
ALL_STAGES = ["trd_buy", "lots", "contracts", "trd_app", "rnu", "subjects", "contract_bank", "lot_plans", "plans"]
# gzipped stages and the columns that identify a row (for de-duplication after restarts)
GZ_KEYS = {"lot_plans": "lotId, id", "plans": "id"}


# ── contact hashing ──────────────────────────────────────────────────────────
# Features only need "do two participants share a phone / account / address", so raw
# contacts never reach the disk. HMAC with a local secret key: phone numbers have too
# little entropy for a plain hash to be irreversible.

_HASH_KEY_FILE = RAW / ".hash_key"


def _hash_key() -> bytes:
    if not _HASH_KEY_FILE.exists():
        _HASH_KEY_FILE.write_text(secrets.token_hex(32))
    return bytes.fromhex(_HASH_KEY_FILE.read_text().strip())


def _h(value: str) -> str:
    return hmac.new(_hash_key(), value.encode(), hashlib.sha256).hexdigest()[:20]


def _hash_list(raw: str | None, normalize) -> list[str]:
    parts = [normalize(p) for p in re.split(r"[,;]", raw or "")]
    return sorted({_h(p) for p in parts if p})


def _norm_phone(p: str) -> str:
    digits = re.sub(r"\D", "", p)
    return digits[-10:] if len(digits) >= 10 else ""      # +7 / 8 prefixes collapse to the same number


def _norm_email(e: str) -> str:
    e = e.strip().lower()
    return e if "@" in e else ""


def _norm_site(s: str) -> str:
    s = re.sub(r"^(https?://)?(www\.)?", "", s.strip().lower()).split("/")[0]
    return s if "." in s else ""


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w]", " ", s.lower())).strip()


def hash_subject(row: dict) -> dict:
    row["phone_h"] = _hash_list(row.pop("phone", None), _norm_phone)
    row["email_h"] = _hash_list(row.pop("email", None), _norm_email)
    row["website_h"] = _hash_list(row.pop("website", None), _norm_site)
    return row


def hash_contract_bank(row: dict) -> dict:
    iik = re.sub(r"\s", "", (row.pop("supplierIik", None) or "")).upper()
    addr = _norm_text(row.pop("supplierLegalAddress", None) or "")
    row["supplierIik_h"] = _h(iik) if iik else None
    row["supplierAddress_h"] = _h(addr) if addr else None
    return row


# ── checkpoint helpers ───────────────────────────────────────────────────────

def load_ckpt() -> dict:
    return json.loads(CKPT.read_text()) if CKPT.exists() else {}


def save_ckpt(ck: dict) -> None:
    CKPT.write_text(json.dumps(ck, indent=2))


def append_jsonl(path: Path, rows: list[dict]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open     # appended gzip members form a valid stream
    with opener(path, "at", encoding="utf-8") as f:
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
                    filter_key: str, chunk: int, transform=None, expand=None, compress=False) -> None:
    """Fetch child entities for every collected trd_buy id, in id chunks.
    `transform` maps a row to a row, `expand` maps a row to a list of rows."""
    out = RAW / (f"{name}.jsonl.gz" if compress else f"{name}.jsonl")
    st = ck.setdefault(name, {"chunk_idx": 0, "rows": 0, "done": False})
    if st["done"]:
        log.info("%s: already done (%d rows)", name, st["rows"]); return
    parent_ids = read_ids(RAW / "trd_buy.jsonl")
    id_chunks = list(chunks(parent_ids, chunk))
    log.info("%s: %d parent ids -> %d chunks, resuming at chunk %d", name, len(parent_ids), len(id_chunks), st["chunk_idx"])
    bar = tqdm(total=len(id_chunks), initial=st["chunk_idx"], unit="chunk", desc=name)
    for i in range(st["chunk_idx"], len(id_chunks)):
        for batch, _ in c.paginate(query, data_key, {"filter": {filter_key: id_chunks[i]}}):
            if transform:
                batch = [transform(r) for r in batch]
            if expand:
                batch = [x for r in batch for x in expand(r)]
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


def stage_subjects(c: OWSClient, ck: dict) -> None:
    out = RAW / "subjects.jsonl"
    st = ck.setdefault("subjects", {"after": 0, "rows": 0, "done": False})
    if st["done"]:
        log.info("subjects: already done (%d rows)", st["rows"]); return
    total = c.total_count(GQL_SUBJECT)
    bar = tqdm(total=total, initial=st["rows"], unit="rows", desc="subjects")
    for batch, last_id in c.paginate(GQL_SUBJECT, "Subjects", after=st["after"]):
        append_jsonl(out, [hash_subject(r) for r in batch])
        st["after"] = last_id; st["rows"] += len(batch); save_ckpt(ck)
        bar.update(len(batch))
    bar.close()
    st["done"] = True; save_ckpt(ck)


def explode_lot_plans(lot: dict) -> list[dict]:
    """One flat row per (lot, linked plan point version); a lot without a plan keeps one bare row."""
    base = {"lotId": lot["id"], "trdBuyId": lot["trdBuyId"],
            "lotPointList": json.dumps(lot.get("pointList") or [])}
    return [{**base, **flatten(p)} for p in lot.get("Plans") or []] or [base]


def stage_plans(c: OWSClient, ck: dict, years: list[int]) -> None:
    """Whole register of current plan points per plan year, not only the ones our lots use."""
    for y in years:
        name = f"plans_{y}"
        out = RAW / f"{name}.jsonl.gz"
        st = ck.setdefault(name, {"after": 0, "rows": 0, "done": False})
        if st["done"]:
            log.info("%s: already done (%d rows)", name, st["rows"]); continue
        flt = {"plnPointYear": [y]}
        total = c.total_count(GQL_PLANS, {"filter": flt})
        log.info("%s: totalCount=%s, resuming after id=%s", name, total, st["after"])
        bar = tqdm(total=total, initial=st["rows"], unit="rows", desc=name)
        for batch, last_id in c.paginate(GQL_PLANS, "Plans", {"filter": flt}, after=st["after"]):
            append_jsonl(out, [flatten(r) for r in batch])
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


def gz_to_parquet(name: str, key: str) -> None:
    """Gzipped stages are too big for pandas; DuckDB streams them and drops rows repeated after restarts."""
    import duckdb
    src, dst = RAW / f"{name}.jsonl.gz", RAW / f"{name}.parquet"
    duckdb.sql("SET enable_progress_bar = false")
    duckdb.sql(f"""
        COPY (SELECT * FROM read_json_auto('{src}', format='newline_delimited', sample_size=-1)
              QUALIFY row_number() OVER (PARTITION BY {key}) = 1)
        TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
    n, cols = duckdb.sql(f"SELECT count(*), (SELECT count(*) FROM (DESCRIBE SELECT * FROM '{dst}')) FROM '{dst}'").fetchone()
    log.info("%s: %d rows x %d cols -> %s (%.1f MB)", name, n, cols, dst.name, dst.stat().st_size / 1e6)


def to_parquet(stages: list[str] = ALL_STAGES) -> None:
    for name in stages:
        if name in GZ_KEYS:
            files = sorted(RAW.glob("plans_*.jsonl.gz")) if name == "plans" else [RAW / f"{name}.jsonl.gz"]
            for f in files:
                if f.exists():
                    gz_to_parquet(f.name.removesuffix(".jsonl.gz"), GZ_KEYS[name])
            continue
        src = RAW / f"{name}.jsonl"
        if not src.exists():
            continue
        rows = [flatten(json.loads(l)) for l in src.open(encoding="utf-8")]
        df = pd.DataFrame(rows)
        key = "pid" if name == "subjects" else "id"
        if key in df:
            df = df.drop_duplicates(key)
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
    ap.add_argument("--plan-years", default="2024", help="comma-separated plan years for the plans stage")
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
        elif s == "subjects":
            stage_subjects(c, ck)
        elif s == "contract_bank":
            stage_by_parent(c, ck, "contract_bank", GQL_CONTRACT_BANK, "Contract", "trdBuyId", 150,
                            transform=hash_contract_bank)
        elif s == "lot_plans":
            stage_by_parent(c, ck, "lot_plans", GQL_LOT_PLANS, "Lots", "trdBuyId", 150,
                            expand=explode_lot_plans, compress=True)
        elif s == "plans":
            stage_plans(c, ck, [int(y) for y in a.plan_years.split(",")])
        else:
            ap.error(f"unknown stage {s}")
    to_parquet(stages)


if __name__ == "__main__":
    main()
