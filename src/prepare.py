"""
Data preparation for Assignment 2: raw OWS snapshot -> one modelling table
(one row = one procurement that produced at least one contract).

Every step is explained in notebooks/02_data_preparation.ipynb; this module only holds
the SQL so the notebook stays readable and the midterm/A3 notebooks can reuse it.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

TABLES = ["trd_buy", "lots", "contracts", "trd_app", "rnu"]

# Final contract statuses (RefContractStatus) that we treat as a failed contract.
FAILED_STATUSES = {
    340: "Terminated unilaterally",
    350: "Terminated by mutual agreement",
    330: "Not concluded",
    345: "Awaiting termination",
    462: "Transferred / terminated unilaterally",
    463: "Transferred / terminated by mutual agreement",
}
# Strict variant for sensitivity analysis: drop "mutual agreement" terminations.
STRICT_FAILED_STATUSES = {340, 330, 345, 462}
PARTIAL_STATUS = 375  # "Partially executed": ambiguous, kept as 0 and reported separately

SUBJECT_TYPES = {1: "Goods", 2: "Works", 3: "Services"}

# First two digits of the KATO code -> region (51 is the pre-2018 code of Shymkent).
REGIONS = {
    "10": "Abai", "11": "Akmola", "15": "Aktobe", "19": "Almaty region", "23": "Atyrau",
    "27": "West Kazakhstan", "31": "Zhambyl", "33": "Zhetisu", "35": "Karaganda",
    "39": "Kostanay", "43": "Kyzylorda", "47": "Mangystau", "51": "Shymkent",
    "55": "Pavlodar", "59": "North Kazakhstan", "61": "Turkestan", "62": "Ulytau",
    "63": "East Kazakhstan", "71": "Astana", "75": "Almaty city", "79": "Shymkent",
}

TRADE_METHODS_EN = {
    3: "Request for quotations",
    6: "Single source after failed tender",
    133: "Framework agreement, 2nd stage",
    2: "Open tender",
    32: "Tender with prequalification",
    188: "Tender with scoring system",
    121: "Tender: school food (goods)",
}

APP_LOTS_SCHEMA = (
    '[{"lotId":"BIGINT","statusId":"BIGINT","price":"DOUBLE","amount":"DOUBLE",'
    '"discountValue":"DOUBLE","discountPrice":"DOUBLE"}]'
)


def connect(raw_dir: Path) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with one view per raw parquet file (<table>_raw)."""
    con = duckdb.connect()
    for t in TABLES:
        con.sql(f"CREATE VIEW {t}_raw AS SELECT * FROM read_parquet('{(raw_dir / f'{t}.parquet').as_posix()}')")
    return con


def _sql_map(mapping: dict, col: str, default: str = "NULL") -> str:
    whens = " ".join(f"WHEN {col} = {k!r} THEN {v!r}" for k, v in mapping.items())
    return f"CASE {whens} ELSE {default} END"


def build_clean_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Typed, de-duplicated, decoded versions of the raw tables (prefix clean_)."""
    region = _sql_map(REGIONS, "substr(kato, 3, 2)", "'Unknown'")
    subject = _sql_map(SUBJECT_TYPES, "refSubjectTypeId", "'Unknown'")
    method = _sql_map(TRADE_METHODS_EN, "refTradeMethodsId", "'Other'")
    con.sql(f"""
        CREATE OR REPLACE TABLE clean_buy AS
        SELECT id, numberAnno, nameRu,
               refTradeMethodsId, RefTradeMethods_nameRu AS trade_method_ru, {method} AS trade_method,
               refSubjectTypeId, {subject} AS subject_type,
               CAST(publishDate AS TIMESTAMP) AS publish_dt,
               CAST(startDate   AS TIMESTAMP) AS start_dt,
               CAST(endDate     AS TIMESTAMP) AS end_dt,
               totalSum AS total_sum, countLots AS count_lots, orgBin AS org_bin,
               CASE WHEN kato = '[null]' THEN 'Unknown' ELSE {region} END AS region,
               singlOrgSign::INT AS single_organizer, isLightIndustry::INT AS light_industry,
               isConstructionWork::INT AS construction, (disablePersonId > 0)::INT AS disabled_org,
               (parentId > 0)::INT AS has_parent,
               refBuyStatusId, RefBuyStatus_nameRu AS buy_status_ru
        FROM trd_buy_raw
        QUALIFY row_number() OVER (PARTITION BY id ORDER BY lastUpdateDate DESC) = 1
    """)
    # last version of every contract chain (rootId groups the original + amendments)
    con.sql("""
        CREATE OR REPLACE TABLE clean_contract AS
        SELECT id, rootId, trdBuyId, supplierBiin, refContractStatusId AS status_id,
               RefContractStatus_nameRu AS status_ru, contractSum AS contract_sum,
               trim(replace(RefContractCancel_nameRu, '§', '')) AS cancel_reason_ru
        FROM contracts_raw
        QUALIFY row_number() OVER (PARTITION BY rootId ORDER BY id DESC) = 1
    """)
    con.sql(f"""
        CREATE OR REPLACE TABLE clean_bid AS
        SELECT a.id AS app_id, a.buyId AS buy_id, a.supplierBinIin AS supplier_bin,
               j.lotId AS lot_id, j.price
        FROM trd_app_raw a, unnest(from_json(a.AppLots, '{APP_LOTS_SCHEMA}')) t(j)
    """)


def build_model_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Join features (known at award time) with the contract-outcome target."""
    failed = ", ".join(map(str, FAILED_STATUSES))
    strict = ", ".join(map(str, STRICT_FAILED_STATUSES))
    return con.sql(f"""
        WITH target AS (
            SELECT trdBuyId AS id,
                   max((status_id IN ({failed}))::INT) AS contract_failed,
                   max((status_id IN ({strict}))::INT) AS contract_failed_strict,
                   max((status_id = {PARTIAL_STATUS})::INT) AS partially_executed,
                   count(*) AS n_contracts
            FROM clean_contract GROUP BY 1
        ),
        lot_agg AS (
            SELECT trdBuyId AS id, count(*) AS n_lots,
                   max(dumping)::INT AS any_dumping,
                   max(psdSign)::INT AS any_design_docs,
                   max(consultingServices)::INT AS any_consulting,
                   string_agg(nameRu, ' | ') AS lot_names
            FROM lots_raw GROUP BY 1
        ),
        bid_lot AS (   -- one row per (lot, bid); price is a unit price -> compare with planned unit price
            SELECT b.buy_id, b.lot_id, b.app_id, b.supplier_bin,
                   b.price * l.count / NULLIF(l.amount, 0) AS price_ratio
            FROM clean_bid b JOIN lots_raw l ON l.id = b.lot_id
            WHERE b.price > 0 AND l.amount > 0 AND l.count > 0
        ),
        lot_comp AS (
            SELECT buy_id, lot_id, count(*) AS bids, min(price_ratio) AS best_ratio,
                   stddev_samp(price_ratio) / NULLIF(avg(price_ratio), 0) AS price_cv
            FROM bid_lot GROUP BY 1, 2
        ),
        bid_agg AS (
            SELECT buy_id AS id, avg(bids) AS bids_per_lot, avg(best_ratio) AS best_bid_ratio,
                   avg(price_cv) AS bid_price_cv
            FROM lot_comp GROUP BY 1
        ),
        app_agg AS (
            SELECT buy_id AS id, count(DISTINCT app_id) AS n_applications,
                   count(DISTINCT supplier_bin) AS n_bidders
            FROM clean_bid GROUP BY 1
        ),
        org_hist AS (  -- organiser activity before this procurement (no outcome information)
            SELECT id, row_number() OVER (PARTITION BY org_bin ORDER BY publish_dt, id) - 1 AS org_prior_buys
            FROM clean_buy
        )
        SELECT b.id, b.publish_dt, b.trade_method, b.subject_type, b.region,
               b.total_sum, b.count_lots,
               date_diff('hour', b.start_dt, b.end_dt) / 24.0 AS bid_window_days,
               date_diff('hour', b.publish_dt, b.start_dt) / 24.0 AS publish_to_start_days,
               dayofweek(b.publish_dt) AS publish_dow,
               b.single_organizer, b.light_industry, b.construction, b.disabled_org, b.has_parent,
               coalesce(la.n_lots, 0) AS n_lots, coalesce(la.any_dumping, 0) AS any_dumping,
               coalesce(la.any_design_docs, 0) AS any_design_docs,
               coalesce(la.any_consulting, 0) AS any_consulting,
               coalesce(aa.n_applications, 0) AS n_applications,
               coalesce(aa.n_bidders, 0) AS n_bidders,
               ba.bids_per_lot, ba.best_bid_ratio, ba.bid_price_cv,
               oh.org_prior_buys, la.lot_names,
               t.contract_failed, t.contract_failed_strict, t.partially_executed, t.n_contracts
        FROM clean_buy b
        JOIN target t USING (id)
        LEFT JOIN lot_agg la USING (id)
        LEFT JOIN app_agg aa USING (id)
        LEFT JOIN bid_agg ba USING (id)
        LEFT JOIN org_hist oh USING (id)
        ORDER BY b.publish_dt, b.id
    """).df()


# ── modelling table used by the midterm / A3 notebooks ──────────────────────

RELEASE_URL = "https://github.com/awzwz/tender-risk-ml/releases/download/data-v1/"

NUM_FEATURES = ["log_total_sum", "n_lots", "bid_window_days", "publish_to_start_days", "n_bidders",
                "bids_per_lot", "best_bid_ratio", "bid_price_cv", "org_prior_buys"]
BIN_FEATURES = ["no_bids", "single_organizer", "construction", "light_industry", "disabled_org",
                "any_dumping", "any_design_docs", "any_consulting"]
CAT_FEATURES = ["trade_method", "subject_type", "region"]
FEATURES = NUM_FEATURES + BIN_FEATURES + CAT_FEATURES
TEXT_FEATURE = "lot_names"
TARGET = "contract_failed"

VAL_START, TEST_START = "2024-03-01", "2024-03-16"   # chronological split boundaries


def finalize_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Post-SQL cleaning steps C5–C9 of the A2 notebook plus the time-based split."""
    import numpy as np

    df = df.copy()
    df["construction"] = (df["construction"] > 0).astype(int)                       # C5 mis-coded flag
    bad = df["best_bid_ratio"].notna() & ~df["best_bid_ratio"].between(0.05, 1.0001)
    df.loc[bad, "best_bid_ratio"] = np.nan                                           # C7 unit errors
    df["log_total_sum"] = np.log10(df["total_sum"])                                  # C8 skewed value
    df["no_bids"] = (df["n_bidders"] == 0).astype(int)                               # C9 structural gap
    df["split"] = np.select([df["publish_dt"] < VAL_START, df["publish_dt"] < TEST_START],
                            ["train", "val"], "test")
    keep = ["id", "publish_dt", "split", TARGET, "contract_failed_strict", "partially_executed",
            TEXT_FEATURE] + FEATURES
    return df[keep].reset_index(drop=True)


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)


def load_model_table(root: Path, rebuild: bool = False) -> pd.DataFrame:
    """
    Return the clean modelling table (one row = one procurement with a contract).

    Order of attempts: local data/processed/model_table.parquet -> the same file from the
    GitHub release -> full rebuild from the raw snapshot (downloaded if missing).
    """
    processed = root / "data" / "processed" / "model_table.parquet"
    if processed.exists() and not rebuild:
        return pd.read_parquet(processed)
    if not rebuild:
        try:
            _download(RELEASE_URL + processed.name, processed)
            return pd.read_parquet(processed)
        except Exception as e:  # no network / asset missing -> rebuild from raw
            print(f"could not download the prepared table ({e}); rebuilding from raw data")
    raw = root / "data" / "raw"
    for t in TABLES:
        if not (raw / f"{t}.parquet").exists():
            _download(RELEASE_URL + f"{t}.parquet", raw / f"{t}.parquet")
    con = connect(raw)
    build_clean_tables(con)
    df = finalize_model_table(build_model_table(con))
    processed.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed, index=False)
    return df
