"""
A3 features added to the 20 midterm features (F20). All are computed as of the procurement's publication date,
from other procurements only, so nothing known after the award can enter (research v3, §11 of
reports/A3_research_ru.md; pilots experiments/a3_pilots/p5_indicators.py and winner.py).

  REL    customer–supplier history (Tender Radar's RECURRING_WINNER, CAROUSEL, SUPPLIER_CONCENTRATION,
         CUSTOMER_WINNER_CONCENTRATION, HIGH_WIN_RATE_FEW_BIDS re-computed without look-ahead)
  PRICE  planned unit price against the January–February median for the same lot name (OVERPRICED_UNIT;
         ENSTRU codes are empty in OWS lots, so the lot name stands in for the product code)

The winner is never taken from the contracts table: the contract signer is partly the outcome (it differs from
the bidders after a refusal). It is the supplier named in the announcement (single source), otherwise the lowest
bidder by discounted price.

    from src.features import load_a3_table, REL, PRICE
    df = load_a3_table(ROOT)        # model table + REL + PRICE, cached in data/processed/a3_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src import prepare as P

REL = ["pair_prior_contracts", "pair_share_of_customer", "customer_hhi", "winner_contracts_30d",
       "winner_top_customer_share", "winner_win_rate"]
PRICE = ["unit_price_vs_market", "unit_price_vs_market_max", "share_lots_benchmarked"]
MIN_HISTORY = 5          # shares and rates need at least this many earlier events, otherwise NaN
MIN_BENCHMARK = 5        # lots with the same name in Jan–Feb needed for a market median

CACHE = "a3_features.parquet"
RAW_TABLES = ["trd_buy", "lots", "contracts", "trd_app"]


def _connect(raw: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("SET enable_progress_bar = false")
    r = raw.as_posix()
    con.sql(f"""CREATE TABLE buy AS
        SELECT id, orgBin AS org, CAST(publishDate AS TIMESTAMP) AS pdt, nullif(biinSupplier, '') AS announced
        FROM read_parquet('{r}/trd_buy.parquet')
        QUALIFY row_number() OVER (PARTITION BY id ORDER BY lastUpdateDate DESC) = 1""")
    con.sql(f"""CREATE TABLE app AS
        SELECT a.buyId AS buy_id, a.supplierBinIin AS sup, CAST(a.dateApply AS TIMESTAMP) AS dt, j.lotId AS lot_id,
               CASE WHEN j.discountPrice > 0 THEN j.discountPrice ELSE j.price END AS p
        FROM read_parquet('{r}/trd_app.parquet') a, unnest(from_json(a.AppLots, '{P.APP_LOTS_SCHEMA}')) t(j)""")
    # one row per contract chain (the original and its amendments share rootId)
    con.sql(f"""CREATE TABLE ctr AS
        SELECT min(trdBuyId) AS buy_id, min(customerBin) AS cust, min(supplierBiin) AS sup,
               min(CAST(signDate AS TIMESTAMP)) AS sdt
        FROM read_parquet('{r}/contracts.parquet') GROUP BY rootId""")
    con.sql(f"CREATE VIEW lots AS SELECT * FROM read_parquet('{r}/lots.parquet')")
    return con


def _award(con: duckdb.DuckDBPyConnection) -> None:
    """Table `award(id, t, winner, cust)`: who won and for whom, without the contracts table.

    Ties are broken deterministically as the procurement rules do: at equal prices the earlier application wins;
    across lots the supplier who won most lots, then the one who applied first.
    """
    con.sql("""CREATE TABLE award AS
        WITH lw AS (SELECT buy_id, lot_id, sup, dt FROM app WHERE p > 0
                    QUALIFY row_number() OVER (PARTITION BY buy_id, lot_id ORDER BY p, dt, sup) = 1),
        bw AS (SELECT buy_id AS id, sup FROM (SELECT buy_id, sup, count(*) AS n, min(dt) AS first_dt FROM lw GROUP BY 1, 2)
               QUALIFY row_number() OVER (PARTITION BY buy_id ORDER BY n DESC, first_dt, sup) = 1),
        cu AS (SELECT trdBuyId AS id, arg_min(customerBin, id) AS cust FROM lots GROUP BY 1)
        SELECT b.id, b.pdt AS t, b.pdt - INTERVAL 30 DAY AS t30, coalesce(b.announced, bw.sup) AS sup, cu.cust
        FROM buy b LEFT JOIN bw USING (id) LEFT JOIN cu USING (id)""")


def _relationship(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """REL: cumulative customer / supplier histories looked up "as of" publication with ASOF joins."""
    # one event per awarded procurement and supplier (a multi-lot procurement is one win, not several)
    con.sql("""CREATE TABLE ev AS
        SELECT e.*, row_number() OVER (PARTITION BY cust, sup ORDER BY sdt, buy_id) AS pair_rn,
               (EXISTS (SELECT 1 FROM app a WHERE a.buy_id = e.buy_id AND a.sup = e.sup))::INT AS competitive
        FROM (SELECT buy_id, cust, sup, min(sdt) AS sdt FROM ctr
              WHERE sdt IS NOT NULL AND cust IS NOT NULL AND sup IS NOT NULL GROUP BY 1, 2, 3) e""")
    con.sql("""CREATE TABLE ev_cust AS   -- running count and running sum of squared supplier counts (for HHI)
        SELECT cust, sdt, count(*) OVER w AS n, sum(2 * pair_rn - 1) OVER w AS sumsq
        FROM ev WINDOW w AS (PARTITION BY cust ORDER BY sdt)""")
    con.sql("CREATE TABLE ev_pair AS SELECT cust, sup, sdt, count(*) OVER (PARTITION BY cust, sup ORDER BY sdt) AS n FROM ev")
    con.sql("""CREATE TABLE ev_sup AS
        SELECT sup, sdt, count(*) OVER w AS n, max(pair_rn) OVER w AS top_cust_n, sum(competitive) OVER w AS comp_wins
        FROM ev WINDOW w AS (PARTITION BY sup ORDER BY sdt)""")
    con.sql("""CREATE TABLE ev_part AS
        SELECT sup, dt, count(*) OVER (PARTITION BY sup ORDER BY dt) AS n
        FROM (SELECT sup, buy_id, min(dt) AS dt FROM app GROUP BY 1, 2)""")

    def asof(tbl, keys, tcol, val, tq="t"):
        on = " AND ".join(f"q.{k} = e.{k}" for k in keys)
        return con.sql(f"SELECT q.id, e.{val} AS v FROM award q ASOF LEFT JOIN {tbl} e ON {on} AND q.{tq} > e.{tcol}").df()

    parts = {
        "cust_n": ("ev_cust", ["cust"], "sdt", "n", "t"), "cust_sumsq": ("ev_cust", ["cust"], "sdt", "sumsq", "t"),
        "pair_n": ("ev_pair", ["cust", "sup"], "sdt", "n", "t"),
        "sup_n": ("ev_sup", ["sup"], "sdt", "n", "t"), "sup_n_30": ("ev_sup", ["sup"], "sdt", "n", "t30"),
        "sup_top": ("ev_sup", ["sup"], "sdt", "top_cust_n", "t"), "sup_compwins": ("ev_sup", ["sup"], "sdt", "comp_wins", "t"),
        "sup_part": ("ev_part", ["sup"], "dt", "n", "t"),
    }
    rel = con.sql("SELECT id FROM award").df()
    for name, args in parts.items():
        rel = rel.merge(asof(*args).rename(columns={"v": name}), on="id", how="left")
    rel = rel.fillna({k: 0 for k in parts})
    enough = lambda n: n >= MIN_HISTORY
    rel["pair_prior_contracts"] = rel.pair_n
    rel["pair_share_of_customer"] = np.where(enough(rel.cust_n), rel.pair_n / rel.cust_n.clip(lower=1), np.nan)
    rel["customer_hhi"] = np.where(enough(rel.cust_n), rel.cust_sumsq / rel.cust_n.clip(lower=1) ** 2, np.nan)
    rel["winner_contracts_30d"] = rel.sup_n - rel.sup_n_30
    rel["winner_top_customer_share"] = np.where(enough(rel.sup_n), rel.sup_top / rel.sup_n.clip(lower=1), np.nan)
    rel["winner_win_rate"] = np.where(enough(rel.sup_part), rel.sup_compwins / rel.sup_part.clip(lower=1), np.nan)
    return rel[["id"] + REL]


def _price(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """PRICE: log ratio of the planned unit price to the Jan–Feb median for the same lot name."""
    return con.sql(f"""
        WITH lp AS (SELECT l.trdBuyId AS id, lower(trim(l.nameRu)) AS nm, l.amount / l.count AS up, b.pdt
                    FROM lots l JOIN buy b ON b.id = l.trdBuyId WHERE l.amount > 0 AND l.count > 0),
        bench AS (SELECT nm, median(up) AS med FROM lp WHERE pdt < TIMESTAMP '{P.VAL_START}'
                  GROUP BY nm HAVING count(*) >= {MIN_BENCHMARK}),
        r AS (SELECT lp.id, ln(lp.up / bench.med) AS lr FROM lp JOIN bench USING (nm) WHERE bench.med > 0),
        nl AS (SELECT trdBuyId AS id, count(*) AS n FROM lots GROUP BY 1)
        SELECT b.id, median(r.lr) AS unit_price_vs_market, max(r.lr) AS unit_price_vs_market_max,
               count(r.lr) / any_value(nl.n) AS share_lots_benchmarked
        FROM buy b LEFT JOIN r USING (id) JOIN nl USING (id) GROUP BY b.id
    """).df()


def winners(raw: Path) -> pd.DataFrame:
    """Leak-free winner and customer of every procurement: columns id, winner, customer."""
    con = _connect(raw)
    _award(con)
    return con.sql("SELECT id, sup AS winner, cust AS customer FROM award").df()


def build_a3_features(raw: Path) -> pd.DataFrame:
    """REL + PRICE for every procurement in trd_buy (one row per id)."""
    con = _connect(raw)
    _award(con)
    return _relationship(con).merge(_price(con), on="id", how="left")


def load_a3_table(root: Path, rebuild: bool = False) -> pd.DataFrame:
    """Model table (F20, target, split, lot names) joined with the A3 features; builds and caches them once."""
    root = Path(root)
    cache = root / "data" / "processed" / CACHE
    if rebuild or not cache.exists():
        raw = root / "data" / "raw"
        for t in RAW_TABLES:
            if not (raw / f"{t}.parquet").exists():
                raw.mkdir(parents=True, exist_ok=True)
                P._download(P.RELEASE_URL + f"{t}.parquet", raw / f"{t}.parquet")
        build_a3_features(raw).to_parquet(cache, index=False)
    mt = P.load_model_table(root)
    return mt.merge(pd.read_parquet(cache), on="id", how="left")
