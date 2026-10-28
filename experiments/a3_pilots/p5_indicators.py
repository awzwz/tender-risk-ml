"""
A3 pilot 5 — which of Tender Radar's rule indicators carry signal for contract failure, once they are
re-computed honestly (only information available at publication / award time, continuous values instead
of thresholded flags)? Also audits the current F20 features for time drift.

Feature groups (all as of the procurement's publish_dt; history comes from other procurements only):
  BID   bid structure:   relative gap between the two lowest prices (TINY_WIN_MARGIN), identical prices
                          from different suppliers (IDENTICAL_BID_PRICES), timing of applications relative
                          to the deadline (LATE_BID_SUBMISSION)
  PROC  procedure:       repeat procurement (REPEAT_TENDER), publication hour / weekend / night
                          (NIGHT_OR_WEEKEND_PUBLISH)
  REL   relationships:   customer–winner history from contracts signed before publication
                          (RECURRING_WINNER, CUSTOMER_WINNER_CONCENTRATION / CAROUSEL as HHI,
                          SUPPLIER_CONCENTRATION, HIGH_WIN_RATE_FEW_BIDS)
  PRICE planned price:   planned unit price vs the Jan–Feb median for the same lot name (OVERPRICED_UNIT;
                          ENSTRU codes are empty in OWS lots, so the lot name stands in for the product code)
  ORG30 drift fix:       organiser's procurements in the previous 30 days instead of the count since 1 Jan

Writes data/processed/p5_features.parquet (reused by p6) and p5_results.json.
"""
import json, sys, warnings
from pathlib import Path

import duckdb, lightgbm as lgb, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import prepare as P
warnings.filterwarnings("ignore"); SEED = 42; out = {}
RAW = (ROOT / "data/raw").as_posix()

con = duckdb.connect()
con.sql(f"CREATE VIEW mt AS SELECT * FROM read_parquet('{(ROOT / 'data/processed/model_table.parquet').as_posix()}')")
con.sql(f"""CREATE TABLE buy AS
    SELECT id, orgBin AS org, CAST(publishDate AS TIMESTAMP) AS pdt, CAST(endDate AS TIMESTAMP) AS end_dt,
           (parentId > 0)::INT AS has_parent
    FROM read_parquet('{RAW}/trd_buy.parquet')
    QUALIFY row_number() OVER (PARTITION BY id ORDER BY lastUpdateDate DESC) = 1""")
con.sql(f"""CREATE TABLE app AS
    SELECT id AS app_id, buyId AS buy_id, supplierBinIin AS sup, CAST(dateApply AS TIMESTAMP) AS dt, AppLots
    FROM read_parquet('{RAW}/trd_app.parquet')""")
con.sql(f"""CREATE TABLE bidlot AS   -- lowest price per (lot, supplier); at_cap = bid equals the planned unit price
    SELECT a.buy_id, j.lotId AS lot_id, a.sup, min(j.price) AS price,
           bool_or(abs(j.price - l.amount / l.count) < 0.005 * l.amount / l.count) AS at_cap
    FROM app a, unnest(from_json(a.AppLots, '{P.APP_LOTS_SCHEMA}')) t(j)
    JOIN read_parquet('{RAW}/lots.parquet') l ON l.id = j.lotId
    WHERE j.price > 0 AND l.amount > 0 AND l.count > 0 GROUP BY 1, 2, 3""")
# one row per contract chain (original + amendments share rootId)
con.sql(f"""CREATE TABLE ctr AS
    SELECT rootId, min(trdBuyId) AS buy_id, min(customerBin) AS cust, min(supplierBiin) AS sup,
           min(CAST(signDate AS TIMESTAMP)) AS sdt
    FROM read_parquet('{RAW}/contracts.parquet') GROUP BY rootId""")
# Winner and customer come from the announcement and the bids, never from contracts (see winner.py).
# The two contract-based definitions are kept only to measure how much they leaked:
#   first_sup — supplier of the first contract;  min_sup — min(supplierBiin), the pilot 4 definition.
sys.path.insert(0, str(Path(__file__).parent))
from winner import create_winner_table
create_winner_table(con, RAW)
con.sql("""CREATE TABLE cur AS   -- the procurement being scored: its customer and winner
    SELECT m.id, m.publish_dt AS t, a.cust, a.winner, c.first_sup, c.min_sup
    FROM mt m LEFT JOIN award a USING (id)
    LEFT JOIN (SELECT buy_id, arg_min(sup, rootId) AS first_sup, min(sup) AS min_sup FROM ctr GROUP BY 1) c
           ON c.buy_id = m.id""")

# ── BID ──────────────────────────────────────────────────────────────────────
bid = con.sql("""
WITH r AS (SELECT buy_id, lot_id, price, row_number() OVER (PARTITION BY lot_id ORDER BY price) AS rn FROM bidlot),
lot_gap AS (SELECT buy_id, lot_id,
                   (max(price) FILTER (WHERE rn = 2) - max(price) FILTER (WHERE rn = 1)) / max(price) FILTER (WHERE rn = 1) AS gap
            FROM r GROUP BY 1, 2 HAVING count(*) >= 2),
gap AS (SELECT buy_id AS id, min(gap) AS win_gap_min, avg(gap) AS win_gap_mean FROM lot_gap GROUP BY 1),
ident AS (SELECT buy_id AS id, (NOT bool_or(at_cap))::INT AS identical_below_cap, bool_or(at_cap)::INT AS identical_at_cap
          FROM bidlot GROUP BY buy_id, lot_id, price HAVING count(DISTINCT sup) > 1),
timing AS (SELECT a.buy_id AS id,
                  avg((a.dt >= b.end_dt - INTERVAL 1 HOUR)::INT) AS late_share_1h,
                  median(date_diff('minute', a.dt, b.end_dt) / 60.0) AS apps_hours_before_end
           FROM app a JOIN buy b ON b.id = a.buy_id GROUP BY 1),
wt AS (SELECT c.id, min(date_diff('minute', a.dt, b.end_dt) / 60.0) AS winner_hours_before_end
       FROM cur c JOIN app a ON a.buy_id = c.id AND a.sup = c.winner JOIN buy b ON b.id = c.id GROUP BY 1)
SELECT m.id, least(g.win_gap_min, 10) AS win_gap_min, least(g.win_gap_mean, 10) AS win_gap_mean,
       coalesce(i.identical_below_cap, 0) AS identical_below_cap, coalesce(i.identical_at_cap, 0) AS identical_at_cap,
       tm.late_share_1h, tm.apps_hours_before_end, wt.winner_hours_before_end
FROM mt m LEFT JOIN gap g USING (id)
LEFT JOIN (SELECT id, max(identical_below_cap) AS identical_below_cap, max(identical_at_cap) AS identical_at_cap
           FROM ident GROUP BY id) i USING (id)
LEFT JOIN timing tm USING (id) LEFT JOIN wt USING (id)
""").df()

# ── PROC ─────────────────────────────────────────────────────────────────────
proc = con.sql("""
SELECT m.id, b.has_parent, hour(m.publish_dt) AS publish_hour, (dayofweek(m.publish_dt) IN (0, 6))::INT AS publish_weekend,
       (hour(m.publish_dt) < 7 OR hour(m.publish_dt) >= 22)::INT AS publish_night
FROM mt m JOIN buy b USING (id)
""").df()

# ── REL: cumulative histories joined "as of" publication with ASOF joins ─────
# one event per awarded procurement and supplier (a multi-lot procurement gives several contracts = one win)
con.sql("""CREATE TABLE ev AS
    SELECT e.*, row_number() OVER (PARTITION BY cust, sup ORDER BY sdt, buy_id) AS pair_rn,
           (EXISTS (SELECT 1 FROM app a WHERE a.buy_id = e.buy_id AND a.sup = e.sup))::INT AS competitive
    FROM (SELECT buy_id, cust, sup, min(sdt) AS sdt FROM ctr
          WHERE sdt IS NOT NULL AND cust IS NOT NULL AND sup IS NOT NULL GROUP BY 1, 2, 3) e""")
con.sql("""CREATE TABLE ev_cust AS   -- customer's contracts so far and sum of squared supplier counts (for HHI)
    SELECT cust, sdt, count(*) OVER w AS n, sum(2 * pair_rn - 1) OVER w AS sumsq
    FROM ev WINDOW w AS (PARTITION BY cust ORDER BY sdt)""")
con.sql("""CREATE TABLE ev_pair AS
    SELECT cust, sup, sdt, count(*) OVER (PARTITION BY cust, sup ORDER BY sdt) AS n FROM ev""")
con.sql("""CREATE TABLE ev_sup AS     -- winner's contracts so far, its top customer, its competitive wins
    SELECT sup, sdt, count(*) OVER w AS n, max(pair_rn) OVER w AS top_cust_n, sum(competitive) OVER w AS comp_wins
    FROM ev WINDOW w AS (PARTITION BY sup ORDER BY sdt)""")
con.sql("""CREATE TABLE ev_part AS    -- distinct procurements a supplier has bid on so far
    SELECT sup, dt, count(*) OVER (PARTITION BY sup ORDER BY dt) AS n
    FROM (SELECT sup, buy_id, min(dt) AS dt FROM app GROUP BY 1, 2)""")
con.sql("""CREATE TABLE ev_org AS
    SELECT org, pdt, count(*) OVER (PARTITION BY org ORDER BY pdt) AS n FROM buy WHERE org IS NOT NULL""")

def asof(tbl, keys, tcol, val, tq="t"):
    """Value of the last history row strictly before the procurement's time (tq = t, or t30 = t - 30 days)."""
    on = " AND ".join(f"q.{k} = e.{k}" for k in keys)
    return f"SELECT q.id, e.{val} AS v FROM q ASOF LEFT JOIN {tbl} e ON {on} AND q.{tq} > e.{tcol}"

con.sql("""CREATE TABLE q AS SELECT c.id, c.t, c.t - INTERVAL 30 DAY AS t30, c.cust, c.winner AS sup, b.org
           FROM cur c JOIN buy b USING (id)""")
parts = {
    "cust_n": asof("ev_cust", ["cust"], "sdt", "n"), "cust_sumsq": asof("ev_cust", ["cust"], "sdt", "sumsq"),
    "pair_n": asof("ev_pair", ["cust", "sup"], "sdt", "n"),
    "sup_n": asof("ev_sup", ["sup"], "sdt", "n"), "sup_top": asof("ev_sup", ["sup"], "sdt", "top_cust_n"),
    "sup_compwins": asof("ev_sup", ["sup"], "sdt", "comp_wins"), "sup_part": asof("ev_part", ["sup"], "dt", "n"),
    "sup_n_30": asof("ev_sup", ["sup"], "sdt", "n", "t30"),
    "org_n": asof("ev_org", ["org"], "pdt", "n"), "org_n_30": asof("ev_org", ["org"], "pdt", "n", "t30"),
}
rel = con.sql("SELECT id FROM q").df()
for name, sql in parts.items():
    rel = rel.merge(con.sql(sql).df().rename(columns={"v": name}), on="id", how="left")
rel = rel.fillna({k: 0 for k in parts})
rel["pair_prior_contracts"] = rel.pair_n
rel["pair_share_of_customer"] = np.where(rel.cust_n >= 5, rel.pair_n / rel.cust_n.clip(lower=1), np.nan)
rel["customer_hhi"] = np.where(rel.cust_n >= 5, rel.cust_sumsq / rel.cust_n.clip(lower=1) ** 2, np.nan)
rel["winner_contracts_30d"] = rel.sup_n - rel.sup_n_30
rel["winner_top_customer_share"] = np.where(rel.sup_n >= 5, rel.sup_top / rel.sup_n.clip(lower=1), np.nan)
rel["winner_win_rate"] = np.where(rel.sup_part >= 5, rel.sup_compwins / rel.sup_part.clip(lower=1), np.nan)
rel["org_buys_30d"] = rel.org_n - rel.org_n_30
REL = ["pair_prior_contracts", "pair_share_of_customer", "customer_hhi", "winner_contracts_30d",
       "winner_top_customer_share", "winner_win_rate"]

# ── PRICE ────────────────────────────────────────────────────────────────────
price = con.sql(f"""
WITH lp AS (SELECT l.trdBuyId AS buy_id, lower(trim(l.nameRu)) AS nm, l.amount / l.count AS up, b.pdt
            FROM read_parquet('{RAW}/lots.parquet') l JOIN buy b ON b.id = l.trdBuyId
            WHERE l.amount > 0 AND l.count > 0),
bench AS (SELECT nm, median(up) AS med FROM lp WHERE pdt < TIMESTAMP '{P.VAL_START}' GROUP BY nm HAVING count(*) >= 5),
r AS (SELECT lp.buy_id AS id, ln(lp.up / bench.med) AS lr FROM lp JOIN bench USING (nm) WHERE bench.med > 0)
SELECT m.id, median(r.lr) AS unit_price_vs_market, max(r.lr) AS unit_price_vs_market_max,
       count(r.lr) / NULLIF(any_value(m.n_lots), 0) AS share_lots_benchmarked
FROM mt m LEFT JOIN r USING (id) GROUP BY m.id
""").df()

# ── W (pilot 4) ──────────────────────────────────────────────────────────────
w = con.sql("""
WITH h AS (SELECT c.id, count(a.sup) AS winner_bids_30d, count(DISTINCT a.buy_id) AS winner_procs_30d
           FROM cur c LEFT JOIN app a ON a.sup = c.winner AND a.dt < c.t AND a.dt >= c.t - INTERVAL 30 DAY GROUP BY 1)
SELECT c.id, CASE WHEN substr(c.winner, 5, 1) IN ('0','1','2','3') THEN 1 ELSE 0 END AS winner_individual,
       h.winner_bids_30d, h.winner_procs_30d
FROM cur c JOIN h USING (id)
""").df()
for col, pre in [("min_sup", "old_"), ("first_sup", "first_")]:   # contract-based winners, leak measurement only
    w = w.merge(con.sql(f"""
    WITH h AS (SELECT c.id, count(a.sup) AS b, count(DISTINCT a.buy_id) AS p
               FROM cur c LEFT JOIN app a ON a.sup = c.{col} AND a.dt < c.t AND a.dt >= c.t - INTERVAL 30 DAY GROUP BY 1)
    SELECT c.id, CASE WHEN substr(c.{col}, 5, 1) IN ('0','1','2','3') THEN 1 ELSE 0 END AS {pre}winner_individual,
           h.b AS {pre}winner_bids_30d, h.p AS {pre}winner_procs_30d
    FROM cur c JOIN h USING (id)""").df(), on="id")

feats = (bid.merge(proc, on="id").merge(rel[["id", "org_buys_30d"] + REL], on="id")
         .merge(price, on="id").merge(w, on="id"))
feats.to_parquet(ROOT / "data/processed/p5_features.parquet", index=False)

BID = ["win_gap_min", "win_gap_mean", "identical_below_cap", "identical_at_cap", "late_share_1h",
       "apps_hours_before_end", "winner_hours_before_end"]
PROC = ["has_parent", "publish_hour", "publish_weekend", "publish_night"]
PRICE = ["unit_price_vs_market", "unit_price_vs_market_max", "share_lots_benchmarked"]
W = ["winner_individual", "winner_bids_30d", "winner_procs_30d"]
W_OLD = ["old_winner_individual", "old_winner_bids_30d", "old_winner_procs_30d"]
W_FIRST = ["first_winner_individual", "first_winner_bids_30d", "first_winner_procs_30d"]

df = P.load_model_table(ROOT).merge(feats, on="id", how="left")
for c in P.CAT_FEATURES: df[c] = df[c].astype("category")
E = np.load(ROOT / "data/processed/rubert_tiny2_cls.npy")
uniq = pd.read_parquet(ROOT / "data/processed/rubert_tiny2_texts.parquet").text
pos = pd.Series(np.arange(len(uniq)), index=uniq.values)
Z = E[pos[df.lot_names.fillna("").str.lower().str.strip()].to_numpy()]
T = [f"emb{k}" for k in range(32)]
df[T] = PCA(32, random_state=SEED).fit(Z[(df.split == "train").to_numpy()]).transform(Z)

y = P.TARGET
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
yt = te[y].to_numpy()

# ── audit of the current features ────────────────────────────────────────────
def psi(a, b, bins=10):
    a, b = a.dropna(), b.dropna()
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a) + 1e-6
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))

num = P.NUM_FEATURES + ["org_buys_30d"] + BID + REL + PRICE + W
out["drift_psi_train_vs_test"] = {c: round(psi(tr[c], te[c]), 3) for c in num}
out["coverage_non_null"] = {c: round(float(df[c].notna().mean()), 3) for c in BID + REL + PRICE}
out["org_prior_buys_median_by_split"] = df.groupby("split").org_prior_buys.median().to_dict()
out["org_buys_30d_median_by_split"] = df.groupby("split").org_buys_30d.median().to_dict()

def at_k(p, k):
    n = int(len(p) * k); idx = np.argsort(-p)[:n]
    return float(yt[idx].mean()), float(yt[idx].sum() / yt.sum())

models = {}
def run(cols, name):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=SEED, verbose=-1)
    m.fit(tr[cols], tr[y], eval_set=[(va[cols], va[y])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
    out[name] = {"n_features": len(cols), "best_iter": m.best_iteration_,
                 "val_pr": average_precision_score(va[y], pv), "test_pr": average_precision_score(yt, pt),
                 "test_roc": roc_auc_score(yt, pt), "test_p_at_1": at_k(pt, 0.01)[0], "test_r_at_5": at_k(pt, 0.05)[1]}
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)
    models[name] = m
    return pt

F20 = P.FEATURES
F20o = [c for c in F20 if c != "org_prior_buys"] + ["org_buys_30d"]
preds = {
    "F20": run(F20, "F20"),
    "F20_org30": run(F20o, "F20_org30"),
    "F20+BID": run(F20 + BID, "F20+BID"),
    "F20+PROC": run(F20 + PROC, "F20+PROC"),
    "F20+REL": run(F20 + REL, "F20+REL"),
    "F20+PRICE": run(F20 + PRICE, "F20+PRICE"),
    "F20+W_old": run(F20 + W_OLD, "F20+W_old"),
    "F20+W_first": run(F20 + W_FIRST, "F20+W_first"),
    "F20+W": run(F20 + W, "F20+W"),
    "F20+W+T": run(F20 + W + T, "F20+W+T"),
    "F20+W+T+ALL": run(F20 + W + T + BID + PROC + REL + PRICE, "F20+W+T+ALL"),
    "F20+T": run(F20 + T, "F20+T"),
    "F20+T+REL+PRICE": run(F20 + T + REL + PRICE, "F20+T+REL+PRICE"),
    "F20+T+REL+PRICE+BID": run(F20 + T + REL + PRICE + BID, "F20+T+REL+PRICE+BID"),
}

# permutation importance of the F20 model on validation (PR-AUC drop, 3 shuffles)
rng = np.random.default_rng(SEED); m = models["F20"]; base_v = out["F20"]["val_pr"]; imp = {}
for c in F20:
    drops = []
    for _ in range(3):
        X = va[F20].copy(); X[c] = X[c].sample(frac=1, random_state=int(rng.integers(1e9))).values
        drops.append(base_v - average_precision_score(va[y], m.predict_proba(X)[:, 1]))
    imp[c] = round(float(np.mean(drops)), 4)
out["perm_importance_F20_val"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

# paired bootstrap on test
idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
def boot(a, b):
    d = np.array([average_precision_score(yt[i], preds[a][i]) - average_precision_score(yt[i], preds[b][i]) for i in idx])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
out["bootstrap"] = {f"{a} vs {b}": boot(a, b) for a, b in [
    ("F20_org30", "F20"), ("F20+BID", "F20"), ("F20+PROC", "F20"), ("F20+REL", "F20"), ("F20+PRICE", "F20"),
    ("F20+W_old", "F20"), ("F20+W_first", "F20"), ("F20+W", "F20"),
    ("F20+W+T", "F20"), ("F20+W+T+ALL", "F20+W+T"),
    ("F20+T", "F20"), ("F20+T+REL+PRICE", "F20+T"), ("F20+T+REL+PRICE+BID", "F20+T+REL+PRICE"),
    ("F20+T+REL+PRICE+BID", "F20")]}
Path(__file__).with_name("p5_results.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out["bootstrap"], indent=1), flush=True)
