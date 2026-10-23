"""
A3 pilot 1b — does NEW information beat tuning? Builds candidate features from data we have not used yet and
measures the gain of LightGBM on validation/test with a paired bootstrap.
  * bid structure per procurement: counts of each application status, price gap winner vs runner-up
  * winner: individual entrepreneur (IIN) vs legal entity (BIN), winner's activity before this procurement
Rare statuses (300/450/480) are checked separately as possible post-award leakage.
"""
import json, warnings
from pathlib import Path
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import prepare as P
warnings.filterwarnings("ignore"); SEED = 42; out = {}

con = duckdb.connect()
con.sql(f"CREATE VIEW mt AS SELECT * FROM read_parquet('{(ROOT / 'data/processed/model_table.parquet').as_posix()}')")
con.sql(f"""CREATE TABLE bid AS SELECT a.buyId AS id, a.supplierBinIin AS sup, CAST(a.dateApply AS TIMESTAMP) AS dt,
              j.lotId AS lot_id, j.statusId AS st, j.price
            FROM read_parquet('{(ROOT / 'data/raw/trd_app.parquet').as_posix()}') a,
                 unnest(from_json(a.AppLots, '[{{"lotId":"BIGINT","statusId":"BIGINT","price":"DOUBLE"}}]')) t(j)""")
con.sql(f"""CREATE TABLE win AS SELECT trdBuyId AS id, min(supplierBiin) AS winner
            FROM read_parquet('{(ROOT / 'data/raw/contracts.parquet').as_posix()}') GROUP BY 1""")
feat = con.sql("""
WITH st AS (
  SELECT id,
    count(*) FILTER (WHERE st = 120) AS n_st120, count(*) FILTER (WHERE st = 310) AS n_st310,
    count(*) FILTER (WHERE st = 330) AS n_st330, count(*) FILTER (WHERE st = 360) AS n_st360,
    count(*) FILTER (WHERE st = 370) AS n_st370,
    count(*) FILTER (WHERE st IN (300, 450, 480)) AS n_st_rare,
    count(*) FILTER (WHERE st = 310) * 1.0 / count(*) AS share_st310
  FROM bid GROUP BY 1),
gap AS (   -- per lot: runner-up price / winner price - 1, averaged over the procurement's lots
  SELECT id, avg(g) AS win_second_gap FROM (
    SELECT id, lot_id, max(price) FILTER (WHERE st = 370) / NULLIF(max(price) FILTER (WHERE st = 360), 0) - 1 AS g
    FROM bid WHERE price > 0 GROUP BY 1, 2) WHERE g IS NOT NULL GROUP BY 1),
w AS (
  SELECT m.id, w.winner,
    CASE WHEN substr(w.winner, 5, 1) IN ('0','1','2','3') THEN 1 ELSE 0 END AS winner_individual
  FROM mt m JOIN win w USING (id)),
hist AS (   -- winner's bids and wins strictly before this procurement was published
  SELECT w.id, count(b.sup) AS winner_prior_bids, count(b.sup) FILTER (WHERE b.st = 360) AS winner_prior_wins
  FROM w JOIN mt m USING (id) LEFT JOIN bid b ON b.sup = w.winner AND b.dt < m.publish_dt
  GROUP BY 1)
SELECT m.id, st.* EXCLUDE (id), gap.win_second_gap, w.winner_individual, hist.winner_prior_bids, hist.winner_prior_wins
FROM mt m LEFT JOIN st USING (id) LEFT JOIN gap USING (id) LEFT JOIN w USING (id) LEFT JOIN hist USING (id)
""").df()

df = P.load_model_table(ROOT).merge(feat, on="id", how="left")
for c in P.CAT_FEATURES: df[c] = df[c].astype("category")
y = P.TARGET

# leakage screen: failure rate by presence of each status family
screen = {}
for c in ["n_st120", "n_st310", "n_st330", "n_st360", "n_st370", "n_st_rare"]:
    has = df[c].fillna(0) > 0
    screen[c] = {"share_with": float(has.mean()), "fail_with": float(df.loc[has, y].mean()), "fail_without": float(df.loc[~has, y].mean())}
out["status_screen"] = screen

tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
def run(cols, name):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=SEED, verbose=-1)
    m.fit(tr[cols], tr[y], eval_set=[(va[cols], va[y])], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
    out[name] = {"val_pr": average_precision_score(va[y], pv), "test_pr": average_precision_score(te[y], pt),
                 "test_roc": roc_auc_score(te[y], pt), "n_features": len(cols)}
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    out[name]["top_gain"] = {k: round(v / imp.sum(), 3) for k, v in imp.head(8).items()}
    return pt

base_cols = P.FEATURES
bid_cols = ["n_st120", "n_st310", "n_st330", "n_st360", "n_st370", "share_st310", "win_second_gap"]
win_cols = ["winner_individual", "winner_prior_bids", "winner_prior_wins"]
p_base = run(base_cols, "lgbm_base")
p_bid = run(base_cols + bid_cols, "lgbm_plus_bid_structure")
p_win = run(base_cols + win_cols, "lgbm_plus_winner")
p_all = run(base_cols + bid_cols + win_cols, "lgbm_plus_both")
run(base_cols + bid_cols + win_cols + ["n_st_rare"], "lgbm_plus_both_plus_rare_status")   # leakage check only

rng = np.random.default_rng(SEED); yt = te[y].to_numpy(); n = len(yt); d = []
for _ in range(1000):
    i = rng.integers(0, n, n); d.append(average_precision_score(yt[i], p_all[i]) - average_precision_score(yt[i], p_base[i]))
out["bootstrap_diff_both_minus_base"] = {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "share_le_0": float(np.mean(np.array(d) <= 0))}
out["winner_individual_rate"] = {"share": float(df.winner_individual.mean()),
                                 "fail_if_individual": float(df.loc[df.winner_individual == 1, y].mean()),
                                 "fail_if_legal_entity": float(df.loc[df.winner_individual == 0, y].mean())}
Path(__file__).with_name("p1b_results.json").write_text(json.dumps(out, indent=2, default=float))
for k, v in out.items(): print(k, json.dumps(v, default=lambda x: round(float(x), 4), ensure_ascii=False)[:600])
