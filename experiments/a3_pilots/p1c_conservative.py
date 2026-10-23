"""
A3 pilot 1c — leakage-safe version of pilot 1b. Uses only bid-time information (prices, counts, dates) and the
winner's identity; no application statuses at all, so nothing can be updated after the award.
"""
import json, warnings
from pathlib import Path
import sys

import duckdb, lightgbm as lgb, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import prepare as P
warnings.filterwarnings("ignore"); SEED = 42; out = {}
con = duckdb.connect()
con.sql(f"CREATE VIEW mt AS SELECT * FROM read_parquet('{(ROOT / 'data/processed/model_table.parquet').as_posix()}')")
con.sql(f"""CREATE TABLE bid AS SELECT a.buyId AS id, a.supplierBinIin AS sup, CAST(a.dateApply AS TIMESTAMP) AS dt,
              j.lotId AS lot_id, j.price
            FROM read_parquet('{(ROOT / 'data/raw/trd_app.parquet').as_posix()}') a,
                 unnest(from_json(a.AppLots, '[{{"lotId":"BIGINT","price":"DOUBLE"}}]')) t(j)""")
con.sql(f"""CREATE TABLE win AS SELECT trdBuyId AS id, min(supplierBiin) AS winner
            FROM read_parquet('{(ROOT / 'data/raw/contracts.parquet').as_posix()}') GROUP BY 1""")
feat = con.sql("""
WITH lot AS (   -- price competition per lot from prices only: gap between the two cheapest offers
  SELECT id, lot_id, count(*) AS k,
         list_sort(list(price))[2] / NULLIF(list_sort(list(price))[1], 0) - 1 AS gap12
  FROM bid WHERE price > 0 GROUP BY 1, 2),
lotagg AS (SELECT id, avg(gap12) AS gap12_mean, min(gap12) AS gap12_min FROM lot WHERE k >= 2 GROUP BY 1),
late AS (   -- how late the applications came relative to the deadline-free window: last-day share
  SELECT b.id, avg(CASE WHEN date_diff('hour', b.dt, m.publish_dt + INTERVAL 1 DAY * m.bid_window_days + INTERVAL 1 DAY * m.publish_to_start_days) <= 24 THEN 1 ELSE 0 END) AS share_last_day
  FROM bid b JOIN mt m USING (id) GROUP BY 1),
w AS (SELECT m.id, w.winner, CASE WHEN substr(w.winner, 5, 1) IN ('0','1','2','3') THEN 1 ELSE 0 END AS winner_individual
      FROM mt m JOIN win w USING (id)),
hist AS (   -- winner's activity in the 30 days before publication (fixed window -> no drift with calendar time)
  SELECT w.id, count(b.sup) AS winner_bids_30d, count(DISTINCT b.id) AS winner_procs_30d
  FROM w JOIN mt m USING (id) LEFT JOIN bid b ON b.sup = w.winner AND b.dt < m.publish_dt AND b.dt >= m.publish_dt - INTERVAL 30 DAY
  GROUP BY 1)
SELECT m.id, lotagg.gap12_mean, lotagg.gap12_min, late.share_last_day, w.winner_individual, hist.winner_bids_30d, hist.winner_procs_30d
FROM mt m LEFT JOIN lotagg USING (id) LEFT JOIN late USING (id) LEFT JOIN w USING (id) LEFT JOIN hist USING (id)
""").df()
df = P.load_model_table(ROOT).merge(feat, on="id", how="left")
for c in P.CAT_FEATURES: df[c] = df[c].astype("category")
y = P.TARGET; tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
def run(cols, name):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=SEED, verbose=-1)
    m.fit(tr[cols], tr[y], eval_set=[(va[cols], va[y])], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
    imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    out[name] = {"val_pr": average_precision_score(va[y], pv), "test_pr": average_precision_score(te[y], pt), "test_roc": roc_auc_score(te[y], pt),
                 "top_gain": {k: round(v / imp.sum(), 3) for k, v in imp.head(6).items()}}
    return pt
price_cols = ["gap12_mean", "gap12_min", "share_last_day"]; win_cols = ["winner_individual", "winner_bids_30d", "winner_procs_30d"]
p0 = run(P.FEATURES, "lgbm_base"); run(P.FEATURES + price_cols, "lgbm_plus_price_competition")
run(P.FEATURES + win_cols, "lgbm_plus_winner_30d"); p3 = run(P.FEATURES + price_cols + win_cols, "lgbm_plus_both_safe")
rng = np.random.default_rng(SEED); yt = te[y].to_numpy(); n = len(yt); d = []
for _ in range(1000):
    i = rng.integers(0, n, n); d.append(average_precision_score(yt[i], p3[i]) - average_precision_score(yt[i], p0[i]))
out["bootstrap_diff_safe_minus_base"] = {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
Path(__file__).with_name("p1c_results.json").write_text(json.dumps(out, indent=2, default=float))
for k, v in out.items(): print(k, json.dumps(v, default=lambda x: round(float(x), 4), ensure_ascii=False))
