"""
A3 pilot 4 — do the two new information sources add up, and does the January history gap matter?
  * W: winner's activity in the 30 days before publication + individual entrepreneur flag (safe set from pilot 1c)
  * T: 32 PCA components of frozen rubert-tiny2 embeddings of lot names (from p2a_embed.py)
LightGBM on F20, F20+W, F20+T, F20+W+T with a paired bootstrap against F20 on test.
History check: the bid data starts on 1 Jan 2024, so a January procurement sees less than 30 days of winner history.
If training on February only (complete history) beats Jan-Feb training for F20+W, the gap hurts and the
Nov-Dec 2023 applications are worth downloading.
"""
import json, warnings
from pathlib import Path
import sys

import duckdb, lightgbm as lgb, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import prepare as P
warnings.filterwarnings("ignore"); SEED = 42; out = {}
con = duckdb.connect()
con.sql(f"CREATE VIEW mt AS SELECT * FROM read_parquet('{(ROOT / 'data/processed/model_table.parquet').as_posix()}')")
con.sql(f"""CREATE TABLE bid AS SELECT buyId AS id, supplierBinIin AS sup, CAST(dateApply AS TIMESTAMP) AS dt
            FROM read_parquet('{(ROOT / 'data/raw/trd_app.parquet').as_posix()}')""")
con.sql(f"""CREATE TABLE win AS SELECT trdBuyId AS id, min(supplierBiin) AS winner
            FROM read_parquet('{(ROOT / 'data/raw/contracts.parquet').as_posix()}') GROUP BY 1""")
feat = con.sql("""
WITH w AS (SELECT m.id, w.winner, CASE WHEN substr(w.winner, 5, 1) IN ('0','1','2','3') THEN 1 ELSE 0 END AS winner_individual
           FROM mt m JOIN win w USING (id)),
hist AS (
  SELECT w.id, count(b.sup) AS winner_bids_30d, count(DISTINCT b.id) AS winner_procs_30d
  FROM w JOIN mt m USING (id) LEFT JOIN bid b ON b.sup = w.winner AND b.dt < m.publish_dt AND b.dt >= m.publish_dt - INTERVAL 30 DAY
  GROUP BY 1)
SELECT m.id, w.winner_individual, hist.winner_bids_30d, hist.winner_procs_30d
FROM mt m LEFT JOIN w USING (id) LEFT JOIN hist USING (id)
""").df()
df = P.load_model_table(ROOT).merge(feat, on="id", how="left")
for c in P.CAT_FEATURES: df[c] = df[c].astype("category")

E = np.load(ROOT / "data/processed/rubert_tiny2_cls.npy")
uniq = pd.read_parquet(ROOT / "data/processed/rubert_tiny2_texts.parquet").text
pos = pd.Series(np.arange(len(uniq)), index=uniq.values)
Z = E[pos[df.lot_names.fillna("").str.lower().str.strip()].to_numpy()]
pca = PCA(32, random_state=SEED).fit(Z[(df.split == "train").to_numpy()])
T = [f"emb{k}" for k in range(32)]
df[T] = pca.transform(Z)

y = P.TARGET; W = ["winner_individual", "winner_bids_30d", "winner_procs_30d"]
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
yt = te[y].to_numpy()

# history gap: share of procurements whose winner has no bids in the previous 30 days, by publication week
wk = df.assign(week=df.publish_dt.dt.isocalendar().week.astype(int), none=(df.winner_bids_30d.fillna(0) == 0))
out["share_no_winner_history_by_week"] = {int(k): round(float(v), 3) for k, v in wk.groupby("week").none.mean().items()}

def at_k(p, k):  # precision in the top k% and recall in the top k%
    n = int(len(p) * k); idx = np.argsort(-p)[:n]
    return float(yt[idx].mean()), float(yt[idx].sum() / yt.sum())

def run(cols, name, train=tr):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=SEED, verbose=-1)
    m.fit(train[cols], train[y], eval_set=[(va[cols], va[y])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
    out[name] = {"n_features": len(cols), "train_rows": len(train), "best_iter": m.best_iteration_,
                 "val_pr": average_precision_score(va[y], pv), "test_pr": average_precision_score(yt, pt),
                 "test_roc": roc_auc_score(yt, pt), "test_p_at_1": at_k(pt, 0.01)[0], "test_r_at_5": at_k(pt, 0.05)[1]}
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)
    return pt

preds = {"F20": run(P.FEATURES, "F20"), "F20+W": run(P.FEATURES + W, "F20+W"),
         "F20+T": run(P.FEATURES + T, "F20+T"), "F20+W+T": run(P.FEATURES + W + T, "F20+W+T")}
feb = tr[tr.publish_dt >= "2024-02-01"]
run(P.FEATURES + W, "F20+W_train_feb_only", feb); run(P.FEATURES, "F20_train_feb_only", feb)

rng = np.random.default_rng(SEED); idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
base = np.array([average_precision_score(yt[i], preds["F20"][i]) for i in idx])
out["bootstrap_vs_F20"] = {}
for k in ("F20+W", "F20+T", "F20+W+T"):
    d = np.array([average_precision_score(yt[i], preds[k][i]) for i in idx]) - base
    out["bootstrap_vs_F20"][k] = {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
Path(__file__).with_name("p4_results.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out["bootstrap_vs_F20"], indent=1), flush=True)
