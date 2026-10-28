"""
A3 pilot 7 — do annual procurement plans carry signal for contract failure?

Every lot is linked to one or more plan points; each point has versions (a plan amendment creates a new
version under the same rootrecordId). The Plans register serves only current versions, so the versions a lot
was linked to come from Lots { Plans } (stage lot_plans in src/collect.py). Only versions created no later
than the procurement's publication are used; the latest of them is the plan "in force" at publication.

  PLAN  lot amount vs planned amount; days since the plan version in force was created; planned advance
        payment; publication month vs planned month; planned vs actual procurement method; multi-year
        funding; transferred point; preliminary plan; funding source; share of lots with a plan in force

A lot turned out to be linked to exactly one version per point before publication, so "amendments before
publication" is always 0 and is only reported. Versions created after publication are reported as a leak
demonstration (failure rate when the plan was changed later), never used as features.

Missingness leak (found in the first run): 3.2 % of lots are linked only to versions created after
publication (median 29 days later; such procurements fail ~2.5x more often). A real system would know the
version in force for every lot, so the gap is an artefact of collecting retrospectively. The honest variants
(1) drop plan_coverage and fill gaps with train medians / modes, and (2) compare on fully covered
procurements only. The leaky variant (NaN gaps + coverage) is kept to show the size of the leak.

Needs data/raw/lot_plans.parquet and data/processed/p5_features.parquet.
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
    SELECT id, CAST(publishDate AS TIMESTAMP) AS pdt, refTradeMethodsId AS method
    FROM read_parquet('{RAW}/trd_buy.parquet')
    QUALIFY row_number() OVER (PARTITION BY id ORDER BY lastUpdateDate DESC) = 1""")
con.sql(f"""CREATE TABLE lp_all AS
    SELECT p.lotId AS lot_id, p.trdBuyId AS buy_id, b.pdt, b.method, p.id AS ver, p.rootrecordId AS root,
           CAST(p.dateCreate AS TIMESTAMP) AS vdt, p.amount, p.prepayment, p.refMonthsId AS plan_month,
           p.refTradeMethodsId AS plan_method, p.refFinsourceId AS finsource, p.transferType, p.sum2, p.sum3,
           p.PlanActs_planPreliminary AS prelim
    FROM read_parquet('{RAW}/lot_plans.parquet') p JOIN buy b ON b.id = p.trdBuyId""")
# versions created after publication are amendments made later — the outcome may already be known
con.sql("CREATE TABLE lp AS SELECT * FROM lp_all WHERE ver IS NOT NULL AND vdt <= pdt")
con.sql("""CREATE TABLE v0 AS   -- per lot and plan point: the version in force at publication
    SELECT *, count(*) OVER w AS n_ver_before, min(vdt) OVER w AS first_vdt
    FROM lp WINDOW w AS (PARTITION BY lot_id, root)
    QUALIFY row_number() OVER (PARTITION BY lot_id, root ORDER BY vdt DESC, ver DESC) = 1""")

out["versions_vs_publication"] = con.sql("""
    SELECT count(*) AS rows, count(*) FILTER (WHERE ver IS NULL) AS lots_without_plan,
           count(*) FILTER (WHERE vdt <= pdt) AS created_before, count(*) FILTER (WHERE vdt > pdt) AS created_after,
           count(DISTINCT lot_id) AS lots, count(DISTINCT lot_id) FILTER (WHERE vdt <= pdt) AS lots_with_plan_in_force
    FROM lp_all""").df().iloc[0].to_dict()

plan = con.sql(f"""
WITH lot AS (
    SELECT v.lot_id, v.buy_id,
           ln(any_value(l.amount) / NULLIF(sum(v.amount), 0)) AS lot_vs_plan,
           min(date_diff('hour', v.vdt, v.pdt)) / 24.0 AS days_since_change,
           min(date_diff('hour', v.first_vdt, v.pdt)) / 24.0 AS days_since_planned,
           max(v.n_ver_before - 1) AS amendments,
           max(v.prepayment) AS prepayment,
           max(month(v.pdt) - v.plan_month) AS month_gap,
           max((v.plan_method <> v.method)::INT) AS method_mismatch,
           max((coalesce(v.sum2, 0) > 0 OR coalesce(v.sum3, 0) > 0)::INT) AS multi_year,
           max((coalesce(v.transferType, 0) > 0)::INT) AS transferred,
           max(coalesce(v.prelim, 0)) AS preliminary,
           mode(v.finsource) AS finsource
    FROM v0 v JOIN read_parquet('{RAW}/lots.parquet') l ON l.id = v.lot_id
    GROUP BY 1, 2)
SELECT m.id,
       median(lot_vs_plan) AS plan_lot_vs_plan, max(lot_vs_plan) AS plan_lot_vs_plan_max,
       min(days_since_change) AS plan_days_since_change, min(days_since_planned) AS plan_days_since_planned,
       max(amendments) AS plan_amendments, max(prepayment) AS plan_prepayment, max(month_gap) AS plan_month_gap,
       max(method_mismatch) AS plan_method_mismatch, max(multi_year) AS plan_multi_year,
       max(transferred) AS plan_transferred, max(preliminary) AS plan_preliminary,
       CAST(mode(finsource) AS VARCHAR) AS plan_finsource,
       count(lot.lot_id) / NULLIF(any_value(m.n_lots), 0) AS plan_coverage
FROM mt m LEFT JOIN lot ON lot.buy_id = m.id GROUP BY m.id
""").df()
plan.to_parquet(ROOT / "data/processed/p7_features.parquet", index=False)

later = con.sql("""SELECT buy_id AS id, 1 AS plan_changed_later FROM lp_all WHERE vdt > pdt GROUP BY 1""").df()

PLAN = ["plan_lot_vs_plan", "plan_days_since_change", "plan_prepayment", "plan_month_gap", "plan_method_mismatch",
        "plan_multi_year", "plan_transferred", "plan_preliminary", "plan_finsource"]
df = (P.load_model_table(ROOT).merge(pd.read_parquet(ROOT / "data/processed/p5_features.parquet"), on="id", how="left")
      .merge(plan, on="id", how="left").merge(later, on="id", how="left"))
df["plan_changed_later"] = df.plan_changed_later.fillna(0)
df["full_plan"] = df.plan_coverage.fillna(0) >= 1 - 1e-9
PLAN_LEAKY = [f"{c}_leaky" for c in PLAN] + ["plan_coverage"]
df[PLAN_LEAKY[:-1]] = df[PLAN]
trn = df.split == "train"
for c in PLAN:   # honest variant: gaps get a typical value, so they cannot flag "changed after publication"
    fill = df.loc[trn, c].mode().iloc[0] if c == "plan_finsource" else df.loc[trn, c].median()
    df[c] = df[c].fillna(fill)
for c in P.CAT_FEATURES + ["plan_finsource", "plan_finsource_leaky"]: df[c] = df[c].astype("category")

y = P.TARGET
out["coverage_non_null"] = {c: round(float(df[c].notna().mean()), 3) for c in PLAN_LEAKY}
out["full_plan_rate_by_split"] = {s: {"share": round(float(g.full_plan.mean()), 4),
                                      "fail_full": round(float(g.loc[g.full_plan, y].mean()), 4),
                                      "fail_gap": round(float(g.loc[~g.full_plan, y].mean()), 4)}
                                  for s, g in df.groupby("split")}
out["amendments_before_publication_nonzero_share"] = float((df.plan_amendments > 0).mean())
out["describe"] = df[[c for c in PLAN if c != "plan_finsource"]].describe().round(3).to_dict()

def psi(a, b, bins=10):
    a, b = a.dropna(), b.dropna()
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3: return 0.0
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a) + 1e-6
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b) + 1e-6
    return float(np.sum((pb - pa) * np.log(pb / pa)))
out["drift_psi_train_vs_test"] = {c: round(psi(df.loc[df.split == "train", c], df.loc[df.split == "test", c]), 3)
                                  for c in PLAN if c != "plan_finsource"}

buckets = {   # failure rate by simple buckets, train only, to read the direction of each signal
    "plan_changed_later (LEAK, not a feature)": df.plan_changed_later,
    "plan_prepayment": pd.cut(df.plan_prepayment, [-1, 0, 30, 100]),
    "plan_month_gap": pd.cut(df.plan_month_gap, [-13, -1, 0, 2, 13]),
    "plan_days_since_change": pd.cut(df.plan_days_since_change, [-1, 1, 7, 30, 1000]),
    "plan_lot_vs_plan": pd.cut(df.plan_lot_vs_plan, [-50, -0.01, 0.01, 50]),
    "plan_method_mismatch": df.plan_method_mismatch, "plan_multi_year": df.plan_multi_year,
    "plan_transferred": df.plan_transferred, "plan_preliminary": df.plan_preliminary,
}
out["rate_by_bucket_train"] = {c: {str(k): v for k, v in df[trn].groupby(b[trn], observed=True)[y]
                                   .agg(["mean", "size"]).round(4).to_dict("index").items()}
                               for c, b in buckets.items()}

E = np.load(ROOT / "data/processed/rubert_tiny2_cls.npy")
uniq = pd.read_parquet(ROOT / "data/processed/rubert_tiny2_texts.parquet").text
pos = pd.Series(np.arange(len(uniq)), index=uniq.values)
Z = E[pos[df.lot_names.fillna("").str.lower().str.strip()].to_numpy()]
T = [f"emb{k}" for k in range(32)]
df[T] = PCA(32, random_state=SEED).fit(Z[trn.to_numpy()]).transform(Z)

def at_k(yt, p, k):
    n = int(len(p) * k); idx = np.argsort(-p)[:n]
    return float(yt[idx].mean()), float(yt[idx].sum() / yt.sum())

models = {}
def run(cols, name, d=df):
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    yt = te[y].to_numpy()
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=SEED, verbose=-1)
    m.fit(tr[cols], tr[y], eval_set=[(va[cols], va[y])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
    out[name] = {"n_features": len(cols), "n_test": len(te), "best_iter": m.best_iteration_,
                 "val_pr": average_precision_score(va[y], pv), "test_pr": average_precision_score(yt, pt),
                 "test_roc": roc_auc_score(yt, pt), "test_p_at_1": at_k(yt, pt, 0.01)[0],
                 "test_r_at_5": at_k(yt, pt, 0.05)[1]}
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)
    models[name] = m
    return pt

F20 = P.FEATURES
REL = ["pair_prior_contracts", "pair_share_of_customer", "customer_hhi", "winner_contracts_30d",
       "winner_top_customer_share", "winner_win_rate"]
PRICE = ["unit_price_vs_market", "unit_price_vs_market_max", "share_lots_benchmarked"]
BASE = F20 + T + REL + PRICE          # best leak-free set from pilot 5
full = df[df.full_plan]                # procurements where every lot has the plan version in force
preds = {"F20": run(F20, "F20"), "F20+PLAN": run(F20 + PLAN, "F20+PLAN"),
         "F20+PLAN_leaky": run(F20 + PLAN_LEAKY, "F20+PLAN_leaky"),
         "BASE": run(BASE, "BASE"), "BASE+PLAN": run(BASE + PLAN, "BASE+PLAN"),
         "BASE+PLAN_leaky": run(BASE + PLAN_LEAKY, "BASE+PLAN_leaky")}
preds_full = {"F20": run(F20, "full:F20", full), "F20+PLAN": run(F20 + PLAN, "full:F20+PLAN", full),
              "BASE": run(BASE, "full:BASE", full), "BASE+PLAN": run(BASE + PLAN, "full:BASE+PLAN", full)}

# permutation importance of the PLAN columns in F20+PLAN on validation (PR-AUC drop, 3 shuffles)
va = df[df.split == "val"]
rng = np.random.default_rng(SEED); m = models["F20+PLAN"]; cols = F20 + PLAN; base_v = out["F20+PLAN"]["val_pr"]; imp = {}
for c in PLAN:
    drops = []
    for _ in range(3):
        X = va[cols].copy(); X[c] = X[c].sample(frac=1, random_state=int(rng.integers(1e9))).values
        drops.append(base_v - average_precision_score(va[y], m.predict_proba(X)[:, 1]))
    imp[c] = round(float(np.mean(drops)), 4)
out["perm_importance_PLAN_val"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

def boot(pr, d, a, b):
    yt = d.loc[d.split == "test", y].to_numpy()
    idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
    diff = np.array([average_precision_score(yt[i], pr[a][i]) - average_precision_score(yt[i], pr[b][i]) for i in idx])
    return {"mean": float(diff.mean()), "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
out["bootstrap"] = {f"{a} vs {b}": boot(preds, df, a, b) for a, b in [
    ("F20+PLAN", "F20"), ("F20+PLAN_leaky", "F20"), ("BASE+PLAN", "BASE"), ("BASE+PLAN_leaky", "BASE")]}
out["bootstrap"].update({f"full: {a} vs {b}": boot(preds_full, full, a, b) for a, b in [
    ("F20+PLAN", "F20"), ("BASE+PLAN", "BASE")]})
Path(__file__).with_name("p7_results.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: out[k] for k in ("versions_vs_publication", "perm_importance_PLAN_val", "bootstrap")},
                 indent=1, default=str), flush=True)
