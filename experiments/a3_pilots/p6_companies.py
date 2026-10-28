"""
A3 pilot 6 — company attributes and "shared requisites" (Tender Radar NEW_COMPANY_BIG_CONTRACT,
COMMON_REQUISITES, BANK_DETAILS_REUSE), re-computed without look-ahead.

  COMP   winner's registration age at publication (re-registrations after publication -> NaN, the register
         is a 2026 snapshot), legal form, sector (KSE), OKED section, supplier type; customer's legal form
  CONT   contacts: another bidder in the same procurement shares the winner's phone / e-mail; any two bidders
         share one; winner shares a contact with the customer; how many registered participants use the
         winner's phone (accountant / registration-agent phones are common, so this is a count, not a flag)
  BANK   the winner's bank account (from its contracts signed before publication) was used by another
         supplier before publication; two bidders have the same legal address (same as-of rule)

Contacts and requisites are the 2026 snapshot hashed at download (src/collect.py); only equality is used.
Needs data/raw/subjects.parquet, data/raw/contract_bank.parquet and data/processed/p5_features.parquet.
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
HAS_BANK = (ROOT / "data/raw/contract_bank.parquet").exists()

con = duckdb.connect()
con.sql(f"CREATE VIEW mt AS SELECT * FROM read_parquet('{(ROOT / 'data/processed/model_table.parquet').as_posix()}')")
con.sql(f"""CREATE TABLE ctr AS
    SELECT id AS cid, rootId, trdBuyId AS buy_id, customerBin AS cust, supplierBiin AS sup, CAST(signDate AS TIMESTAMP) AS sdt
    FROM read_parquet('{RAW}/contracts.parquet')""")
sys.path.insert(0, str(Path(__file__).parent))
from winner import create_winner_table   # winner and customer without contracts (see winner.py)
create_winner_table(con, RAW)
con.sql("CREATE TABLE cur AS SELECT m.id, m.publish_dt AS t, a.cust, a.winner FROM mt m LEFT JOIN award a USING (id)")
con.sql(f"""CREATE TABLE bidder AS SELECT DISTINCT buyId AS id, supplierBinIin AS sup
            FROM read_parquet('{RAW}/trd_app.parquet')""")
# one row per BIN/IIN; subjects carry either bin (legal entities) or iin (individuals)
con.sql(f"""CREATE TABLE subj AS
    SELECT coalesce(nullif(bin, ''), iin) AS biin, any_value(TRY_CAST(regdate AS TIMESTAMP)) AS regdate,
           any_value(refKopfCode) AS kopf, any_value(kseCode) AS kse, any_value(typeSupplier) AS stype,
           any_value(nullif(regexp_extract(CAST(okedList AS VARCHAR), '(\\d{2})', 1), '')) AS oked2,
           any_value(from_json(phone_h, '["VARCHAR"]')) AS phone_h,     -- lists are JSON strings in parquet
           any_value(from_json(email_h, '["VARCHAR"]')) AS email_h
    FROM read_parquet('{RAW}/subjects.parquet')
    WHERE coalesce(nullif(bin, ''), iin) IS NOT NULL GROUP BY 1""")

comp = con.sql("""
SELECT c.id,
       CASE WHEN s.regdate <= c.t THEN date_diff('day', s.regdate, c.t) / 365.25 END AS winner_age_years,
       (s.regdate > c.t)::INT AS winner_reregistered_later,
       (s.biin IS NOT NULL)::INT AS winner_in_register,
       s.kopf AS winner_kopf, CAST(s.kse AS VARCHAR) AS winner_kse, CAST(s.stype AS VARCHAR) AS winner_stype,
       s.oked2 AS winner_oked2, cs.kopf AS customer_kopf
FROM cur c LEFT JOIN subj s ON s.biin = c.winner LEFT JOIN subj cs ON cs.biin = c.cust
""").df()

con.sql("""CREATE TABLE contact AS   -- (participant, hashed contact); phones and e-mails pooled
    SELECT biin, unnest(phone_h) AS h, 'p' AS kind FROM subj
    UNION ALL SELECT biin, unnest(email_h), 'e' FROM subj""")
cont = con.sql("""
WITH deg AS (SELECT h, count(DISTINCT biin) AS n FROM contact WHERE kind = 'p' GROUP BY h),
bc AS (SELECT b.id, b.sup, k.h FROM bidder b JOIN contact k ON k.biin = b.sup),
pair AS (SELECT x.id, max((x.sup = c.winner OR y.sup = c.winner)::INT) AS winner_shares_contact_with_bidder,
                1 AS bidders_share_contact
         FROM bc x JOIN bc y ON x.id = y.id AND x.h = y.h AND x.sup < y.sup JOIN cur c ON c.id = x.id GROUP BY 1),
wc AS (SELECT c.id, max(1) AS winner_shares_contact_with_customer
       FROM cur c JOIN contact a ON a.biin = c.winner JOIN contact b ON b.biin = c.cust AND b.h = a.h GROUP BY 1),
wd AS (SELECT c.id, max(d.n) AS winner_phone_degree
       FROM cur c JOIN contact k ON k.biin = c.winner AND k.kind = 'p' JOIN deg d USING (h) GROUP BY 1)
SELECT c.id, coalesce(p.bidders_share_contact, 0) AS bidders_share_contact,
       coalesce(p.winner_shares_contact_with_bidder, 0) AS winner_shares_contact_with_bidder,
       coalesce(wc.winner_shares_contact_with_customer, 0) AS winner_shares_contact_with_customer,
       wd.winner_phone_degree
FROM cur c LEFT JOIN pair p USING (id) LEFT JOIN wc USING (id) LEFT JOIN wd USING (id)
""").df()

feats = comp.merge(cont, on="id")
# The register is a 2026 snapshot: "missing from it" (liquidated) or "re-registered after publication" can be a
# consequence of the failure. Those two facts are reported but never used; for such winners the attributes are
# replaced by the training-period median / mode so the model cannot tell them apart.
COMP = ["winner_age_years", "winner_kopf", "winner_kse", "winner_stype", "winner_oked2", "customer_kopf"]
LEAKY = (feats.winner_in_register == 0) | (feats.winner_reregistered_later == 1)
train_ids = set(con.sql("SELECT id FROM mt WHERE split = 'train'").df().id)
ref = feats[feats.id.isin(train_ids) & ~LEAKY]
for c in COMP[:5]:
    fill = ref[c].median() if c == "winner_age_years" else ref[c].mode().iloc[0]
    feats.loc[LEAKY, c] = fill
CONT = ["bidders_share_contact", "winner_shares_contact_with_bidder", "winner_shares_contact_with_customer",
        "winner_phone_degree"]
BANK = []
if HAS_BANK:
    con.sql(f"""CREATE TABLE req AS   -- first time each (account / address, supplier) appears in a signed contract
        SELECT k, kind, sup, min(sdt) AS first_seen FROM (
            SELECT b.supplierIik_h AS k, 'iik' AS kind, b.supplierBiin AS sup, c.sdt
            FROM read_parquet('{RAW}/contract_bank.parquet') b JOIN ctr c ON c.cid = b.id
            UNION ALL
            SELECT b.supplierAddress_h, 'addr', b.supplierBiin, c.sdt
            FROM read_parquet('{RAW}/contract_bank.parquet') b JOIN ctr c ON c.cid = b.id)
        WHERE k IS NOT NULL AND sdt IS NOT NULL GROUP BY 1, 2, 3""")
    bank = con.sql("""
    WITH wi AS (SELECT c.id, max(1) AS winner_account_shared
                FROM cur c JOIN req a ON a.sup = c.winner AND a.kind = 'iik' AND a.first_seen < c.t
                JOIN req o ON o.k = a.k AND o.kind = 'iik' AND o.sup <> c.winner AND o.first_seen < c.t GROUP BY 1),
    ba AS (SELECT b.id, b.sup, r.k FROM bidder b JOIN cur c USING (id)
           JOIN req r ON r.sup = b.sup AND r.kind = 'addr' AND r.first_seen < c.t),
    sa AS (SELECT x.id, max(1) AS bidders_share_address FROM ba x JOIN ba y ON x.id = y.id AND x.k = y.k AND x.sup < y.sup GROUP BY 1)
    SELECT c.id, coalesce(wi.winner_account_shared, 0) AS winner_account_shared,
           coalesce(sa.bidders_share_address, 0) AS bidders_share_address
    FROM cur c LEFT JOIN wi USING (id) LEFT JOIN sa USING (id)
    """).df()
    feats = feats.merge(bank, on="id"); BANK = ["winner_account_shared", "bidders_share_address"]

feats.to_parquet(ROOT / "data/processed/p6_features.parquet", index=False)
df = (P.load_model_table(ROOT).merge(pd.read_parquet(ROOT / "data/processed/p5_features.parquet"), on="id", how="left")
      .merge(feats, on="id", how="left"))
CATS = P.CAT_FEATURES + ["winner_kopf", "winner_kse", "winner_stype", "winner_oked2", "customer_kopf"]
for c in CATS: df[c] = df[c].astype("category")

y = P.TARGET
out["coverage"] = {c: round(float(df[c].notna().mean()), 3) for c in COMP + CONT + BANK}
out["rate_by_flag"] = {c: {str(k): v for k, v in df.groupby(c)[y].agg(["mean", "size"]).round(4).to_dict("index").items()}
                       for c in ["winner_reregistered_later", "winner_in_register"] + CONT[:3] + BANK}
clean = df[(df.winner_in_register == 1) & (df.winner_reregistered_later != 1)]   # real (not imputed) ages only
out["rate_by_age_bucket"] = {str(k): v for k, v in clean.groupby(pd.cut(clean.winner_age_years, [0, 1, 2, 5, 10, 100]))[y]
                             .agg(["mean", "size"]).round(4).to_dict("index").items()}

E = np.load(ROOT / "data/processed/rubert_tiny2_cls.npy")
uniq = pd.read_parquet(ROOT / "data/processed/rubert_tiny2_texts.parquet").text
pos = pd.Series(np.arange(len(uniq)), index=uniq.values)
Z = E[pos[df.lot_names.fillna("").str.lower().str.strip()].to_numpy()]
T = [f"emb{k}" for k in range(32)]
df[T] = PCA(32, random_state=SEED).fit(Z[(df.split == "train").to_numpy()]).transform(Z)

tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
yt = te[y].to_numpy()

def at_k(p, k):
    n = int(len(p) * k); idx = np.argsort(-p)[:n]
    return float(yt[idx].mean()), float(yt[idx].sum() / yt.sum())

def run(cols, name):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=SEED, verbose=-1)
    m.fit(tr[cols], tr[y], eval_set=[(va[cols], va[y])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = m.predict_proba(va[cols])[:, 1], m.predict_proba(te[cols])[:, 1]
    out[name] = {"n_features": len(cols), "best_iter": m.best_iteration_,
                 "val_pr": average_precision_score(va[y], pv), "test_pr": average_precision_score(yt, pt),
                 "test_roc": roc_auc_score(yt, pt), "test_p_at_1": at_k(pt, 0.01)[0], "test_r_at_5": at_k(pt, 0.05)[1]}
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items()}, flush=True)
    return pt

F20 = P.FEATURES
REL = ["pair_prior_contracts", "pair_share_of_customer", "customer_hhi", "winner_contracts_30d",
       "winner_top_customer_share", "winner_win_rate"]
PRICE = ["unit_price_vs_market", "unit_price_vs_market_max", "share_lots_benchmarked"]
BASE = F20 + T + REL + PRICE          # best leak-free set from pilot 5
preds = {"F20": run(F20, "F20"), "F20+COMP": run(F20 + COMP, "F20+COMP"), "F20+CONT": run(F20 + CONT, "F20+CONT"),
         "BASE": run(BASE, "BASE"), "BASE+COMP+CONT": run(BASE + COMP + CONT, "BASE+COMP+CONT")}
if BANK:
    preds["F20+BANK"] = run(F20 + BANK, "F20+BANK")
    preds["BASE+COMP+CONT+BANK"] = run(BASE + COMP + CONT + BANK, "BASE+COMP+CONT+BANK")

rng = np.random.default_rng(SEED); idx = [rng.integers(0, len(yt), len(yt)) for _ in range(1000)]
def boot(a, b):
    d = np.array([average_precision_score(yt[i], preds[a][i]) - average_precision_score(yt[i], preds[b][i]) for i in idx])
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
pairs = [("F20+COMP", "F20"), ("F20+CONT", "F20"), ("BASE+COMP+CONT", "BASE")]
if BANK: pairs += [("F20+BANK", "F20"), ("BASE+COMP+CONT+BANK", "BASE")]
out["bootstrap"] = {f"{a} vs {b}": boot(a, b) for a, b in pairs}
Path(__file__).with_name("p6_results.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out["bootstrap"], indent=1), flush=True)
