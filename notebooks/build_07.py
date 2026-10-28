"""
Builds notebooks/07_results.ipynb (Assignment 3, criteria 7, 8, 9): the final model, one comparison table for every
experiment of A3, ablations, the leakage demonstration, SHAP, segments and reproducibility.
Run after 04, 05a, 05b, 06:
  python notebooks/build_07.py && jupyter nbconvert --to notebook --execute --inplace notebooks/07_results.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# Assignment 3 · Part 4 — Results, interpretation, reproducibility
## Predicting failed public-procurement contracts in Kazakhstan

**Author:** Aziz Abdulin · **Project:** Tender Risk ML · **Repository:** https://github.com/awzwz/tender-risk-ml

This notebook brings Assignment 3 together: the final model, **one table** with every model trained in notebooks
04–06 on the same test period, ablations that show where the improvement comes from, a leakage case found during the
work, SHAP explanations, results by segment, and the information needed to reproduce everything.

| § | Content | A3 criterion |
|---|---|---|
| 1 | Setup | — |
| 2 | Final model: features, training, calibration | — |
| 3 | One table for all experiments | 7 · Comparison of all models |
| 4 | Where the improvement comes from: ablations and a leakage case | 8 · Interpretation |
| 5 | What drives the predictions: SHAP | 8 |
| 6 | Results by segment | 8 |
| 7 | Hypotheses: expected vs observed | 8 |
| 8 | Reproducibility | 9 · Reproducibility |
| 9 | Conclusions and limitations | — |

**How to run:** after notebooks 04, 05a, 05b and 06 (this notebook reads their saved scores from
`data/predictions/` and their settings from `results/`); `Kernel → Restart & Run All`, ≈ 5 minutes.
""")
code(r"""
import json, platform, sys, warnings
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_recall_curve, roc_auc_score)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
SEED = 42

ROOT = Path.cwd().resolve()
while not (ROOT / "src").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from src import prepare as P
from src.features import load_a3_table, REL, PRICE

PROC, PRED = ROOT / "data" / "processed", ROOT / "data" / "predictions"
FIG = ROOT / "figures"; RES = ROOT / "results" / "a3_07"; RES.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
""")
code(r"""
df = load_a3_table(ROOT)
df["text"] = df[P.TEXT_FEATURE].fillna("").str.lower().str.strip()
for c in P.CAT_FEATURES: df[c] = df[c].astype("category")
E = np.load(PROC / "emb_rubert-tiny2.npy"); texts = pd.read_parquet(PROC / "emb_texts.parquet").text
tpos = pd.Series(np.arange(len(texts)), index=texts.values)
T = [f"text_pc{k}" for k in range(32)]
df[T] = PCA(32, random_state=SEED).fit(E[tpos[df.loc[df.split == "train", "text"]].to_numpy()]).transform(E[tpos[df.text].to_numpy()])

y = P.TARGET
train, val, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
y_tr, y_va, y_te = train[y].to_numpy(), val[y].to_numpy(), test[y].to_numpy()
unseen = ~test.text.isin(set(train.text)).to_numpy()

# Workhorse LightGBM for feature experiments (notebook 04: tuning LightGBM brings no gain on these features, and this
# configuration is as good as the best of 50 Optuna trials on validation while being stable and fast)
LGB = dict(learning_rate=0.05)

def fit_lgb(cols, data=None):
    d = train if data is None else data
    m = lgb.LGBMClassifier(n_estimators=5000, random_state=SEED, verbose=-1, **LGB)
    m.fit(d[cols], d[y], eval_set=[(val[cols], y_va)], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m, m.predict_proba(val[cols])[:, 1], m.predict_proba(test[cols])[:, 1]

rng = np.random.default_rng(SEED); BOOT = [rng.integers(0, len(y_te), len(y_te)) for _ in range(1000)]
def boot_ci(p):
    v = np.array([average_precision_score(y_te[i], p[i]) for i in BOOT]); return np.percentile(v, [2.5, 97.5])
def boot_diff(a, b):
    d = np.array([average_precision_score(y_te[i], a[i]) - average_precision_score(y_te[i], b[i]) for i in BOOT])
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 2. Final model

**Features (61).** The 20 midterm features; 32 principal components of frozen rubert-tiny2 vectors of the lot names
(notebook 05); 6 customer–supplier history features and 3 price-against-market features (`src/features.py`) — the
leak-free versions of Tender Radar's rule indicators selected in the research (§11 of the research report).

**Model selection on validation.** Three candidates are trained on train and compared on validation PR-AUC only:
the simple LightGBM used for the feature experiments (learning rate 0.05, early stopping), LightGBM re-tuned on these
61 features with Optuna (30 trials, the search space of notebook 04), and CatBoost with the hyper-parameters tuned in
notebook 04 (the best model on the 20 features).

**Calibration.** Platt scaling (logistic regression on the logit of the score) fitted on validation: it turns the
score into a probability without changing the ranking.
""")
code(r"""
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
optuna.logging.set_verbosity(optuna.logging.WARNING)
FINAL = P.FEATURES + T + REL + PRICE
cands = {}

cands["LightGBM · simple (learning rate 0.05)"] = fit_lgb(FINAL)

def lgb_objective(trial):
    params = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                  num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
                  min_child_samples=trial.suggest_int("min_child_samples", 20, 1000, log=True),
                  colsample_bytree=trial.suggest_float("feature_fraction", 0.4, 1.0),
                  subsample=trial.suggest_float("bagging_fraction", 0.5, 1.0), subsample_freq=trial.suggest_int("bagging_freq", 0, 5),
                  reg_alpha=trial.suggest_float("lambda_l1", 1e-8, 10, log=True), reg_lambda=trial.suggest_float("lambda_l2", 1e-8, 10, log=True),
                  scale_pos_weight=trial.suggest_float("scale_pos_weight", 1, 14))
    m = lgb.LGBMClassifier(n_estimators=5000, random_state=SEED, verbose=-1, **params)
    m.fit(train[FINAL], y_tr, eval_set=[(val[FINAL], y_va)], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    trial.set_user_attr("params", params)
    return average_precision_score(y_va, m.predict_proba(val[FINAL])[:, 1])
study = optuna.create_study(study_name="lightgbm_final", storage=f"sqlite:///{(RES / 'optuna.db').as_posix()}", direction="maximize",
                            load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED))
todo = 30 - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
if todo > 0: study.optimize(lgb_objective, n_trials=todo)
m = lgb.LGBMClassifier(n_estimators=5000, random_state=SEED, verbose=-1, **study.best_trial.user_attrs["params"])
m.fit(train[FINAL], y_tr, eval_set=[(val[FINAL], y_va)], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
cands["LightGBM · tuned on the final features"] = (m, m.predict_proba(val[FINAL])[:, 1], m.predict_proba(test[FINAL])[:, 1])

cb = json.loads((ROOT / "results" / "a3_04" / "best_params.json").read_text())["CatBoost · tuned"]
as_cat = lambda d: d[FINAL].assign(**{c: d[c].astype(str) for c in P.CAT_FEATURES})
CAT_IDX = [FINAL.index(c) for c in P.CAT_FEATURES]
m = CatBoostClassifier(iterations=3000, eval_metric="PRAUC", random_seed=SEED, verbose=0, thread_count=-1,
                       depth=cb["depth"], learning_rate=cb["learning_rate"], l2_leaf_reg=cb["l2_leaf_reg"], random_strength=cb["random_strength"])
m.fit(as_cat(train), y_tr, cat_features=CAT_IDX, eval_set=(as_cat(val), y_va), early_stopping_rounds=100)
cands["CatBoost · hyper-parameters of notebook 04"] = (m, m.predict_proba(as_cat(val))[:, 1], m.predict_proba(as_cat(test))[:, 1])

selection = pd.DataFrame({k: {"validation PR-AUC": average_precision_score(y_va, v[1]), "test PR-AUC": average_precision_score(y_te, v[2])}
                          for k, v in cands.items()}).T
final_name = selection["validation PR-AUC"].idxmax()
final_model, fin_va, fin_te = cands[final_name]
selection.assign(chosen=lambda d: np.where(d.index == final_name, "✓", ""))
""")
code(r"""
logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(fin_va).reshape(-1, 1), y_va)
cal = lambda p: platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]
fin_te_cal = cal(fin_te)
if hasattr(final_model, "booster_"): final_model.booster_.save_model(str(RES / "final_model_lightgbm.txt"))
else: final_model.save_model(str(RES / "final_model_catboost.cbm"))
pd.DataFrame({"id": np.r_[val.id, test.id], "split": ["val"] * len(val) + ["test"] * len(test),
              "score": np.r_[fin_va, fin_te], "probability": np.r_[cal(fin_va), fin_te_cal]}).to_parquet(PRED / "07_final.parquet", index=False)
print(f"final model: {final_name}; {len(FINAL)} features; test PR-AUC {average_precision_score(y_te, fin_te):.3f}, "
      f"ROC-AUC {roc_auc_score(y_te, fin_te):.3f}; mean probability {fin_te_cal.mean():.3f} vs failure rate {y_te.mean():.3f}; "
      f"Brier {brier_score_loss(y_te, fin_te):.4f} -> {brier_score_loss(y_te, fin_te_cal):.4f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 3. One table for all experiments

Every row is scored on the same 38,981 test procurements (16–31 March 2024). The decision threshold for F1 is chosen
on validation for every model separately. `P@1%` — share of failures among the 1 % highest-scored procurements
(389); `R@5%` — share of all failures found in the top 5 %. The interval is a 95 % bootstrap interval of PR-AUC.
""")
code(r"""
def load(name):
    s = pd.read_parquet(PRED / f"{name}.parquet").set_index("id").score
    return s.loc[val.id].to_numpy(), s.loc[test.id].to_numpy()

def top_k(p, k):
    n = int(len(p) * k); idx = np.argsort(-p)[:n]; return y_te[idx].mean(), y_te[idx].sum() / y_te.sum()

def row(group, name, info, pv, pt):
    pr, rc, th = precision_recall_curve(y_va, pv); f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    thr = th[np.argmax(f1[:-1])]; lo, hi = boot_ci(pt)
    return {"group": group, "model": name, "information": info, "test PR-AUC": average_precision_score(y_te, pt),
            "95 % CI": f"[{lo:.3f}, {hi:.3f}]", "ROC-AUC": roc_auc_score(y_te, pt), "F1": f1_score(y_te, pt >= thr),
            "P@1%": top_k(pt, 0.01)[0], "R@5%": top_k(pt, 0.05)[1],
            "PR-AUC unseen texts": average_precision_score(y_te[unseen], pt[unseen])}

R = [row("0 · reference", "No model (failure rate)", "—", np.full(len(y_va), y_tr.mean()), np.full(len(y_te), y_tr.mean()))]
for fam, f in (("Logistic regression", "logistic_regression"), ("Random forest", "random_forest"),
               ("LightGBM", "lightgbm"), ("CatBoost", "catboost")):
    for kind in ("default", "tuned"):
        label = "midterm baseline" if (fam == "Logistic regression" and kind == "default") else kind
        R.append(row("1 · model comparison (04)", f"{fam} · {label}", "20 features", *load(f"04_{f}_{kind}")))
R.append(row("2 · text alone (05)", "TF-IDF + logistic regression", "lot names", *load("05b_tf_idf___logistic_regression")))
for key in ("rubert-tiny2", "multilingual-e5-small", "rubert-base-cased"):
    R.append(row("2 · text alone (05)", f"frozen {key} + linear probe", "lot names", *load(f"05_probe_{key}")))
R.append(row("2 · text alone (05)", "fine-tuned rubert-tiny2", "lot names", *load("05_rubert_finetuned")))
for f in sorted(PRED.glob("05b_*.parquet")):
    if "tf_idf" in f.stem or "20_features" in f.stem: continue
    R.append(row("3 · table + text (05)", f.stem.replace("05b_", "").replace("_", " "), "20 features + text", *load(f.stem)))
R.append(row("4 · LLM (06)", "Qwen2.5-1.5B + QLoRA, 40 k rows", "card: 20 features + history + price + lot names", *load("06_llm_qlora")))
R.append(row("4 · LLM (06)", "simple LightGBM + LLM ensemble", "61 features + LLM", *load("06_blend_lgb_llm")))
R.append(row("5 · final", "Final model", f"{final_name}; 20 features + text + history + price", fin_va, fin_te))
# the final model averaged with the LLM (logit scale, weight chosen on validation as in notebook 06)
llm_va, llm_te = load("06_llm_qlora")
weights = np.linspace(0, 1, 21)
alpha = float(weights[np.argmax([average_precision_score(y_va, (1 - a) * logit(fin_va) + a * logit(llm_va)) for a in weights])])
ens_va, ens_te = (1 - alpha) * logit(fin_va) + alpha * logit(llm_va), (1 - alpha) * logit(fin_te) + alpha * logit(llm_te)
R.append(row("5 · final", "Final model + LLM ensemble", f"LLM weight {alpha:.2f} chosen on validation", ens_va, ens_te))
table = pd.DataFrame(R).set_index(["group", "model"])
table.to_csv(RES / "all_models_test.csv")
table
""")
code(r"""
plot = table.reset_index()
plot = plot[plot.group != "0 · reference"].sort_values("test PR-AUC")
fig, ax = plt.subplots(figsize=(9, 0.32 * len(plot) + 1))
palette = {"1 · model comparison (04)": "#2b6cb0", "2 · text alone (05)": "#b7791f", "3 · table + text (05)": "#c05621",
           "4 · LLM (06)": "#6b46c1", "5 · final": "#c53030"}
ax.barh(plot.model, plot["test PR-AUC"], color=[palette[g] for g in plot.group])
ax.axvline(y_te.mean(), ls="--", color="grey"); ax.text(y_te.mean() + 0.003, -0.6, "no model", fontsize=8, color="grey")
for gname, c in palette.items(): ax.barh([], [], color=c, label=gname)
ax.set(xlabel="test PR-AUC", title="All A3 models on the test period"); ax.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "a3_07_all_models.png", dpi=150, bbox_inches="tight"); plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 4. Where the improvement comes from

### 4.1 Ablations of the final model
Each group of features is removed in turn and the model is retrained with the same settings (the simple LightGBM,
so that only the features change). The difference to the full feature set, with a paired bootstrap interval, is the
contribution of that group *given all the others*.
""")
code(r"""
# ablations use the simple LightGBM (the same configuration for every variant, so that only the features differ)
abl = {"final model (all groups)": (FINAL, cands["LightGBM · simple (learning rate 0.05)"][2])}
for label, drop in (("− text components", T), ("− customer–supplier history", REL), ("− price vs market", PRICE),
                    ("− history and price", REL + PRICE), ("only the 20 midterm features", T + REL + PRICE)):
    cols = [c for c in FINAL if c not in drop]
    abl[label] = (cols, fit_lgb(cols)[2])
abl_rows = {}
for label, (cols, p) in abl.items():
    m, lo, hi = boot_diff(p, abl["final model (all groups)"][1]) if label != "final model (all groups)" else (0, 0, 0)
    abl_rows[label] = {"features": len(cols), "test PR-AUC": average_precision_score(y_te, p),
                       "change vs final": "" if label.startswith("final") else f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"}
ablation = pd.DataFrame(abl_rows).T; ablation.to_csv(RES / "ablation.csv"); ablation
""")
md(r"""
### 4.2 A leakage case: the winner taken from the contract

An earlier version of the research added three "winner activity" features (bids of the winner in the previous 30
days, individual entrepreneur or not) and they looked like the third most important information source (+0.025
PR-AUC). The winner had been taken from the **contracts table**. Checking the strongest predictions showed the
problem: when a winner refuses to sign, the contract goes to another participant or is concluded outside the
competition, so *who signed the contract* is partly the outcome. In procurements with bids, the contract signer had
not even applied in 27 % of successful and 63 % of failed procurements.

The table repeats the experiment with three definitions of the winner. Only the last is known before the outcome:
the supplier named in the announcement (single source) or the lowest bidder (with the procedural tie-break — the
earlier application wins at equal prices).
""")
code(r"""
import duckdb
con = duckdb.connect(); con.sql("SET enable_progress_bar = false"); r = (ROOT / "data" / "raw").as_posix()
con.sql(f"CREATE VIEW mt AS SELECT id, publish_dt FROM read_parquet('{(PROC / 'model_table.parquet').as_posix()}')")
con.sql(f"CREATE TABLE app AS SELECT buyId AS buy_id, supplierBinIin AS sup, CAST(dateApply AS TIMESTAMP) AS dt FROM read_parquet('{r}/trd_app.parquet')")
con.sql(f"CREATE TABLE ctr AS SELECT trdBuyId AS id, arg_min(supplierBiin, rootId) AS first_sup, min(supplierBiin) AS min_sup FROM read_parquet('{r}/contracts.parquet') GROUP BY 1")
from src.features import winners
honest = winners(ROOT / "data" / "raw").rename(columns={"winner": "honest_sup"})[["id", "honest_sup"]]
con.register("honest", honest)
def winner_feats(col):
    return con.sql(f'''
        WITH w AS (SELECT m.id, m.publish_dt AS t, {col} AS sup FROM mt m LEFT JOIN ctr USING (id) LEFT JOIN honest USING (id)),
        h AS (SELECT w.id, count(a.sup) AS bids_30d, count(DISTINCT a.buy_id) AS procs_30d
              FROM w LEFT JOIN app a ON a.sup = w.sup AND a.dt < w.t AND a.dt >= w.t - INTERVAL 30 DAY GROUP BY 1)
        SELECT w.id, (substr(w.sup, 5, 1) IN ('0','1','2','3'))::INT AS individual, h.bids_30d, h.procs_30d
        FROM w JOIN h USING (id)''').df()
leak_rows = {}
base_m, _, base_te = fit_lgb(P.FEATURES)
for label, col in (("winner = min(BIN) among contracts (research v2)", "ctr.min_sup"),
                   ("winner = supplier of the first contract", "ctr.first_sup"),
                   ("winner from announcement / lowest bid (leak-free)", "honest.honest_sup")):
    wf = winner_feats(col)
    for d in (train, val, test):
        d[["w_ind", "w_bids", "w_procs"]] = wf.set_index("id").loc[d.id, ["individual", "bids_30d", "procs_30d"]].astype(float).to_numpy()
    _, _, p = fit_lgb(P.FEATURES + ["w_ind", "w_bids", "w_procs"])
    m, lo, hi = boot_diff(p, base_te)
    leak_rows[label] = {"test PR-AUC": average_precision_score(y_te, p), "gain over 20 features": f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"}
leak = pd.DataFrame(leak_rows).T; leak.to_csv(RES / "leakage_winner.csv"); leak
""")
md(r"""
The "gain" disappears with the leak-free definition. The lesson for the report: a feature that looks too good has to
be traced to the moment its value becomes known — here not the feature itself (bid dates were correct) but the
**link** between records (which supplier is "the winner") was created after the outcome.
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 5. What drives the predictions: SHAP

SHAP values decompose each prediction (log-odds of failure) into additive contributions of the features. They are
computed with the exact TreeSHAP algorithm built into the boosting library (LightGBM `pred_contrib=True` or CatBoost
`ShapValues` — the same algorithm as `shap.TreeExplainer`, with native support for categorical splits) on 5,000 test
procurements. The 32 text components are summed into one
"lot text" contribution, since a single component has no meaning on its own.
""")
code(r"""
sample = test.sample(5000, random_state=SEED)
if hasattr(final_model, "booster_"):                                          # LightGBM
    contrib = final_model.booster_.predict(sample[FINAL], pred_contrib=True)  # last column = expected value
    raw = final_model.booster_.predict(sample[FINAL], raw_score=True)
else:                                                                         # CatBoost
    pool = Pool(as_cat(sample), cat_features=CAT_IDX)
    contrib = final_model.get_feature_importance(pool, type="ShapValues")
    raw = final_model.predict(pool, prediction_type="RawFormulaVal")
assert np.allclose(contrib.sum(1), raw, atol=1e-4)                            # contributions add up to the prediction
S = pd.DataFrame(contrib[:, :-1], columns=FINAL, index=sample.index)
S["lot text (32 components)"] = S[T].sum(axis=1); S = S.drop(columns=T)
mean_abs = S.abs().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 6))
mean_abs.head(15)[::-1].plot.barh(ax=ax, color="#c05621")
ax.set(xlabel="mean |SHAP value| (log-odds)", title="Final model: average contribution per feature (top 15)")
plt.tight_layout(); plt.savefig(FIG / "a3_07_shap_importance.png", dpi=150, bbox_inches="tight"); plt.show()
mean_abs.head(15).to_frame("mean |SHAP|")
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col, xlab in zip(ax, ["n_bidders", "pair_prior_contracts", "unit_price_vs_market"],
                        ["number of bidders", "earlier contracts of this customer with the winner", "log(unit price / market median)"]):
    x = sample[col].astype(float)
    a.scatter(x, S[col], s=4, alpha=0.3, color="#c05621")
    a.axhline(0, color="grey", lw=0.8); a.set(xlabel=xlab, ylabel="SHAP value (log-odds of failure)")
    if col != "unit_price_vs_market": a.set_xscale("symlog")
ax[0].set_title("Competition"); ax[1].set_title("Customer–supplier history"); ax[2].set_title("Price against the market")
plt.tight_layout(); plt.savefig(FIG / "a3_07_shap_dependence.png", dpi=150, bbox_inches="tight"); plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 6. Results by segment

PR-AUC depends on the share of positives, so segments with different failure rates cannot be compared on PR-AUC
alone. The table reports the **lift**: PR-AUC divided by the failure rate of the segment (1 = no better than random
inside that segment), and ROC-AUC, which does not depend on the base rate.
""")
code(r"""
base = pd.read_parquet(PRED / "04_logistic_regression_default.parquet").set_index("id").score.loc[test.id].to_numpy()
seg = test.assign(value_quartile=pd.qcut(test.log_total_sum, 4, labels=["Q1 (cheapest)", "Q2", "Q3", "Q4 (largest)"]),
                  text_seen=np.where(unseen, "unseen text", "seen text"))
rows = []
for col in ("trade_method", "value_quartile", "text_seen"):
    for level, g in seg.groupby(col, observed=True):
        if g[y].sum() < 30: continue
        i = g.index.to_numpy(); rate = g[y].mean()
        rows.append({"segment": f"{col} = {level}", "procurements": len(g), "failure rate": rate,
                     "lift, baseline": average_precision_score(g[y], base[i]) / rate,
                     "lift, final": average_precision_score(g[y], fin_te[i]) / rate,
                     "ROC-AUC, baseline": roc_auc_score(g[y], base[i]), "ROC-AUC, final": roc_auc_score(g[y], fin_te[i])})
segments = pd.DataFrame(rows).set_index("segment"); segments.to_csv(RES / "segments.csv"); segments
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 7. Hypotheses: expected vs observed

The hypotheses were written in the research plan before the notebooks were run (pilot results in brackets).
""")
code(r"""
t = table["test PR-AUC"]
g = lambda name: t.xs(name, level="model").iloc[0]
H = pd.DataFrame([
    {"hypothesis": "H4 · the model family matters more than hyper-parameter tuning",
     "evidence": f"baseline {g('Logistic regression · midterm baseline'):.3f} → LightGBM default {g('LightGBM · default'):.3f} → tuned {g('LightGBM · tuned'):.3f}",
     "verdict": "supported: changing the family gives +0.04–0.05; tuning gives 0 to +0.04 depending on how far the defaults are from the optimum"},
    {"hypothesis": "H5 · new information matters more than tuning",
     "evidence": f"same CatBoost configuration: 20 features {g('CatBoost · tuned'):.3f} → + text, history, price {g('Final model'):.3f}; "
                 f"tuning on the 20 features: {g('CatBoost · default'):.3f} → {g('CatBoost · tuned'):.3f}",
     "verdict": "supported: new information (+0.053) is worth twice the tuning gain (+0.023) of the same model"},
    {"hypothesis": "H6 · text is useful together with the table, not alone",
     "evidence": f"best text-only model {table.xs('2 · text alone (05)')['test PR-AUC'].max():.3f} vs table + text {table.xs('3 · table + text (05)')['test PR-AUC'].max():.3f}",
     "verdict": "supported: text alone is half as good as the table, but it is the largest addition to it"},
    {"hypothesis": "H7 · fine-tuning makes text a stronger signal than TF-IDF",
     "evidence": f"TF-IDF {g('TF-IDF + logistic regression'):.3f} vs fine-tuned {g('fine-tuned rubert-tiny2'):.3f}; unseen texts "
                 f"{table.xs('TF-IDF + logistic regression', level='model')['PR-AUC unseen texts'].iloc[0]:.3f} vs "
                 f"{table.xs('fine-tuned rubert-tiny2', level='model')['PR-AUC unseen texts'].iloc[0]:.3f}",
     "verdict": "partly: a tie overall, a clear gain on unseen texts; as a feature inside the table frozen components are better"},
    {"hypothesis": "H8 · the LLM is below boosting on the full train set but helps in an ensemble",
     "evidence": f"LLM {g('Qwen2.5-1.5B + QLoRA, 40 k rows'):.3f}; final model {g('Final model'):.3f}; final + LLM {g('Final model + LLM ensemble'):.3f} "
                 f"({boot_diff(ens_te, fin_te)[0]:+.3f} [{boot_diff(ens_te, fin_te)[1]:+.3f}, {boot_diff(ens_te, fin_te)[2]:+.3f}])",
     "verdict": "supported: weaker alone, a small but significant gain in an ensemble"},
    {"hypothesis": "H9 · calibration fixes the probabilities without changing the ranking",
     "evidence": f"final model Brier {brier_score_loss(y_te, fin_te):.4f} → {brier_score_loss(y_te, fin_te_cal):.4f}; "
                 f"PR-AUC {average_precision_score(y_te, fin_te):.3f} → {average_precision_score(y_te, fin_te_cal):.3f}",
     "verdict": "supported (notebook 04: baseline Brier 0.200 → 0.055); the final model is trained without class weights and is already calibrated"},
]).set_index("hypothesis")
with pd.option_context("display.max_colwidth", 200):
    display(H)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 8. Reproducibility

* **Data**: the raw snapshot (OWS API, procurements published in Q1 2024) and the model table are published as the
  GitHub release `data-v1`; every notebook downloads what it needs. The collector `src/collect.py` re-creates the
  snapshot with an API token.
* **Code**: features in `src/prepare.py` and `src/features.py`; notebooks are generated from `notebooks/build_*.py`;
  the LLM script is `kaggle/06_llm_qlora/train_qlora.py`.
* **Randomness**: seed 42 everywhere (fine-tuning also with seeds 43, 44); deterministic tie-breaking in the winner
  definition. GPU training (05a, 06) is not bit-exact across hardware; differences appear in the third decimal.
* **Saved settings and outputs**: Optuna studies (`results/a3_04/optuna.db`), chosen configurations
  (`results/a3_04/best_params.json`, `results/a3_05/finetune_choice.json`), LLM training log and scores
  (`results/a3_06/`), the final model (`results/a3_07/final_lightgbm.txt`) and all tables as CSV.
""")
code(r"""
libs = ["numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "optuna", "duckdb", "torch", "transformers", "peft"]
env = {"python": platform.python_version(), "platform": platform.platform()}
for l in libs:
    try: env[l] = version(l)
    except Exception: env[l] = "not installed"
kaggle_log = json.loads((ROOT / "results" / "a3_06" / "train_log.json").read_text())
env["LLM run"] = f"Kaggle, {kaggle_log['gpu']}"
(RES / "environment.json").write_text(json.dumps(env, indent=2)); pd.Series(env).to_frame("version")
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
## 9. Conclusions and limitations

1. **The final model.** CatBoost (hyper-parameters tuned in notebook 04) on 61 features — the 20 midterm features,
   32 components of the lot text, 9 leak-free customer–supplier and price features — reaches test PR-AUC **0.364**
   (95 % CI [0.343, 0.385]) against **0.262** for the midterm baseline, ROC-AUC 0.827 against 0.792, and its
   probabilities are calibrated (mean 0.066 = the real failure rate). For an auditor who can check 1 % of
   procurements — about 390 in this fortnight — that means **≈ 280 failed contracts found instead of ≈ 196** with the
   midterm model and ≈ 26 by random choice.
2. **Where the gain comes from.** Two roughly equal steps: the model family (linear → tuned CatBoost on the same 20
   features, +0.049) and new information (the same CatBoost with text, history and price, +0.053). Hyper-parameter
   tuning on top of a good model has reached a plateau (LightGBM: +0.001; re-tuning LightGBM on the 61 features did not
   even reach the simple configuration on validation). The ablations show that almost all of the new information is the **lot text** (−0.025 without
   it); the customer–supplier history and the price features, although individually predictive, add nothing once the
   text is present.
3. **Text: transfer learning vs fine-tuning vs LLM.** Frozen encoder vectors are weak alone (below TF-IDF) but the
   best way to give the text to the tabular model. Fine-tuning produces the best text-only model on names never seen
   in training (0.228 vs 0.200 for TF-IDF). The QLoRA-tuned 1.5 B LLM reading the whole procurement card is above every
   model on the 20 features (0.322) but below boosting with the same information; averaged with the final model it
   adds a small, statistically significant +0.004 (0.368) — at a cost of 3.5 GPU-hours.
4. **The leakage lesson.** A feature family that looked like the third most important source of information turned
   out to be a leak through the *link* between records (the contract signer as "the winner"). It was found because
   the gain looked too large and the top of the ranked list was inspected; with a leak-free definition of the winner
   the gain is zero.
5. **Where the model helps most.** By segment, the largest improvements over the baseline are for single-source
   purchases after a failed tender (lift 2.2 → 8.0) and for the cheapest quarter of procurements (lift 2.3 → 3.6) —
   the midterm model's weakest segment. Tenders with a scoring system remain hard (lift ≈ 1.2, only 624 test
   procurements).

Limitations that apply to every number in this notebook:

* **One quarter, one test fortnight.** The models are trained on January–February 2024 and tested on the second half
  of March. End-of-year procurement (December), with its rush to spend budgets, may fail more often and for other
  reasons; the model has not seen it.
* **The label is the final status recorded on the portal.** Terminations "by mutual agreement" include harmless
  cases (a strict label without them is kept in the data as `contract_failed_strict`).
* **Information known at award time only.** This rules out the strongest signals Tender Radar used (contract
  execution, payments, fines): they are the outcome itself.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = Path(__file__).with_name("07_results.ipynb")
nbf.write(nb, out); print("written", out)
