"""
A3 pilot 1 — tabular models. Answers four planning questions with real numbers:
  1. How much does LightGBM out of the box beat the midterm baseline?
  2. How much headroom does a short Optuna search show?
  3. Learning curve: would more training data (another quarter) still help?
  4. How noisy are test metrics (bootstrap CIs, paired difference)?
Run: python experiments/a3_pilots/p1_tabular.py   (writes experiments/a3_pilots/p1_results.json)
"""
import json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import prepare as P

warnings.filterwarnings("ignore"); optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42; rng = np.random.default_rng(SEED); out = {}

df = P.load_model_table(ROOT)
for c in P.CAT_FEATURES:
    df[c] = df[c].astype("category")
tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
X, y = P.FEATURES, P.TARGET

def scores(p_val, p_test):
    return {"val_roc": roc_auc_score(va[y], p_val), "val_pr": average_precision_score(va[y], p_val),
            "test_roc": roc_auc_score(te[y], p_test), "test_pr": average_precision_score(te[y], p_test)}

# 0 — midterm baseline (same pipeline as notebooks/03)
pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), P.NUM_FEATURES),
                         ("bin", "passthrough", P.BIN_FEATURES),
                         ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), P.CAT_FEATURES)])
lr = make_pipeline(pre, LogisticRegression(class_weight="balanced", max_iter=2000)).fit(tr[X], tr[y])
p_lr_val, p_lr_test = lr.predict_proba(va[X])[:, 1], lr.predict_proba(te[X])[:, 1]
out["logreg_baseline"] = scores(p_lr_val, p_lr_test)

# 1 — LightGBM out of the box (library defaults) and with early stopping
t0 = time.time(); m = lgb.LGBMClassifier(random_state=SEED, verbose=-1).fit(tr[X], tr[y])
out["lgbm_defaults"] = {**scores(m.predict_proba(va[X])[:, 1], m.predict_proba(te[X])[:, 1]), "fit_s": time.time() - t0}

def fit_es(params, train_part):
    m = lgb.LGBMClassifier(n_estimators=5000, random_state=SEED, verbose=-1, **params)
    m.fit(train_part[X], train_part[y], eval_set=[(va[X], va[y])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m
t0 = time.time(); m_es = fit_es({"learning_rate": 0.05}, tr)
p_es_val, p_es_test = m_es.predict_proba(va[X])[:, 1], m_es.predict_proba(te[X])[:, 1]
out["lgbm_lr005_earlystop"] = {**scores(p_es_val, p_es_test), "fit_s": time.time() - t0, "best_iter": int(m_es.best_iteration_)}

# 2 — short Optuna search (25 trials) to gauge tuning headroom
def objective(trial):
    params = {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
              "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
              "min_child_samples": trial.suggest_int("min_child_samples", 20, 1000, log=True),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
              "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
              "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
              "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 14)}
    m = fit_es(params, tr)
    return average_precision_score(va[y], m.predict_proba(va[X])[:, 1])
t0 = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=25)
m_opt = fit_es({**study.best_params, "subsample_freq": 1}, tr)
p_opt_val, p_opt_test = m_opt.predict_proba(va[X])[:, 1], m_opt.predict_proba(te[X])[:, 1]
out["lgbm_optuna25"] = {**scores(p_opt_val, p_opt_test), "search_s": time.time() - t0, "best_params": study.best_params,
                        "val_pr_by_trial": [round(t.value, 4) for t in study.trials]}

# 3 — learning curve: random fractions of train, fixed params, early stopping on val
lc = []
for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
    part = tr.sample(frac=frac, random_state=SEED) if frac < 1 else tr
    m = fit_es({"learning_rate": 0.05}, part)
    lc.append({"frac": frac, "rows": len(part), "val_pr": average_precision_score(va[y], m.predict_proba(va[X])[:, 1]),
               "val_roc": roc_auc_score(va[y], m.predict_proba(va[X])[:, 1])})
out["learning_curve"] = lc

# 4 — bootstrap on test: CIs and paired difference (LightGBM-Optuna minus baseline)
B = 1000; yt = te[y].to_numpy(); n = len(yt); boot = {"lr": [], "lgbm": [], "diff": []}
for _ in range(B):
    i = rng.integers(0, n, n)
    a, b = average_precision_score(yt[i], p_lr_test[i]), average_precision_score(yt[i], p_opt_test[i])
    boot["lr"].append(a); boot["lgbm"].append(b); boot["diff"].append(b - a)
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
out["bootstrap_test_pr"] = {"logreg_ci95": ci(boot["lr"]), "lgbm_ci95": ci(boot["lgbm"]), "diff_ci95": ci(boot["diff"]),
                            "share_diff_le_0": float(np.mean(np.array(boot["diff"]) <= 0)), "B": B}

(Path(__file__).with_name("p1_results.json")).write_text(json.dumps(out, indent=2, default=float))
for k, v in out.items():
    if isinstance(v, dict): print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("val_pr_by_trial",)})
    else: print(k, [{kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()} for r in v])
