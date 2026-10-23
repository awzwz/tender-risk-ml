"""
A3 pilot 2 — text. Does a pretrained Russian encoder understand UNSEEN lot names better than TF-IDF?
  * TF-IDF (word 1-2 grams + char 3-5 grams) + logistic regression        -> no transfer
  * frozen rubert-tiny2 CLS embeddings + logistic regression               -> transfer learning (feature extraction)
  * LightGBM on tabular features + 32 PCA components of the embeddings     -> does text add to the tabular model?
Every model is scored on the whole test split and on the 8.7k test procurements whose text never occurs in train.
Embeddings come from p2a_embed.py (run it first; torch is kept out of this process, see there).
"""
import gc, json, warnings
from pathlib import Path
import sys

import lightgbm as lgb, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import prepare as P
warnings.filterwarnings("ignore"); SEED = 42; out = {}

df = P.load_model_table(ROOT)
df["text"] = df.lot_names.fillna("").str.lower().str.strip()
tr, va, te = (df[df.split == s].copy() for s in ("train", "val", "test"))
y = P.TARGET
unseen = ~te.text.isin(set(tr.text))
out["test_unseen_rows"] = int(unseen.sum()); out["test_unseen_fail_rate"] = float(te.loc[unseen, y].mean())

def score(name, p_val, p_test):
    out[name] = {"val_pr": average_precision_score(va[y], p_val), "test_pr": average_precision_score(te[y], p_test),
                 "test_roc": roc_auc_score(te[y], p_test),
                 "test_pr_seen": average_precision_score(te.loc[~unseen, y], p_test[~unseen.to_numpy()]),
                 "test_pr_unseen": average_precision_score(te.loc[unseen, y], p_test[unseen.to_numpy()]),
                 "test_roc_unseen": roc_auc_score(te.loc[unseen, y], p_test[unseen.to_numpy()])}
    print(name, {k: round(v, 4) for k, v in out[name].items()}, flush=True)

# 1 — TF-IDF (no transfer)
w = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=50_000)
c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, max_features=100_000)
Xtr = hstack([w.fit_transform(tr.text), c.fit_transform(tr.text)]).tocsr()
Xva, Xte = (hstack([w.transform(d.text), c.transform(d.text)]).tocsr() for d in (va, te))
m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(Xtr, tr[y])
score("text_tfidf_logreg", m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1])
del Xtr, Xva, Xte, w, c, m; gc.collect()

# 2 — frozen rubert-tiny2 embeddings (computed once per unique text by p2a_embed.py)
E = np.load(ROOT / "data/processed/rubert_tiny2_cls.npy")
uniq = pd.read_parquet(ROOT / "data/processed/rubert_tiny2_texts.parquet").text
pos = pd.Series(np.arange(len(uniq)), index=uniq.values)
emb = lambda d: E[pos[d.text].to_numpy()]
m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(emb(tr), tr[y])
score("text_rubert_frozen_logreg", m.predict_proba(emb(va))[:, 1], m.predict_proba(emb(te))[:, 1])

# 3 — tabular LightGBM with and without 32 PCA components of the embeddings (PCA fitted on train texts only)
pca = PCA(32, random_state=SEED).fit(emb(tr))
for d in (tr, va, te):
    Z = pca.transform(emb(d))
    for k in range(32): d[f"emb{k}"] = Z[:, k]
    for col in P.CAT_FEATURES: d[col] = d[col].astype(pd.CategoricalDtype(sorted(df[col].unique())))
def lgbm(cols, name):
    mm = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=SEED, verbose=-1)
    mm.fit(tr[cols], tr[y], eval_set=[(va[cols], va[y])], eval_metric="average_precision", callbacks=[lgb.early_stopping(100, verbose=False)])
    score(name, mm.predict_proba(va[cols])[:, 1], mm.predict_proba(te[cols])[:, 1])
lgbm(P.FEATURES, "lgbm_tabular")
lgbm(P.FEATURES + [f"emb{k}" for k in range(32)], "lgbm_tabular_plus_rubert_pca32")

Path(__file__).with_name("p2_results.json").write_text(json.dumps(out, indent=2, default=float))
