"""
A3 pilot 2a — torch half of the text pilot (run before p2_text.py).
  * frozen rubert-tiny2 CLS embeddings for the 60.9k unique lot-name texts -> data/processed/rubert_tiny2_*.{npy,parquet}
  * times fine-tuning steps of rubert-tiny2 on the Mac GPU (MPS) to size the real fine-tuning run
Kept in its own process: torch and scikit-learn/LightGBM each bundle libomp, and loading both in one process
segfaults on macOS.
"""
import json, time
from pathlib import Path
import sys

import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src import prepare as P
SEED = 42; torch.manual_seed(SEED); out = {}
DEV, NAME = "mps", "cointegrated/rubert-tiny2"

df = P.load_model_table(ROOT)
df["text"] = df.lot_names.fillna("").str.lower().str.strip()
tr = df[df.split == "train"]; y = P.TARGET

tok = AutoTokenizer.from_pretrained(NAME); enc = AutoModel.from_pretrained(NAME).to(DEV).eval()
uniq = pd.Series(df.text.unique()); t0 = time.time(); embs = []
with torch.no_grad():
    for i in range(0, len(uniq), 512):
        b = tok(list(uniq[i:i + 512]), padding=True, truncation=True, max_length=64, return_tensors="pt").to(DEV)
        e = enc(**b).last_hidden_state[:, 0]
        embs.append(torch.nn.functional.normalize(e, dim=-1).cpu())
E = torch.cat(embs).numpy()
out["embed_unique_texts"] = {"n": len(uniq), "seconds": time.time() - t0, "dim": E.shape[1]}
np.save(ROOT / "data/processed/rubert_tiny2_cls.npy", E)
uniq.to_frame("text").to_parquet(ROOT / "data/processed/rubert_tiny2_texts.parquet")
print("embeddings", out["embed_unique_texts"], flush=True)

# time fine-tuning steps (batch 64, length 64): 3 warm-up steps, then 10 timed
ft = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2).to(DEV).train()
opt = torch.optim.AdamW(ft.parameters(), lr=5e-5, weight_decay=0.01)
b = tok(list(tr.text.iloc[:64]), padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(DEV)
lab = torch.tensor(tr[y].iloc[:64].to_numpy(), dtype=torch.long, device=DEV)
for i in range(13):
    if i == 3: torch.mps.synchronize(); t0 = time.time()
    loss = ft(**b, labels=lab).loss; loss.backward(); opt.step(); opt.zero_grad()
torch.mps.synchronize(); s = (time.time() - t0) / 10
out["finetune_speed"] = {"s_per_step_bs64": s, "samples_per_s": 64 / s, "minutes_per_epoch_78584": 78_584 / (64 / s) / 60,
                         "params": sum(p.numel() for p in ft.parameters())}
Path(__file__).with_name("p2a_results.json").write_text(json.dumps(out, indent=2, default=float))
print("finetune", out["finetune_speed"], flush=True)
