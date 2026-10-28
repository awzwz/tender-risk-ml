"""
A3 criterion 5 — QLoRA fine-tuning of Qwen2.5-1.5B as a classifier of procurement cards. Runs on Kaggle (T4, 16 GB);
pushed and collected from notebooks/06_llm_qlora.ipynb with the Kaggle API.

Input  /kaggle/input/<dataset>/cards.parquet : id, split (train / val / test), text (Russian card), label
Output /kaggle/working/llm_predictions.parquet (id, split, score), train_log.json, adapter/ (LoRA weights, ~70 MB)

Configuration (justified in reports/A3_research_ru.md §6.6, speed measured on Kaggle on 27 Oct 2026):
4-bit NF4 base with double quantisation, fp16 compute (T4 has no bf16 tensor cores); LoRA r=16, alpha=32,
dropout 0.05 on all seven linear projections; the new classification head is trained in full (modules_to_save);
paged 8-bit AdamW, lr 2e-4, cosine decay with 3 % warm-up, effective batch 64 (16 × 4 accumulation), 1 epoch,
gradient clipping 1.0, dynamic padding up to 224 tokens (lot names come last, so only they are ever cut).
"""
import json, math, subprocess, sys, time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes==0.50.2", "peft==0.20.0"], check=True)

import numpy as np
import pandas as pd
import torch
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

SEED, NAME, MAX_LEN = 42, "Qwen/Qwen2.5-1.5B", 224         # cards: median 164 tokens, 90 % ≤ 194
BATCH, ACCUM, LR, WARMUP, EPOCHS = 16, 4, 2e-4, 0.03, 1
TIME_LIMIT_H = 9.0                                         # stop training early rather than lose the 12-hour session
OUT = Path("/kaggle/working"); torch.manual_seed(SEED); np.random.seed(SEED)
log = {"config": dict(model=NAME, max_len=MAX_LEN, batch=BATCH, accumulation=ACCUM, lr=LR, warmup=WARMUP, epochs=EPOCHS,
                      lora=dict(r=16, alpha=32, dropout=0.05), quantization="nf4 + double quant, fp16 compute"),
       "gpu": torch.cuda.get_device_name(0), "steps": []}

cards = pd.read_parquet(next(Path("/kaggle/input").rglob("cards.parquet")))
tr = cards[cards.split == "train"].sample(frac=1, random_state=SEED).reset_index(drop=True)
ev = cards[cards.split != "train"].reset_index(drop=True)
log["rows"] = {"train": len(tr), "val+test": len(ev), "train failure share": float(tr.label.mean())}

tok = AutoTokenizer.from_pretrained(NAME)
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                       bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2, quantization_config=q, device_map={"": 0})
model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS", modules_to_save=["score"],
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
log["parameters"] = {"trainable": trainable, "total (4-bit weights counted as stored)": total}
print(log["parameters"], flush=True)

def batches(texts, size):
    for i in range(0, len(texts), size):
        yield i, tok(list(texts[i:i + size]), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")

params = [p for p in model.parameters() if p.requires_grad]
opt = bnb.optim.PagedAdamW8bit(params, lr=LR, weight_decay=0.0)
steps = EPOCHS * math.ceil(len(tr) / (BATCH * ACCUM))
sched = get_cosine_schedule_with_warmup(opt, int(WARMUP * steps), steps)
scaler = torch.amp.GradScaler("cuda")
labels = torch.tensor(tr.label.to_numpy(), dtype=torch.long)
t0, step, run_loss = time.time(), 0, []
model.train()
for i, b in batches(tr.text.to_numpy(), BATCH):
    b = {k: v.cuda() for k, v in b.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(**b, labels=labels[i:i + BATCH].cuda()).loss / ACCUM
    scaler.scale(loss).backward(); run_loss.append(loss.item() * ACCUM)
    if (i // BATCH + 1) % ACCUM == 0:
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); step += 1
        if step % 25 == 0:
            rec = {"step": step, "of": steps, "loss": float(np.mean(run_loss)), "lr": sched.get_last_lr()[0],
                   "hours": (time.time() - t0) / 3600}
            log["steps"].append(rec); run_loss = []; print(rec, flush=True)
        if step % 200 == 0:
            model.save_pretrained(OUT / "adapter")
        if (time.time() - t0) / 3600 > TIME_LIMIT_H:
            log["stopped_early_at_step"] = step; break
log["train_hours"] = (time.time() - t0) / 3600
model.save_pretrained(OUT / "adapter")

# inference on validation + test, sorted by length so that padding stays small
model.eval(); order = ev.text.str.len().sort_values().index.to_numpy(); scores = np.empty(len(ev)); t1 = time.time()
texts = ev.text.to_numpy()[order]
with torch.no_grad():
    for i, b in batches(texts, 64):
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(model(**b).logits.float(), -1)[:, 1]
        scores[order[i:i + 64]] = p.cpu().numpy()
log["inference_hours"] = (time.time() - t1) / 3600
pd.DataFrame({"id": ev.id, "split": ev.split, "score": scores}).to_parquet(OUT / "llm_predictions.parquet", index=False)
(OUT / "train_log.json").write_text(json.dumps(log, indent=2, default=float))
print("done", {k: log[k] for k in ("train_hours", "inference_hours")}, flush=True)
