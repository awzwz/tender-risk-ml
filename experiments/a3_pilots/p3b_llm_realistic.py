"""
A3 pilot 3b — realistic local LoRA run for Qwen2.5-0.5B on the Mac GPU (MPS).
Settings as for real training: bfloat16 weights, gradient checkpointing, LoRA r=16 on all linear layers,
sequence length 160, batch 8. Times training steps and inference on real procurement "cards" built from the
modelling table, then extrapolates to the planned 40k-example run and to scoring validation + test.
Writes p3b_results.json.
"""
import json, time
from pathlib import Path
import sys

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import prepare as P

torch.manual_seed(42)
DEV, NAME, SEQ, BS = "mps", "Qwen/Qwen2.5-0.5B", 160, 8
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def card(r):
    """Serialise one procurement into the short Russian text the LLM reads."""
    ratio = "нет данных" if np.isnan(r.best_bid_ratio) else f"{r.best_bid_ratio:.0%}"
    return (f"Способ: {r.trade_method}. Предмет: {r.subject_type}. Регион: {r.region}. "
            f"Сумма: {10 ** r.log_total_sum:,.0f} тенге, лотов: {int(r.n_lots)}. "
            f"Участников: {int(r.n_bidders)}, лучшая цена к плановой: {ratio}. "
            f"Приём заявок: {r.bid_window_days:.0f} дн. Лоты: {str(r.lot_names)[:200]}")


df = P.load_model_table(ROOT)
sample = df[df.split == "train"].sample(64, random_state=42)
texts = [card(r) for r in sample.itertuples()]
labels = sample[P.TARGET].to_numpy().astype("int64")  # target is int32; cross-entropy needs int64

tok = AutoTokenizer.from_pretrained(NAME)
lengths = [len(tok(t).input_ids) for t in texts]
out = {"model": NAME, "seq_len": SEQ, "batch": BS, "card_example": texts[0],
       "card_tokens": {"median": float(np.median(lengths)), "max": int(max(lengths))}}

model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2, dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=TARGETS, task_type="SEQ_CLS")
model = get_peft_model(model, lora)
out["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
out["total_params"] = sum(p.numel() for p in model.parameters())
model.to(DEV)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)


def make_batch(i):
    enc = tok(texts[i:i + BS], padding="max_length", truncation=True, max_length=SEQ, return_tensors="pt")
    return enc.to(DEV), torch.tensor(labels[i:i + BS], device=DEV)


def sync():
    torch.mps.synchronize()


model.train()
step_times = []
for step in range(10):
    enc, y = make_batch((step * BS) % (len(texts) - BS))
    t0 = time.time()
    loss = model(**enc, labels=y).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    sync()
    if step >= 2:  # first two steps are warm-up (kernel compilation)
        step_times.append(time.time() - t0)
out["train_sec_per_step"] = float(np.median(step_times))
out["train_examples_per_sec"] = BS / out["train_sec_per_step"]
out["peak_mps_mem_gb"] = torch.mps.driver_allocated_memory() / 1e9

model.eval()
infer_bs, infer_times = 32, []
with torch.no_grad():
    for rep in range(4):
        enc = tok(texts[:infer_bs], padding="max_length", truncation=True, max_length=SEQ,
                  return_tensors="pt").to(DEV)
        t0 = time.time()
        model(**enc)
        sync()
        if rep >= 1:
            infer_times.append(time.time() - t0)
out["infer_examples_per_sec"] = infer_bs / float(np.median(infer_times))

# Extrapolate: one epoch over a 40k balanced training sample, and scoring validation + test (92,495 rows)
out["hours_per_epoch_40k"] = 40_000 / out["train_examples_per_sec"] / 3600
out["hours_to_score_val_test"] = 92_495 / out["infer_examples_per_sec"] / 3600

Path(__file__).with_name("p3b_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in out.items() if k != "card_example"}, indent=2))
