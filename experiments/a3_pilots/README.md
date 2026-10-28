# A3 pilot experiments

Short experiments that answer one question each before the Assignment 3 notebooks are built. All use the A2 time
split (train Jan–Feb, validation 1–15 Mar, test 16–31 Mar 2024), seed 42 and test PR-AUC with a paired bootstrap
(1,000 resamples) for differences. Results are written to `<script>_results.json` next to each script; the write-up
is in [reports/A3_research_ru.md](../../reports/A3_research_ru.md).

| Script | Question | Key result (test PR-AUC) |
|---|---|---|
| `p1_tabular.py` | model family vs tuning vs training size | LR 0.262 → LightGBM 0.311; Optuna +0.003; learning curve flat after 50 % of train |
| `p1b_new_info.py` | what else is in applications and contracts; leakage screen | status counts +0.032 but post-award (excluded); rare statuses leak |
| `p1c_conservative.py` | the same with bid-time features only | winner's 30-day activity +0.025 — **superseded by p5**: the winner came from contracts (leak) |
| `p2a_embed.py` | frozen `rubert-tiny2` embeddings; fine-tuning speed on Apple M4 | 61k texts in 19 s; fine-tuning 742 samples/s (1.8 min/epoch) |
| `p2_text.py` | TF-IDF vs frozen embeddings, seen vs unseen texts | text alone 0.15 (TF-IDF) / 0.13 (frozen); 32 PCA components in LightGBM +0.036 |
| `p3b_llm_realistic.py` | LoRA speed of `Qwen2.5-0.5B` on Apple M4 | 3.6 samples/s training, 13.8 samples/s inference; 8.8M trainable parameters |
| `p4_combined.py` | do winner and text features add up; does the January history gap matter | F20 + winner + text 0.355 (+0.044) — **superseded by p5**: winner features leak through the contracts table |
| `winner.py` | helper: winner and customer without contracts (announced supplier, else lowest discounted bid) | contract signer did not bid in 27 % of successful vs 63 % of failed procurements, so it cannot define the winner |
| `p5_indicators.py` | Tender Radar's rule indicators re-computed as of publication; audit of F20 | honest winner features −0.004; customer–supplier history +0.012, price vs market +0.008, bid structure +0.005, publication time −0.003; best set F20 + text 0.347, + history + price 0.351 |
| `p6_companies.py` | company register (age, legal form, sector), shared contacts and bank details | company attributes −0.002, shared contacts −0.013, shared bank account / address 0.000 (too rare) |
| `p7_plans.py` | annual plans: the plan version in force at publication (age, advance payment, planned vs actual method, month, funding) | F20 + plan +0.004 (CI [0.001, 0.008]), best set + plan +0.002 (CI incl. 0). Missingness leak: 3.2 % of lots are linked only to versions created later (median 29 days); those procurements fail 15–17 % vs 6.3 %, and keeping the gaps inflates the gain ~2× |

Run order: `p2a_embed.py` before `p2_text.py`, `p4_combined.py`, `p5_indicators.py` (they read the saved embeddings);
`p5` before `p6` (p6 reads `data/processed/p5_features.parquet`). `p6` needs `python -m src.collect --stages
subjects,contract_bank` (contacts and bank accounts are stored only as keyed hashes), `p7` needs
`python -m src.collect --stages lot_plans` and runs after `p5`. Each script
runs in its own process: torch and scikit-learn/LightGBM bundle different copies of libomp, and loading both in one
process segfaults on macOS.

```bash
python experiments/a3_pilots/p1_tabular.py
python experiments/a3_pilots/p2a_embed.py
python experiments/a3_pilots/p4_combined.py
```

`p1b`, `p1c` and `p4` also read the raw applications and contracts (`data/raw/trd_app.parquet`,
`data/raw/contracts.parquet`); download them from the `data-v1` release if they are missing. The other scripts need
only the model table, which `src.prepare.load_model_table` fetches by itself.
