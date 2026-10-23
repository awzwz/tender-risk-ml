# A3 pilot experiments

Short experiments that answer one question each before the Assignment 3 notebooks are built. All use the A2 time
split (train Jan–Feb, validation 1–15 Mar, test 16–31 Mar 2024), seed 42 and test PR-AUC with a paired bootstrap
(1,000 resamples) for differences. Results are written to `<script>_results.json` next to each script; the write-up
is in [reports/A3_research_ru.md](../../reports/A3_research_ru.md).

| Script | Question | Key result (test PR-AUC) |
|---|---|---|
| `p1_tabular.py` | model family vs tuning vs training size | LR 0.262 → LightGBM 0.311; Optuna +0.003; learning curve flat after 50 % of train |
| `p1b_new_info.py` | what else is in applications and contracts; leakage screen | status counts +0.032 but post-award (excluded); rare statuses leak |
| `p1c_conservative.py` | the same with bid-time features only | winner's 30-day activity +0.025; price gaps and late bids add nothing |
| `p2a_embed.py` | frozen `rubert-tiny2` embeddings; fine-tuning speed on Apple M4 | 61k texts in 19 s; fine-tuning 742 samples/s (1.8 min/epoch) |
| `p2_text.py` | TF-IDF vs frozen embeddings, seen vs unseen texts | text alone 0.15 (TF-IDF) / 0.13 (frozen); 32 PCA components in LightGBM +0.036 |
| `p3b_llm_realistic.py` | LoRA speed of `Qwen2.5-0.5B` on Apple M4 | 3.6 samples/s training, 13.8 samples/s inference; 8.8M trainable parameters |
| `p4_combined.py` | do winner and text features add up; does the January history gap matter | F20 + winner + text 0.355 (+0.044); training on February only gains +0.008 with winner features |

Run order: `p2a_embed.py` before `p2_text.py` and `p4_combined.py` (they read the saved embeddings). Each script
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
