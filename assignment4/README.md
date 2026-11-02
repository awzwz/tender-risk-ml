# Assignment 4 — Course practicals on new datasets

The practical exercises of the course, repeated on three open datasets that were not used in class. Each notebook
contains the working code, its executed output and short conclusions. Russian reading copies: `*_ru.ipynb`.

| # | Practical | Dataset | Notebook | Main result (test) |
|---|---|---|---|---|
| 1 | Data preparation and EDA | Hotel Booking Demand | `P1_data_preparation.ipynb` | 5 leaking columns found and removed; 119,208 bookings × 34 features, time-based split |
| 2 | Baseline and metrics | Hotel Booking Demand | `P2_baseline_metrics.ipynb` | logistic regression ROC-AUC 0.849, PR-AUC 0.806; random CV overstates (0.907 vs 0.849) |
| 3 | Model comparison and hyperparameter tuning | Hotel Booking Demand | `P3_model_comparison_tuning.ipynb` | LightGBM PR-AUC 0.817 vs 0.806; tuning matters only for the decision tree (+0.222) |
| 4 | Transfer learning | Beans (leaf photos) | `P4_transfer_learning.ipynb` | frozen EfficientNet-B0 + linear 91.4 %; with 10 % of data frozen ResNet-18 80.5 % vs scratch CNN 71.1 % |
| 5 | Fine-tuning | Beans (leaf photos) | `P5_fine_tuning.ipynb` | fine-tuned ResNet-18 96.9 % vs 89.8 % frozen |
| 6 | LLM + LoRA | Twitter Financial News | `P6_llm_lora.ipynb` | LoRA on Qwen2.5-0.5B: macro-F1 0.869 vs 0.715 frozen + linear and 0.336 zero-shot |

## Datasets

| Dataset | Size | Task | Source, licence |
|---|---|---|---|
| Hotel Booking Demand | 119,390 bookings, 32 columns, 2015–2017 | binary: will the booking be cancelled | Antonio, Almeida & Nunes, *Data in Brief* 22 (2019), CC BY 4.0; CSV from the TidyTuesday repository |
| Beans | 1,295 leaf photos, 3 classes | image classification: angular leaf spot / bean rust / healthy | Makerere AI Lab (2020), MIT; Hugging Face `AI-Lab-Makerere/beans` |
| Twitter Financial News Sentiment | 11,931 tweets, 3 classes | text classification: bearish / bullish / neutral | zeroshot (2022), MIT; Hugging Face `zeroshot/twitter-financial-news-sentiment` |

Every notebook downloads its data into `data/` on the first run.

## How to run

Environment: `requirements.txt` of the repository plus `torchvision`. Run from this folder.

* **P1 → P2 → P3** (CPU, a few minutes each): P1 writes `data/hotels_clean.parquet`, which P2 and P3 read.
* **P4 → P5** (GPU; ~10 min each on Apple MPS and heavy for a laptop — a Kaggle GPU is the better place): P5 reads
  `results/p4_summary.json` for its comparison table.
* **P6** needs a GPU: `python kaggle_p6.py && kaggle kernels push -p kaggle/a4_p6` runs it on a free Kaggle T4; the
  executed notebook comes back with `kaggle kernels output azizabdulin/a4-p6-llm-lora -p kaggle/a4_p6/output`.

Notebooks are generated from `build_p*.py` (markdown in English and Russian, shared code); `finalize.py` updates the
markdown of executed notebooks and writes the Russian copies without re-running them. P4–P6 use PyTorch and do not
import scikit-learn (two OpenMP runtimes crash on macOS); their metrics come from `a4_metrics.py`, checked against
scikit-learn in `tests/test_a4_metrics.py`.
