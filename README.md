# tender-risk-ml

Predicting failed public-procurement contracts in Kazakhstan (goszakup.gov.kz) — a semester ML project.
Русская версия этого файла: [README_ru.md](README_ru.md).

Course pipeline: **A1 Problem & Dataset → A2 Data Preparation & Baseline → Midterm → A3 Model Training & Fine-Tuning.**

## Problem in one paragraph

Kazakhstan's public procurement portal publishes ~1.8M procurements per year through the OWS v3 GraphQL API.
Manual audit is impossible at that scale. This project builds a **binary classifier** that, from information
available at the moment the winner is selected (tender attributes, lots, bidding pattern), predicts whether the
procurement will end in a **failed contract** — one that is terminated or never concluded. Rule-based risk indicators
from an earlier engineering project (Tender Radar) are treated as candidate features, not as labels, so the model is
trained against a real observed outcome.

## Results (test period, 16–31 March 2024)

| Model | PR-AUC [95 % CI] | ROC-AUC | failures among the top 1 % |
|---|---|---|---|
| No model (failure rate) | 0.066 | 0.500 | 6 % |
| Midterm baseline: logistic regression, 20 features | 0.262 [0.246, 0.280] | 0.792 | 50 % |
| Tuned CatBoost, 20 features (A3, notebook 04) | 0.311 [0.293, 0.331] | 0.807 | 57 % |
| Qwen2.5-1.5B fine-tuned with QLoRA (A3, notebook 06) | 0.322 [0.302, 0.342] | 0.808 | 59 % |
| **Final model: CatBoost, 61 features incl. lot text (A3, notebook 07)** | **0.364 [0.343, 0.385]** | **0.827** | **72 %** |

At an audit budget of 1 % of procurements the final model finds ≈ 280 failed contracts per fortnight against ≈ 196 for
the midterm baseline. About half of the gain comes from the model family, half from new information (mostly the lot
text); hyper-parameter tuning adds little. Details: `reports/A3_report.pdf`.

## Repository layout

```
├── data/
│   ├── raw/                        # OWS snapshot (parquet) — not in git, downloaded from the release
│   └── processed/                  # model_table.parquet — not in git, built by A2 or downloaded
├── src/
│   ├── ows_client.py               # GraphQL client for OWS v3 (cursor pagination, retries)
│   ├── collect.py                  # dataset collection, resumable (contacts / bank accounts stored only as hashes)
│   ├── prepare.py                  # cleaning, feature table, target, time split
│   ├── features.py                 # A3: leak-free winner, customer–supplier history, price vs market
│   ├── metrics.py                  # PR-AUC / ROC-AUC in NumPy for torch code (tested against scikit-learn)
│   ├── llm_cards.py, labels_ru.py  # A3: Russian procurement cards for the LLM
├── notebooks/
│   ├── 02_data_preparation.ipynb   # A2: quality, cleaning, EDA, features, target, split
│   ├── 03_baseline.ipynb           # Midterm: baseline, metrics, error analysis
│   ├── 04_model_comparison_tuning.ipynb  # A3: 4 models, grid search + Optuna, threshold, calibration
│   ├── 05a_text_encoders_finetune.ipynb  # A3: transfer learning (3 frozen encoders) and fine-tuning (torch only)
│   ├── 05b_text_comparison.ipynb         # A3: TF-IDF vs encoders, text inside the tabular model
│   ├── 06_llm_qlora.ipynb                # A3: Qwen2.5-1.5B + QLoRA (trained on Kaggle), evaluation, ensemble
│   ├── 07_results.ipynb                  # A3: final model, all-models table, ablations, leakage case, SHAP
│   ├── build_*.py                  # the notebooks are generated from these scripts
│   ├── *_ru.ipynb                  # Russian reading copies (not for submission)
│   └── make_ru_02.py, make_ru_03.py, make_ru_a3.py (+ ru_a3/)
├── kaggle/06_llm_qlora/            # the QLoRA training script run on Kaggle through the API
├── experiments/a3_pilots/          # pilot experiments behind the A3 plan (see its README)
├── results/                        # A3 outputs: Optuna studies, chosen settings, LLM log and scores, tables
├── reports/                        # proposal, reports, slides (English + *_ru copies)
├── figures/                        # charts saved by the notebooks (figures/ru: Russian charts for the report)
└── collect_supervisor.sh           # restarts the collector after network failures
```

## Setup

```bash
python -m venv ~/.venvs/tender-risk-ml && source ~/.venvs/tender-risk-ml/bin/activate
pip install -r requirements.txt
```

Then open a notebook and run `Kernel → Restart & Run All`. No paths need editing and no API token is needed: the
notebooks download the data from the GitHub release `data-v1` on first run.

## Data

Source: OWS v3 GraphQL API (`https://ows.goszakup.gov.kz/v3/graphql`), the official open-data interface of the
procurement portal. Scope: procurements published in **Q1 2024** plus their lots, applications (bids), contracts,
and the register of unreliable suppliers (RNU).

| Table | Rows |
|---|---|
| `trd_buy` (procurements) | 630,762 |
| `lots` | 1,286,364 |
| `trd_app` (applications) | 1,066,963 |
| `contracts` | 555,723 |
| `rnu` | 17,343 |

The modelling table has 358,789 procurements that produced a contract; 6.6 % of them failed.

To collect the snapshot again (needs a personal token in `.env`, see `.env.example`):

```bash
python -m src.collect --date-from 2024-01-01 --date-to 2024-03-31
```

The script is resumable (checkpoints in `data/raw/_checkpoint.json`).

## Assignment map

| Assignment | Deliverable | Where |
|---|---|---|
| A1 Problem & Dataset ✅ | proposal (2–4 pages) + collected dataset | `reports/A1_proposal.docx` / `.pdf`, `src/collect.py` |
| A2 Data Preparation & Baseline ✅ | notebook + brief report | `notebooks/02_data_preparation.ipynb` (HTML: `reports/A2_notebook.html`), `reports/A2_report.pdf` |
| Midterm ✅ | working code, baseline model, ≥2 metrics, slides | `notebooks/03_baseline.ipynb` (HTML: `reports/midterm_notebook.html`), `reports/midterm_slides.pptx`, `reports/midterm_metrics.csv` |
| A3 Training & Fine-Tuning ✅ | model comparison, tuning, transfer learning, fine-tuning, LLM + QLoRA, final model, report | `notebooks/04`–`07`, `kaggle/06_llm_qlora/`, `reports/A3_report.pdf`; research and pilots: `reports/A3_research_ru.md`, `experiments/a3_pilots/` |

Russian reading copies of every deliverable are listed in [README_ru.md](README_ru.md).
