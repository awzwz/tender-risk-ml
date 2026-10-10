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

## Results so far (test period, 16–31 March 2024)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| No model (prior) | 0.500 | 0.066 | – |
| Logistic regression (baseline) | 0.792 | 0.262 | 0.324 |
| Random forest | 0.804 | 0.298 | 0.368 |
| Logistic regression + lot text (TF-IDF) | 0.802 | 0.282 | 0.332 |

Inspecting the 5 % of procurements with the highest baseline score finds 26 % of all failed contracts.

## Repository layout

```
├── data/
│   ├── raw/                        # OWS snapshot (parquet) — not in git, downloaded from the release
│   └── processed/                  # model_table.parquet — not in git, built by A2 or downloaded
├── src/
│   ├── ows_client.py               # GraphQL client for OWS v3 (cursor pagination, retries)
│   ├── collect.py                  # dataset collection, resumable
│   └── prepare.py                  # cleaning, feature table, target, time split
├── notebooks/
│   ├── 02_data_preparation.ipynb   # A2: quality, cleaning, EDA, features, target, split
│   ├── 03_baseline.ipynb           # Midterm: baseline, metrics, error analysis
│   ├── build_02.py, build_03.py    # the notebooks are generated from these scripts
│   ├── *_ru.ipynb                  # Russian reading copies (not for submission)
│   └── make_ru_02.py, make_ru_03.py
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
| A3 Training & Fine-Tuning | improved model + experiments | planned |

Russian reading copies of every deliverable are listed in [README_ru.md](README_ru.md).
