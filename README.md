# tender-risk-ml

Predicting risky public procurements in Kazakhstan (goszakup.gov.kz) — a semester ML project.

Course pipeline: **A1 Problem & Dataset → A2 Data Preparation & Baseline → Midterm → A3 Model Training & Fine-Tuning.**

## Problem in one paragraph

Kazakhstan's public procurement portal publishes ~1.8M procurements per year through the OWS v3 GraphQL API.
Manual audit is impossible at that scale. This project builds a **binary classifier** that, from information
available at publication/award time (tender attributes, lot descriptions, bidding pattern, supplier/customer history),
predicts whether the procurement will end in a **problematic outcome** — the resulting contract is terminated or never
concluded. Rule-based risk indicators from an earlier engineering project (Tender Radar) are reused as features,
not as labels, so the model is trained against a real observed outcome.

## Repository layout

```
├── data/
│   ├── raw/          # OWS snapshot (jsonl/parquet) — not in git, see "Data"
│   └── processed/    # cleaned tables produced in A2
├── src/
│   ├── ows_client.py # thin sync GraphQL client for OWS v3 (limit/after pagination, retries)
│   ├── collect.py    # dataset collection script, resumable
│   ├── features.py   # feature engineering (A2)
│   └── labels.py     # target construction (A2)
├── notebooks/
│   ├── 01_collect.ipynb          # A1  — collection & first look
│   ├── 02_prepare_eda.ipynb      # A2  — cleaning, EDA, features, target, split
│   ├── 03_baseline.ipynb         # Mid — baseline models + metrics
│   └── 04_tuning_finetune.ipynb  # A3  — tuning, transfer learning, fine-tuning
├── reports/
│   ├── A1_proposal.md
│   ├── A2_report.md
│   ├── midterm_slides.md
│   └── A3_results.md
└── figures/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # put your OWS_TOKEN there
```

## Data

Source: OWS v3 GraphQL API (`https://ows.goszakup.gov.kz/v3/graphql`). Scope: procurements published in **Q1 2024**
(`TrdBuy.publishDate` in 2024-01-01..2024-03-31) plus their lots, applications (bids), contracts, and the full
register of unreliable suppliers (RNU).

```bash
python -m src.collect --date-from 2024-01-01 --date-to 2024-03-31
```

The script is resumable (checkpoints in `data/raw/_checkpoint.json`); rerunning continues where it stopped.
Raw snapshot is kept out of git because of size — it is published as a GitHub release asset.

## Assignment map

| Assignment | Deliverable | Where |
|---|---|---|
| A1 Problem & Dataset | proposal (2–4 pages) + collected dataset | `reports/A1_proposal.md`, `src/collect.py` |
| A2 Data Preparation & Baseline | notebook + brief report | `notebooks/02_*.ipynb`, `reports/A2_report.md` |
| Midterm | working code, baseline model, ≥2 metrics, slides | `notebooks/03_*.ipynb`, `reports/midterm_slides.md` |
| A3 Training & Fine-Tuning | improved model + experiments | `notebooks/04_*.ipynb`, `reports/A3_results.md` |
