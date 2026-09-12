# Project Proposal — Assignment 1: Problem & Dataset

**Author:** Aziz Abdulin · **Repository:** https://github.com/awzwz/tender-risk-ml

---

## 1. Project title

**Tender Risk ML — predicting problematic public procurements in Kazakhstan from open procurement data**

## 2. Problem description

Kazakhstan's public procurement system (goszakup.gov.kz) processes roughly **1.8 million procurements per year**
(12.9 million in total as of September 2026, with over 40 million supplier bids), spending a significant share of the state budget. Every procurement,
lot, bid and contract is published through the open **OWS v3 GraphQL API**. Oversight bodies, journalists and
auditors, however, work by hand: they can inspect a few hundred procurements a year, while the signs of a
problematic procurement — a single bidder, an implausible price discount, a competitive procedure that "fails" and is
re-awarded to a single source, a supplier with a history of terminated contracts — are scattered across several
linked tables and are invisible at the level of one record.

The practical consequence is that a large share of contracts end badly. In a sample of FY2024 contracts, about
**one in five was terminated or never concluded**, and about **7 % of procurements were cancelled** after publication.
Each such case means a repeated procedure, delayed goods or services, and often wasted budget. Detecting the
procurements likely to fail *before* they fail would let a controlling body concentrate its limited attention where it
matters.

An earlier engineering project by the author (Tender Radar) approached this with 31 hand-written risk rules. Rules are
transparent but arbitrary: their thresholds were chosen by intuition, and there was no way to measure whether a
"risky" procurement actually turned out badly. This project replaces intuition with a **supervised model trained on
observed outcomes**, using the rules only as candidate features.

## 3. Project goal

Build and evaluate a machine-learning classifier that, using only information available at the time a procurement
is published and awarded, predicts whether the procurement will have a **problematic outcome** — defined as the
resulting contract being terminated or never concluded — and identify which characteristics of a procurement are most
predictive of that outcome.

## 4. Project objectives

1. **Dataset construction.** Collect a linked snapshot of procurements published in Q1 2024 from the OWS v3 API —
   procurements, lots, bids, contracts, and the register of unreliable suppliers — and publish it as a reproducible,
   documented dataset (collection script + data dictionary).
2. **Data preparation and EDA.** Clean and normalise the data (missing values, duplicates, currency/date formats,
   text normalisation for Russian/Kazakh lot names), explore distributions and class balance, and define the target
   variable rigorously with a leakage-safe **time-based train/validation/test split**.
3. **Feature engineering.** Derive tabular features (price, number of bidders, discount, trade method, customer and
   supplier history, the 31 Tender Radar indicators) and text features (TF-IDF over lot names/descriptions).
4. **Baseline modelling and evaluation.** Train and compare classical models (Logistic Regression, Random Forest,
   gradient boosting) on the tabular + TF-IDF features; report ROC-AUC, PR-AUC and F1 on the held-out test period,
   with an interpretation of feature importance.
5. **Model improvement.** Perform hyperparameter optimisation of the best classical model and add a text branch by
   **fine-tuning a pre-trained multilingual transformer** (transfer learning from `xlm-roberta-base` /
   `rubert-tiny2`) on lot descriptions; compare with, and ensemble into, the tabular model.

## 5. Type of AI task

**Binary classification** (supervised) on tabular data, with an **NLP component**: lot names and descriptions are
free text in Russian and Kazakh, vectorised with TF-IDF for the baseline and, in the fine-tuning stage, encoded by a
pre-trained transformer. Because positive cases are the minority (~20 %), evaluation focuses on ROC-AUC, PR-AUC and F1
rather than accuracy.

## 6. Dataset

### 6.1 Source and access

- **Source:** OWS v3 GraphQL API, `https://ows.goszakup.gov.kz/v3/graphql` — the official open-data interface of the
  public procurement portal. Access via a personal API token (valid until 13 Nov 2026). Data is public.
- **Scope:** all procurements with `publishDate` in **2024-01-01 … 2024-03-31** — **630 762 procurements**, about 35 %
  of the 1 818 344 FY2024 procurements — and every linked record. Q1 2024 is old enough that every contract has had > 2 years
  to reach a final status, so the outcome label is settled.
- **Collection:** `src/collect.py` — resumable pagination (`limit/after`), stored as JSONL and Parquet. The snapshot
  is published as a GitHub release asset; the script reproduces it from scratch.

### 6.2 Entities

| Table | Grain | Approx. rows (Q1 2024) | Key fields |
|---|---|---|---|
| `trd_buy` | one procurement | 630 762 | id, nameRu, trade method, status, totalSum, countLots, customer BIN, winner BIN, publish/start/end dates |
| `lots` | one lot | ~1 000 000 | trdBuyId, nameRu/nameKz, descriptionRu, amount, dumping flag, KTRU code (`enstruList`), status |
| `trd_app` | one bid | ~2 500 000 | buyId, supplier BIN, per-lot price / discount |
| `contracts` | one contract | ~470 000 | trdBuyId, supplier BIN, status, cancellation reason, sums, plan/actual execution dates |
| `rnu` | one supplier entry | 17 638 (all-time) | supplier BIN, start/end dates in the register |

The `trd_buy` count is exact (API `totalCount`); the others are extrapolated from a pilot sample of 150
  procurements (1.6 lots and 0.75 contracts per procurement) and from the bid/procurement ratio; exact counts will be
  given in A2.

### 6.3 Target variable

**Primary target — `contract_failed` (binary, per procurement):**
1 if the contract originating from the procurement has status *Terminated* (`Расторгнут`, code 350), *Awaiting
termination* (345) or *Not concluded* (330); 0 if it is *Active*, *Amended* or *Executed*. Procurements without any
contract record are labelled separately (see below) and excluded from the primary task.

Sample distribution (first 400 FY2024 contracts): 350/345/330 → **23 %** positive; 455 Amended 68 %; 190/390/460
Active/Executed 6 %.

**Auxiliary labels** kept in the dataset for extended analysis:
- `tender_cancelled` — procurement status *Cancelled* (430) or *Refused* (410): ~7 % of procurements.
- `winner_in_rnu` — the winning supplier appears in the register of unreliable suppliers at any later date (rare).
- `single_source_after_failed` — trade method "single source after a failed procedure": ~19 % of procurements — an
  indicator that competition failed.

Why an *observed outcome* rather than the rule-based risk score: a model trained on labels produced by hand-written
rules only learns those rules. Contract termination is a real, independently recorded event, so the model's quality
can be measured honestly and the rules can be tested as features.

### 6.4 Leakage control

Only fields known at publication/award time enter the feature set: procurement and lot attributes, bids, winner
identity, and the *history* of the customer and the supplier **strictly before** the procurement's publish date.
Contract execution fields (`faktSum`, `faktExecDate`, status) are used for the label only. The split is by time:
train on January–February, validate/test on March 2024, so that no future information leaks into training.

### 6.5 Feature groups (preliminary)

- **Procurement:** trade method, subject type, total sum, number of lots, planned duration, repeat flag, single-organiser flag, construction/light-industry flags, region (KATO).
- **Competition:** number of bidders, number of bids per lot, winner's discount, dumping flag, price dispersion among bids.
- **Supplier / customer history (prior to publish date):** number of prior contracts, share terminated, share of wins with the same customer, presence in RNU.
- **Rule indicators:** the 31 Tender Radar indicators re-implemented in pandas.
- **Text:** TF-IDF (word + char n-grams) of lot `nameRu` / `descriptionRu`; transformer embeddings in A3.

## 7. Plan and deliverables

| Stage | Weeks | Deliverable |
|---|---|---|
| A1 Problem & Dataset | 1–2 | this proposal; raw Q1 2024 snapshot collected and published |
| A2 Data Preparation & Baseline | 3–4 | notebook: structure, missing values, duplicates, cleaning, EDA, visualisations, features + target, time split; brief report |
| Midterm | 5 | baseline LogReg + Random Forest, ROC-AUC / PR-AUC / F1, slides, justification of model choice |
| A3 Training & Fine-Tuning | after midterm | model comparison incl. LightGBM, hyperparameter optimisation, fine-tuned transformer on lot text, ensemble; experimental results |

## 8. Expected results

- A public, reproducible dataset of Q1 2024 procurements with a documented outcome label.
- A classifier with ROC-AUC clearly above the rule-based score used as a baseline comparator.
- A ranked list of the procurement characteristics most associated with contract failure — the practically useful
  output for an auditor.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| API token expires 13 Nov 2026 | full snapshot collected in September; all later work is offline |
| Class imbalance | PR-AUC/F1 as metrics; class weights; threshold tuning |
| Outcome label noise (termination "by mutual agreement" is not always a failure) | keep cancellation reason codes; report results for the strict subset (terminated for breach / not concluded) as a sensitivity analysis |
| Text in two languages | multilingual models (XLM-R); Kazakh-only lots handled via `nameRu` which is almost always present |
| Compute for transformer fine-tuning | `rubert-tiny2` fits on CPU/Colab; LoRA for larger models |
