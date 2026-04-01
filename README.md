# Hospital Supply Chain Analytics Platform

A production-quality data platform for predicting and visualizing supply stockout risk across a multi-facility hospital network. Ingests heterogeneous procurement, inventory, and supplier data; trains a gradient boosting model to forecast shortfalls 14 days in advance; and surfaces risk scores through an operator-facing real-time dashboard.

Built to demonstrate the full FDSE stack: data engineering → ML → deployed decision-support tooling.

---

## Overview

Hospital supply chains fail quietly. A facility runs low on nitrile gloves or IV bags, procurement staff don't notice until it's too late, and the downstream effects — delayed procedures, staff exposure, emergency spot-purchases at 3× cost — are entirely preventable with the right signals 2 weeks earlier.

This platform operationalizes those signals. It processes order histories, inventory trajectories, and supplier delivery performance across 8 facilities and 10 critical SKU categories, builds a feature-rich ML model that predicts stockout probability at the facility × SKU level, and exposes the results through a dashboard designed around how operations managers actually make decisions.

---

## Architecture

```
data/
  generate_data.py          Synthetic data generator (3 years × 8 facilities × 10 SKUs)
  procurement_records.csv   Raw purchase order data
  inventory_snapshots.csv   Weekly inventory levels per facility × SKU
  supplier_deliveries.csv   Delivery events with actual vs. promised lead times

pipeline/
  etl.py                    Ingest → validate → transform → load (SQLite)

model/
  train_model.py            Feature engineering + GradientBoosting training + evaluation
  score.py                  Batch inference → JSON for dashboard consumption
  stockout_model.pkl        Serialized trained model + decision threshold
  feature_names.txt         Ordered feature list for inference reproducibility
  evaluation_report.txt     Full classification report, confusion matrix, feature importances

dashboard/
  public/
    index.html              Single-file React dashboard (no build step)
    scores.json             Model output consumed by dashboard
```

---

## Quickstart

**Requirements:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`

```bash
# 1. Generate synthetic data
python data/generate_data.py

# 2. Run ETL pipeline (validates, transforms, loads to SQLite)
python pipeline/etl.py --data-dir data/ --db supply_chain.db

# 3. Train the stockout prediction model
python model/train_model.py --db supply_chain.db

# 4. Generate current risk scores
python model/score.py --db supply_chain.db --out dashboard/public/scores.json

# 5. Launch the dashboard
cd dashboard/public && python -m http.server 8080
# Open http://localhost:8080
```

---

## Data Pipeline

`etl.py` implements a four-stage pipeline:

**Ingest** — loads the three raw CSV exports, enforces column types, and logs shape and null counts for each source.

**Validate** — asserts referential integrity (every delivery PO must exist in procurement), business rule constraints (quantities > 0, fill rates in [0,1]), and date parseability. Exits with a descriptive error on any violation.

**Transform** — parses dates, computes derived fields (order week/month, delay severity flags, stock coverage buckets), and normalizes data into a star schema with separate dimension tables for facilities, SKUs, and suppliers.

**Load** — writes 7 tables to SQLite with appropriate indexes on facility, SKU, date, and supplier columns. The resulting database is ~4 MB for 3 years of data across 8 facilities.

---

## Machine Learning Model

### Problem framing

Binary classification: will a given facility × SKU combination experience a stockout event in the next 14 days? A "stockout event" is defined as inventory falling below 30% of the reorder point in any snapshot within the horizon.

### Features (19 total)

**Inventory trajectory:**
- Current units on hand and days of supply
- Days-of-supply at 1, 2, and 4 weeks prior (lag features)
- 4-week rolling mean and standard deviation of days of supply
- 2-week trend (current DoS minus DoS 14 days ago)
- Recent stockout count (rolling 4-week sum)

**Procurement velocity (28-day lookback):**
- Number of purchase orders placed
- Total units ordered
- Average fill rate on deliveries received
- Average delivery delay in days
- Fraction of deliveries with severe delays (≥5 days late)

**Temporal:**
- Month of year, ISO week, Q4 flag (flu season / holiday demand spike)

**Categorical:**
- Facility ID (label-encoded)

### Training

Chronological 80/20 train-test split (no shuffling — preserves temporal ordering to prevent lookahead leakage). Decision threshold tuned on the validation set to maximize recall at ≥60% precision.

### Results

| Metric             | Value          |
|--------------------|----------------|
| ROC-AUC            | 0.979          |
| Recall (stockout)  | 0.81+          |
| 5-fold CV recall   | 0.897 ± 0.019  |
| Prediction horizon | 14 days        |

The model's top predictors are current `units_on_hand`, `days_of_supply`, and the 4-week rolling mean — confirming that inventory trajectory is a stronger signal than supplier history alone, though supplier fill rate and delay history meaningfully improve precision.

### Swap in XGBoost

The model backend is a single constant in `train_model.py`. To use XGBoost:

```python
# In model/train_model.py, replace:
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=300, ...)

# With:
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                      subsample=0.8, eval_metric="logloss", random_state=42)
```

Everything downstream (scoring, serialization, dashboard) is unchanged.

---

## Dashboard

A single-file React dashboard (`dashboard/public/index.html`) — no Node, no build step, no bundler. Loads `scores.json` at runtime and renders:

**Stat strip** — total items monitored, critical count, high-risk count, average risk score across the network.

**Facility selector** — click any facility to filter the risk table. Shows per-facility critical/high counts and average risk score at a glance.

**Supplier delay alerts** — flags any supplier-facility pair with on-time rate below 80%, showing on-time rate, average delay, and severe-delay count.

**Risk table** — all facility × SKU combinations ranked by risk score. Filterable by risk level, supply category, and free-text search. Each row shows units on hand, days of supply, fill rate, a proportional risk bar, and a severity badge.

The dashboard falls back to realistic synthetic data if `scores.json` is not served — useful for demos without a local server.

---

## Design decisions

**SQLite over a hosted database.** For a project at this scale (sub-10MB, single analyst), SQLite eliminates deployment complexity while preserving full SQL expressiveness. In production this schema maps directly to PostgreSQL.

**Single-file dashboard.** React via CDN with Babel transpilation — no npm, no webpack. Keeps the dashboard fully portable while still being written in idiomatic React with hooks and component composition.

**Chronological split, not random.** Random shuffling would leak future inventory states into training. The 80/20 split treats the last ~7 months as held-out test data, simulating real deployment conditions.

**Threshold tuning.** A stockout in a hospital is far more costly than a false alarm. The decision threshold is tuned to maximize recall at a minimum acceptable precision, rather than using the default 0.5 cutoff.

---

## Extensions

- **Automated retraining** — schedule `train_model.py` weekly via cron or Airflow as new procurement data arrives
- **Real data integration** — ETL pipeline accepts any CSV export matching the schema; column mapping config can be added to `etl.py`
- **REST API wrapper** — expose `score.py` as a FastAPI endpoint for EHR/ERP integration
- **Push alerts** — extend `score.py` to POST critical-risk items to Slack or email

---

## Author

Parker Boyle · Amherst College · CS + Mathematics · parkerboyle212@gmail.com
