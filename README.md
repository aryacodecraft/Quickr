# Quickr — Quick-Commerce Delivery Intelligence System

Quickr is a BDA practical project that analyses real food-delivery records with **PySpark**, trains **scikit-learn** models, stores analytical results in **Apache Cassandra**, and serves a seven-page **Streamlit** dashboard with a Blinkit-inspired yellow/forest-green/ivory theme.

The project does not fabricate records, predictions, or metrics. Every historical KPI comes from the cleaned dataset and Cassandra; every model score comes from a saved held-out evaluation.

---

## 1. What is included

| Area | Implementation |
|---|---|
| Main analytics pipeline | `notebooks/Quickr_BDA_Analysis.ipynb` |
| Dashboard | `app/app.py` plus `app/styles.py`, `app/cassandra_client.py`, `app/prediction_service.py` |
| Database schema | `cassandra/schema.cql` |
| Schema initialiser | `scripts/init_cassandra.py` |
| Saved models | `models/quickr_delivery_time_model.joblib`, `models/quickr_delay_risk_model.joblib` |
| Saved evaluation | `models/metrics.json`, `models/risk_metrics.json`, `models/confusion_matrix.csv`, `models/test_predictions.csv` |
| Tests | `tests/` |
| Clean data and summaries | `data/processed/quickr_clean.csv`, `data/processed/analytics/` |

### Dashboard pages

1. **Overview** — operational summary and category comparisons.
2. **Delivery Performance** — five-card scorecard, including a configurable exact duration threshold.
3. **Delivery Analytics** — interactive traffic/weather/vehicle/city segment filters.
4. **Prediction** — delivery-duration estimate from the saved Random Forest pipeline.
5. **Delivery Risk** — three-band, duration-based risk assessment from the saved logistic-regression classifier.
6. **Model Performance** — metrics-only report; **no charts**.
7. **Data Quality** — cleaning funnel, missing-value counts, source and limitations.

---

## 2. Dataset and target

- Source: project dataset `data/raw/food_delivery.csv`
- Raw file: `data/raw/food_delivery.csv`
- Raw shape: **45,593 rows × 20 columns**
- Target: `Time_taken(min)`, parsed from text such as `(min) 24` into `duration_min`
- Clean shape: **40,016 rows**
- Cleaning rules:
  - valid duration: **1–120 minutes**
  - valid Indian GPS coordinates
  - straight-line route distance: **0.1–100 km**
  - complete `City`, `Weatherconditions`, `Road_traffic_density`
  - valid partner age/rating/vehicle-condition/concurrent-delivery ranges
  - duplicate rows removed

The dataset has **no promised delivery deadline, no SLA, and no official late/on-time label**. All "late" language in this project therefore refers to a documented analytical duration threshold, not a customer promise.

### Actual scorecard results

| KPI | Actual value |
|---|---:|
| Valid records | 40,016 |
| Average delivery time | 26.43 min |
| Median delivery time | 26.0 min |
| Fastest / slowest | 10.0 / 54.0 min |
| Deliveries over 33 min | 8,922 (22.30%) |
| Selected model MAE | 3.13 min |

The Delivery Performance page recalculates the count and percentage for any integer threshold from 1 to 54 minutes using exact per-threshold counts stored in Cassandra. The 33-minute threshold is the documented P75 reference; it is **not** an official late rate.

---

## 3. Models

### 3.1 Delivery-duration regression

- Model: **Random Forest Regressor**
- 100 trees, `random_state=42`, `n_jobs=-1`
- Features: partner age, partner rating, route distance, vehicle condition, concurrent deliveries, weather, traffic, order type, vehicle type, festival, city
- Preprocessing: median/most-frequent imputation and one-hot encoding inside the sklearn pipeline
- Split: reproducible **80/20 random holdout** (`random_state=42`)
- Training / test rows: **32,012 / 8,004**
- Actual held-out results:
  - **MAE: 3.126 min**
  - **RMSE: 3.922 min**
  - **R²: 0.819**
  - **MAPE: 13.47%**

### 3.2 Duration-risk classifier

- Model: **Logistic Regression** (`max_iter=1000`, `random_state=42`)
- Positive class: `duration_min > 33 minutes`, where 33 is the **75th percentile of the training split only**
- No target leakage: actual duration, IDs, timestamps, and pickup/completion fields are excluded from model inputs
- Preprocessing: median/most-frequent imputation, standard scaling for numeric features, one-hot encoding for categorical features
- Evaluation: same held-out 20% split
- The split is unstratified because the positive label is defined after the training split; both class rates are reported.
- Class counts: **7,228 / 32,012 positive in train**; **1,694 / 8,004 positive in test**
- Actual held-out results at the 0.50 evaluation cut-off:
  - accuracy **0.874**
  - precision **0.729**
  - recall **0.640**
  - F1 **0.682**
  - ROC-AUC **0.925**
  - confusion matrix: TN 5,908 / FP 402 / FN 610 / TP 1,084

Display bands: **Low < 0.40**, **Medium 0.40–0.65**, **High ≥ 0.65**. These cut-offs are configurable on the Delivery Risk page. The score is **not calibrated**, so the UI calls it a model-estimated risk score and never presents it as a guaranteed probability.

---

## 4. Architecture

```
raw CSV
   │
   ├─ pandas cleaning and feature engineering in the notebook
   ├─ PySpark RDD MapReduce analytics
   ├─ scikit-learn training and held-out evaluation
   │
   ├─ data/processed/quickr_clean.csv + analytics CSVs
   ├─ models/*.joblib + metrics JSON/CSV
   │
   └─ Cassandra quickr keyspace
         ├─ delivery_summary_by_category
         ├─ delivery_segment_summary
         ├─ delivery_duration_distribution
         ├─ delivery_duration_thresholds   (exact configurable-threshold counts)
         ├─ delivery_metrics               (quality, model, risk, metadata)
         └─ delivery_predictions           (dashboard predictions, UTC day partitions)

Streamlit dashboard ── Cassandra reads ── cached summary tables
                    └─ saved joblib pipelines for predictions
```

The dashboard never reruns the Spark notebook on page load. Cassandra is queried through prepared statements; the app has no mock-data fallback.

---

## 5. Exact setup commands (Windows PowerShell)

From `D:\Quickr`:

```powershell
# 1. Create and activate the Python 3.11 environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Start the local Cassandra development node
docker compose up -d

# 4. Create the keyspace and six tables
.\.venv\Scripts\python.exe scripts\init_cassandra.py

# 5. Execute the notebook end-to-end
#    (~2–5 minutes; writes all metrics, models, CSVs, and Cassandra rows)
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\Quickr_BDA_Analysis.ipynb

# 6. Start the dashboard
.\.venv\Scripts\streamlit.exe run app\app.py
```

Open <http://localhost:8501>.

> Use `streamlit run app\app.py`, not `python app\app.py`. Do not open `app/app.py` as a plain Python script.

### Cassandra host/port

The development container exposes Cassandra on **host port 9043** (container port 9042). Defaults are:

```text
CASSANDRA_HOST=127.0.0.1
CASSANDRA_PORT=9043
CASSANDRA_KEYSPACE=quickr
CASSANDRA_DATACENTER=datacenter1
```

They can be overridden with environment variables or a `.env` file.

### If Cassandra is unavailable

The dashboard deliberately shows a setup message. Start the container and initialise the schema:

```powershell
docker compose up -d
docker ps --filter name=quickr-cassandra
.\.venv\Scripts\python.exe scripts\init_cassandra.py
```

No synthetic records or metrics are substituted.

---

## 6. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Last verified result:

```text
55 passed, 1 warning in 61.36s (0:01:01)
```

The suite covers:

- raw dataset shape/schema and cleaning funnel
- valid/invalid duration handling
- validation of prediction and direct-distance risk inputs
- saved regression and risk model contracts
- target-leakage prevention (duration absent from model inputs)
- scorecard metrics recomputed from the cleaned CSV
- exact threshold counts and percentage changes
- Cassandra helpers, threshold reads, and live round-trip checks
- all seven dashboard pages plus a real Delivery Risk form submission
- confirmation that Model Performance contains **no Plotly charts**

The dashboard smoke test is skipped only when Cassandra or the saved model artifacts are unavailable; it does not use mock data.

---

## 7. Reproducibility and artifacts

The notebook writes:

- `data/processed/quickr_clean.csv`
- `data/processed/analytics/category_summary.csv`
- `data/processed/analytics/segment_summary.csv`
- `data/processed/analytics/duration_distribution.csv`
- `data/processed/analytics/delivery_metrics.csv`
- `models/quickr_delivery_time_model.joblib`
- `models/quickr_delay_risk_model.joblib`
- `models/model_metadata.json`
- `models/metrics.json`
- `models/risk_metrics.json`
- `models/test_predictions.csv`
- `models/confusion_matrix.csv`

Cassandra write-back is verified in the notebook's read-back section (table counts and sample traffic rows).

---

## 8. Limitations and honest interpretation

- The source has **no promised delivery time**, so the project cannot calculate a real late-delivery rate or SLA breach rate.
- "Risk" is a duration-based proxy, not an official late label.
- Route distance is straight-line Haversine distance; no road network or live traffic is available.
- Weather/traffic associations are observational, not causal.
- Partner "experience" is not in the data; age and rating are the only partner signals used.
- The data has no price, quantity, product, or category fields, so revenue/product-sales analysis is not possible.
- The model is a course project trained on one historical dataset; it is not production-ready and should not be used for real dispatch decisions.
- Missing age/rating/concurrent-delivery values are imputed inside the sklearn pipelines, fitted on training data only.

---

## 9. Short viva questions

**Why Cassandra?**  
It is a distributed, query-oriented store with simple prepared writes and bounded partition reads for dashboard summaries. The project uses one keyspace and small query-specific tables; it does not scan the full dataset on every page load.

**Why is the risk label a proxy?**  
The CSV has actual duration but no promised deadline or official late flag. The notebook therefore defines risk as duration above the training-set P75 and states clearly that this is analytical.

**How is target leakage prevented?**  
The feature list is explicit: it contains no `duration_min`, `Time_taken(min)`, IDs, order timestamps, or pickup/completion fields. The classifier is trained only on pre-delivery attributes.

**Why three risk bands?**  
Logistic Regression gives a continuous score. The score is uncalibrated, so the dashboard groups it into Low/Medium/High bands. The cut-offs (0.40/0.65) are configurable and are not presented as calibrated probabilities.

**Why no charts on Model Performance?**  
The page is intentionally a metrics-only report: four KPI cards, a genuine held-out results table, model information, and a plain-English interpretation. Other pages retain charts where they support exploration.

**Why does the scorecard threshold not equal an official late rate?**  
Because no SLA exists. The page labels the card "duration threshold" and explains that it counts observed deliveries longer than the selected number of minutes.

---

## 10. Author

- **Name:** Arya Mahajan
- **Class:** B3
- **Batch:** B1
- **Roll no.:** 03
- **Practical:** Big Data Analytics
