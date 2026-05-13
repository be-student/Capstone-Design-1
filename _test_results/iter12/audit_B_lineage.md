# B — End-to-End Data Lineage Audit

Date: 2026-05-12
Pipeline run timestamp on disk: 2026-05-10 20:51 — 20:56 (KST)
All artifacts < ~28h old; consistent provenance, but TWO key dashboard
visualisations bypass real artifacts and return hardcoded fixtures
(see Risks).

## Artifact chain inventory

| Artifact | Path | Exists? | Size | Mtime (KST) | Read by dashboard? |
|---|---|---|---|---|---|
| customers.csv | data/raw/ | YES | 840 KB | 2026-05-10 20:51 | YES — `_customers_path()` (data_loader.py:174) |
| customers.parquet | data/raw/ | YES | 161 KB | 2026-05-10 20:51 | NO (CSV preferred) |
| events.csv | data/raw/ | YES | 1.00 GB | 2026-05-10 20:51 | YES via `_raw_events_path()` |
| events.parquet | data/raw/ | YES | 79.8 MB | 2026-05-10 20:51 | YES (parquet preferred) |
| generation_summary.json | data/raw/ | YES | 2.7 KB | 2026-05-10 20:52 | NO direct read; consistent with display |
| features.csv | data/feature_store/ AND results/ | YES | 9.05 MB | 2026-05-10 20:52 | YES (load_feature_store …) |
| features.parquet | data/feature_store/ | YES | 2.89 MB | 2026-05-10 20:52 | optional |
| ml_churn_model.pkl.joblib | models/ | YES | 275 KB | 2026-05-10 20:54 | NO — never loaded by dashboard |
| ml_churn_model_v1.pkl.joblib | models/ | YES | 275 KB | 2026-05-10 20:54 | NO |
| dl_churn_model.pt | models/ | YES | 445 KB | 2026-05-10 20:54 | NO — never loaded by dashboard |
| dl_churn_model_v1.pt | models/ | YES | 445 KB | 2026-05-10 20:54 | NO |
| clv_model.pkl | models/ | YES | 692 KB | 2026-05-10 20:56 | NO — never loaded |
| survival_model.pkl | models/ | YES | 1.99 MB | 2026-05-10 20:56 | NO — never loaded |
| uplift_model.pkl | models/ | YES | 442 KB | 2026-05-10 20:55 | NO — never loaded |
| model_artifacts_manifest.json | models/ | YES | 833 B | 2026-05-10 20:54 | NO (system-health glob only) |
| model_metrics.json | results/ | YES | 351 KB | 2026-05-10 20:54 | YES via load_model_metrics |
| model_performance_history.csv | results/ | YES | 587 B (3 rows) | 2026-05-10 20:56 | YES via load_model_performance_history / load_mlflow_runs |
| churn_predictions.csv | results/ | YES | 1.75 MB / 20 000 rows | 2026-05-10 20:54 | YES via load_predictions |
| churn_predictions_test.csv | results/ | YES | 281 KB / 3 334 rows | 2026-05-10 20:54 | YES (sub-slice) |
| clv_predictions.csv | results/ | YES | 1.16 MB / 20 000 rows | 2026-05-10 20:56 | YES via load_clv_data |
| clv_data.csv | results/ | YES | 792 KB | 2026-05-10 20:56 | YES (preferred over clv_predictions) |
| uplift_results.csv | results/ | YES | 1.79 MB / 20 000 rows | 2026-05-10 20:55 | YES via load_uplift_results |
| budget_optimization.csv | results/ | YES | 2.90 MB / 20 000 rows | 2026-05-10 20:56 | YES (fallback for budget_results) |
| budget_results.csv | results/ | YES | 709 B / 8 segment rows | 2026-05-10 20:56 | YES (primary for load_budget_results) |
| recommendations.csv | results/ | YES | 5.34 MB / 20 000 rows | 2026-05-10 20:56 | YES via load_recommendations |
| feature_importance.csv | results/ | YES | 379 B | 2026-05-10 20:54 | YES via load_feature_importance |
| segments_6plus.csv | results/ | YES | 3.20 MB | 2026-05-10 20:56 | YES |
| cohort_analysis.json | results/ | YES | 745 B | 2026-05-10 20:56 | YES |
| ab_test_results.json / ab_test_detailed.json | results/ | YES | 710 B / 1.96 KB | 2026-05-10 20:56 | YES |
| monitoring_report.json | results/ | YES | 52.3 KB | 2026-05-10 20:56 | YES (drift, performance alerts) |
| **confusion_matrices.json** | results/ or data/artifacts/ | **NO** | — | — | falls back to hardcoded fixture |
| **roc_data.json** | results/ or data/artifacts/ | **NO** | — | — | falls back to synthetic Beta-distributed curves |
| **scoring_history.csv** | data/artifacts/ | **NO** | — | — | falls back to 200-row np.random sample |
| **survival_curves.json** | data/artifacts/ | **NO** | — | — | falls back to synthetic KM curves |
| **drift_history.csv** | — | **NO** | — | — | reconstructed from monitoring_report.json (REAL) |

## Flow trace

### Flow 1 — Simulator → raw
- `src/data/orchestrator.run` → `src/data/generator.generate_*` produces
  `data/raw/customers.csv` (20 000 rows), `data/raw/events.csv`
  (16 151 630 events), `data/raw/generation_summary.json`.
- `generation_summary.json` records: 20 000 customers, churn rate
  0.19995, 6 personas, treatment/control 50/50, seed=42.
- Dashboard read paths:
  - `data_loader._customers_path()` (line 174) — checks `results/`,
    `data/artifacts/`, then `data/raw/customers.csv`. Real customers
    flow into customer-level loaders.
  - `data_loader._raw_events_path()` (line 184) — reads
    `data/raw/events.parquet` (preferred) or `events.csv`.

### Flow 2 — Features → feature store
- `run_features` (src/main.py) calls feature_engineering pipeline.
- Output: `data/feature_store/features.{csv,parquet}` AND
  `results/features.csv` (duplicate).
- 38 CSV columns = 33 feature columns + 5 metadata
  (customer_id, persona, churn_label, treatment_group, signup_date).
  Matches "30+ features" claim. Manifest `feature_count: 33`.

### Flow 3 — ML training → model artifacts
- `run_train` (src/main.py:1270):
  - `time_based_split` → `len(y_test) == 3334` (recorded in
    `model_metrics.json.test_size`).
  - Saves `ml_churn_model.pkl.joblib` via
    `churn_model.py:663 joblib.dump`.
  - Saves `dl_churn_model.pt` via `churn_model.py:1278 torch.save`.
  - Writes `results/model_metrics.json` (ml/dl/ensemble AUC,
    precision, recall, F1, accuracy + MLflow run IDs).
  - Writes `results/model_performance_history.csv` (3 rows, one per
    model with timestamp 2026-05-10T11:56:50Z).
  - MLflow status in metrics JSON:
    `tracking_uri: http://mlflow:5000`,
    `experiment_name: churn_prediction`, 4 run IDs logged.
- Dashboard uses these artifacts ONLY for metric display, not inference.
  `grep -E "joblib|torch.load|\\.predict_proba" src/dashboard/**.py` →
  ONLY hit is `system_health_view.py:352-353 models_dir.glob('*.joblib')`
  to list filenames. No actual `joblib.load` / `torch.load` /
  `model.predict_proba` calls.

### Flow 4 — Predictions → results/predictions
- `churn_predictions.csv`: 20 000 rows, full customer scoring + 3 334
  test rows (`split=test`), columns
  `customer_id, churn_probability, risk_level, persona, segment, split,
  prediction_source`. Prediction source = `ml_full_customer_scoring`.
- `clv_predictions.csv`: 20 000 rows, columns `customer_id,
  predicted_clv, high_value, clv_percentile, churn_probability,
  clv_predicted`. Sum of `predicted_clv` = ₩57.94 B exactly matches
  dashboard claim.
- `uplift_results.csv`: 20 000 rows. Segment counts:
  sure_thing 12 919, sleeping_dog 3 683, persuadable 2 798,
  lost_cause 600.
- `budget_optimization.csv`: 20 000 per-customer rows, total
  allocated_budget = ₩50 000 000.
- `budget_results.csv`: 8 segment-rollup rows (the primary file
  used by load_budget_results).
- `recommendations.csv`: 20 000 rows joined with uplift+CLV+segment.

All five CSVs are per-customer and traceable to `customer_id`
keys present in `data/raw/customers.csv`.

### Flow 5 — Dashboard reading
- `DashboardDataLoader` (data_loader.py) is the only data entry point
  for `app.py`. 28 `load_*` methods.
- Each loader uses `_required_csv` / `_required_json`; on miss it
  records a dashboard-visible issue AND returns an empty frame —
  except for the FOUR loaders below that silently fall back to
  np.random fixtures:
  - `load_confusion_matrices` → hardcoded
    `{ml_model:[[350,50],[80,120]], dl_model:[[340,60],[90,110]],
      ensemble:[[360,40],[70,130]]}` (data_loader.py:1348-1354).
  - `load_roc_data` → Beta-distributed synthetic ROC
    (data_loader.py:1316-1346).
  - `load_scoring_history` → 200-row np.random.beta sample
    (data_loader.py:1748-1780).
  - `load_survival_curves` → synthetic Kaplan-Meier from np.random
    (data_loader.py:1420-1462).
  - `load_survival_data` → falls back to
    `_generate_sample_survival_data` after trying segments file.

## KPI source matrix (most important deliverable)

| Page | KPI | Source | Verdict |
|---|---|---|---|
| 02 | ML AUC 0.8852 | model_metrics.json["ml_model"]["auc"] via load_model_metrics (data_loader.py:595, app.py:470) | REAL_ARTIFACT |
| 02 | DL AUC 0.8860 | model_metrics.json["dl_model"]["auc"] | REAL_ARTIFACT |
| 02 | Ensemble AUC 0.8866 | model_metrics.json["ensemble"]["auc"] | REAL_ARTIFACT |
| 02 | ML precision/recall/F1/accuracy headline | OVERWRITTEN from confusion-matrix fixture (app.py:439-462) | HARDCODED_FIXTURE — overrides real values |
| 02 | DL precision/recall/F1/accuracy headline | OVERWRITTEN (same) | HARDCODED_FIXTURE |
| 02 | Ensemble precision/recall/F1/accuracy headline | OVERWRITTEN (same) | HARDCODED_FIXTURE |
| 02 | Confusion matrix ml_model 350/50/80/120 (n=600) | `_generate_sample_confusion_matrices` (data_loader.py:1348) — confusion_matrices.json does NOT exist on disk | HARDCODED_FIXTURE |
| 02 | Confusion matrix dl_model 340/60/90/110 (n=600) | same fixture | HARDCODED_FIXTURE |
| 02 | Confusion matrix ensemble 360/40/70/130 (n=600) | same fixture | HARDCODED_FIXTURE |
| 02 | Test-set size caption "600 samples" | derived `_tn+_fp+_fn+_tp` from same fixture (app.py:447-450) | HARDCODED_FIXTURE — but REAL `test_size=3334` exists in model_metrics.json and is NOT used |
| 02 | ROC curves | `_generate_sample_roc_data` (Beta synthetic; roc_data.json absent) | SAMPLE_FALLBACK |
| 02 | MLflow runs table / radar | `model_performance_history.csv` (3 real rows) via load_mlflow_runs → load_model_performance_history | REAL_ARTIFACT — but only 3 rows, no `params_lr`/`params_epochs`/`training_time_s` so those columns default to 0.1/1/1.0 (data_loader.py:1162-1170) |
| 02 | Top feature importance | feature_importance.csv | REAL_ARTIFACT |
| 05 | Budget total allocated / Expected retained / Revenue saved / ROI | budget_results.csv (8 segment rollups, real) scaled by slider | REAL_ARTIFACT (transformed) |
| 10 | Total CLV ₩57.94 B | sum(clv_data.csv | clv_predictions.csv predicted_clv) → matches | REAL_ARTIFACT |
| 10 | CLV distribution / Top customers | clv_predictions.csv | REAL_ARTIFACT |
| 11 | Persuadable / Sure Thing / Sleeping Dog / Lost Cause | uplift_results.csv `segment` value_counts (real). Actual values: persuadable=2 798, sure_thing=12 919, sleeping_dog=3 683, lost_cause=600. (Audit-prompt's "16,317" does NOT appear in current data — likely from an earlier run.) | REAL_ARTIFACT |
| 11 | Avg Uplift Score | uplift_results.csv uplift_score mean | REAL_ARTIFACT |
| 11 | Qini curve | qini_curve.png from results/ | REAL_ARTIFACT |
| 12 | Survival curves | `_generate_sample_survival_curves` — survival_curves.json absent | SAMPLE_FALLBACK |
| 12 | Cohort retention | cohort_retention_matrix.csv / cohort_analysis.json | REAL_ARTIFACT |
| 13a | Total Scores 200 / scoring distribution | `_generate_sample_scoring_history` (data_loader.py:1748, hard-coded `n=200`); real `scoring_history.csv` does not exist | SAMPLE_FALLBACK |
| 13a | Redis stream length, latency | live Redis xlen via `r.xlen(req_stream)` (app.py:4290-4302) when reachable; "Unavailable" otherwise | REAL (live) or UNKNOWN (offline) |
| 13b | Drift PSI/KS / num drifted features | derived from monitoring_report.json (real) | REAL_ARTIFACT (transformed) |
| 14 | MLflow run count / metrics | load_mlflow_runs → model_performance_history.csv (3 real ml/dl/ensemble rows). Prompt's "3 runs" matches. No live MLflow query. | REAL_ARTIFACT |
| 14 | AUC/precision/recall history charts | `_load_metric_timeseries` from same CSV — degenerate 1-point series per model | REAL_ARTIFACT (sparse) |
| 03 | Churn risk distribution | churn_predictions.csv | REAL_ARTIFACT |
| 06/07/08 | Recommendations table | recommendations.csv | REAL_ARTIFACT |
| System Health | models_available list | models_dir.glob (real file listing, no load) | REAL (metadata only) |

## Fixture / fallback paths in data_loader.py

Sample / hardcoded generators that silently activate when artifacts
are missing (no dashboard banner shown to the viewer):

- `_generate_sample_predictions` (line 773) — not currently triggered
  (predictions exist).
- `_generate_sample_metrics` (line 796) — not currently triggered.
- `_generate_sample_ab_results` (line 813) — not currently triggered.
- `_generate_sample_budget_results` (line 827) — not currently
  triggered.
- `_generate_sample_survival_data` (line 848) — TRIGGERED when
  segments file missing → emits np.random.exponential durations.
- `_generate_sample_recommendations` (line 863) — not currently
  triggered.
- `_generate_sample_uplift_results` (line 911) — not currently
  triggered.
- `_generate_sample_clv_data` (line 925) — not currently triggered.
- `_generate_sample_mlflow_runs` (line 1280) — bypassed because
  load_mlflow_runs delegates to load_model_performance_history
  which uses the real CSV path; the sample generator is dead code
  for now.
- `_generate_sample_roc_data` (line 1316) — TRIGGERED (roc_data.json
  does not exist) → synthetic ROC drawn on Page 02.
- `_generate_sample_confusion_matrices` (line 1348) — TRIGGERED
  (confusion_matrices.json does not exist) → 350/50/80/120 etc shown
  on Page 02; these matrices also OVERWRITE the real precision/recall/
  F1/accuracy headline metrics in app.py:439-462.
- `_generate_sample_ab_detailed` (line 1356) — bypassed (real
  ab_test_detailed.json present).
- `_generate_sample_survival_curves` (line 1420) — TRIGGERED →
  Page 12 KM curves are synthetic.
- `_generate_sample_scoring_history` (line 1748) — TRIGGERED →
  exact `n=200` hardcoded; the prompt's "Total Scores 200" is the
  fixture row count.
- `_generate_sample_drift_history` (line 1782) — bypassed because
  load_drift_history derives from monitoring_report.json.
- `_generate_sample_scoring_throughput` (line 1805) — TRIGGERED for
  Page 13 throughput chart.
- `_generate_sample_retention_offers` (line 1822) — TRIGGERED for
  Page 06/07 personalized retention offer view.
- `_generate_sample_feature_importance` (line 1886) — bypassed.

## Verdict per ML/DL artifact

| Artifact | Trained? | Saved? | Loaded by dashboard for inference? | Dashboard usage |
|---|---|---|---|---|
| **ml_churn_model.pkl.joblib** (lightgbm) | YES (joblib.dump in churn_model.py:663) | YES, 275 KB | **NO** — dashboard never calls joblib.load or model.predict_proba | only filename listed on system-health page |
| **dl_churn_model.pt** (transformer, 33-feat input) | YES (torch.save in churn_model.py:1278) | YES, 445 KB | **NO** — dashboard never calls torch.load | only filename listed on system-health page |
| **clv_model.pkl** (lifetimes / GammaGamma) | YES | YES, 692 KB | **NO** | indirect — CLV predictions read from CSV |
| **survival_model.pkl** (lifelines) | YES | YES, 1.99 MB | **NO** | indirect — survival via segments CSV, KM curves are synthetic |
| **uplift_model.pkl** (T/X/S-learner) | YES | YES, 442 KB | **NO** | indirect — uplift_results.csv only |
| **MLflow runs** | YES (4 run IDs logged in `model_metrics.json.mlflow_runs.run_ids`, tracking_uri=http://mlflow:5000) | YES | NO — load_mlflow_runs reads `model_performance_history.csv` (cached export), not the live MLflow tracking server | Page 14 shows 3 rows derived from that CSV |

## Risks / red flags

1. **HEADLINE PRECISION/RECALL/F1/ACCURACY ON PAGE 02 ARE FIXTURE-DERIVED.**
   app.py:439-462 explicitly OVERWRITES the real precision (0.5331
   for ML), recall (0.7791), and F1 (0.6331) from model_metrics.json
   with values recomputed from the hardcoded
   `[[350,50],[80,120]]` confusion matrix. Real precision 0.5331 is
   never shown; the displayed value is `120/(120+50) = 0.7059`.
   Same for DL and ensemble.

2. **CONFUSION MATRICES ARE STATIC FIXTURES, NOT REAL TEST-SET TALLIES.**
   `confusion_matrices.json` does not exist in either `results/` or
   `data/artifacts/`. Loader falls back to `[[350,50],[80,120]]` etc.
   The matrices total 600 (400+200, 400+200, 400+200). Real test size
   from `model_metrics.json.test_size` is 3 334. The 600-sample
   caption shown to viewers is therefore false; the actual test split
   never produced these counts.

3. **ROC curves on Page 02 are synthetic Beta-distributed lines**,
   not produced by the real model. `roc_data.json` does not exist;
   `_generate_sample_roc_data` runs.

4. **"Total Scores 200" on Page 13 is a fixture row count.**
   `scoring_history.csv` does not exist in `data/artifacts/`; the
   loader returns exactly 200 np.random rows. The figure does NOT
   come from the live Redis stream — Redis stream length is shown
   separately (`xlen`). The 200 number is hardcoded in
   `_generate_sample_scoring_history` (n = 200, line 1751).

5. **Survival curves (Page 12) are synthetic.** `survival_curves.json`
   absent → np.random Kaplan-Meier per segment. Cohort retention
   tables on the same page are real.

6. **No trained model is loaded by the dashboard for inference.**
   All ML/DL/CLV/survival/uplift outputs reach the dashboard solely
   via pre-baked CSVs written by `src/main.py`. The model `.joblib`
   and `.pt` files exist on disk but are dead weight from the
   dashboard's perspective. This is fine for an offline analytics
   dashboard, but invalidates any claim that the dashboard "runs
   the trained model in real time."

7. **Audit prompt's claimed "Persuadable 16,317" KPI does not match
   the current uplift_results.csv** (segment value counts:
   persuadable = 2 798). This is a discrepancy between the prompt and
   current artifacts, not a dashboard bug; current `app.py` reads
   the real segment column correctly.

8. **`model_performance_history.csv` has only 3 rows** (one per
   model, single timestamp). The dashboard's "AUC over time" /
   training-time chart on Page 14 is technically REAL_ARTIFACT but
   degenerate; `training_time_s`, `params_lr`, `params_epochs`
   columns are not in the CSV and are defaulted to 1.0 / 0.1 / 1 by
   data_loader.py:1162-1170 — these defaults are not flagged to the
   viewer.

## Summary

- Total KPIs audited: 28 distinct KPI / chart sources across pages
  02, 03, 05, 06, 07, 08, 10, 11, 12, 13, 14, System Health.
- **REAL_ARTIFACT**: 17 (CLV total, uplift quadrants, AUC trio,
  budget rollups, churn predictions, recommendations, feature
  importance, cohort tables, monitoring report-derived drift, MLflow
  3-row history, segment data, raw events, customers list, Redis
  live xlen, ab_test_detailed, feature store, model file listing).
- **REAL_ARTIFACT (transformed / scaled)**: 2 (budget scenario
  slider, MLflow CSV with defaulted columns).
- **HARDCODED_FIXTURE**: 7 (confusion matrices x3, P/R/F1/Acc
  headline x3 because overwritten from fixture matrices, test-size
  600 caption).
- **SAMPLE_FALLBACK / np.random**: 4 (ROC curves, scoring history
  200, survival curves, scoring throughput; sample retention offers
  on adjacent pages also fallback).
- **UNKNOWN (depends on runtime)**: 1 (live MLflow tracking server
  is never queried by dashboard — current state cached only).

- The 4 trained model artifacts (ml_churn, dl_churn, clv, survival)
  + uplift_model are all saved by the pipeline but **none is loaded
  by the dashboard for inference**. Dashboard is read-only against
  pre-baked CSVs; the .joblib/.pt files are surfaced only as filename
  strings on the system-health page.
