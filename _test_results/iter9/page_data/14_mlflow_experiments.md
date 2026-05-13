# Page 14 — MLflow Experiments (full data dump)

## Banners
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- 🚨 "**MLflow tracking server not available. Showing cached experiment data from artifacts.**"

## MLflow Configuration (visible JSON)
```json
{
  "tracking_uri": "sqlite:///mlflow/mlflow.db",
  "experiment_name": "churn_prediction",
  "log_models": true,
  "log_artifacts": true
}
```

## KPI cards
| Label | Value |
|---|---|
| Total Runs | **3** |
| Best AUC | 0.8866 |
| Best Model | ensemble |
| Total Training Time | **3s** |

## AUC by Model Type (bar)
| Model | AUC |
|---|---:|
| ensemble | 0.8866 |
| dl_model | 0.8860 |
| ml_model | 0.8852 |

Threshold line at 0.78.

## Learning Rate vs AUC (scatter)
- 3 model types plotted; Learning Rate axis labels show 0.01, 0.1, 1 (log-scale tick marks 2/5 in between).
- AUC y-range very narrow: 0.8855–0.8865.
- 3 points cluster tightly — no genuine sweep.

## Epochs vs AUC (size = training time)
- 3 points
- AUC range 0.8855–0.8865
- Epochs axis 0–2

## AUC vs Training Time
- 3 points
- AUC range 0.8855–0.8865
- Training Time 0–2 seconds

## AUC per Training Second (Efficiency)
| Model | auc/sec |
|---|---:|
| ensemble | 0.8866 (assuming time=1s) |
| dl_model | 0.8860 |
| ml_model | 0.8852 |

🚨 If all models trained in 1s, "AUC per second" = AUC. Chart is uninformative.

## Experiment Timeline
- x = `11:56:50.2892 → 11:56:50.2893` May 10 2026 — **0.1ms x-axis**
- y = AUC 0.78–0.88
- 3 model points plotted as a temporal series

🚨 **3 timestamps within 0.1 milliseconds plotted as time-series** = three sequential function calls dressed as a longitudinal experiment timeline.

## MLflow Run Performance Comparison (radar)
H3 exists, axes are auc/precision/recall/f1_score/accuracy 0–1, three models plotted. From DOM extractor, no value annotations rendered → likely intact.

## Issues
1. **MLflow tracking server NOT available** — page is in fallback mode showing cached artifacts. Procurement-blocking for any SaaS that promises model lineage.
2. **3 runs total = no real experiment history** — cached snapshot, not a live registry.
3. **Hyperparameter "sweep" is degenerate** — AUC range 0.0014 across 3 runs.
4. **Time axis 0.1ms wide** — not a temporal series.
5. **Total Training Time 3s** — synthetic floor (1s × 3 models).
6. **No model registry stages** (Staging / Production / Archived) visible.
7. **No model promotion / rollback history**.
8. **No experiment metadata** (training data version, feature set ID, code commit SHA).
