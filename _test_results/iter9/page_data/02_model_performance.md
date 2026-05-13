# Page 02 — Model Performance (full data dump)

## Banners
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- "Ensemble AUC: 0.8866 (>= 0.78 threshold)"
- "ML Weight: 0.6 | DL Weight: 0.4"

## Section structure
- H2: Model Performance
- H3: Performance Comparison
- H3: Metrics Comparison Chart
- H3: ROC Curves
- H3: Confusion Matrices
- H3: Model Capability Radar
- H3: MLflow Experiment Runs
- H3: Ensemble Configuration

## KPI cards (top)
| Label | Value |
|---|---|
| ML Model AUC | 0.8852 |
| DL Model AUC | 0.8860 |
| Ensemble AUC | 0.8866 |
| Best Model | ensemble |

## Metrics comparison chart
auc / precision / recall / f1_score / accuracy across ml_model, dl_model, ensemble. (Numerical values match Page 01's per-model KPI cards.)

## ROC curves
- ml_model AUC=0.885
- dl_model AUC=0.886
- ensemble AUC=0.887
- Random AUC=0.5

## Confusion matrices (BIG INCONSISTENCY vs headline KPIs)

| Model | TN | FP | FN | TP | Acc | Prec | Rec |
|---|---:|---:|---:|---:|---:|---:|---:|
| ml_model | 350 | 50 | 80 | 120 | 78.33% | 70.59% | 60.00% |
| dl_model | 340 | 60 | 90 | 110 | 75.00% | 64.71% | 55.00% |
| ensemble | 360 | 40 | 70 | 130 | 81.67% | 76.47% | 65.00% |

**Total cases on each matrix = 600** — not 20,000. This is NOT the same evaluation set as the headline AUC table.

**Headline-vs-matrix discrepancy (CRITICAL):**

| Model | Headline Precision | Matrix Precision | Headline Recall | Matrix Recall |
|---|---:|---:|---:|---:|
| ml_model | 0.5331 | 0.7059 | 0.7791 | 0.6000 |
| dl_model | 0.6759 | 0.6471 | 0.6318 | 0.5500 |
| ensemble | 0.6426 | 0.7647 | 0.6621 | 0.6500 |

→ Headline KPI table and confusion matrix below it report different precision and recall for the same model. **At least one is wrong**, or they're computed on different test splits/thresholds without disclosure.

## MLflow Experiment Runs
- Training Time: ml_model 1.0s, dl_model 1.0s, ensemble 1.0s — **identical synthetic floor**.
- AUC vs Training Time Trade-off: 3 points all at x=1, y between 0.8855–0.886 → **degenerate scatter**.

## Ensemble Configuration
- ML Weight 60% | DL Weight 40%
- Pie chart "Ensemble Weight Distribution" (60/40 slice)

## Cross-page consistency check
- Headline AUCs 0.8852/0.8860/0.8866 match Page 01 ✓
- Confusion matrix on this page conflicts with Precision/Recall on Page 01 (which match this page's headline) — so the conflict is **inside this page**, not across pages.
