# Page 08 — Model Monitoring (full data dump)

## Banners
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- "Best model by AUC: ensemble (AUC = 0.8866)"
- "No performance degradation detected for ensemble."

## KPI cards
| Group | Label | Value |
|---|---|---|
| Drift | Total Checks | **1** |
| Drift | Current Status | **RED** |
| Drift | Red Alerts | 1 |
| Drift | Yellow Alerts | 0 |
| Throughput | Avg Requests/min | 49.0 |
| Throughput | Peak Requests/min | 83.3 |
| Throughput | Avg Latency | 19.1 ms |
| Throughput | Avg Error Rate | **0.0103** (1.03%) |

🚨 **"Total Checks = 1, Status = RED"** — single observation reported as a status. With 1 sample any "trend" is degenerate.

🚨 **Headline says "No performance degradation detected"** while drift status = **RED**. Direct contradiction.

## Drift charts (single-point)
1. **Drift Alert Timeline**: x = `11:56:50.282 → .2835` May 10 2026 (**1.5ms x-axis span!**), y=Number of Drifted Features 6–8.
2. **Mean PSI Over Time**: same 1.5ms x-axis, y=PSI 0–0.25 with thresholds Yellow=0.1, Red=0.25.
3. **Mean KS Statistic Over Time**: same 1.5ms x-axis, y=KS 0.02–0.05 with Warning=0.05.

🚨 All 3 "trend" charts have a **1.5-millisecond x-axis** = single data point, not a trend.

## Model Performance Comparison
Same numbers as Page 02 (auc/prec/rec/f1/accuracy for ml/dl/ensemble).

## Training Run History
x-axis: `11:56:50.2892 → 11:56:50.2893` May 10 2026 (**0.1ms x-axis**!) Three runs plotted as a temporal series.

## Scoring Throughput / Latency / Error Rate (all dated Oct 15-16, 2024)
- **Throughput**: x = Oct 15 2024 00:00 → 18:00, y = 0–80 req/min (matches Avg 49 / Peak 83.3 KPI).
- **Latency**: x = Oct 15 2024 timestamps, y = 15–35 ms. Avg KPI says 19.1 ms.
- **Error Rate**: x = Oct 15 → Oct 16 2024, y = 0%-2.0%. Avg KPI says 1.03%.

🚨 **Time anchor mismatch**: Drift charts dated **May 10, 2026 11:56:50** (1.5ms span). Throughput/latency/error charts dated **Oct 15-16, 2024** (24-hour span). **~19 months gap on the same monitoring page.**

## Survival curves duplicated from Page 07
"Kaplan-Meier Survival Curves by Segment" reused with same 6 behavioral segments. Page-scope creep.

## Monitoring Configuration (visible JSON)
```json
PSI: {n_bins:10, binning_strategy:"quantile", yellow_threshold:0.1, red_threshold:0.25}
KS:  {warning_threshold:0.05, drift_threshold:0.01}
Settings: {alert_on_yellow:false, alert_on_red:true, log_to_mlflow:true}
```

## Issues
1. **"No degradation detected" vs "Status: RED"** — direct contradiction.
2. **Drift "trends" on 1.5ms x-axis** — single-point.
3. **Time anchor mismatch** drift (May 2026) vs throughput (Oct 2024) — 19-month gap.
4. **Error rate 1.03%** is 10× SaaS SLO target (<0.1%).
5. **Page 07 KM curves duplicated** — scope creep.
6. **Training Run History x-axis 0.1ms wide** — 3 calls plotted as time series.
