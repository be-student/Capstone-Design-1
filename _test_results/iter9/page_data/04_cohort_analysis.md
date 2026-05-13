# Page 04 — Cohort Analysis (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## Section structure
- H2: Cohort Analysis
- H3: Cohort Overview / Retention Heatmap / Retention Curves by Cohort / Average Retention Curve / Period-over-Period Retention Change / Retention Matrix (Raw Data)

## KPI cards
| Label | Value |
|---|---|
| Total Cohorts | **4** (only Jan/Feb/Mar/Apr 2024) |
| Periods Tracked | 13 |
| Avg Period-1 Retention | 99.0% |
| Avg Final Retention | **2.5%** |

## Retention Heatmap (full matrix)

| Cohort | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan 2024 | 100.0% | 99.1% | 98.3% | 97.5% | 96.2% | 94.9% | 93.9% | 92.4% | 91.1% | 89.3% | 88.4% | 88.2% | **10.2%** |
| Feb 2024 | 100.0% | 99.1% | 98.5% | 97.6% | 96.4% | 94.9% | 93.3% | 91.8% | 90.4% | 89.4% | 89.1% | **7.9%** | 0.0% |
| Mar 2024 | 100.0% | 99.0% | 98.3% | 97.3% | 95.6% | 94.3% | 92.5% | 91.0% | 89.9% | 89.3% | **9.8%** | 0.0% | 0.0% |
| Apr 2024 | 100.0% | 98.9% | 98.1% | 96.5% | 95.5% | 93.5% | 92.4% | 91.0% | **92.1%** | 12.9% | 0.0% | 0.0% | 0.0% |

🚨 **CRITICAL DATA ANOMALIES:**
1. **Apr 2024 cohort**: Period 7 = 91.0% → Period 8 = **92.1%** (retention monotonicity violation: cannot go UP).
2. **Last surviving period before zero-truncation**: Jan P12 = 10.2%, Feb P11 = 7.9%, Mar P10 = 9.8%, Apr P9 = 12.9% — all are sudden ~80-percentage-point drops, suggesting these are **data-window-end artifacts** (denominator collapse), not real attrition events.
3. **0.0% trailing cells** are because the cohort hasn't been observed for that many months yet — these zeros are mixed into "Avg Final Retention 2.5%" without filtering.

## Period-over-Period Retention Change
| P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +0.0% | -1.0% | -0.8% | -1.1% | -1.3% | -1.5% | -1.4% | -1.5% | -0.7% | -20.7% | -23.4% | -22.8% | -21.5% |

→ Average drop of -1% per period for P1–P8, then suddenly -20%+ drops in P9+ — the latter are the truncation artifacts.

## Retention Matrix (Raw Data)
H3 exists but raw data section not surfaced via DOM extractor (likely a dataframe).

## Cross-page consistency
- Total Customers (page 00/01/03) = 20,000. Cohort cohorts sum (visible from heatmap P0 = 100% for each of 4 cohorts) — population per cohort not shown.

## Key issues
1. **Only 4 cohorts** (Jan-Apr 2024) — below SaaS norm of ≥6 monthly cohorts for any longitudinal narrative.
2. **Avg Final Retention 2.5%** is computed by averaging 0.0% trailing cells (truncation artifact).
3. **Retention monotonicity violation** in Apr 2024 (91.0% → 92.1%, P7 → P8).
4. No CIs on retention curves; no per-cohort sample size (n=?) shown; no segment/feature drill-down.
