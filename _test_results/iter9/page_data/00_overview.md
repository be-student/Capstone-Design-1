# Page 00 — Overview (full data dump)

**Captured:** 2026-05-10 from `http://localhost:8501` (FULL mode, n=20,000)
**scrollHeight:** 3,701 px

## Banners (alerts)
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- "Churn predictions cover all 20,000 customers."

## Section structure
- H2: Churn Prediction Overview
- H3: Churn Probability Distribution
- H3: Risk Level Distribution
- H3: Average Churn Probability by Segment
- H3: Feature Importance
- H3: Customer Segment Overview
- H3: Individual Customer Lookup
- H3: Customer: C000000

## KPI cards
| Label | Value |
|---|---|
| Total Customers | 20,000 |
| Avg Churn Prob | 31.31% |
| High Risk | 5,717 |
| Total CLV | 57,936,514,970 KRW |
| Churn Probability (Customer C000000) | 3.09% |
| Risk Level (Customer C000000) | LOW |
| Segment (Customer C000000) | bargain_hunter |
| Predicted CLV (Customer C000000) | 2,716,186 KRW |
| Recommended Action (Customer C000000) | N/A |
| Days Since Purchase (Customer C000000) | 0 |

## Charts (plotly)
1. **Distribution of Churn Probabilities** (histogram). x=Churn Probability (0–0.9+), y=count (0–4000). Mass at 0–0.05 ≈ 4,000; right tail spike ≈ 1,700 at p≈0.9 → **bimodal**.
2. **Customer Risk Levels** (donut/pie). low 57.9% · critical 18% · medium 13.5% · high 10.6%.
3. **Churn Rate by Customer Segment** (bar). x=Segment {bargain_hunter, dormant, explorer, new_customer, regular_loyal, vip_loyal}, y=Avg Churn Prob (0–0.8).
4. **Top 10 Feature Importance Scores** (horizontal bar). Features (low → high importance): session_duration_change, purchase_cycle_anomaly, tenure_days, stage_tenure_days, avg_session_duration, frequency, avg_purchase_cycle_days, monetary, recency, sequence_length.
5. Customer-level churn-risk gauge for selected customer C000000 (3.1).

## Sidebar context (visible globally)
- Churn Definition: No purchase 30 days, No login 60 days, Operator OR
- Budget Total: 50,000,000 KRW
- Ensemble Weights: ML 0.6 | DL 0.4

## Inferred consistency cross-checks
- Risk donut: 18% critical → **3,600 customers** (matches Page 01 "Critical (>75%) = 3,596")
- Risk donut: 10.6% high → **2,120 customers** (Page 01 says High>50% = 5,717 which includes Critical). High-only = 5,717 - 3,596 = 2,121 ≈ matches 10.6%. ✓
- Risk donut: 57.9% low → 11,580. Page 01 only surfaces High and Critical counts, not Low or Medium.
- Total CLV / 20,000 = 2,896,826 KRW per customer = matches order of magnitude vs Customer C000000's predicted CLV 2,716,186.

## What is NOT shown (but should be on a SaaS-grade Overview)
- "as of" timestamp (data freshness)
- Model version / scoring batch ID
- Period-over-period delta (vs last week / last month)
- Threshold definitions (what does "High Risk" mean? — only revealed on Page 01)
- Segment / date / cohort filter
- Confidence interval on Avg Churn Prob (31.31% point estimate)
- Class-balance disclosure (positive class rate)
- Confusion matrix or backtest of yesterday's predictions vs today's outcomes
- Methodology / model card link
