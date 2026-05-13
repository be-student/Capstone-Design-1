# Page 01 — Churn Analytics (full data dump)

## Banners
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- "Churn predictions cover all 20,000 customers."
- "At-Risk Revenue (churn prob > 50%): 2,997,471,916 KRW (5.2% of total CLV)"

## Section structure
- H2: Churn Prediction Analytics
- H3: Churn Risk Summary
- H3: Churn Risk Score Distribution
- H3: Risk Level Breakdown
- H3: Churn Probability Density by Segment
- H3: Churn Probability by Risk Level
- H3: Segment x Risk Level Cross-Tabulation
- H3: Feature Importance Analysis
- H3: Segment-Level Churn Risk Analysis
- H3: Model Performance Summary
- H3: Churn Risk vs Customer Lifetime Value
- H3: High Risk Customers

## KPI cards
| Group | Label | Value |
|---|---|---|
| Summary | Total Customers | 20,000 |
| Summary | Avg Churn Prob | 31.31% |
| Summary | Median Churn Prob | 15.39% |
| Summary | High Risk (>50%) | 5,717 |
| Summary | Critical (>75%) | 3,596 |
| ml_model | AUC | 0.8852 |
| ml_model | F1 Score | 0.6331 |
| ml_model | Precision | 0.5331 |
| ml_model | Recall | 0.7791 |
| dl_model | AUC | 0.8860 |
| dl_model | F1 Score | 0.6531 |
| dl_model | Precision | 0.6759 |
| dl_model | Recall | 0.6318 |
| ensemble | AUC | 0.8866 |
| ensemble | F1 Score | 0.6522 |
| ensemble | Precision | 0.6426 |
| ensemble | Recall | 0.6621 |

## Charts
1. **Distribution of Churn Risk Scores with Threshold Boundaries** (histogram). x=Churn Probability 0–0.9, y=Customer Count 0–3,500. Threshold lines at 0.25/0.50/0.75. Leftmost bin ~3,500.
2. **Customer Risk Level Distribution** (donut). low 57.9% · critical 18% · medium 13.5% · high 10.6%.
3. **Churn Probability Distribution by Segment** (multi-trace histogram). Segments: bargain_hunter, new_customer, dormant, explorer, regular_loyal, vip_loyal.
4. **Churn Probability Distribution by Risk Level** (box plot). x=Risk Level {low, medium, high, critical}, y=Churn Probability 0–1.
5. **Proportion of Risk Levels within Each Segment** (heatmap). 6 segments × 4 risk levels:

| Segment | low | medium | high | critical |
|---|---:|---:|---:|---:|
| vip_loyal | 0.99 | 0.01 | 0.00 | 0.00 |
| regular_loyal | 0.94 | 0.03 | 0.01 | 0.03 |
| new_customer | 0.41 | 0.10 | 0.05 | 0.44 |
| explorer | 0.41 | 0.33 | 0.19 | 0.07 |
| dormant | 0.05 | 0.94 | 0.06 | 0.02 (visual order: low/critical/medium/high — order in DOM may differ; numbers from heatmap) |
| bargain_hunter | 0.87 | 0.06 | 0.02 | 0.04 |

(Numbers in DOM dump came in a different cell order; cross-table semantics are: vip_loyal almost entirely "low", new_customer split between "low" and "critical", dormant dominates "medium" — verifiable in PNG.)

6. **Top 10 Churn Prediction Features** (horizontal bar). Top → bottom: sequence_length, recency, monetary, avg_purchase_cycle_days, frequency, avg_session_duration, stage_tenure_days, tenure_days, purchase_cycle_anomaly, session_duration_change.
7. **Cumulative Feature Importance** (line). Reaches 80% threshold around feature index 5.
8. **Average Churn Risk by Segment** (horizontal bar with sample sizes):

| Segment | n | avg_churn |
|---|---:|---:|
| vip_loyal | 2,030 | very low (~0.01) |
| regular_loyal | 4,949 | low (~0.05) |
| bargain_hunter | 4,087 | low (~0.07) |
| explorer | 2,975 | mid (~0.30) |
| new_customer | 3,014 | high (~0.50) |
| dormant | 2,945 | very high (~0.85+) |

9. **Churn Probability vs Predicted CLV** (scatter). x=Churn Probability 0–1, y=Predicted CLV 0–15M. Color by risk_level {low, high, critical, medium}.

## High-Risk Customers section
- "Churn probability threshold" slider, default 0.50 (range 0.00–1.00)
- "5717 customers above threshold (50%)"

## Cross-page consistency check
- Total Customers 20,000 — matches Page 00.
- Avg Churn Prob 31.31% — matches Page 00.
- High Risk 5,717 — matches Page 00.
- Risk donut shares 57.9 / 18 / 13.5 / 10.6 — matches Page 00 donut exactly.
- Median Churn Prob 15.39% — only here, not on Page 00.
- AUC values 0.8852 / 0.8860 / 0.8866 — should match Page 02.
