# Page 07 — Survival Analysis (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## Section structure
- H2: Survival Analysis
- H3: Kaplan-Meier Survival Curves by Segment / Median Survival Time by Segment / Average Survival Probability by Segment / Estimated Hazard Rate by Segment / Event Rate by Segment / Customer Lifetime Duration Distribution / Duration Distribution by Segment / Survival Model Configuration

## KPI cards
| Label | Value |
|---|---|
| Total Customers | 20,000 |
| Events Observed (Churn) | **5,717** |
| Event Rate | **28.59%** |
| Median Duration | **309 days** |

🚨 **Churn count = 5,717 matches Page 01 "High Risk (>50%) = 5,717"** — same number used as both predicted-high-risk and observed-events. Probably one population is being mislabeled as the other.

## KM curves x-axis: 0–350 days. Median 309d on a ~350d horizon → right-censoring artifact.

## Average Survival Probability by Segment (uplift segments — different taxonomy than Page 03!)
| Segment | Avg Survival |
|---|---:|
| high_value_sure_thing | 97.68% |
| mid_value_sure_thing | 86.98% |
| low_value_sure_thing | 73.04% |
| high_value_persuadable | 39.84% |
| mid_value_persuadable | 39.78% |
| sleeping_dog | 38.81% |
| high_value_lost_cause | 37.23% |
| low_value_persuadable | 22.28% |

🚨 **Two different segment taxonomies on the dashboard**: Page 03 uses {vip_loyal, regular_loyal, bargain_hunter, explorer, dormant, new_customer} (6 behavioral); Page 07 uses {high/mid/low × value × persuadable/sure_thing/lost_cause/sleeping_dog} (8 uplift). No crosswalk visible.

## Daily Hazard Rate by Segment (behavioral segments — yet ANOTHER taxonomy!)
| Segment | Hazard/day |
|---|---:|
| dormant | 0.00254 |
| new_customer | 0.00166 |
| bargain_hunter | 0.00120 |
| explorer | 0.00099 |
| regular_loyal | 0.00062 |
| vip_loyal | 0.00023 |

🚨 **Same page mixes BOTH taxonomies** (uplift in Avg Survival, behavioral in Daily Hazard). Different segments → not comparable.

## Event Rate by Segment (binary 0/1 anomalies)
| Segment | Event Rate |
|---|---:|
| high_value_lost_cause | 100.0% |
| high_value_persuadable | 100.0% |
| high_value_sure_thing | 0.0% |
| low_value_persuadable | 100.0% |
| low_value_sure_thing | 0.0% |
| mid_value_persuadable | 100.0% |
| mid_value_sure_thing | 0.0% |
| sleeping_dog | 65.8% |

🚨 **Binary 0% / 100% / 0% / 100% pattern** — 7 of 8 segments show exactly 0% or 100%. Either segment definition perfectly correlates with churn (which would be tautological) or sample sizes are extremely small. This is **suspicious as a production signal**.

## Customer Duration Distribution
x: duration_days 0–350, y: count 0–2,500. mid_value_sure_thing dominates the 350-day bin (~2,500) — heavy right-censoring at the window boundary.

## Survival Model Configuration
```json
{
  "penalizer": 0.01,
  "l1_ratio": 0,
  "alpha": 0.05
}
```

## Issues
1. **Churn count 5,717 = High Risk count 5,717** (Page 01) — coincidence or "events" is just count of high-prob predictions, not real outcomes.
2. **Two segment taxonomies on the same page**.
3. **Event Rate binary 0%/100% for 7 of 8 segments** — likely a labeling tautology.
4. **Median 309 days near 350-day horizon** = right-censoring artifact, no annotation.
5. No **at-risk table** (n at each timepoint) under the KM curves.
6. No **95% confidence bands** on the curves.
7. No **log-rank** test between segments.
