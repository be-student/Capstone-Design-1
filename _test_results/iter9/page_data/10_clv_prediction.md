# Page 10 — CLV Prediction (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## KPI cards
| Label | Value |
|---|---|
| Total CLV | **57,936,514,970 KRW** |
| Average CLV | 2,896,826 KRW |
| Median CLV | **1,701,727 KRW** |
| CLV Std Dev | 3,575,497 KRW |

Internal sanity: Total CLV / 20,000 = 2,896,825.7 → matches Average CLV ✓.

## CLV Distribution
- Mean (annotation in chart): 2,896,826
- Median (annotation in chart): 1,701,727
- Histogram x = 0–15M, y = 0–4,000

## Average CLV by Segment (8 uplift segments)
| Segment | Mean CLV (KRW) |
|---|---:|
| high_value_persuadable | 8,563,357 |
| high_value_sure_thing | 8,416,210 |
| high_value_lost_cause | 4,153,154 |
| mid_value_sure_thing | 2,443,032 |
| mid_value_persuadable | 1,756,438 |
| low_value_sure_thing | 1,171,391 |
| sleeping_dog | 722,502 |
| low_value_persuadable | 424,653 |

## Total CLV by Segment
| Segment | Total CLV (KRW) |
|---|---:|
| high_value_sure_thing | 33,639,590,866 |
| mid_value_sure_thing | 16,942,427,813 |
| sleeping_dog | 2,546,096,653 |
| low_value_sure_thing | 2,512,633,819 |
| low_value_persuadable | 1,176,712,448 |
| mid_value_persuadable | 1,097,773,503 |
| high_value_lost_cause | 4,153,154 |
| high_value_persuadable | 17,126,714 |

🚨 **Total CLV by Segment sum check**: 33.64B + 16.94B + 2.55B + 2.51B + 1.18B + 1.10B + 4.15M + 17.1M = **57.93B** ≈ matches Total CLV 57.94B ✓.

🚨 **high_value_persuadable**: Mean CLV = 8,563,357 but Total = only 17,126,714 → only **2 customers** in this segment! 

🚨 **high_value_lost_cause**: Mean = 4,153,154, Total = 4,153,154 → only **1 customer**!

🚨 So "high_value_persuadable" with 2 customers received 31,000 KRW from the LP (Page 05). On Page 11 it's part of the "Persuadable 16,317" cohort — yet only 2 customers fit "high_value_persuadable". The taxonomies don't match cohort sizes.

## CLV Tier Distribution
4 tiers: Platinum / Gold / Silver / Bronze, each 25%.

## CLV Percentile Analysis
| Percentile | CLV (KRW) |
|---|---:|
| P10 | 66,322 |
| P25 | 973,223 |
| P50 | 1,701,727 |
| P75 | 3,075,712 |
| P90 | 6,767,562 |
| P95 | 13,032,750 |
| P99 | 14,716,426 |

## CLV vs Churn Risk scatter
x = Churn Probability **−0.5 to 1.5** ⚠ probability shouldn't be negative.

## Issues
1. **Segment sizes wildly imbalanced**: high_value_persuadable n=2, high_value_lost_cause n=1 (vs total 20k).
2. **CLV vs Churn x-axis includes negative probabilities** (-0.5 to 1.5 range).
3. **CLV Tier Distribution all exactly 25%** — quartile split, but "Tier" implies business-defined; here it's just statistical quartiles.
4. **No CIs** on percentile values.
5. **No model version / training date / data freshness** on this page.
6. **Both behavioral and uplift taxonomies absent**: only uplift segments shown here, but Page 03 Segmentation page used behavioral.
