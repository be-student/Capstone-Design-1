# Page 12 — CLV & Retention Campaign (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## Section structure (4-section narrative)
- 1. Customer Lifetime Value Overview
- 2. Uplift Modeling & Treatment Effectiveness
- 3. Budget Optimization Outcomes
- 4. Campaign ROI Metrics

## KPI cards (per section)
| Section | Label | Value |
|---|---|---|
| 1 CLV | Total CLV | 57,936,514,970 KRW |
| 1 CLV | Avg CLV | 2,896,826 KRW |
| 1 CLV | At-Risk CLV | 2,997,471,916 KRW |
| 1 CLV | At-Risk CLV % | 5.2% |
| 2 Uplift | Avg Uplift | **0.0434** |
| 2 Uplift | Max Uplift | 0.6874 |
| 2 Uplift | Treatable Customers | **16,317 (81.6%)** |
| 3 Budget | Budget Allocated | 50,000,000 KRW |
| 3 Budget | Revenue Saved | **192,155,554 KRW** |
| 3 Budget | Customers Retained | **122.29548658078494** ⚠ float leak |
| 3 Budget | Overall ROI | **3.8x** |

## CRITICAL UI bug
🚨 **Customers Retained = "122.29548658078494"** — full precision Python float spilled into a customer-facing KPI card. Should be `122` or `122.3`.

## ROI by Segment (full numbers)
| Segment | ROI |
|---|---:|
| high_value_persuadable | 8.0x |
| high_value_sure_thing | 5.1x |
| mid_value_persuadable | 4.1x |
| low_value_persuadable | 3.8x |
| mid_value_sure_thing | 3.5x |
| low_value_sure_thing | 3.1x |
| sleeping_dog | 0.0x |
| high_value_lost_cause | 0.0x |

🚨 **"Overall ROI 3.8x" exactly matches low_value_persuadable's segment ROI 3.8x** — coincidence or display bug?

## Expected Revenue Saved by Segment
| Segment | Revenue Saved (KRW) |
|---|---:|
| mid_value_persuadable | 86,389,749 |
| low_value_persuadable | 53,124,598 |
| mid_value_sure_thing | 44,134,867 |
| low_value_sure_thing | 4,794,955 |
| high_value_sure_thing | 3,463,472 |
| high_value_persuadable | 247,912 |
| high_value_lost_cause | 0 |
| sleeping_dog | 0 |

Sum: 192,155,553 ≈ matches "Revenue Saved 192,155,554" (off by 1 KRW rounding).

## Cost per Retained Customer
| Segment | Cost/Retention (KRW) |
|---|---:|
| high_value_sure_thing | 768,851 |
| mid_value_persuadable | 557,158 |
| high_value_persuadable | 539,865 |
| mid_value_sure_thing | 422,742 |
| low_value_sure_thing | 420,766 |
| low_value_persuadable | 309,586 |
| high_value_lost_cause | 0 |
| sleeping_dog | 0 |

## Cumulative Uplift Curve (Qini-style)
- x = % Customers Treated 0-90
- y = Cumulative Uplift 0-1200

## Cross-page consistency
- Total CLV 57,936,514,970 — matches Page 10 ✓
- Avg CLV 2,896,826 — matches Page 10 ✓
- Avg Uplift 0.0434 — matches Page 11 ✓
- Treatable 16,317 — matches Page 11 Persuadable count ✓
- Budget 50,000,000 — matches Page 05 ✓
- Revenue Saved 192,155,554 — matches Page 05 192,155,551 within 3 KRW ✓
- **Overall ROI 3.8x here vs 3.5x on Page 05 vs 9.0x on Page 09** ⚠
- **Avg Uplift 0.0434 here vs Avg Expected Uplift 6.36% on Page 09** ⚠ different units (decimal vs percent), but 0.0434 ≠ 0.0636

## Issues
1. **Customers Retained = 122.29548658078494** (14 decimals) — format helper bug.
2. **3 different "Overall ROI" values** across Pages 05/09/12: 3.5 / 9.0 / 3.8.
3. **Avg Uplift 0.0434 ≠ Page 09 Avg Expected Uplift 6.36%** — same population, different value.
4. **Section 1 CLV chart uses behavioral segments** (vip_loyal, dormant…) while Sections 2-4 use uplift segments (high/mid/low_value_*) — taxonomy mixing on one page.
5. **Campaign Effectiveness Radar** chart H3 exists but no numeric values rendered (likely n=0 traces).
