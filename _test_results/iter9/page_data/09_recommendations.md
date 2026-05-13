# Page 09 — Recommendations (full data dump)

## Banner
- 🧪 "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## KPI cards (top)
| Label | Value |
|---|---|
| Total Recommendations | 20,000 |
| Avg Expected Uplift | **6.36%** |
| Top Action Type | No Action |
| High Priority | **16,106** |

## Cost-Benefit KPI cards (mid-page)
| Label | Value |
|---|---|
| Total Campaign Cost | 1,211,055 KRW |
| Est. Revenue Saved | 10,893,463 KRW |
| Overall ROI | **9.0x** |
| Avg Expected Uplift | **10.88%** ⚠ different from top KPI 6.36% |

## Recommendation Distribution
| Action | Count | Share |
|---|---:|---:|
| no_action | 16,602 | 83% |
| coupon | 3,398 | 17% |
| **Sum** | **20,000** | 100% |

## Avg Expected Uplift by Action
| Action | Avg Uplift |
|---|---:|
| coupon | 16.12% |
| no_action | 4.36% |

## Cost-by-Offer-Type
| Offer Type | Cost (KRW) |
|---|---:|
| premium_discount | 559,170 |
| discount_coupon | 528,658 |
| engagement_email | 108,831 |
| loyalty_points | 14,396 |

Sum: 1,211,055 ✓ matches Total Campaign Cost.

## ROI by Offer Type
| Offer Type | ROI |
|---|---:|
| premium_discount | 10.1x |
| loyalty_points | 9.8x |
| engagement_email | 9.3x |
| discount_coupon | 7.8x |

## Priority Score Distribution
- Range: 0–1
- Mean: 0.81
- High Priority threshold: implicit (16,106 of 20,000 = 80.5%)

## Issues
1. **Two "Avg Expected Uplift" KPIs**: 6.36% (top, all 20k) vs 10.88% (mid, treated only). Same label, different values, no footnote.
2. **High Priority 16,106 vs Coupon recipients 3,398** → 12,708 high-priority customers receive **no_action**. The dashboard does not explain this gap inline.
3. **ROI 9.0x** here vs **3.5x** on Page 05 vs (TBD) on Page 12 — definition trap.
4. **No churn-prob threshold floor** for coupon eligibility shown.
5. **No model version / data freshness**.
