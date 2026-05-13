# A5 — Recommendations / CLV / Retention Campaign

Auditor stance: independent reviewer, no prior context. All numbers cited are pulled directly from the PNG screenshots and the matching `page_data/*.md` dumps under `C:\Users\yoonc\Capstone-Design-1\_test_results\`.

---

## Page 09 — Recommendations

### Visible KPIs
- Top KPI strip: Total Recommendations **20,000** | Avg Expected Uplift **6.36%** | Top Action Type **No Action** | High Priority **16,106**
- Cost-Benefit strip (mid-page): Total Campaign Cost **1,211,055 KRW** | Est. Revenue Saved **10,893,463 KRW** | Overall ROI **9.0x** | Avg Expected Uplift **10.88%**
- Recommendation Distribution: no_action **16,602 (83%)**, coupon **3,398 (17%)**.
- Avg Uplift by Action: coupon **16.12%**, no_action **4.36%**.
- Cost-by-Offer-Type: premium_discount 559,170 / discount_coupon 528,658 / engagement_email 108,831 / loyalty_points 14,396 (sum 1,211,055 ✓).
- ROI by Offer Type: premium_discount **10.1x**, loyalty_points **9.8x**, engagement_email **9.3x**, discount_coupon **7.8x**.
- Priority Score Distribution mean ≈ 0.81; High Priority bucket = 16,106 / 20,000 = 80.5%.

### Wrong / Unreliable
- **Two "Avg Expected Uplift" KPIs with the same label, different values**: 6.36% (top, computed across all 20,000 customers) vs 10.88% (mid, computed across the 3,398 treated customers). Same wording, no footnote, no scope qualifier — an analyst will not be able to tell which is which without reading source code. This is the core money-domain trust failure on this page.
- **High Priority 16,106 vs coupon recipients 3,398** → **12,708 "high priority" customers receive `no_action`**. The page never reconciles "priority" with "treated"; if priority is supposed to drive intervention, ~79% of the priority queue is being explicitly ignored, and that is silent.
- **Overall ROI 9.0x** disagrees with Page 05's 3.5x and Page 12's 3.8x for the same campaign. With Cost 1,211,055 and Revenue Saved 10,893,463 the arithmetic on this page (10,893,463 / 1,211,055 = 8.99) is internally consistent, so the divergence is a definitional problem (treated subset only vs full budget vs full population), not a math bug — but it is shown without that qualification.
- ROI 9.0x is suspicious next to Page 12's "Revenue Saved 192M / Budget 50M" → it suggests this page only counts the 1.2M actually spent on coupons, not the 50M budget envelope. That is a defensible analytic choice but it is undocumented on the card.
- Avg Uplift by Action: no_action shows **4.36%** uplift — uplift on customers who received no treatment should be 0 by definition; this is the "predicted uplift if treated" for the not-treated population, again unlabelled.

### Missing
- No model/version, training timestamp, data freshness, or cohort-as-of date.
- No threshold definition for "High Priority" (the 0.81 cut is implicit).
- No churn-probability floor for coupon eligibility shown.
- No CIs / variance bands on uplift.
- No reconciliation panel between "High Priority" count and "treated" count.

---

## Page 10 — CLV Prediction

### Visible KPIs
- Top KPI strip: Total CLV **57,936,514,970 KRW** | Average CLV **2,896,826 KRW** | Median CLV **1,701,727 KRW** | CLV Std Dev **3,575,497 KRW**.
- 57,936,514,970 / 20,000 = 2,896,825.7 ≈ Average CLV ✓.
- CLV by Segment (Mean | Total):
  - high_value_persuadable 8,563,357 | **17,126,714**
  - high_value_sure_thing 8,416,210 | 33,639,590,866
  - high_value_lost_cause 4,153,154 | **4,153,154**
  - mid_value_sure_thing 2,443,032 | 16,942,427,813
  - mid_value_persuadable 1,756,438 | 1,097,773,503
  - low_value_sure_thing 1,171,391 | 2,512,633,819
  - sleeping_dog 722,502 | 2,546,096,653
  - low_value_persuadable 424,653 | 1,176,712,448
- Percentiles: P10 66,322 / P25 973,223 / P50 1,701,727 / P75 3,075,712 / P90 6,767,562 / P95 13,032,750 / P99 14,716,426.
- CLV Tier Distribution: Platinum/Gold/Silver/Bronze each 25% (statistical quartile, not business tier).

### Wrong / Unreliable
- **Tiny segments hidden in the headline**:
  - high_value_persuadable: Total 17,126,714 / Mean 8,563,357 → **n = 2 customers**.
  - high_value_lost_cause: Total 4,153,154 = Mean 4,153,154 → **n = 1 customer**.
  - These two segments together hold 3 customers but are presented next to a 33.6B / 16.9B segment with no count column highlighted. The "Mean CLV by Segment" bar chart treats n=1 and n=4,000 segments as visually equivalent — a buyer-trust hazard if anyone sorts or budgets by mean.
- **CLV vs Churn Risk scatter x-axis spans roughly −0.5 to 1.5** — a probability axis must be [0, 1]. Either auto-padding from matplotlib leaked through to the customer view, or the underlying churn_probability column has values outside [0,1]. Either way it makes the page look unvalidated.
- **CLV Tier Distribution exactly 25/25/25/25** — labelled "Tier Classification" but is just a quartile cut on the predicted CLV column. A SaaS buyer reading "Platinum/Gold/Silver/Bronze" will assume business definitions (e.g. revenue, tenure) and be misled.
- The 8 uplift-segment taxonomy used here does not match the behavioral-segment taxonomy used on Page 03 / Section 1 of Page 12; the dashboard has two parallel customer ontologies and no map between them.

### Missing
- **No segment counts column** in the by-segment table (the very thing that would expose n=1 and n=2).
- No model version, training date, churn-prob model that feeds the scatter, or data-freshness stamp.
- No CIs on percentiles or means.
- No outlier flag on the n=1/n=2 segments.
- Top/Bottom 10 customer tables show `nan` in the churn_probability column (visible in the PNG) — silent NaN propagation.

---

## Page 12 — CLV & Retention Campaign

### Visible KPIs
- Section 1 (CLV): Total CLV **57,936,514,970 KRW** | Avg CLV **2,896,826 KRW** | At-Risk CLV **2,997,471,916 KRW** | At-Risk CLV % **5.2%**.
- Section 2 (Uplift): Avg Uplift **0.0434** | Max Uplift **0.6874** | Treatable Customers **16,317 (81.6%)**.
- Section 3 (Budget): Budget Allocated **50,000,000 KRW** | Revenue Saved **192,155,554 KRW** | Customers Retained **122.29548658078494** | Overall ROI **3.8x**.
- ROI by Segment: high_value_persuadable 8.0x, high_value_sure_thing 5.1x, mid_value_persuadable 4.1x, low_value_persuadable 3.8x, mid_value_sure_thing 3.5x, low_value_sure_thing 3.1x, sleeping_dog 0.0x, high_value_lost_cause 0.0x.
- Expected Revenue Saved by Segment sums to 192,155,553 ≈ headline 192,155,554 (1 KRW rounding ✓).

### Wrong / Unreliable
- **`Customers Retained = 122.29548658078494`** — full IEEE-754 float spilled into a customer-facing KPI card (14 decimal digits). This is a one-line format helper bug and is the single most damaging visual on the entire money domain — it screams "untested."
- **"Overall ROI 3.8x" is bit-for-bit identical to low_value_persuadable's segment ROI 3.8x.** Either the "Overall" tile is accidentally pulling that one segment's ROI, or it is a coincidence that should be footnoted.
- **Three different "Overall ROI" values across the product for the same campaign**: Page 05 = 3.5x, Page 09 = 9.0x, Page 12 = 3.8x. No definition shown next to any of the three.
- **Two "uplift" headline metrics that should reconcile but don't**:
  - Page 09 Avg Expected Uplift = 6.36% (≈ 0.0636).
  - Page 12 Avg Uplift = 0.0434.
  Both are advertised as "average uplift over the same 20k population" with no scope qualifier; 0.0434 ≠ 0.0636 even after unit normalization.
- **Taxonomy mixing within a single page**: Section 1 ("Customer Lifetime Value Overview") chart uses behavioral segments (vip_loyal, dormant, …); Sections 2-4 use uplift segments (high/mid/low_value_persuadable, sure_thing, lost_cause, sleeping_dog). A reader cannot trace a customer between sections.
- Campaign Effectiveness Radar in Section 4 renders the polygon outline but no numeric scale labels — visually present, analytically opaque.
- Sleeping_dog and high_value_lost_cause both show ROI 0.0x and Cost/Retention 0 — i.e. they are excluded from spend, but the ROI table still lists them at the bottom rather than in a separate "Excluded" panel, mixing "earned 0" with "deliberately not treated."

### Missing
- No definitional footnote on "Overall ROI" (denominator: spend? budget? at-risk CLV?).
- No definitional footnote on "Avg Uplift" (population: all 20k? treatable 16,317? treated 3,398?).
- Customers Retained has no integer formatter and no CI (122 is itself a model-derived expectation, not a count).
- No data freshness / model version on the page.
- No reconciliation panel pointing to Pages 05 / 09 / 11.

---

## Cross-page money KPI table

| KPI | P05 (LP) | P09 (Reco) | P10 (CLV) | P12 (Retention) |
|---|---|---|---|---|
| Total CLV | — | — | 57,936,514,970 KRW | 57,936,514,970 KRW |
| Avg CLV | — | — | 2,896,826 KRW | 2,896,826 KRW |
| Median CLV | — | — | 1,701,727 KRW | — |
| At-Risk CLV | — | — | — | 2,997,471,916 KRW (5.2%) |
| Total Recommendations | — | 20,000 | — | — |
| High Priority count | — | 16,106 | — | — |
| Coupons issued | — | 3,398 | — | — |
| Treatable Customers | — | — | — | 16,317 (81.6%) |
| Avg Expected Uplift / Avg Uplift | — | **6.36%** (top) / **10.88%** (mid) | — | **0.0434** |
| Max Uplift | — | — | — | 0.6874 |
| Budget Allocated | 50,000,000 KRW | — | — | 50,000,000 KRW |
| Total Campaign Cost (spent) | — | 1,211,055 KRW | — | — |
| Revenue Saved | ~192,155,551 KRW | 10,893,463 KRW | — | 192,155,554 KRW |
| Overall ROI | **3.5x** | **9.0x** | — | **3.8x** |
| Customers Retained | 118 | — | — | **122.29548658078494** |

Three different "Overall ROI" values, three different "Revenue Saved" magnitudes (because two of them are scoped to "spent" vs "budget"), two different "Customers Retained" counts, and two different "Avg Uplift" headlines — none of these are reconciled in the UI.

---

## SaaS-readiness verdict — Money domain

**Verdict: NOT READY for paid customer use.** The arithmetic inside any single page is internally consistent (sums tie within rounding, Total ÷ N = Mean, segment revenue sums to the headline), but the *cross-page money story is incoherent* and at least one cosmetic bug (`122.29548658078494`) makes the product look like a beta. A finance or growth buyer who clicks Page 05 → 09 → 12 sees three ROI values and two uplift values and will not sign the PO.

**Top 3 buyer-trust blockers:**
1. **Three "Overall ROI" values for one campaign (3.5x / 9.0x / 3.8x) with no definition footnote anywhere.** Same label, different denominators, no glossary. This is the #1 reason a financial reviewer will reject the dashboard.
2. **`Customers Retained = 122.29548658078494`** on Page 12. A 14-decimal float in a customer-count card is the kind of artifact that triggers a full QA re-review and erases trust in every adjacent number.
3. **Tiny segments masquerade as headline segments.** Page 10 shows high_value_persuadable (n=2) and high_value_lost_cause (n=1) on the same axis as segments of n=4,000+, with no count column emphasized — and Page 09 has 16,106 "High Priority" but only 3,398 coupons, leaving 12,708 high-priority customers silently in `no_action`. A SaaS buyer reading either chart will reach a wrong commercial decision.

Honourable mentions (fixable in <1 day each): negative probabilities on the Page 10 churn scatter x-axis, dual "Avg Expected Uplift" cards on Page 09 with the same label, and the behavioral-vs-uplift taxonomy mixing inside Page 12.
