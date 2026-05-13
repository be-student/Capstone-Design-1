# A5 — Recommendations / CLV / Retention Campaign

## Page 09 — Recommendations

### Visible KPIs
- Banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- Total Recommendations: **20,000**
- Avg Expected Uplift: **6.36%**
- Top Action Type: **No Action**
- High Priority: **16,106**
- Recommendation Type Distribution (donut): no_action **83%**, coupon **17%**
- Recommendations by Type (bar): no_action **16,602**, coupon **3,398**
- Sidebar: Churn Definition (No purchase: 30 days, No login: 60 days, OR), Budget 50,000,000 KRW, Ensemble Weights ML 0.6 / DL 0.4

### Wrong / suspicious
- **"Top Action Type: No Action" with 83% share is a CFO red flag**. The headline KPI a buyer reads first effectively says "the system recommends doing nothing for 4 out of 5 customers." There is no threshold sensitivity panel, no rationale callout — buyer cannot tell if this is well-tuned or a broken classifier biased toward the negative class.
- **High-priority vs No-action contradiction**: 16,106 customers are flagged "High Priority" yet only **3,398** receive an actionable recommendation (coupon). That means **~12,708 high-priority customers receive "no_action"** — a direct logical contradiction between the priority flag and the action recommendation, with no inline explanation.
- **Donut/bar arithmetic mismatch**: Donut shows 83% / 17%, but bar counts are 16,602 / 3,398 = **83.01% / 16.99%**. The donut rounds, the bar does not — minor, but the page never reconciles them.
- **"Avg Expected Uplift 6.36%" has no denominator/definition** — uplift over what baseline? Per-customer? Population-weighted? Treated-only? No footnote.

### Unreliable
- Top Action Type as a single-string KPI ("No Action") is information-poor — collapses a full distribution to its mode. A CFO needs the share, not the winner.
- Avg Expected Uplift presented without confidence interval, sample size of treated cohort, or model version.
- "High Priority" count (16,106) has no definition of what threshold makes a customer high-priority.

### Missing
- No revenue / KRW impact KPI on this page (no "Expected Revenue Saved", no "Budget Utilized", no "Cost per Retained Customer").
- No ROI figure at all on the Recommendations page despite this being the action page.
- No threshold sensitivity panel to justify the 83% no-action share.
- No model version / training timestamp / data freshness stamp.
- Action types are limited to {no_action, coupon} — no email, discount tier, concierge, or escalation paths shown; either the model is degenerate or the UI hides them.

---

## Page 10 — CLV Prediction

### Visible KPIs
- Banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- Total CLV: **57,936,514,970 …** (truncated with ellipsis)
- Average CLV: **2,896,826 KRW**
- Median CLV: **1,701,727 KRW**
- CLV Std Dev: **3,575,497 KRW**
- Customer Lifetime Value Distribution histogram (Median line near ~1.7M, range 0–~15M)
- CLV Distribution by Segment box plot across 7+ segments (high_value_sure_thing, high_value_persuadable, high_value_lost_cause, mid_value_sure_thing, sleeping_dog, low_value_sure_thing, low_value_persuadable)
- CLV by Segment section (Average and Total, charts truncated below fold)

### Wrong / suspicious
- **Total CLV is truncated as "57,936,514,970 …"** — the headline money number on the CLV page literally does not fit in its container. A CFO sees an ellipsis on a revenue figure and stops reading. Format helper untested.
- **Histogram median annotation reads "Median ≈ 0…1,701,826"** (cut text on the chart label) — the median number on the chart (~1,701,826) does **not match** the KPI tile Median CLV of **1,701,727**. ~99 KRW drift, consistent with separate calculations on different sample slices, but the page does not reconcile.
- **Segment label collisions**: x-axis labels on the box plot are rotated and overlap; the legend lists "high_value_lost_cause" — a CFO will ask why the system spends compute predicting CLV for customers it has labeled "lost cause".
- **Std Dev (3.58M) > Mean (2.90M)** — distribution is heavy-tailed/log-normal. Reporting Average CLV as a primary KPI on a long-tail distribution is misleading; Median is the correct central tendency, and the page leads with Average.

### Unreliable
- "Total CLV ~ 57.9B KRW" with no time horizon. CLV over what window — 12 months? Lifetime? Discounted? No discount rate disclosed.
- No model name, training cohort, or holdout R² / MAE for the CLV regressor — buyers cannot trust 57.9B without knowing model error bars.
- "CLV Std Dev" exposed as a customer-facing KPI is unusual — it is a model diagnostic, not a buyer KPI.

### Missing
- No currency-format toggle / unit (KRW vs M-KRW vs B-KRW); the truncation problem above proves the absence.
- No CLV-vs-actual back-test panel (predicted vs realized).
- No "Top 10% of CLV captures X% of revenue" Pareto figure — standard CLV deck content.
- No At-Risk CLV figure on this page (it appears on Page 12 but not here — see consistency section).

---

## Page 12 — CLV & Retention Campaign

### Visible KPIs
- Banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- Section 1: Customer Lifetime Value Overview
  - Total CLV: **57,936,514,970 …** (truncated)
  - Avg CLV: **2,896,826 KRW**
  - At-Risk CLV: **2,997,471,916 K…** (truncated)
  - At-Risk CLV %: **5.2%**
- CLV Distribution by Risk Level histogram (low / high / critical / medium)
- Segment CLV vs Churn Risk bubble chart, segments: bargain_hunter, dormant, explorer, new_customer, regular_loyal, vip_loyal
- Section 2: Uplift Modeling & Treatment Effectiveness
  - Avg Uplift: **0.0434**
  - Max Uplift: **0.6874**
  - Treatable Customers: **16,317 (81.6%)**

### Wrong / suspicious
- **Two of four headline KPIs are visually truncated** ("57,936,514,970 …" and "2,997,471,916 K…"). The dollar/won figures the buyer cares most about are the ones that don't fit. P0 layout bug.
- **At-Risk CLV % math check**: 2,997,471,916 / 57,936,514,970 = **5.173%** → rounds to 5.2% ✓ arithmetic is internally consistent, but the truncation makes the buyer have to re-derive it.
- **Segmentation taxonomy mismatch with Page 10**. Page 10 segments customers as {high/mid/low_value × sure_thing/persuadable/lost_cause/sleeping_dog} (uplift quadrants). Page 12 segments customers as {bargain_hunter, dormant, explorer, new_customer, regular_loyal, vip_loyal} (behavioral personas). **Same 20,000 customers, two different segmentation schemes, no key/crosswalk shown.** A CFO will ask which one drives the budget.
- **Avg Uplift 0.0434 (≈4.34%) vs Page 09 "Avg Expected Uplift 6.36%"** — same population, two different averages, no reconciliation. ~46% relative gap. Likely "all customers" vs "treated customers", but the pages do not say.
- **Treatable Customers 16,317 (81.6%) vs Page 09's no_action share 83%**. 16,317 treatable + 3,398 actionable coupons = different numerators of the same idea. 16,317 ≠ 16,602 (no_action count) and ≠ 16,106 (high priority) — three close but distinct integers in the same ~16k band, none reconciled.
- **No "Overall ROI" / "Avg ROI" / "Total Revenue Saved" / "Customers Retained" KPI tile is visible** in the captured viewport — the campaign page lacks the very money-domain KPIs the audit was scoped to find. Either they live below the fold (then the headline above the fold is incomplete) or they are absent.

### Unreliable
- "Treatable Customers 81.6%" without a definition — treatable by what action, at what cost, with what minimum uplift? No footnote.
- Max Uplift 0.6874 is a single-customer outlier number; presenting it as a headline KPI invites cherry-pick suspicion.
- Bubble chart "size = customers" but no scale legend — buyer cannot read absolute counts off bubble area.

### Missing
- **No Overall ROI** anywhere in the visible viewport — for a "Retention Campaign" page, this is the missing centerpiece.
- **No Total Revenue Saved** in KRW.
- **No Customers Retained** count.
- **No Budget vs Spend tile** (the sidebar shows 50M KRW budget; nothing on this page consumes it).
- No campaign time window / horizon.
- No reconciliation between the two segmentation schemes.

---

## Cross-page ROI/CLV consistency check

| KPI label | Page 09 | Page 10 | Page 12 | Verdict |
|---|---|---|---|---|
| Total CLV | — | 57,936,514,970 … (truncated) | 57,936,514,970 … (truncated) | Same underlying number, **both truncated** — P1 |
| Average / Avg CLV | — | 2,896,826 KRW | 2,896,826 KRW | Match |
| Median CLV | — | 1,701,727 KRW (KPI) vs ~1,701,826 (chart) | — | **~99 KRW drift between tile and chart annotation** — P2 |
| At-Risk CLV | — | not shown | 2,997,471,916 K… (truncated) | Truncation P0; not cross-checked |
| At-Risk CLV % | — | not shown | 5.2% | Arithmetic OK |
| Avg Uplift | 6.36% (Avg Expected Uplift) | — | 0.0434 = 4.34% (Avg Uplift) | **Mismatch — 6.36% vs 4.34% on same n=20,000** — P0 |
| Max Uplift | — | — | 0.6874 | Single-source |
| "Treatable / no-action / high-priority" cohort sizing | High Priority 16,106; no_action 16,602; coupon 3,398 | — | Treatable 16,317 (81.6%) | **Three close-but-different ~16k integers, no reconciliation** — P0 |
| Top Action Type | "No Action" (83%) | — | — | Single-source |
| Total Revenue Saved | **NOT SHOWN** | NOT SHOWN | NOT SHOWN above fold | Missing — P0 |
| Customers Retained | NOT SHOWN | NOT SHOWN | NOT SHOWN above fold | Missing — P0 |
| Overall ROI | NOT SHOWN | NOT SHOWN | NOT SHOWN above fold | Missing — P0 |
| Total Recommendations | 20,000 | n=20,000 (banner) | n=20,000 (banner) | Match |

P0 mismatches:
1. Avg uplift differs across pages (6.36% vs 4.34%) for the same population with no footnote.
2. Three different ~16k cohort counts (16,106 / 16,317 / 16,602) without a crosswalk.
3. Money KPIs (ROI, Revenue Saved, Customers Retained) are absent above the fold on the very page titled "CLV & Retention Campaign".
4. Total CLV and At-Risk CLV are truncated with ellipsis on the headline tile.

---

## SaaS-readiness verdict — Money domain

**Verdict: DO-NOT-SHIP**

**Top 3 buyer-trust blockers:**
1. **Headline money KPIs are visually truncated.** Total CLV "57,936,514,970 …" and At-Risk CLV "2,997,471,916 K…" are cut off mid-digit on Pages 10 and 12. A CFO will not sign a contract on a dashboard that cannot render its own revenue number. P0 format-helper bug.
2. **Same population, different uplift averages and different cohort sizes across pages with no reconciliation.** Avg Expected Uplift 6.36% (P09) vs Avg Uplift 4.34% (P12); High Priority 16,106 vs Treatable 16,317 vs no_action 16,602. With no footnote defining denominators, every cross-page click erodes trust. P0.
3. **The "Retention Campaign" page has no ROI, no Revenue Saved, no Customers Retained tile above the fold, while 12,708 high-priority customers are simultaneously assigned "no_action".** The product cannot answer the only two CFO questions on a pricing call: "what does this earn me?" and "why is the system telling me to do nothing for 80% of my best-priority list?" P0.

Secondary issues to clear before re-review: median tile vs chart drift (~99 KRW), Std Dev exposed as a buyer KPI on a log-normal distribution, two incompatible segmentation taxonomies between P10 and P12 with no crosswalk, no model version / training-date / horizon disclosure on any CLV figure.
