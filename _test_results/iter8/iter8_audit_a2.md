# A2 — Customer Segmentation / Cohort / Budget Optimization

## Page 03 — Customer Segmentation

### Visible KPIs
- Total Segments: **6**
- Total Customers: **20,000**
- Highest Risk Segment: **dormant**
- Segment Distribution (pie chart): regular_loyal **24.7%**, bargain_hunter (slice unlabeled, ~ adjacent), new_customer **15.1%**, explorer **14.7%**, dormant **14.9%** (visible label), vip_loyal **10.2%**, with one slice at **20.4%** (likely bargain_hunter from color order)
- "Customers per Segment" bar chart: bars top out around 5,000 (regular_loyal); others appear in roughly 2,000–3,000 range
- Top banner: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."
- Sidebar: Churn Definition (No purchase: 30 days; No login: 60 days; Operator: OR), Budget 50,000,000 KRW, Ensemble Weights ML: 0.6 | DL: 0.4

### Wrong / suspicious
- Pie chart percentages from labels visible: 24.7 + 20.4 + 15.1 + 14.9 + 14.7 + 10.2 = **100.0%** — arithmetic checks out, but the "Highest Risk Segment = dormant" label is asserted at the top while only **14.9%** of the population is dormant; a "highest risk segment" KPI with no risk score, count, or churn-probability number alongside it is meaningless to a buyer.
- The bar chart on the right and pie chart on the left appear to convey the same underlying counts twice — duplicate visualization without added analytic value.
- Bar chart labels for "regular_loyal" appear to peak ~5,000 → 5,000/20,000 = 25% which lines up with 24.7%, but the bargain_hunter bar is taller than new_customer in the bars, while the pie suggests bargain_hunter (20.4%) > new_customer (15.1%) > explorer (14.7%) — visually consistent but not explicitly labeled, so a buyer cannot reconcile counts to percentages without hovering.

### Unreliable
- Six segment shares (10.2 / 14.7 / 14.9 / 15.1 / 20.4 / 24.7) for n=20,000 is a suspiciously clean rank-ordered distribution — looks like a deterministic synthetic prior rather than empirical behavior data.
- "FULL mode (n=20000). All KPIs are simulator-generated" is an explicit disclosure that all numbers are synthetic; nothing on this page can be cited as evidence of real-world segment behavior.

### Missing
- **No segment definitions**: what behavioral rule defines "regular_loyal" vs "bargain_hunter" vs "vip_loyal" vs "explorer"? A SaaS buyer cannot evaluate segment quality without the rule set.
- **No churn probability per segment** on this view (header says "Segment Churn Risk Analysis" but the actual chart is cut off below the fold).
- **No CLV / revenue weighting** per segment — segmentation without monetization context is half a product.
- **No confidence intervals** on segment shares.
- **No drilldown** from a segment to its membership criteria or example customers.
- **Pie chart legend overlaps the chart**, and one slice's percentage is not visible in the screenshot — accessibility/readability issue.

---

## Page 04 — Cohort Analysis

### Visible KPIs
- Total Cohorts: **4**
- Periods Tracked: **13**
- Avg Period-1 Retention: **99.0%**
- Avg Final Retention: **2.5%**
- Cohort labels in heatmap: **Jan 2024, Feb 2024, Mar 2024, Apr 2024**
- Heatmap row values (Period 0 → Period 12), reading off the visible tiles:
  - Apr 2024: 100.0, 98.9, 98.1, 96.5, 95.5, 93.5, 92.4, 92.1, 92.1, 12.9, 0.0, 0.0, 0.0
  - Mar 2024: 100.0, 99.0, 98.5, 97.5, 96.5, 94.3, 93.5, 92.5, 91.0, 89.0, 19.5, 0.0, 0.0
  - Feb 2024: 100.0, 99.0, 98.5, 97.5, 96.2, 94.5, 93.3, 92.5, 91.5, 90.4, 86.1, 7.9, 0.0
  - Jan 2024: 100.0, 99.1, 98.5, 97.5, 96.2, 94.5, 93.3, 91.1, 89.5, 86.5, 88.3, 65.2, 10.7
- Sidebar identical to page 03.

### Wrong / suspicious
- **Cliff drops to 0.0% are physically implausible**. Apr 2024 cohort goes 92.1% → **12.9%** → **0.0%** in two periods, and stays at 0.0% for the remaining periods. That is not retention; that is the dataset's right edge being reported as zero retention. The chart is conflating "no observation yet" with "zero retained users."
- Same pattern in every cohort: Mar drops to 0.0% at Period 11, Feb at Period 12. This is a **censoring / truncation bug** rendered as a value.
- "Avg Final Retention: 2.5%" is computed across these spurious 0.0% trailing cells, so the headline KPI is **arithmetically wrong**: it averages real retention with not-yet-observed periods.
- "Avg Period-1 Retention: 99.0%" — implausibly high for any commerce/SaaS product; combined with the synthetic-data banner, this is a generator default, not a measurement.
- Jan 2024 Period 10 = **88.3%** is *higher* than Period 9 = **86.5%**. Retention is monotonically non-increasing for a fixed cohort definition; an uptick means either the cohort definition is being recomputed each period (definitional churn) or the values are noise from the generator. Either way, a buyer cannot trust the curve.

### Unreliable
- **Only 4 monthly cohorts** (Jan–Apr 2024) is below the SaaS threshold of ≥6 cohorts needed for "longitudinal insight." This is too few to detect seasonality or sustained retention shifts.
- Period-1 values clustered at 99.0–99.1% and Period-2 at 98.1–98.5% across all four cohorts is a suspiciously tight band — looks like a closed-form decay function, not measured data with cohort-to-cohort variance.
- The heatmap is dense and colorful but conveys little because all four rows trace nearly identical curves until they hit the truncation cliff — single decay shape replicated four times.

### Missing
- **No confidence intervals** or sample-size-per-cell. Cohort retention without n per cell is not auditable.
- **No distinction between "censored" and "zero retained"** — a buyer cannot tell whether the 0.0% cells mean churn-out or no-data-yet.
- **No cohort sizes** (how many customers are in Jan vs Feb vs Mar vs Apr 2024 cohort).
- **≥6 cohorts** needed; only 4 shown.
- **No definition of "retention"** (active = login? purchase? either?). Sidebar shows churn definition but not retention definition; they are not the same thing.
- **No segment × cohort breakdown** — retention often differs by acquisition channel or segment; not exposed here.

---

## Page 05 — Budget Optimization

### Visible KPIs
- Total Budget (KRW) slider: **50,000,000** (default 50,000,000 KRW)
- Cost Multiplier: **1.00**
- Uplift Multiplier: **1.00**
- Allocation Summary:
  - Total Allocated: **50,000,000 KRW**
  - Expected Retained: **118**
  - Revenue Saved: **192,155,551 KRW**
  - Avg ROI: **3.5x**
- Budget Allocation by Segment (table, allocated_budget_krw / expected_retained / expected_revenue_saved_krw / roi):
  - 0 high_value_lost_cause: **0 / 0 / 0 / 0.00**
  - 1 high_value_persuadable: **31,000 / 0 / 247,912 / 8.00**
  - 2 high_value_sure_thing: **680,000 / 0 / 3,463,472 / 5.09**
  - 3 low_value_persuadable: **13,922,000 / 44 / 53,124,598 / 3.82**
  - 4 low_value_sure_thing: **1,524,000 / 3 / 4,794,754 / 3.15**
  - 5 mid_value_persuadable: **21,062,000 / 49 / 86,389,749 / 4.10**
  - 6 mid_value_sure_thing: **12,781,000 / 22 / 44,134,866 / 3.45**
  - 7 sleeping_dog: (visible row, values cut off)

### Wrong / suspicious
- **Budget allocation to high-ROI segments is near-zero**:
  - high_value_persuadable has **ROI 8.00x** (the highest visible ROI on the page) but is allocated only **31,000 KRW** — 0.062% of the 50M budget.
  - high_value_sure_thing has **ROI 5.09x** but gets only **680,000 KRW** — 1.36% of budget.
  - Meanwhile low_value_persuadable (ROI 3.82x) gets **13,922,000 KRW** and mid_value_persuadable (ROI 4.10x) gets **21,062,000 KRW** — together 70% of budget on lower-ROI segments. **This is the opposite of what an LP optimizer should do.** Either the optimizer is broken, the ROI column is computed post-hoc with diminishing returns not shown, or there is a constraint (e.g., segment size cap) that is hidden from the buyer.
- **"Expected Retained" rows of 0 with positive Revenue Saved**: high_value_persuadable shows 0 retained but 247,912 KRW revenue saved; high_value_sure_thing shows 0 retained but 3,463,472 KRW revenue saved. **Mathematically impossible** unless retention and revenue are decoupled (fractional retained customers rounded down? then Revenue Saved should also be the fractional product). This is a definition bug.
- **Total expected_retained from visible rows: 0+0+0+44+3+49+22 = 118**, which matches the headline "Expected Retained: 118" — but only because sleeping_dog (row 7) and any further rows contribute 0. Buyer cannot verify without the cut-off rows.
- **Sum of allocated budget visible rows**: 0 + 31,000 + 680,000 + 13,922,000 + 1,524,000 + 21,062,000 + 12,781,000 = **50,000,000 KRW** exactly. Headline matches, but the suspicious thing is that there is no slack/unallocated bucket — 100% of budget is forced to spend even when high-ROI segments are starved.
- **Avg ROI 3.5x** definition trap: with Revenue Saved 192,155,551 and Total Allocated 50,000,000, aggregate ROI = 192.16M / 50M = **3.84x**, not 3.5x. So "Avg ROI" is the **mean of segment ROIs** (0 + 8.00 + 5.09 + 3.82 + 3.15 + 4.10 + 3.45 + sleeping_dog ÷ 8 ≈ 3.5 if sleeping_dog ROI is small), which is a different and arguably wrong number to headline.

### Unreliable
- ROI "8.00" is suspiciously round; combined with the simulator-generated banner, it looks like a parameter, not a model output.
- The optimizer's allocation pattern (most budget to mid-tier, almost none to top-ROI) suggests the LP is using a closed-form per-segment formula with a size or saturation cap that is not exposed — the page is presenting a decision without showing why.
- No sensitivity curve as Cost Multiplier or Uplift Multiplier varies — sliders are present but no chart shows ROI/retained as a function of those knobs.

### Missing
- **No channel breakdown** (email vs push vs SMS vs paid retargeting) despite this being a "budget optimization" page — campaign budget without channel unit economics is incomplete.
- **No scenario tooltip explaining ROI denominator** (revenue saved ÷ allocated? ÷ total budget? net of cost multiplier?).
- **No definition of "Avg ROI"** — mean-of-segments vs revenue/budget envelope vs uplift-weighted is a $$ definition trap; current page uses an inconsistent definition versus the headline aggregate.
- **No constraints panel**: are there minimum-spend-per-segment, maximum-reach, or contact-fatigue constraints? The fact that high-ROI segments get tiny allocations strongly implies a hidden constraint that is not disclosed.
- **No counterfactual / no-action baseline**: "Revenue Saved: 192,155,551 KRW" — saved versus what? No baseline retention rate or do-nothing scenario.
- **No confidence intervals** on Expected Retained or Revenue Saved.
- **sleeping_dog row is cut off** at the bottom — page does not show all 8+ segments in the visible viewport.

---

## SaaS-readiness verdict

**Verdict:** DO-NOT-SHIP

**Top 3 blockers:**
1. **Cohort analysis is statistically broken**: only 4 cohorts (need ≥6), and trailing 0.0% cells from data truncation are being averaged into the "Avg Final Retention: 2.5%" headline. The Jan 2024 Period 9→10 uptick (86.5% → 88.3%) violates monotonicity, signaling a definitional/measurement bug. A buyer reading retention here gets misinformation.
2. **Budget optimizer allocates against ROI rank**: high_value_persuadable at ROI 8.00x receives 31,000 KRW (0.06%) while lower-ROI segments absorb 70% of the 50M KRW budget. Either the LP is wrong or there is an undisclosed constraint — in either case, unshippable without explanation. Compounded by an "Avg ROI 3.5x" headline that does not equal aggregate revenue/budget (3.84x), exposing a definition trap.
3. **All KPIs are simulator-generated synthetic data** (banner explicitly says so), with no segment definitions, no retention definition, no confidence intervals, no channel breakdown, and no counterfactual baseline. The product currently demonstrates UI scaffolding, not analytical evidence a paying customer can act on.
