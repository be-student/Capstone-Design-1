# A2 — Segmentation / Cohort / Budget

Independent audit of pages 03, 04, 05. Image + structured-data dump cross-checked against on-page math.

---

## Page 03 — Customer Segmentation

### Visible KPIs
- Total Segments: **6**
- Total Customers: **20,000**
- Highest Risk Segment: **dormant**
- Segment shares (donut): regular_loyal 24.7%, bargain_hunter 20.4%, new_customer 15.1%, explorer 14.9%, dormant 14.7%, vip_loyal 10.2%
- Segment counts (bar): vip_loyal 2,030 | regular_loyal 4,949 | bargain_hunter 4,087 | explorer 2,975 | new_customer 3,014 | dormant 2,945
- Mean CLV (KRW): vip_loyal 12,760,815 | regular_loyal 3,248,249 | bargain_hunter 1,932,498 | new_customer 1,503,177 | explorer 1,120,117 | dormant 66,362
- Definitions table rows: vip_loyal, loyal_customer, potential_loyalist, at_risk, hibernating, explorer, new_customer, bargain_hunter (8 rows)

### Wrong / Unreliable
- **Definitions table mismatch (NAMING SCHISM).** The Segment Definitions & Retention Actions table lists `vip_loyal, loyal_customer, potential_loyalist, at_risk, hibernating, explorer, new_customer, bargain_hunter` (8 names). But the Distribution / Statistics charts use `vip_loyal, regular_loyal, bargain_hunter, new_customer, explorer, dormant` (6 names). Only **4 names overlap** (vip_loyal, explorer, new_customer, bargain_hunter). `regular_loyal` and `dormant` are nowhere in the definitions table; conversely `loyal_customer / potential_loyalist / at_risk / hibernating` never appear in any chart. A user clicking the headline "Highest Risk Segment = dormant" cannot find dormant in the definitions table at all.
- **Counts vs. shares × 20,000** all reconcile: 2,030+4,949+4,087+2,975+3,014+2,945 = **20,000 exact** ✓; share% × 20,000 within ±0.05pp of reported counts. So sample-size math is fine, but it is glued to a definitions table that doesn't describe the same six segments.
- **"Highest Risk Segment = dormant"** has no risk score next to it. The reader has to read the bar chart to see that dormant ≈ 0.85+. KPI is qualitative-only.
- Avg CLV for **dormant = 66,362 KRW** (≈ $50). Two orders of magnitude below the next segment (explorer 1.12M). Plausible for a dormant cohort but worth flagging — extreme tail driving any "weighted" rollup.

### Missing
- No segment risk score in the headline KPI (just a label).
- No CIs / sample-weighted CLV.
- No date / model version stamp on the segmentation run.
- Definitions table renders Korean column ("loyal_program", "engagement_campaign", …) but no English fallback; appears localized but only partially.

---

## Page 04 — Cohort Analysis

### Visible KPIs
- Total Cohorts: **4** (2024-01, 2024-02, 2024-03, 2024-04)
- Periods Tracked: **13**
- Avg Period-1 Retention: **99.0%**
- Avg Final Retention: **2.5%**

### Wrong / Unreliable
- **Only 4 cohorts.** Below SaaS norm of ≥6–12 monthly cohorts for any longitudinal narrative. With n=20,000 customers in the simulator this is a generation-side limitation, not a UI limitation.
- **Retention monotonicity violation — Apr 2024 cohort: P7 = 91.0% → P8 = 92.1%.** Retention curves are by construction non-increasing (you can't un-churn within the same lookback window). This is either a bug in cohort accounting or noise that the page does not flag. Reported as a **CRITICAL anomaly** in the data dump and visible in the heatmap.
- **"Avg Final Retention = 2.5%" is a garbage average.** The matrix is right-truncated:
  - Jan 2024 P12 = 10.2%, but Feb's P12, Mar's P11–P12, Apr's P10–P12 are all **0.0%** because those cohorts have **not yet been observed that long** — they are NaN-as-zero, not real attrition.
  - Naive mean across the last column = (10.2 + 0 + 0 + 0) / 4 = **2.55%** → matches the displayed 2.5%. This confirms the KPI is averaging zero-filled future cells.
  - Same artifact contaminates Period-over-Period: P9 −20.7%, P10 −23.4%, P11 −22.8%, P12 −21.5% — all denominator-collapse artifacts, not real ~20pp monthly churn.
- **"Last surviving" cells are themselves implausible:** Jan P12 = 10.2%, Feb P11 = 7.9%, Mar P10 = 9.8%, Apr P9 = 12.9% — each cohort drops ~80pp in a single period right at the right edge. That is the data-window cutoff bleeding into the visible matrix; should be masked, not displayed.

### Missing
- No per-cohort sample sizes (n=?) on the heatmap or curves.
- No CI bands on the retention curves.
- No filter/mask on truncated cells; no "as-of" date so the reader cannot tell that Apr's P10–P12 are unobserved.
- No segment × cohort cross-tab — there is no path from page 03's `dormant` segment into a cohort view.

---

## Page 05 — Budget Optimization

### Visible KPIs
- Total Allocated: **50,000,000 KRW**
- Expected Retained: **118**
- Revenue Saved: **192,155,551 KRW**
- Avg ROI: **3.5x**

### Wrong / Unreliable
- **Sum of segment allocations ≠ Total Allocated.**
  21,062,000 + 13,922,000 + 12,781,000 + 1,524,000 + 680,000 + 31,000 + 0 + 0 = **50,000,000 KRW exactly** (the data dump's note of 49M is itself a tally error; my recomputation matches the 50M headline). However the **% column rounds to 100.002%** (42.124 + 27.844 + 25.562 + 3.048 + 1.36 + 0.062 + 0 + 0). Within rounding, segment allocations reconcile. Calling this **green for sum, but the data-dump note was wrong**.
- **Avg ROI 3.5x ≠ aggregate ROI 3.84x.** Revenue Saved / Total Allocated = 192,155,551 / 50,000,000 = **3.843x**. The headline says **3.5x**, i.e. it is showing the unweighted mean of per-segment ROIs, while the aggregate (revenue/spend) is 3.84x. Two definitions of "ROI" living on the same KPI strip.
- **Baseline Retained 122 ≠ Allocation Summary Expected Retained 118.** The "Expected Retained Customers by Scenario" table reports Baseline = 122 and "Current Selection" = 122. But the headline KPI says **118**. The headline is the same scenario as Current Selection (both at 50M total, no multiplier). 4-customer mismatch is unexplained.
- **high_value_persuadable** receives only **31,000 KRW (0.062%)** while the ROI bar shows it ≈ **8x** (highest on the chart). Either the LP has a segment-size cap that isn't surfaced, or the optimizer is mis-weighting. No "constraint binding" indicator on the page.
- **Channel-Level Cost Breakdown** section is rendered as an empty H3 with a banner: "Channel configuration not found in config. Add `budget.channels` to simulator_config.yaml…" — empty section = visible hole for an end-user.
- **Cost Reduction scenario shows Total Allocated = 50M (same as Baseline) but distinct ROI** — the bar chart treats "Cost Reduction" as a different point even though the budget is unchanged. Either the multiplier isn't displayed, or the scenario isn't actually reducing cost.

### Missing
- No CIs on retained-customer point estimates ("Aggressive +50% retains 220" is a single number).
- No display of LP constraints or which constraint is binding (segment caps, min spend, etc.).
- No reconciliation between the two ROI definitions; the headline doesn't say "mean of segment ROIs" anywhere.
- No `budget.channels` configured → entire channel-breakdown section is dead.

---

## Cross-page consistency check

| Check | Page A | Page B | Result |
|---|---|---|---|
| Total customers 20,000 | 03 (KPI) | implicit on 04 (cohort base) | ✓ matches simulator n=20,000 |
| Segment names | 03 charts: 6 names | 03 definitions table: 8 names | ✗ **only 4 names overlap** |
| Segment names | 03: regular_loyal, dormant, bargain_hunter, … | 05: high_value_*, mid_value_*, low_value_*, sleeping_dog, lost_cause | ✗ **completely disjoint taxonomies** — page 03 uses behavioral segments, page 05 uses uplift segments. No bridge. A user cannot ask "how is `dormant` being treated in the budget?" |
| Headline "Expected Retained" | 05 KPI: 118 | 05 scenario table Baseline / Current Selection: 122 | ✗ same scenario, two values (Δ=4) |
| ROI definition | 05 KPI: 3.5x | 05 derivable: 192,155,551 / 50,000,000 = 3.843x | ✗ definition trap |
| Cohort coverage | 04: only 4 cohorts (Jan–Apr 2024) | 03 implies 20,000 customers exist | ✗ thin cohort coverage given simulator data volume |
| Retention monotonicity | 04 Apr 2024: P7 91.0% → P8 92.1% | universal cohort math: must be non-increasing | ✗ violation, not flagged |

---

## SaaS-readiness verdict

**Verdict: NOT SHIP-READY.** Three independent numerical-integrity defects across the trio (P03 segment-name schism, P04 monotonicity violation + zero-truncation in the headline KPI, P05 ROI/retained-count mismatches) plus a complete taxonomy disconnect between segmentation (P03) and budget (P05). For an executive-facing SaaS dashboard these are credibility-killers: any analyst checking the math will find at least one contradiction inside 60 seconds.

**Top 3 blockers:**
1. **Page 04 — "Avg Final Retention 2.5%" is averaging zero-filled unobserved future cells** AND the matrix contains a monotonicity violation (Apr P7 91.0% → P8 92.1%) that is mathematically impossible for retention. Mask truncated cells, fix or flag the monotonicity bug, and recompute the KPI.
2. **Page 05 — three KPI internal contradictions in one card strip**: Avg ROI 3.5x vs aggregate 3.84x; Expected Retained 118 vs Baseline-scenario 122; high_value_persuadable shows ~8x ROI but receives 0.062% of spend with no explanation. Pick one ROI definition, reconcile retained counts, surface LP constraints.
3. **Pages 03 ↔ 05 use disjoint segment taxonomies** (behavioral vs. uplift) and Page 03's own definitions table only overlaps 4 of 6 chart segments. Either unify the taxonomy, or display both on each page with an explicit mapping; users currently cannot trace `dormant` from segmentation into budget.

---

### 5-line summary
1. Page 03 counts add to 20,000 exactly, but the Definitions table names 8 segments that overlap only 4 of the 6 charted segments — `regular_loyal` and `dormant` (the headline "highest risk") are not defined on the page.
2. Page 04 has only 4 cohorts, contains a retention monotonicity violation (Apr 2024: P7 91.0% → P8 92.1%), and the headline "Avg Final Retention 2.5%" is `(10.2+0+0+0)/4` — averaging zero-filled future cells.
3. Page 05's "Avg ROI 3.5x" contradicts the aggregate 192,155,551/50,000,000 = 3.843x; "Expected Retained 118" contradicts the Baseline/Current-Selection scenario value of 122 on the same page.
4. Pages 03 and 05 use entirely disjoint segment taxonomies (behavioral vs. uplift); a user cannot trace any single segment across the two pages.
5. Verdict: NOT ship-ready — three independent numerical-integrity defects and a taxonomy disconnect; fix monotonicity/zero-truncation on P04, reconcile P05's ROI + retained counts, and unify the segment vocabulary between P03 and P05 before exposing this to a customer.
