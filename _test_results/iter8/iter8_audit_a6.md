# A6 — Real-Time Scoring (3 tabs)

> Source of truth: the three PNGs only. All values transcribed verbatim.
> Banner on every tab: "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated."

## Tab a — Live Scoring Status

### Visible KPIs
- Page title: **Real-Time Scoring & Recommendations** with sub-line "Live scoring status, personalized retention offers, and model monitoring dashboards."
- Tabs: Live Scoring Status (active) | Retention Offer Recommendations | Model Monitoring.
- Section: **Service Health**
  - Redis: **Connected** (green pill)
  - Request Stream: **0**
  - Response Stream: **0**
  - Consumer Group: **scoring_consu...** (truncated)
- Collapsible: "Redis Configuration Details" (collapsed).
- Section: **Scoring Throughput & Latency**
  - Chart 1: "Scoring Requests per Minute" — blue line, y-axis ~20 to ~80 requests/min, x-axis Oct 15 2024 (00:00 → 18:00 → next day). Visibly bursty/diurnal shape.
  - Chart 2: "Response Latency & Error Rate" — orange line for Avg Latency (ms) on left axis (~10 to ~30 ms range with spikes to ~30+), dotted line for Error Rate (%) on right axis (0 → 3.5 visible scale).

### Wrong / suspicious
- **Stream depth contradictions are real and present.** Request Stream: 0 and Response Stream: 0 are shown as flat KPI cards with no qualifier (no "right now" / "queue depth" label). With a working scorer there is normally either a small in-flight count or an explicit "lifetime processed" KPI alongside. Here the page shows neither — it just declares zero on both sides while the chart below shows ~20–80 req/min of synthetic traffic. The two views contradict each other unless these cards mean "current queue depth," which is never disclosed.
- The Consumer Group label is **truncated mid-word** ("scoring_consu...") — operators cannot copy-paste an identifier they cannot read.
- Latency chart is **labeled "Response Latency & Error Rate"** but the right-axis scale "0 / 1 / 2 / 3 / 3.5" with no explicit "%" tick formatting risks reading 3.5% as 3.5 — and 3.5% error rate would be a hard SLO miss (criterion threshold: >1% = deal blocker). Without a clear unit on the right axis, a buyer cannot tell if this is .035, 3.5, or 3.5%.
- Time axis on both charts is fixed to **Oct 15–16 2024**, not "now" — this is a real-time tab. The data is stale by ~19 months versus the dashboard date (May 10 2026 per page metadata) and the Model Monitoring tab clock (May 10 2026, see tab c).

### Unreliable
- "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated." banner means the latency curve, error-rate curve, and throughput curve are not telemetry from a running scorer — they are simulator output. Any operational claim drawn from them is unfalsifiable.
- No p50 / p95 / p99 latency numbers — only an unlabeled line. A SaaS buyer cannot check the avg<50ms / peak<100ms criterion without numerics. Visual inspection suggests avg ~15–20 ms and peaks ~30 ms which would pass, **but the numbers are not surfaced**.
- Error-rate curve oscillates between ~0 and ~3.5 (units ambiguous). If it is %, this fails SLO; if it is count, it is meaningless without volume. Cannot tell.

### Missing
- Per-criterion: there is **no "Total Scores" KPI on this tab** (the audit prompt expected one). Only Request Stream / Response Stream / Consumer Group appear.
- No p50/p95/p99 latency, no QPS instantaneous reading, no last-request timestamp, no model version pinned to the live serving path.
- No "queue depth right now" vs "lifetime processed" disambiguation.
- No success-rate / 2xx-rate KPI — only error-rate as a chart.

---

## Tab b — Retention Offer Recommendations

### Visible KPIs
- Header: **Personalized Retention Offer Recommendations** with sub-line "AI-driven retention offers optimized per customer based on churn risk segment, CLV, and expected uplift."
- Filter pills row 1 — "Filter by Risk Tier": critical, high, medium (low not shown / off).
- Filter pills row 2 — "Filter by Offer Type": bargain_keeper, discount, engagement, loyalty_points, regular_keeper (some pills selected).
- KPI strip:
  - Total Offers: **44**
  - Total Cost: **1,196,659 KRW**
  - Expected Revenue Saved: **10,752,341 KRW**
  - Expected ROI: **8.0x**
- Charts (visible at bottom of crop): "Offer Type Distribution" (donut/pie) and "Average Expected Uplift by Segment" (bar chart, partially visible).

### Wrong / suspicious
- **44 total offers from a 20,000-row simulator.** That is a 0.22% activation rate. Either the filters are excluding most of the population (in which case the KPI cards should reflect filter state) or the recommendation engine is only firing for an extreme tail. No "X of Y customers" denominator is shown.
- **8.0x ROI is suspiciously round.** 10,752,341 / 1,196,659 = 8.985…, which rounds to 9.0x not 8.0x. The ROI card and the underlying ratio do not reconcile. Either the ROI is computed on a different basis (e.g., net revenue minus cost) or the displayed value is hand-set.
- The audit prompt's "no_action priority 1.00" check cannot be evaluated — no per-customer quick-lookup widget is visible on this crop. If it exists below the fold, it was not captured. Flag: **content not verifiable from the provided PNG.**
- Pills appear pre-filtered (e.g., engagement and loyalty_points appear as the "selected" set in offer-type) but the KPI numbers don't update visibly to reflect filter state — there is no "filtered: 44 / total: N" pattern.

### Unreliable
- Synthetic banner applies. The 1.2M cost / 10.75M saved / 8x ROI all come from simulator math, not booked revenue.
- "Expected Revenue Saved" is the most dangerous metric on the page — it is a model-output, not realized revenue, and the label does not say "modeled / projected."
- No confidence interval on uplift, no holdout validation, no campaign-period anchor. A pricing buyer cannot tell whether 8x is over a quarter, a year, or the lifetime of the offer.

### Missing
- No customer-level row (Quick Lookup) visible on this crop, so the "no_action priority 1.00" mismatch cannot be checked here.
- No "Last refreshed" timestamp.
- No model version (which uplift model produced these 44 offers?).
- No filter state echo on the KPI cards ("44 offers under current filters").
- No baseline/control comparator — uplift relative to what?

---

## Tab c — Model Monitoring (within Real-Time Scoring page)

### Visible KPIs
- Section: **Model Monitoring Dashboard** with sub-line "Track model drift (PSI & KS), alert history, and scoring quality over time to ensure reliable predictions."
- KPI strip:
  - Total Drift Checks: **1**
  - Red Alerts: **1**
  - Yellow Warnings: **0**
  - Latest Alert Level: **RED**
- Section: **Drift Alert Timeline** → chart "Drift Alerts Over Time"
  - Y-axis: "# Drifted Features" with ticks 6 / 6.5 / 7 / 7.5 / 8.
  - X-axis: timestamps "11:56:50.282", "11:56:50.2825", "11:56:50.283", "11:56:50.2835" — **microsecond-resolution ticks across an essentially zero-width time window**.
  - Date label: **May 10, 2026**.
  - Single red dot at y=7 plotted in the middle of that microsecond window.

### Wrong / suspicious
- **Single-point "trend" failure.** Audit-prompt rule explicitly: 1 observation is not a trend. The page plots 1 point on a continuous time axis instead of showing "Insufficient history." This is a textbook case the rubric calls out.
- **X-axis at microsecond resolution.** 11:56:50.282 → .2835 spans ~1.5 milliseconds. With Total Drift Checks = 1, matplotlib/plotly auto-scaled to a meaningless horizontal range — operators reading this will think drift sampling happens on a millisecond cadence, which is false.
- **100% red-alert rate.** 1 of 1 checks is red. With no priors and no history, RED status is uncalibrated alarmism. The KPI strip would scream "production model is broken" to an on-call engineer who skims it.
- "Latest Alert Level: RED" but the live tab reports Redis: Connected and a healthy throughput chart. The page does not reconcile "system healthy AND model drifting hard" vs "single red drift datapoint, ignore for now."

### Unreliable
- Synthetic banner applies — the single drift event is simulator-generated, not observed.
- No PSI or KS numerical value printed despite the sub-line promising "PSI & KS."
- No feature-level breakdown of which 7 features drifted.
- No alert thresholds disclosed (what PSI cutoff = RED?).

### Missing
- A "needs N more checks before trend is meaningful" disclaimer.
- Linkage to which model_version this drift was measured against.
- A second observation (only then is a timeline a timeline).
- An incident/ack workflow — RED alert with no "acknowledge" or "page on-call" affordance.

---

## Cross-tab consistency check

| KPI | Tab a (Live) | Tab b (Offers) | Tab c (Monitoring) | Mismatch |
|---|---|---|---|---|
| Synthetic-data banner | Present, n=20000 | Present, n=20000 | Present, n=20000 | OK |
| Total Scores | **not shown** (only Request Stream:0, Response Stream:0) | n/a | n/a | Audit-prompt expected 200 vs 0 contradiction; on this build neither value is rendered, so the contradiction is *latent but unresolvable* — the page doesn't show "Total Scores: 200" anywhere, so a buyer can't verify lifetime throughput at all. |
| Latest Drift Alert Level | n/a | n/a | RED | Single source — no cross-check possible |
| Model version | not shown | not shown | not shown | All three tabs hide the live model_version → cannot confirm the drift alert (tab c), the offer engine (tab b), and the scoring service (tab a) are talking about the same artifact |
| Time anchor | Oct 15–16 2024 | not shown | May 10 2026 11:56:50 | **Hard mismatch** — live throughput chart is from Oct 2024 while drift monitoring is at May 10 2026. The two tabs cannot both be "live." |
| Currency / units | n/a | KRW (1.2M cost / 10.75M saved) | n/a | Internal consistency of tab b: 10,752,341 / 1,196,659 = 8.985x but card says 8.0x — **arithmetic mismatch within tab b** |
| Filter-state echo | n/a | KPIs do not visibly change with filter pills | n/a | Filters look decorative |

Headline mismatches:
1. **Time anchors disagree by ~19 months** between tab a and tab c.
2. **ROI math doesn't reconcile** (8.985x → displayed 8.0x).
3. **Model version absent on all three tabs** — operators cannot tell which artifact is live.

---

## SaaS-readiness verdict — Real-time serving

**Verdict: DO-NOT-SHIP**

**Top 3 production-readiness blockers:**
1. **No real telemetry plumbing.** Banner declares "All KPIs are simulator-generated"; tab a shows Oct 2024 traffic while tab c shows May 2026 timestamps; Request/Response Stream both fixed at 0. The page is a mock with charts, not a serving console. A buyer cannot use it to operate a live model.
2. **Single-observation "drift trend" with RED status.** Tab c plots one datapoint on a 1.5-millisecond-wide x-axis and declares Latest Alert Level: RED with 100% red-alert rate. This violates the rubric's "1 observation isn't a trend → say Insufficient history" rule and would mis-page on-call from day one.
3. **Numerical/labeling defects that fail buyer scrutiny.** ROI card shows 8.0x while the underlying ratio is 8.985x; error-rate axis lacks units (could be 3.5% which is a hard SLO miss); Consumer Group identifier is truncated; no model_version on any tab so operator cannot tie offers ↔ scores ↔ drift to one artifact.

Secondary issues that would need fixing before a NEEDS-DISCLAIMER ship: surface p50/p95/p99 latency numerically; show "filtered N of total M" on offer KPIs; add "last refreshed" timestamp; replace single-point drift chart with a "needs ≥N checks" placeholder; reconcile Stream:0 KPIs with the throughput chart by relabeling them "current queue depth."
