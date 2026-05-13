# A6 — Real-Time Scoring (3 tabs)

Source artifacts:
- Tab a (Live): `_test_results/dashboard_pages/13_realtime_scoring_a_live.png`
- Tab b (Offers): `_test_results/dashboard_pages/13_realtime_scoring_b_offers.png`
- Tab c (Monitoring): `_test_results/dashboard_pages/13_realtime_scoring_c_monitoring.png`
- Combined data MD: `_test_results/page_data/13_realtime_scoring.md`

Page banner (all tabs): "Synthetic data — FULL mode (n=20000). All KPIs are simulator-generated." + "Redis: Connected" + "Recommended Offer: no_action".

---

## Tab a — Live Scoring Status

### Visible KPIs
- **Service Health**: Redis Connected; Request Stream = **0**; Response Stream = **0**; Consumer Group = `scoring_consumers`.
- **Recent Scoring History**: Total Scores = **200**; Avg Churn Prob = **27.30%**; High/Critical Risk = **17**; Primary Model = `ensemble`.
- Charts: Scoring Requests per Minute (x-axis Oct 15 2024); Response Latency & Error Rate dual-axis (Oct 15→Oct 16 2024, latency 15–35 ms, error 0–5%); Churn Probability Distribution histogram of 200 scores; Risk Level Distribution bars (low/medium/high/critical). Detailed Scoring Log expander (Latest 50).

### Wrong / contradictory
- **Request Stream = 0 AND Response Stream = 0 vs Total Scores = 200** — direct KPI contradiction. With both queues empty, a SaaS SRE would conclude the consumer is dead; yet 200 scores apparently flowed through. Either the labels mean different things (queue depth vs lifetime processed) and the UI fails to disambiguate, or the streams really are dead and 200 is stale fixture data.
- **Time anchor is stale**: throughput/latency charts plot **Oct 15–16, 2024** as "Real-time". On a page titled *Real-Time Scoring* this is ~19 months out of date relative to the system clock used elsewhere on the page (May 10, 2026, see Tab c).
- **High/Critical Risk = 17 of 200 = 8.5%** is presented as an absolute number with no rate, no trend, and no SLA threshold.

### Unreliable
- **"200" Total Scores** with no time window label — is this lifetime, last hour, or last refresh? The chart x-axis spans ~24h but the KPI is undated.
- Latency chart axis 15–35 ms with no SLO line and no p50/p95/p99 separation — error rate axis 0–5% with no alert threshold.
- Histogram of 200 churn probabilities is too small a sample to draw distribution conclusions for a 20,000-customer population.

### Missing
- No p50/p95/p99 latency split, no QPS counter, no error budget burn, no scored-vs-failed split, no model version pinned to the displayed scores, no last-scored timestamp, no consumer lag (only "stream depth = 0").
- No link from Total Scores 200 to the 20,000-customer population denominator.
- No "as of" timestamp on the Service Health card.

---

## Tab b — Retention Offer Recommendations

### Visible KPIs
- **Total Offers = 44**; **Total Cost = 1,196,659 KRW**; **Expected Revenue Saved = 10,752,341 KRW**; **Expected ROI = 8.0x**.
- Filters: Risk Level (critical, high, medium), Segment (bargain_hunter, dormant, explorer, new_customer, regular_loyal, vip_loyal), Offer Type (discount_coupon, engagement_email, loyalty_points, premium_discount).
- Charts: Offer Type Distribution pie; Average Expected Uplift by Segment bar; Cost vs Expected Revenue Saved by Segment grouped bar; Churn Probability vs Expected Uplift scatter.
- Detailed Offer Recommendations table (priority_score, customer_id, segment, risk_level, churn_probability, offer_type, offer_detail, expected_uplift, estimated_cost…).
- **Quick Recommendation Lookup** for customer 138000: Recommendation = **no_action**; Expected Uplift = **1.46%**; Priority Score = **1.00**; banner "Recommended Offer: no_action".

### Wrong / contradictory
- **ROI math is wrong on the card.** 10,752,341 / 1,196,659 = **8.985x** (≈ 9.0x). Card displays **8.0x** — off by ~11%. Either the number is truncated (not rounded), or one of the three numerator/denominator/ROI tiles is from a different snapshot.
- **"no_action" with Priority Score 1.00 and Expected Uplift 1.46%** is an internal contradiction in the Quick Lookup card. If priority is maxed at 1.00, the recommendation should be the highest-EV offer, not "no_action". If "no_action" is correct (uplift 1.46% < cost threshold), then Priority Score 1.00 is mislabeled — it is acting as a churn risk score, not an action priority.
- The page-level banner "Recommended Offer: no_action" is rendered globally even before any customer is selected, making it look like the system's default recommendation for everyone.

### Unreliable
- **44 offers — denominator missing.** 44 of 200 recently scored = 22%; 44 of 17 high/critical = 259% (impossible, so the denominator is not "high/critical risk"); 44 of 20,000 population = 0.22%. The card never says.
- **Total Cost 1,196,659 KRW vs Budget 50,000,000 KRW** (sidebar) — only 2.4% of budget consumed. No utilization indicator.
- ROI 8x / 9x quoted with no confidence band, no holdout-vs-counterfactual evidence, and no per-segment ROI distribution.
- Average Expected Uplift by Segment chart shows uplifts up to ~5% but the lookup card shows 1.46% — no anchor to the segment baseline.

### Missing
- No "44 of N scored" denominator on the Total Offers card; no "as of" timestamp; no offer-policy version; no link from a row in the table to the customer record.
- No "next best action" alternative shown when the recommendation is `no_action`, no reason code, no estimated cost when no_action is chosen.
- No conversion-rate / accept-rate feedback loop visible — the dashboard never closes the loop on whether yesterday's offers worked.

---

## Tab c — Model Monitoring

### Visible KPIs
- **Total Drift Checks = 1**; **Red Alerts = 1**; **Yellow Warnings = 0**; **Latest Alert Level = RED**.
- Charts: Drift Alerts Over Time (single point, May 10 2026 11:56:50, x-axis spans ~1.5 ms); PSI Trend (single point, "Alert PSI 0.25" line shown); KS Statistic Trend (single point); Scoring Volume Over Time bar chart with ~50 hourly buckets all at value **4**; Mean Churn Probability Over Time line+band (~50 points, Oct 1–Oct 15 2024); Model Type Usage in Recent Scoring pie (ensemble / lightgbm / xgboost). Drift Detection History (Full) and Monitoring Configuration expanders.

### Wrong / contradictory
- **Total Drift Checks = 1, Latest Alert = RED.** A red alarm fired on the very first sample with no comparison history. There is no "is this n-of-n statistically significant" guard — a SaaS monitoring tool fires RED on a single observation, which is a false-positive factory.
- **"Drift Alerts Over Time" with a 1.5 ms x-axis = exactly 1 datapoint.** Calling that a "trend" is misleading; the chart should not render as a time series at n=1.
- **PSI Trend / KS Trend = single point each**, yet the chart panes have full axes implying multi-week tracking.
- **Time anchor mismatch with Tab a.** Drift charts dated **May 10, 2026 11:56:50** (current); throughput/latency charts on Tab a dated **Oct 15–16, 2024**; "Mean Churn Probability Over Time" on this same Tab c also dated **Oct 1 – Oct 15 2024**. Same "real-time" surface, two clocks ~19 months apart.
- **Scoring Volume Over Time = 4 across every one of the ~50 hourly buckets.** Uniform-4 is synthetic placeholder data, not telemetry. A real scoring service has Poisson-shaped traffic; a flat constant betrays the seed.

### Unreliable
- KPIs do not surface the actual PSI / KS values that triggered the RED alert — only the count. An on-call engineer cannot triage from this card.
- "Latest Alert Level = RED" with no baseline period, no feature name, no reference window — it is a status with no payload.
- Model Type Usage pie has no time window and no sample count tied to it.

### Missing
- No alert payload (which feature drifted, PSI/KS value, threshold crossed, baseline window, current window).
- No mute / acknowledge state, no incident link, no alert age.
- No model version, no training date, no last-retrain timestamp tied to the drift event.
- No SLA on monitoring freshness ("last drift check ran X minutes ago").

---

## Cross-tab consistency check (timestamps, model versions, total counts)

| Surface | Timestamp anchor | Sample count |
|---|---|---|
| Tab a — Scoring Requests per Minute | Oct 15 2024 | (per-minute trace, ~24h) |
| Tab a — Latency & Error Rate | Oct 15–16 2024 | (per-minute trace, ~24h) |
| Tab a — Total Scores card | undated | **200** |
| Tab b — Total Offers card | undated | **44** |
| Tab c — Drift Alerts Over Time | **May 10 2026 11:56:50** (1.5 ms span) | **1** |
| Tab c — Mean Churn Prob Over Time | Oct 1 – Oct 15 2024 | ~50 hourly points |
| Tab c — Scoring Volume Over Time | Oct 1 – Oct 15 2024 | ~50 buckets, **all = 4** |

- **Two clock universes on one "real-time" page**: Oct 2024 (throughput / latency / mean-churn / volume) and May 2026 (drift alerts). These cannot both be "current".
- **Counts cannot be reconciled**: 200 recent scores (Tab a) vs 44 offers (Tab b) vs 1 drift check (Tab c) vs ~50 buckets × 4 scores = 200 (Tab c) — only the last ties back to Tab a, and only if you ignore that the buckets are uniform-4 placeholders.
- **Model version is never pinned**. Tab a says "Primary Model: ensemble"; Tab c pie shows ensemble + lightgbm + xgboost mixed. No version string, no training date, no model-id tying the 200 scores, the 44 offers, and the 1 drift check together.

---

## SaaS-readiness verdict — Real-time serving

**Verdict: NOT production-ready.** This is a demo surface dressed as a real-time console. The streams are at 0 while the score counter shows 200, ROI is mis-arithmetic'd on the headline card (8.0x vs the 8.99x the underlying numerator/denominator imply), the drift "trend" is a single point on a 1.5-millisecond axis, scoring volume is a flat constant of 4 across every hour, and two of the three tabs are dated 19 months apart from the third. The customer-lookup card simultaneously reports Priority 1.00 (max) and recommends `no_action` with 1.46% uplift, which is the user-visible giveaway that the priority score is mis-wired to risk rather than to action EV. Nothing on the page would let an on-call SRE answer "is the scoring service healthy right now and what changed in the last hour?".

**Top 3 production-readiness blockers:**
1. **KPI integrity is broken at the headline level.** Request/Response Stream = 0 while Total Scores = 200 (Tab a); Total Offers 44 with no denominator and Expected ROI 8.0x while the displayed numerator/denominator imply 8.99x (Tab b); Total Drift Checks = 1 firing RED on a single sample (Tab c). Three of the four tile groups disagree with their own underlying numbers — fix labels, units, denominators, and rounding before any of this can be trusted.
2. **Time-anchor incoherence and synthetic telemetry.** Throughput/latency dated Oct 15–16 2024, drift dated May 10 2026 11:56:50 (1.5 ms span), Scoring Volume = 4 uniformly across ~50 hourly buckets. The page must commit to one clock, show "as of" on every card, and replace placeholder uniform-4 volume with real consumer-group counters before this is a monitoring tool.
3. **Recommendation engine UX contradicts itself.** Quick Lookup returns `no_action` + Priority Score 1.00 + Expected Uplift 1.46% with no reason code, no alternative offer, and a global banner that defaults every visitor to `no_action`. Either Priority Score is mis-defined (it's a churn-risk score, not an action priority) or the policy is broken — either way it is not a recommendation a CRM operator can act on. Add reason codes, the chosen offer's counterfactual, and a model+policy version stamp before shipping.
