# Customer Simulator — SaaS Deployment Review

**Reviewed:** `config/simulator_config.yaml` (542 lines), `src/data/generator.py` (897 lines), `src/data/orchestrator.py` (468 lines)
**Question:** Would the current simulator pass muster as the data engine for a real SaaS churn product (demos, POCs, model training, customer-facing analytics)?
**Verdict:** **DO-NOT-DEPLOY** as the SaaS production data source. **Acceptable** as a reference fixture for pipeline tests, sales demos with disclosure, and code regressions. The gap to SaaS-grade is large and structural — it is not a tuning problem.

---

## 1. What the simulator does well

| Capability | Status | Note |
|---|:---:|---|
| 6 named personas with proportions summing to 1.00 | ✅ | `vip_loyal/regular_loyal/bargain_hunter/explorer/dormant/new_customer`. Names match `require.md`. |
| 8+ event types | ✅ | `page_view, search, purchase, add_to_cart, remove_from_cart, coupon_use, cs_contact, review` |
| Treatment / control split + balance check | ✅ | `treatment_ratio: 0.50`, persona-balanced, SMD threshold |
| Marketing response taxonomy | ✅ | `conversion / no_response / adverse` per event |
| Configurable churn definition | ✅ | `no_purchase_days, no_login_days, operator: OR/AND` |
| Future-window prediction split | ✅ | T = end_date − max(no_purchase, no_login); features ≤ T, label > T |
| Reproducibility | ✅ | Single `random_seed` makes the run deterministic |
| Per-persona heterogeneous treatment effect | ✅ | `coupon_conversion_lift` ranges 0.05–0.25 across personas |
| Validation gates on output | ✅ | `target_churn_rate.min/max`, `min_group_size`, `group_size_check` |

**Bottom line on strengths**: it is a credible educational / reference fixture. It produces internally consistent customer–event data that correctly exercises the downstream pipeline (RFM, sequence, uplift, survival, A/B). For someone learning the codebase or running CI integration tests, it does its job.

---

## 2. Critical gaps for SaaS deployment

These are structural — none are configuration tweaks. Each is a category a real customer-data engine would have, that this simulator does not.

### 2.1 Distribution stationarity (no time structure)
- `grep season|holiday|day_of_week|drift|trend` in `src/data/generator.py` returns **zero hits**.
- Real customers have weekly patterns (weekend spikes), monthly cycles (payday), holiday discontinuities (Black Friday, Lunar New Year), and shocks (COVID, inflation).
- Models trained on stationary sim data **will fail on real data** because real recency/frequency features have strong day-of-week structure that synthetic recency does not.
- **Impact**: any model performance number this simulator produces is an upper bound that does not transfer to production.

### 2.2 No covariate drift across the simulation horizon
- The persona mix and behavior parameters are fixed at generation time. Customer #5,000 has the same persona-distribution origin as customer #1.
- Real SaaS deployments see **acquisition-channel mix shifts** quarter-over-quarter (paid → organic → referral mix changes), and **persona-distribution drift** during product changes.
- Drift detection (PSI/KS pages) currently runs on a single timestamp because there is no temporal split that produces drift to detect.

### 2.3 No demographic / contextual features
- `grep device|gender|age|geo|country|region|locale` → **zero hits**.
- Real churn models lean heavily on:
  - **device class** (mobile vs desktop heavily predictive of engagement),
  - **acquisition channel** (paid-search vs organic vs referral correlate with retention),
  - **geographic region** (LTV varies 3–10× by country),
  - **customer-tier metadata** (free / paid / enterprise).
- Without these, the model has no proxy for the **single biggest real-world predictor** of churn (cohort/channel) and the dashboard cannot show geo-faceted views that customers expect.

### 2.4 Treatment effect calibrated 3× too aggressive
- Per-persona `coupon_conversion_lift` ranges 0.05–0.25, multiplied by 0.3 in the churn-prob path → realized **A/B lift ≈ 33%** in the dashboard.
- Real e-commerce coupon-retention literature reports lift in the **5–15%** range. 33% on synthetic data trains stakeholders to expect treatment effects that real campaigns will never deliver.
- Anyone benchmarking budget ROI against this number will set spend targets that real campaigns underperform on day one.

### 2.5 Single-product, single-channel monolith
- Every customer interacts with one synthetic product through one synthetic interface.
- Real SaaS:
  - **Multi-product catalog** (cross-sell / upsell are core to retention strategy),
  - **Multi-channel touchpoints** (email + push + in-app + SMS, each with different lift and cost),
  - **Multi-tenant variance** for B2B products (each tenant has its own behavior distribution).
- Recommendations on this simulator can only ever be "send coupon vs no_action". Real recommendation systems pick *which offer* from a catalog of 20+.

### 2.6 No adverse data scenarios
Production data engines must handle:
- **Late-arriving events** (5-min lag through 24-hour backfill)
- **Duplicate events** (idempotency key collisions)
- **Schema migrations** (event_type taxonomy changes mid-simulation)
- **GDPR right-to-deletion** mid-window (customer disappears retroactively)
- **Data corruption** (timestamps in the future, negative amounts)

The simulator generates a perfectly clean snapshot. Any model that uses it will silently fail the first time real data hits it with one of these.

### 2.7 Latent customer state machine is not modeled
- Real customer journey: `signup → trial → active → power_user → at_risk → churned ↔ won_back`.
- This sim has only **persona** as durable state; persona is fixed at customer creation.
- That means the sim cannot generate "win-back" customers (churn → reactivate), "expanding" customers (silver → gold tier), or "downgrading" customers (gold → silver). Real churn models predict transitions; this simulator can only predict an absorbing-state recency threshold.

### 2.8 Event arrival process is too clean
- Events are generated by per-day Poisson-ish sampling with persona-conditioned rates.
- Real customer events follow **Hawkes-like clustering** (one purchase triggers more browsing in the next hour). Real session boundaries are heavy-tailed (some sessions 30s, some 4h). Real intra-day distribution clusters around lunch / evening / payday — not uniform.
- Model features like `pageviews_per_session`, `session_duration` are computed against unrealistically uniform distributions, so feature-importance rankings transfer poorly.

### 2.9 Scale + privacy are unaddressed
- FULL mode is 20,000 customers. SMALL mode (current) is 5,000.
- Real B2C churn problems are 1M–100M customers. The pipeline has not been pressure-tested at the scale a real SaaS lives at.
- No PII generation (realistic-looking fake names, emails, addresses). Privacy-handling bugs (PII leaking into logs / artifacts / dashboards) cannot be tested before production.

### 2.10 Stochasticity is single-source / single-seed
- One `random_seed` controls everything.
- Real systems need bootstrap variance: how stable is the AUC across resamples? What is the 95% CI on the A/B lift? Currently every run is identical or has only one source of variance.

---

## 3. SaaS-deployment scoring (10-axis rubric)

| Axis | Score | Comment |
|---|:---:|---|
| Data realism (distributions) | 4/10 | Persona-driven, but no time structure / drift |
| Temporal structure | 1/10 | No seasonality, no weekly patterns, stationary throughout |
| Demographic coverage | 0/10 | No geo / device / age / channel attributes |
| Multi-product / multi-channel | 1/10 | Single product, single coupon channel |
| Customer state machine | 2/10 | Persona-fixed, no win-back / upgrade / downgrade transitions |
| Adverse data scenarios | 0/10 | Perfectly clean — no late / dup / corrupted / deleted events |
| Treatment-effect realism | 4/10 | Has heterogeneity, but calibrated 3× too aggressive |
| Scale | 3/10 | 5K–20K is a development scale, not a SaaS scale |
| Privacy / PII | 0/10 | Synthetic IDs only; no PII handling tested |
| Determinism / variance | 6/10 | Reproducible (good for tests) but only one seed (no CI / variance) |
| **Overall** | **21/100** | Reference fixture grade, not SaaS-production grade |

---

## 4. Acceptable vs unacceptable use today

### ✅ Acceptable
- Sales demo with explicit "synthetic data" banner (the dashboard now has this — good).
- CI / regression testing for the data pipeline.
- Onboarding new engineers to the codebase.
- Smoke-testing model training infrastructure (does the LightGBM/PyTorch path execute?).
- Internal pilot to validate end-to-end orchestration.

### ❌ NOT acceptable today
- Training a churn model and pointing it at real customers → catastrophic distribution shift.
- Quoting AUC / ROI / lift numbers to investors or customers → numbers don't transfer.
- Vendor benchmark for procurement → favors models that overfit to the simulator's quirks.
- Research publication or external paper → reviewers will reject the methodology.
- Internal performance dashboards for ops teams → would build false confidence.

---

## 5. Remediation roadmap (priority order)

### Tier S — required before any real-customer model
1. **Add seasonality**: weekly cycle (weekend lift), monthly cycle (payday spike), at minimum 1 holiday window.
2. **Add covariate drift**: persona mix should drift smoothly across the simulation horizon (e.g., new_customer share rises over time).
3. **Add demographic features**: geo (3–5 countries), device class, acquisition channel, customer tier. Make them predictive of churn.
4. **Calibrate treatment effect to 5–15% lift**: scale `coupon_conversion_lift` × 0.3 → × 0.1, OR halve the per-persona lift values.

### Tier A — required before any external pilot
5. **Multi-product catalog**: 5–10 product SKUs with cross-sell relationships.
6. **Multi-channel offers**: at minimum email + push + SMS with different cost/lift.
7. **Customer state machine**: transitions among `trial / active / power / at_risk / churned / won_back`.
8. **Adverse events**: random 5% of events arrive late, 1% duplicated, 0.5% corrupted.

### Tier B — for production readiness
9. **Scale stress test**: 1M customer mode (with sampling for development).
10. **PII generator**: realistic fake names/emails/addresses to test masking / GDPR pipelines.
11. **Multi-seed bootstrap mode**: run N seeds, report metric distributions with CIs.
12. **Hawkes-style event arrivals**: replace per-day Poisson with self-exciting process.

### Tier C — research-grade fidelity
13. **Latent factor model** for customers (continuous traits) instead of 6 discrete personas.
14. **Conformal prediction support** so dashboards can show calibrated uncertainty intervals.
15. **Confounded A/B simulation**: include scenarios where treatment assignment correlates with covariates (test causal-inference robustness).

---

## 6. One-paragraph summary for a CTO

> The current simulator is a credible reference fixture but **not a SaaS data engine**. It produces internally consistent customer-behavior data that exercises the entire pipeline — generator → features → ML/DL → uplift → CLV → survival → recommendations → dashboard — and it now correctly models a future-window label/feature time split (Iter 1-redo) so the model no longer learns the tautology recency=label. However, it has zero seasonality, zero covariate drift, no demographic features, no multi-product or multi-channel structure, no adverse-event scenarios, and a treatment-effect calibration that is 3× more aggressive than industry benchmarks. Models trained on this data and pointed at real customers will collapse on day one, and any AUC / ROI / lift number it produces will not survive due diligence. **Use it for demos with disclosure, integration testing, and onboarding. Do not let model artifacts cross the line into customer-facing decisions until Tier S+A remediation lands.**
