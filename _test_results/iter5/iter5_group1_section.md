## G1 — Data / Feature / Segmentation

### Test results
| Suite | Tests | Pass | Fail | Skip | Duration | Status |
|---|---:|---:|---:|---:|---:|:-:|
| G1 | 168 | 65 | 0 (103 errors) | 0 | 27.33s | FAIL |

**Root cause (single bug, mass-affects 103 tests):** test fixtures open `config/simulator_config.yaml` with `open(path, "r")` — no `encoding="utf-8"`. Windows defaults to cp949 and the YAML now contains a UTF-8 byte (`0xe2`, em-dash/quote) at offset 2057. iter3 ran the same suite green (168/168) — this is a fresh regression introduced by the staged edit to `config/simulator_config.yaml` (visible in `git status`).

Affected files (all errors share the same UnicodeDecodeError stack):
- `tests/test_segmentation.py` — 28 errors (TestSegmentationConfig, RFMScoring, SegmentAssignment, SegmentSummary, ValueUpliftSegmentation, RetentionActions, SegmentationEdgeCases)
- `tests/test_orchestrator.py` — 18 errors (Initialization, FullPipelineExecution, TreatmentControlSplit, PersonaAssignment, OutputFiles, PipelineState, Reproducibility)
- `tests/test_data_generator.py` — 11 errors (TestSingleCustomerEventGenerator + TestDataOutput)
- `tests/test_make_dataset.py` — 2 errors
- `tests/test_generation_summary_schema.py` — 2 errors

Single-line fix (out of scope here): change the fixtures to `open(config_path, "r", encoding="utf-8")`.

The 65 tests that **did** pass cover: persona generation, treatment/control split, churn labels in target range, marketing response, reproducibility (full-data path), preprocessing (load, missing values, outliers, splits, sequences, scaling, validation), feature engineering (RFM, behavioral change, purchase-cycle anomaly, session quality, sequence, time-based, journey stage, no-NaN/no-inf, feature store, ≥30 features, all groups present), and segmentation imports + custom config + KMeans + reproducibility-with-seed. So **the segmentation logic itself is unverified for this run** — only import/instantiation and the seeded-reproducibility path executed.

### Dashboard slice (Page 03 Customer Segmentation)
- **Banner (top):** "Synthetic data — SMALL mode (n=5000). Numbers shown are illustrative; they do NOT represent production performance. Group-size validation: FAILED."
- **KPI strip:** Total Segments = 6 · Total Customers = 5,000 · Highest Risk Segment = dormant.
- **Customers per Segment (bar):** regular_loyal 1,246 · bargain_hunter 977 · explorer 782 · dormant 740 · new_customer 730 · vip_loyal ≈525 (visible in pie/bar but truncated in PNG bar legend).
- **Avg Churn Probability by Segment (bar):** dormant ≈0.95 · explorer ≈0.276 · new_customer ≈0.174 · bargain_hunter ≈0.080 · regular_loyal ≈0.066 · vip_loyal ≈0.007.
- **Segment Statistics table:** counts × {Avg Churn, Min Churn, Max Churn, Std Churn} — dormant row shows 0.94/95% range, vip_loyal shows ~0.0074.
- **Avg CLV by Segment:** vip_loyal far highest (~3.5M+); new_customer ~3.5M visible peak; bargain_hunter ~2.2M; explorer ~1.3M; regular_loyal modest; **dormant ~107k** (lowest).
- **Risk Level Distribution:** dormant column saturated red ("critical" ≈692/740); other 5 segments dominated by green ("low").
- **Segment Definitions table:** 6 rows in Korean — vip_loyal/loyal_customer/potential_loyalist/at_risk/new_customer/bargain_hunter — each with retention action (exclusive_rewards, loyalty_program, engagement_campaign, win_back_campaign, premium_pet_pack, targeted_promotion). Note table lists `at_risk` not `dormant` while the charts show `dormant` — minor naming inconsistency between definition table and runtime segment label. (`explorer` and `dormant` from charts are not in the definitions table.)

**Do KPIs match tested logic?** Partially — `test_at_least_six_segments`, `test_segment_names_match_config`, `test_summary_counts_sum_to_total`, `test_summary_percentages_sum_to_100`, `test_at_risk_has_low_recency_score`, `test_vip_loyal_has_high_rfm` would all be the load-bearing assertions, and **all are erroring this run** due to the YAML fixture bug. The on-screen pattern (dormant ≈95% churn, vip_loyal <1% churn) is qualitatively consistent with tested invariants from iter3's green run, but unverified now.

### SaaS-readiness verdict — Segmentation domain
**Verdict:** NEEDS-DISCLAIMER

**Rationale (3 bullets):**
- Segmentation business logic was previously fully exercised (168/168 in iter3) and the dashboard renders sensible structure: 6 distinct segments, monotonic churn-risk ordering (vip_loyal < regular_loyal < bargain_hunter < new_customer < explorer < dormant), CLV vs risk inversion that makes business sense, no NaN/inf surfaced. Banner explicitly self-discloses "SMALL mode, illustrative, group-size validation FAILED" — that disclosure is the right pattern for a SaaS demo tier.
- Current run has a real regression: 103 errors from a single missing `encoding="utf-8"` in test fixtures triggered by a UTF-8 char added to `config/simulator_config.yaml`. This will reproduce on any non-UTF-8 default tenant runtime (Windows ko/ja/zh-CN locales). For a multi-tenant SaaS this is a blocker if customers self-host on Windows; for a hosted Linux/macOS deployment (utf-8 default) it would silently pass.
- Dashboard has two cosmetic SaaS-grade issues: (a) `dormant` and `explorer` segments shown in charts but not in the "Segment Definitions & Retention Actions" reference table — table lists `at_risk` instead, so customers cannot look up the action playbook for the cohort they care about most; (b) `vip_loyal` truncated from the top-N legend in a couple of bar traces.

**Top 3 deployment blockers:**
1. **Windows-locale UTF-8 read regression** — fix the YAML fixture (`encoding="utf-8"`) and audit all `open(...)` call sites in production code for the same bug; otherwise non-UTF-8-default tenants get 60%+ test failures and likely runtime config-load failures.
2. **Definitions/runtime label mismatch** — reconcile the 6 segment names emitted by `CustomerSegmenter` (vip_loyal, regular_loyal, bargain_hunter, explorer, dormant, new_customer) with the 6 names in the definitions table (… at_risk, loyal_customer, potential_loyalist …). A SaaS customer reading the dashboard cannot map "dormant" to a retention playbook.
3. **"Group-size validation: FAILED" banner is honest but blocks paid-tier framing** — needs a clearer two-tier UX: "Demo (SMALL, statistics not validated)" vs "Production (LARGE, validated)" toggle, with the dashboard refusing to display per-segment churn-rate bars in SMALL mode beyond an illustrative watermark. Currently the same charts render in both modes with only a top-banner caveat.
