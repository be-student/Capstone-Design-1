## G4 — Budget / Recommendations / What-if

### Test results

Aggregate: **72 pass, 0 fail, 381 errors, 0 skip — 4.59s wall** (`_test_results/iter5_group4.xml`).

| Suite | Tests | Pass | Fail | Err | Skip | Duration | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| test_budget_cost_config | 60 | 25 | 0 | 35 | 0 | 0.37s | ERR |
| test_budget_lp_solver | 28 | 0 | 0 | 28 | 0 | 0.00s | ERR |
| test_budget_optimization | 69 | 5 | 0 | 64 | 0 | 3.09s | ERR |
| test_budget_optimization_logic | 49 | 5 | 0 | 44 | 0 | 0.01s | ERR |
| test_budget_optimizer | 25 | 0 | 0 | 25 | 0 | 0.00s | ERR |
| test_lp_budget_optimizer | 43 | 0 | 0 | 43 | 0 | 0.00s | ERR |
| test_recommendations | 33 | 0 | 0 | 33 | 0 | 0.00s | ERR |
| test_recommendations_view | 38 | 33 | 0 | 5 | 0 | 0.11s | ERR |
| test_whatif_analysis | 24 | 0 | 0 | 24 | 0 | 0.00s | ERR |
| test_whatif_budget_optimizer | 33 | 4 | 0 | 29 | 0 | 0.00s | ERR |
| test_whatif_scenario | 51 | 0 | 0 | 51 | 0 | 0.00s | ERR |
| **TOTAL** | **453** | **72** | **0** | **381** | **0** | **4.59s** | **BLOCKED** |

**Root cause of the 381 errors — single shared bug, not 11 independent failures.**
Every error is a setup-phase `UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 2057` raised from a `config` fixture that does `open(CONFIG_PATH, "r")` without `encoding="utf-8"`. On Windows the default codec is cp949 (Korean), and `config/simulator_config.yaml` contains a UTF-8 em-dash (`0xE2 0x80 0x94`) at byte 2057 (line "ted safety net — left at 0"). One-line fix: `open(CONFIG_PATH, "r", encoding="utf-8")` in every Group-4 test file's `config` fixture (11 files). The 72 tests that *did* pass are the ones whose tests don't depend on the `config` fixture (e.g. pure-math LP tests, dataframe-shape recommendation-view tests).

This is a test-harness bug, not a product bug. Treat the 381 errors as **un-evaluated** rather than failed. Until the fixture is fixed I cannot certify Budget/Reco/What-if behaviour at all.

### Dashboard slice

**Page 05 (Budget Optimization):**
- KPIs: Total Allocated 50,000,000 KRW · Expected Retained 123 · Revenue Saved 171,628,990 KRW · **Avg ROI 5.2x**.
- 8-segment LP allocation: low_value_persuadable 16.76M · low_value_sure_thing 10.79M · high_value_sure_thing 5.90M · **high_value_persuadable 470k (despite 21.55x ROI)** · high_value_lost_cause 0 (correct).
- "Channel-Level Cost Breakdown" header is empty under a banner: *"Channel configuration not found in config. Add budget.channels to simulator_config.yaml."* Disclosure is graceful but the empty H3 above the banner is a UX miss.
- Budget sweep curve is **perfectly linear** from 10M → 50M (25 retained → 125 retained, 34.3M → 171.6M saved). Synthetic-model output, not a fit.

**Page 09 (Recommendations):**
- KPIs: Total Recommendations 5,000 · Avg Expected Uplift 8.84% · **Top Action Type "No Action"** (4,319 of 5,000, 86%) · High Priority 4,095 · Total Campaign Cost 1,211,055 KRW · Est. Revenue Saved 10,893,463 KRW · **Overall ROI 9.0x** · Avg Expected Uplift (campaign-only) 10.88%.
- Only 681 customers receive a coupon, almost all from `low_value_persuadable` (590); `high_value_persuadable` gets 9, others 0. Coupon-uplift is sane (median ~0.22 vs no_action 0.07).
- Two adjacent KPI cards both labeled "Avg Expected Uplift" with different values (whole-pop vs treated-pop). UX confusion.

**Page 12 (CLV & Retention Campaign):**
- KPIs: Total CLV 17.34B · At-Risk CLV 773.3M (4.5%) · Avg Uplift 0.0714 · Treatable 4,143 (82.9%) · Budget 50M · Revenue Saved 171,628,993 · **Customers Retained 125.66457549932906** · **Overall ROI 3.4x**.
- Two confirmed UI defects: (1) full-precision float on the Customers Retained KPI; (2) "Campaign Effectiveness by Segment" radar shows 0 trace points — broken chart.
- This page splices Sections 1-4 from pages 10/11/05/09 with its own ROI denominator → produces a third ROI value.

### SaaS-readiness verdict — Budget/Reco domain
**Verdict: DO-NOT-SHIP**

**Rationale:**
The Group-4 test base is currently un-evaluable (84% errored) because of a Windows encoding bug in 11 fixtures. Even setting that aside, the dashboard surface has three independent SaaS blockers — inconsistent ROI definitions across pages, a raw-float KPI leak, and an undisclosed broken radar — plus two model-quality concerns (perfectly linear budget sweep; 86% no_action) that the existing test base does not explicitly guard against. A paying customer staring at "Overall ROI 5.2x / 9.0x / 3.4x" on three views of the same campaign will not trust the product.

**Top deployment blockers:**
1. **Test fixture bug — 381 errors masking real coverage.** All Group-4 `config` fixtures call `open(CONFIG_PATH, "r")` without `encoding="utf-8"`. On Windows (cp949 default), the UTF-8 em-dash at byte 2057 of `simulator_config.yaml` raises `UnicodeDecodeError`. One-line fix per file.
2. **No test pins ROI definition consistency across views.** The dashboard reports three different "ROI" KPIs from three different formulas (see analysis below). `tests/test_dashboard.py:1383,1404` only checks `overall_roi > 0` and `> 1.0` for one definition; nothing asserts the three KPIs agree or are labeled differently.
3. **No test enforces "coupon recipients have churn_prob > threshold."** `test_recommendations.py::test_uses_churn_probability` only asserts mean-score correlation (line 432-447), not a hard floor for coupon eligibility. The iter3 audit claim that coupons-to-low-risk is "0% now" is not regression-locked.
4. **No format/precision test on KPI cards.** Nothing in Group-4 catches the `125.66457549932906` float leak. The page-12 KPI uses `f"{total_retained:,}"` while other pages use `:,.0f` or `:.1f`. A snapshot/format test would fix this.
5. **Linear budget sweep not flagged.** `test_budget_lp_solver.py::TestBudgetMonotonicity::test_roi_monotone_in_budget` and `test_budget_optimization_logic.py::test_diminishing_marginal_returns` (line 201) test *non-decreasing* and *non-increasing-marginal* — the synthetic linear sweep technically passes both. No test asserts strict concavity / saturation, so the unrealistic linear curve sails through.
6. **high_value_persuadable underfunded vs ROI.** 21.55x segment receives 0.94% of budget. Without seeing the LP solve (tests errored), I cannot tell if this is correct (segment customer count is small) or a solver bug. `test_budget_lp_solver.py::TestBudgetMonotenicity` would have answered this — currently un-evaluated.
7. **Channel chart empty due to missing config.** The banner is graceful but the empty section header above it is not. Either hide the H3 or rephrase as "Channel breakdown unavailable — add `budget.channels`."
8. **86% no_action.** The model is conservative by design (budget is allocated by uplift × CLV with a positive-ROI threshold), but with no test asserting a *minimum* recommendation rate, a small parameter shift could push it to 100% no_action and the dashboard would still render "successfully."

**ROI-consistency analysis — three formulas, three denominators:**

| Page | Source | Formula | Denominator | Numerator | Result |
|---|---|---|---|---|---|
| 05 Budget | `app.py:850` | `display_results["roi"].mean()` | per-segment cost (8 rows) | per-segment revenue saved | **5.2x** = unweighted mean of segment ROIs (skewed up by 21.55x outlier in high_value_persuadable) |
| 09 Reco | `recommendations_view.py:363` | `total_revenue / max(total_cost, 1)` | sum of `estimated_cost_krw` over the **681 offers actually issued** (1.21M) | sum of `estimated_revenue_save_krw` (10.89M) | **9.0x** = aggregate over treated customers only |
| 12 Campaign | `app.py:2673` | `total_rev_saved / total_allocated` | full LP-allocated 50M budget | sum of LP `expected_revenue_saved_krw` (171.6M) | **3.4x** = aggregate over the entire 50M budget envelope |

These are three legitimate-but-different KPIs (mean of per-segment ratios vs aggregate ratio over recommendations vs aggregate ratio over full budget). All three are labeled "Overall ROI" / "Avg ROI" without a footnote. **A SaaS user has no way to know which to trust.** Fix options: (a) rename to "Avg Segment ROI" / "Treated-Pop ROI" / "Budget-Envelope ROI" with tooltip definitions; (b) collapse to a single canonical definition; (c) add `tests/test_dashboard_roi_consistency.py` that fails if the three numbers diverge by > X% on identical input. None of these exists today.

### Bottom line
Domain code is largely passing where evaluable (72/72 of fixture-free tests green), but the test base is currently providing **near-zero coverage** because of the encoding bug, and the dashboard's three-different-ROI / float-leak / broken-radar / linear-sweep trio are SaaS-trust killers. Fix the fixture in one PR, then a follow-up PR adds the four missing assertion tests (ROI consistency, coupon eligibility floor, KPI format snapshot, sweep concavity). Until both land: **DO-NOT-SHIP**.
