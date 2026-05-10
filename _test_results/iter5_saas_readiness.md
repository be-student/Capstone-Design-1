# Iter 5 — SaaS Deployment Readiness Audit (6-way Parallel)

**Date:** 2026-05-10
**Method:** 6 sub-agents ran the existing iter3 test partition in parallel (`venv/Scripts/python.exe -m pytest …`) and audited the 14 dashboard PNGs in `_test_results/dashboard_pages/` plus the 3 iter2 overview PNGs in repo root, against SaaS-deployment criteria.
**Bottom line:** **DO-NOT-SHIP for external SaaS pilot.** Major regression vs iter3 + multiple cross-page UX defects + zero cross-page invariant tests.

---

## 1. Test result matrix — iter5 vs iter3

| Group | Domain | Tests | Pass | Fail | Errors | Pass% | iter3 ref |
|---:|---|---:|---:|---:|---:|---:|---|
| G1 | data / segmentation | 168 | 65 | 0 | 103 | 38.7% | 168/168 ✅ |
| G2 | churn / DL / CLV / survival | 207 | 31 | 0 | 175 | 15.0% | 207/207 ✅ |
| G3 | A/B / cohort / uplift | 381 | 218 | 2 | 161 | 57.2% | 381/381 ✅ |
| G4 | budget / reco / what-if | 453 | 72 | 0 | 381 | 15.9% | 453/453 ✅ |
| G5 | dashboard views | 652 | 378 | 3 | 271 | 58.0% | 651/652 ⚠️ |
| G6 | infra / realtime / MLflow | 730 | 456 | 2 | 272 | 62.5% | 688/730 ⚠️ (42 MLflow leak) |
| **Total** | — | **2,591** | **1,220** | **7** | **1,363** | **47.1%** | iter3 ≈ 98.3% |

**Regression severity:** iter3 was 2,547/2,590 green (98.3%). iter5 is 1,220/2,591 green (47.1%). **Net loss: 1,327 previously-passing tests.**

### 1.1 Single root cause for ~1,360 of the 1,370 non-passes

```
UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 2057
  → File "<fixture>", in load_config: yaml.safe_load(open(CONFIG_PATH, "r"))
  → File "config/simulator_config.yaml", line "ted safety net — left at 0"
```

- `config/simulator_config.yaml` is **currently staged as modified** (`git status` shows `M config/simulator_config.yaml`). The recent edit introduced a UTF-8 em-dash (`0xe2 0x80 0x94`) at byte 2057.
- `open(path, "r")` without `encoding="utf-8"` falls back to the locale codec; on Korean Windows that's cp949, which can't decode `0xe2`.
- Affected layers:
  - **Production code:** `src/dashboard/app.py:83` — `with open(CONFIG_PATH, "r") as f: return yaml.safe_load(f)`
  - **Test fixtures:** ≥ 12 files across 6 groups (every `config` fixture that loads the YAML)
- Docker (Linux, UTF-8 default) **masks** this entirely. Any Windows / Korean / Japanese / Chinese self-host tenant breaks on first config load.

### 1.2 Real (non-environment) failures still standing — iter5

| File | Test | Reason |
|---|---|---|
| `test_dashboard.py` | 3 failures | Fail at runtime — independent from the cp949 cascade (per G5 section) |
| `test_dockerfile_syntax_no_empty_from` | 1 failure | Reads Dockerfile.dashboard without `encoding="utf-8"` (G6) |
| `test_entrypoint.py::test_script_syntax_valid` | 1 failure | Windows path mangled to Git-Bash (`C:Users…` lost backslashes) |
| `test_cohort_computations.py` | 2 failures | Same encoding root cause inside non-fixture path |

### 1.3 What did NOT recur from iter3

- **MLflow `Run already active` leak:** 0 instances (iter3: 42). The earlier-stage cp949 errors prevent fixtures from ever entering the MLflow context, masking the leak. **Not actually fixed** — masked by a worse upstream regression.
- `test_clv_predictions_positive` (LARGE-mode bound, the 1 non-MLflow failure in iter3): not in scope of any iter5 group's evaluable subset.

---

## 2. Per-domain SaaS-readiness verdicts

| Group | Domain | Verdict | Headline issue |
|---|---|:--:|---|
| G1 | Segmentation | NEEDS-DISCLAIMER | Definitions table lists `at_risk` but runtime emits `dormant` / `explorer` — customer can't look up retention playbook |
| G2 | Models | NEEDS-DISCLAIMER | Models clear AUC≥0.78 baseline, but `training_time=1s`, `Median Survival 357d` (right-censored window ceiling), CLV-vs-Churn null-X join all leak to UI |
| G3 | Experiments | **DO-NOT-SHIP** | Page 11 plots `uplift_score == treatment_effect` as two independent variables (y=x scatter); page 04 shows 2 cohorts vs SaaS-norm 6-12; page 06 zero-state KPIs read as "broken" |
| G4 | Budget / Reco | **DO-NOT-SHIP** | Three different "Overall ROI" formulas across pages 05/09/12 (5.2× / 9.0× / 3.4×); float-precision leak on page 12; LP solver coverage un-evaluable due to fixture errors |
| G5 | Dashboard UX | **DO-NOT-SHIP** | Synthetic-data banner appears in only 1 of 3 iter2 PNGs (intermittent across refresh/restart); zero cross-page invariant tests |
| G6 | Operations | **DO-NOT-SHIP** | MLflow server unreachable; single-point "trend over time" charts; error rate ~1.5% (15× SaaS SLO); flat synthetic "Scoring Volume = 4" |

**Aggregate verdict: DO-NOT-SHIP for external SaaS pilot. Internal capstone-demo OK with verbal disclaimer.**

---

## 3. Consolidated SaaS blockers (P0 → P2)

### P0 — environment regression
| # | Blocker | Evidence | Fix scope |
|---|---|---|---|
| P0.1 | `open(...)` without `encoding="utf-8"` causes Windows cp949 crash on UTF-8 yaml | 1,363 test errors; `src/dashboard/app.py:83` and ≥12 fixtures | 1 line × ~13 sites + lint guard |

### P0 — load-bearing trust failures (no environment dependency)
| # | Blocker | Evidence | Code/page |
|---|---|---|---|
| P0.2 | Synthetic-data banner is **intermittent** across refresh/restart | iter2_overview_banner.png (no banner) + iter2_overview_after_restart.png (no banner) + iter2_overview_after_refresh.png (banner present) | Render-time guard, not boot-time. Dockerfile hot-reload patch from iter3 does NOT fix this. |
| P0.3 | "Overall ROI" reported as 3 different numbers on 3 pages | 5.2× page 05 (`app.py:850` mean of segment ROIs) / 9.0× page 09 (`recommendations_view.py:363` aggregate over treated) / 3.4× page 12 (`app.py:2673` aggregate over full budget) | Single `compute_overall_roi()` helper + cross-page consistency test |

### P1 — visible quality smells
| # | Blocker | Evidence | Code/page |
|---|---|---|---|
| P1.1 | `uplift_score == treatment_effect` plotted as 2 axes | KPIs `0.0714 == 0.0714`; scatter on y=x | Page 11; remove duplicate or compute genuine ATE |
| P1.2 | Float-precision leak | `Customers Retained = 125.66457549932906` | `app.py:2686` (`f"{x:,}"` should be `:.0f`) |
| P1.3 | "Trend over time" with n=1 datapoint | PSI/KS/Drift on pages 08, 13, 14 | Render `st.info("Insufficient history")` when len(timeseries) < 5 |
| P1.4 | Empty/broken charts when n=0 | page 06 A/B (1 chart, all-zero KPIs); page 10 CLV-vs-Churn (`x_sample=null`); page 12 radar (n=0); page 14 radar (n=0) | Shared `render_empty_state()` component |
| P1.5 | "Request Stream 0" KPI next to "Total Scores 200" KPI | Page 13 service-health tab | Reconcile queue-depth vs lifetime-processed labels |
| P1.6 | 3 microsecond-spaced timestamps plotted as time series | Pages 08, 14 "Model Performance Over Time" | Either show real training history or remove the chart |
| P1.7 | Definitions/runtime-label mismatch | Page 03 table lists `at_risk` but charts emit `dormant` / `explorer` | Reconcile naming or make the table dynamic |

### P2 — quality / scale concerns
| # | Blocker | Observed vs SaaS target |
|---|---|---|
| P2.1 | Error rate | 1.03–1.86% observed vs <0.1% SaaS SLO (15× over) |
| P2.2 | MLflow tracking server | Offline — page 14 in cache-fallback ("Showing cached experiment data from artifacts") |
| P2.3 | Hyperparameter sweep | All 3 runs at LR=0.1 / epochs=1 — not a sweep |
| P2.4 | Cohort count | 2 cohorts vs SaaS-norm 6–12 |
| P2.5 | Synthetic "Scoring Volume = 4" flat line | All 50 hourly buckets identical — clearly generator output |
| P2.6 | Median Survival 357d ≈ 365d window ceiling | Right-censoring artifact rendered as "Median Duration" without annotation |
| P2.7 | `training_time = 1s` for all 3 models | Synthetic-second floor leaks into customer-facing page 02 |
| P2.8 | Probability bimodality 79.8% | Plausible for dormant mass-labeling; uncalibrated and untested for non-bimodality / Brier |
| P2.9 | Page 10 CLV-vs-Churn scatter `x_sample=null` × 5 per trace | View-layer join issue, no test coverage |
| P2.10 | A/B observed lift 33.8% vs literature 5–15% | Disclosed in iter3 audit but no in-page disclaimer |
| P2.11 | "Group-size validation: FAILED" banner phrasing | Correct disclosure for SMALL but technical for non-eng buyers |
| P2.12 | Two coexisting segment taxonomies | behavioral (vip_loyal/regular_loyal/dormant/explorer/new_customer/bargain_hunter) and uplift (high_value_sure_thing/high_value_persuadable/sleeping_dog/lost_cause) — no legend |

---

## 4. Test-coverage gaps surfaced by the audit

These would each be a single-file-or-snippet addition. Their absence let real product issues escape iter3:

| Gap | Would catch | Suggested test |
|---|---|---|
| No `treatment_effect != uplift_score` assertion | Page 11 false-equivalence | `assert not np.allclose(uplift, te, atol=1e-6)` |
| No uplift uniqueness floor | Future re-collapse to <1000 unique values | `assert df['uplift_score'].nunique() >= 1000` |
| No KPI-vs-chart consistency check | Page 04 "Avg Final Retention 41.4%" vs chart's ~90% terminal | Snapshot of KPI value == terminal point of chart |
| No A/B empty-state assertion | Page 06 zero-KPI broken-state | Assert empty-state element when `len(experiments)==0` |
| No A/B lift-realism guard | 33.8% synthetic vs 5–15% literature | `assert observed_lift <= 0.20` or warning render test |
| No coupon-eligibility floor | iter3-fixed coupon-to-low-risk regression isn't locked | `assert (recos_with_coupon['churn_prob'] >= threshold).all()` |
| No KPI format test | float leak (125.66457549932906) | Snapshot/regex test on rendered KPI strings |
| No cross-page ROI consistency | 5.2× / 9.0× / 3.4× discrepancy | Single test calls `compute_overall_roi()` from each surface |
| No CLV↔Churn join null check | Page 10 `x_sample=null` | `assert df.churn_probability.notna().all()` after join |
| No Brier / reliability calibration | Page 01 U-shaped probability histogram | `assert brier_score(y, p) <= 0.20` |
| No banner-persistence Playwright test | Banner missing 2/3 iter2 PNGs | `expect(page.locator('.synthetic-banner')).to_be_visible()` after `page.reload()` |
| No `encoding="utf-8"` lint | Today's regression | grep CI guard or ruff PT004 + RUF rule |

---

## 5. iter3_final.md claims vs iter5 reality

| iter3 claim | iter5 evidence |
|---|---|
| "Synthetic banner: present (auto)" | **CONTRADICTED** — iter2_overview_banner.png and iter2_overview_after_restart.png both render WITHOUT the banner; only iter2_overview_after_refresh.png shows it. The Dockerfile patch (`runOnSave=true --server.fileWatcherType=poll`) addresses code-edit hot-reload, not banner persistence. |
| "5 high-revenue-impact bugs closed" | Largely holds at the model layer (AUC, uplift collapse, coupon spray to zero-risk). But the **uplift==treatment_effect duplication** is a 6th bug that was not enumerated, and remains. |
| "42 MLflow `Run already active` failures (test infra)" | **MASKED, not fixed** — the cp949 fixture cascade prevents MLflow from ever opening a run, so the leak doesn't trigger. Will reappear once the encoding bug is fixed. |
| "SHIP-READY for internal pilot" | Holds for **internal capstone demo with verbal disclaimer**. Does NOT hold for external SaaS pilot. |
| "98.3% pass rate" | Was true at iter3 commit. iter5 with current staged YAML edit drops to 47.1%. |

---

## 6. Recommended path forward (smallest delta to credible external-pilot)

In order:

1. **Single-PR encoding fix** — `encoding="utf-8"` everywhere `open(...)` reads YAML/Dockerfile/scripts. Add CI lint that fails on missing-encoding text-mode opens. Expected impact: restore ~1,360 errored tests to green, re-expose the 42-failure MLflow leak.
2. **Fix the MLflow leak** in `tests/conftest.py` (per iter3_final.md, "one-fixture fix").
3. **Banner persistence** — render-time guard, plus Playwright test asserting `.synthetic-banner` visible after `page.reload()` and after compose restart.
4. **ROI single-source-of-truth** — one helper, one test that fails on cross-page divergence > 1%.
5. **Page 11 fix** — drop the duplicate Treatment-Effect KPI/scatter OR plug in a real ATE from the experimental control. Add `assert not np.allclose(uplift, te)` regression.
6. **Empty-state component** — render `render_empty_state(message, suggested_action)` on pages 06, 10, 12, 14.
7. **`format_count(x)` helper** + lint banning raw `f"{x}"` for KPI-card numerics.
8. **Drift-trend guard** — `st.info("Insufficient history (need ≥7 observations)")` when `len(history) < 5`.
9. **Operational basics for SaaS pilot** — real MLflow server up, real Redis-derived volume metric, error rate <0.5%.

After 1–8 land, re-run the 6-group audit; if test pass rate ≥ 98% and the P0/P1 dashboard items are closed, escalate to external-pilot review.

---

## 7. Reference artifacts

### Per-group sections (full detail)
- [G1 — Data / Feature / Segmentation](iter5_group1_section.md)
- [G2 — Model Training](iter5_group2_section.md)
- [G3 — A/B / Cohort / Uplift](iter5_group3_section.md)
- [G4 — Budget / Recommendations / What-if](iter5_group4_section.md)
- [G5 — Dashboard Views / UX](iter5_group5_section.md)
- [G6 — Infra / Realtime / MLflow](iter5_group6_section.md)

### JUnit XML (per group)
- `_test_results/iter5_group1.xml` … `iter5_group6.xml`

### Source PNGs reviewed
- `_test_results/dashboard_pages/01_churn_analytics.png` … `14_mlflow_experiments.png` (16 captures including 13a/b/c tab variants)
- `iter2_overview_banner.png`, `iter2_overview_after_refresh.png`, `iter2_overview_after_restart.png` (repo root)

### Prior-iteration baselines
- `_test_results/iter3_final.md` — iter3 convergence record (claims SHIP-READY)
- `_test_results/iter3_realism_audit.md`, `iter2_realism_audit.md`, `iter1_redo_realism_audit.md`, `iter1_realism_audit.md`, `realism_synthesis.md`
- `_test_results/DASHBOARD_PAGES.md` — full dashboard reference (used by all 6 agents)
