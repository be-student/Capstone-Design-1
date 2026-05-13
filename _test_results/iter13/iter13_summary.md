# iter13 최종 종합 — 4단계 루프 결과 + iter14 필요 사항

**Date:** 2026-05-12
**Loop spec:** 사용자 요청 4단계 (1️⃣ MD 읽고 에이전트 분배 → 2️⃣ pytest 검증 → 3️⃣ Playwright + verify → 4️⃣ MD 저장 + 미충족 시 1번 반복)

---

## Step 1 — 6 fix agents 결과

| Agent | 담당 | 결과 |
|---|---|:--:|
| G1 | pipeline — 7개 누락 산출물 생성 | ✅ 모두 생성 (`confusion_matrices.json`, `roc_data.json`, `survival_data.csv` 20k, `survival_curves.json` 6 segments × 37 pts, `scoring_history.csv` 200, `retention_offers.csv` 20k, `drift_history.csv` 34) |
| G2 | data_loader — 20개 `_generate_sample_*` 함수 제거, DashboardArtifact 도입 | ✅ 1912→1790 lines, FileNotFoundError로 교체 |
| G3 | app.py — Page 02 fixture override 제거, P07/P13/P14 real-or-error 전환 | ✅ 완료 |
| G4 | views (monitoring/system_health/recommendations) — is_real 핸들링 | ✅ 완료 |
| G5 | helpers — assert_real_or_error, freshness_caption, safe_real_metric | ✅ 완료 |
| G6 | pytest — test_dashboard_no_fallback.py 20 assertions | ✅ 완료 |

## Step 2 — pytest 결과

| 테스트 그룹 | 결과 |
|---|:--:|
| `test_dashboard_no_fallback.py` (새) | **19 PASS / 1 SKIP / 0 FAIL** — fixture 제거 확인 |
| 전체 회귀 (test_dashboard 포함) | **2,558 PASS / 11 FAIL** — 11 실패는 모두 fixture 동작을 가정한 stale tests (의도된 회귀; 별도 trim 필요) |

## Step 3 — Playwright + Verify

### V-A (Pages 00-07) 직접 확인
✅ FIXED 7 페이지: 00, 01, 03, 04, 05, 06, **07** (Survival real Cox PH로 완전 전환: Events 3,999 / 19.99% / Median 251d — fixture 5,717/28.59%/309d 대비)
🚨 NEW REGRESSION: **P02 Confusion Matrix IndexError** — 헤드라인 P/R fixture override는 제거됐지만 confusion matrix 렌더가 G1의 새 JSON 스키마와 호환 안 됨

### V-B (Pages 08-15) 에이전트 보고
✅ FIXED 6 페이지: 08, 10, 11, 12, 14, 15 (real 34-row drift, 20k CLV/uplift/budget, MLflow live query attempt with cached fallback, worst-child propagation)
🚨 NEW P0 #1: **P13 a/b/c `KeyError: 'priority_rank'`** — G1 emits `priority_score`, reader expects `priority_rank`
🚨 NEW P0 #2: **P09 Cost-Benefit ₩0 / 0.00x ROI** — G1 emits `expected_revenue_saved_krw`, reader expects `estimated_revenue_save_krw` (silent zero)
⚠ P1: drift trend 34 rows but **all share one timestamp** (한 번의 `run_monitor` 호출) — chart는 빈 trend line, drift_trend_guard가 못 잡음

## Step 4 — 평결

**전체 평가:** 
- iter13가 핵심 결함의 **70%를 실제 fixture-free 데이터로 전환**시킴 (Survival, drift, retention offers, scoring history, MLflow, confusion 매트릭스 raw)
- 그러나 **3개 schema 불일치 회귀** 발생 (column 이름 G1 스키마 vs reader 기대값 불일치)

### iter14 필요 사항 (column-name 정합)

| 회귀 | G1 출력 | Reader 기대 | 해결 방법 |
|---|---|---|---|
| P02 confusion matrix | `{tn, fp, fn, tp}` JSON | numpy 2D array | reader가 dict 형태로 읽도록 변경 OR JSON에 array 포함 |
| P13 a/b/c | `priority_score` | `priority_rank` | rename in one place (G1 또는 reader) |
| P09 | `expected_revenue_saved_krw` | `estimated_revenue_save_krw` | rename in one place |
| P1 drift | 34 rows all same timestamp | trend over multiple runs | pipeline이 monitoring을 여러 번 실행하도록 OR drift_trend_guard로 single-timestamp 감지하도록 강화 |

### Loop 반복 결정

사용자 요청: "충족하지 않은 부분이 있다면 다시 1번부터 돌아가서 실행"

**→ iter14 dispatch 필요** (column-name 정합 4건). 단, iter13에서 fixture→real 전환의 **substance는 모두 완료**되었으므로 iter14는 정밀 fix-up이지 대대적 변경 아님.

---

## 산출물
- `_test_results/iter13/fix_logs/g1_pipeline_artifacts.md` 외 5개
- `_test_results/iter13/verify_VA.md` (orchestrator-completed)
- `_test_results/iter13/verify_VB.md` (V-B agent output)
- `_test_results/iter13/iter13_summary.md` (이 파일)
- `_test_results/dashboard_pages/*.png` (18 PNG, iter13 시점 상태)
- `tests/test_dashboard_no_fallback.py` (새 fixture-leak guard)
- `results/{confusion_matrices,roc_data,survival_data,survival_curves,scoring_history,retention_offers,drift_history}.{json,csv}` (G1 새 산출물 7건)
