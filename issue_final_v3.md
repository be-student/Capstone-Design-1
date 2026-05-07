# issue_final_v3.md

## 목적

이번 문서는 `require.md`를 필수 가드레일로 두고, `issue_final.md`와 `issue_final_v2.md` 이후 반영 상태를 6개 fresh-context agent가 다시 검증한 결과를 정리한다. 이 사이클은 다음 수정 작업의 기준점이며, 다음 검증은 새 문서 `issue_final_v4.md`로 이어간다.

검증 조건:
- 각 agent는 이전 대화 컨텍스트 없이 read-only로 검증했다.
- `require.md`, `issue_final.md`, `issue_final_v2.md`, `docs/requirement_traceability.md`를 확인했다.
- full test suite는 실행하지 않았다.

## 6개 agent 판정

| 관점 | Agent | 판정 | 핵심 결론 |
| --- | --- | --- | --- |
| CLI / Docker / pipeline / artifact completeness | Cicero | FAIL | Docker/full pipeline wiring은 개선됐지만, 현재 artifact evidence는 small-mode이고 cohort/checklist/artifact mirror가 stale 상태다. |
| Simulator / cohort / journey / feature engineering | Lovelace | FAIL | config와 feature engineering은 대체로 맞지만, 현재 cohort artifact가 M1/M3/M6/M12와 journey outputs를 증명하지 못한다. |
| ML / DL / SHAP / uplift / CLV | Sagan | FAIL | ML/DL/uplift/CLV 구조는 대부분 통과하지만, local SHAP artifact, CLV proxy validation, uplift learner selection, full-mode evidence가 남았다. |
| Segmentation / budget / recommendations / A-B | Popper | FAIL | budget/recommendation은 개선됐지만, high-value/high-risk actionable segment evidence와 A/B power/sample-size evidence가 부족하다. |
| Dashboard / loader / monitoring | Laplace | FAIL | monitoring report는 좋지만, churn overview가 전체 고객이 아니고 dashboard sample fallback/artifact mirror 문제가 남았다. |
| Docs / traceability / hygiene / commit readiness | Carson | FAIL | tracked diff는 clean하나 `.audit/`이 남아 있고, docs가 현재 stale artifacts보다 앞선 증거를 주장한다. |

## 종합 판정

현재 코드 개선은 진행됐지만 `require.md` 제출 기준으로는 아직 PASS가 아니다. 가장 중요한 공통 이슈는 네 가지다.

1. 현재 generated evidence가 full mode가 아니라 small mode다.
2. 현재 `results/`와 `data/artifacts/`가 stale/mismatched 상태라 checklist와 dashboard evidence가 신뢰되지 않는다.
3. cohort/journey artifact는 현재 파일 기준으로 invalid인데 checklist/docs는 이를 통과처럼 기록한다.
4. 일부 요구사항은 구조는 있으나 제출 evidence가 부족하다: local SHAP, CLV actual-vs-predicted 방식, uplift best learner selection, A/B power/sample-size, high-value actionable segmentation.

## 현재 통과로 볼 수 있는 항목

- Docker 기본은 full mode 방향이다. `SMALL=${SMALL:-false}`와 entrypoint small flag passthrough가 확인됐다.
- `run_all`은 주요 단계인 simulate/features/train/uplift/clv/segment/optimize/recommend/cohort/ab_test/survival/monitor를 포함한다.
- Feature engineering은 33개 feature dictionary, feature store CSV/Parquet, RFM/change/session/sequence/time/journey 계열 evidence가 있다.
- Recommendation은 current `results/recommendations.csv` 기준 negative uplift/sleeping dog active action 위반이 0건이다.
- Budget은 50/100/200 what-if, ROI, sleeping dog zero budget evidence가 있다.
- Monitoring report는 PSI/KS feature alerts와 alert level, latest performance rows를 포함한다.
- ML/DL/uplift/CLV 기본 산출물은 small-mode 기준 존재한다.

## Blocker 1: full-mode evidence 부재

증거:
- `data/raw/generation_summary.json`은 5,000 customers, 2,500/2,500 treatment/control, `generation_mode: small`을 기록한다.
- `group_size_check.passed`가 false다.
- full requirement는 20,000 customers, 12개월 이상, treatment/control 각각 10,000명 이상이다.

필요 조치:
- 최종 제출 전 clean full run을 한 번 실행해 full-mode `generation_summary.json`과 downstream artifacts를 다시 생성해야 한다.
- checklist가 `generation_summary.json`의 full/small context를 명시적으로 검증해야 한다.

## Blocker 2: cohort/journey current artifacts invalid

증거:
- 일부 agent가 본 현재 artifact는 `cohort_retention_matrix.csv`가 period `0.0`만 갖고, `cohort_milestones.csv`의 M1/M3/M6/M12가 blank였다.
- `cohort_analysis.json`에 churn sequence/pre-churn/journey funnel error가 기록되어 있었다.
- 다만 fresh in-memory recomputation은 현재 코드로 valid cohort outputs를 만들 수 있어, 핵심 문제는 stale/invalid artifact 관리와 fail-fast 부족이다.

필요 조치:
- `run_cohort` 시작 시 기존 cohort/journey outputs를 정리하거나 run-id를 붙여 stale 파일이 남지 않게 한다.
- required cohort/journey outputs 생성 실패 시 mode/stage를 성공 처리하지 말고 실패시킨다.
- checklist가 `cohort_analysis.json`의 `*_error` 필드, retention period 수, milestone null 여부, churn sequence top-5를 검증해야 한다.
- checklist와 required artifacts 사이 freshness/run-id를 검증해야 한다.

## Blocker 3: artifact checklist와 mirror 신뢰성 부족

증거:
- `required_artifacts_checklist.json`은 25/25로 기록되어도 이후 invalid cohort files와 맞지 않을 수 있다.
- `data/artifacts/recommendations.csv`는 10 synthetic rows인데 `results/recommendations.csv`는 5,000 rows였다.
- `data/artifacts/budget_optimization.csv`는 1 synthetic row인데 `results/budget_optimization.csv`는 5,000 rows였다.

필요 조치:
- pipeline 종료 시 `results/`에서 `data/artifacts/`로 강제 동기화한다.
- checklist는 results/artifacts mirror hash 또는 row/schema equality를 검증한다.
- dashboard loader는 required artifact가 stale/invalid이면 sample fallback으로 숨기지 말고 visible warning/fail state를 반환해야 한다.
- 테스트가 real `data/artifacts/`를 오염시키지 않게 temp dirs를 쓰도록 보장한다.

## Blocker 4: dashboard evidence gaps

증거:
- `results/churn_predictions.csv`가 834 rows로, `data/raw/customers.csv` 5,000 rows 전체를 커버하지 않는다.
- model monitoring page는 `load_mlflow_runs()`를 쓰며, `model_performance_history.csv` 기반 history가 페이지에 직접 쓰이지 않는다.
- sample fallback path가 여전히 required dashboard evidence를 가릴 수 있다.

필요 조치:
- `run_train`이 전체 고객용 churn prediction artifact를 저장하고, test-only metrics와 split은 별도 column으로 구분해야 한다.
- monitoring view가 `load_auc_history()`, `load_precision_history()`, `load_recall_history()`를 직접 쓰거나 `load_mlflow_runs()`가 real performance history를 adapter로 반환해야 한다.
- required dashboard loader들은 missing/invalid artifact일 때 generated sample이 아니라 explicit empty/error state를 반환해야 한다.

## Blocker 5: segmentation high-value actionable evidence 부족

증거:
- `segments_6plus.csv`는 6 segments와 `priority_score == uplift_score * clv`는 만족한다.
- 그러나 high-value + high-risk + positive-uplift 고객이 0건이고, `high_value_persuadable` 또는 `high_value_lost_cause` evidence가 없다.

필요 조치:
- segmentation artifact에 high-value at-risk population absence를 명시적으로 기록하거나, threshold/calibration을 통해 high-value actionable segments가 생성되도록 해야 한다.
- segment validation에서 high-value persuadable/lost-cause evidence 또는 absence report를 요구한다.

## Blocker 6: A/B detailed evidence 부족

증거:
- `ab_test_results.json`은 p-value, confidence interval, required sample size를 포함한다.
- `ab_test_detailed.json`의 observed power가 0.6909, 0.7297로 문서상 0.80 target보다 낮다.
- detailed experiment rows가 required sample size를 포함하지 않는다.

필요 조치:
- `ab_test_detailed.json`에 per-experiment `required_sample_size_per_group`, `required_total_sample_size`, `observed_power`, `design_power`, `is_underpowered`를 포함한다.
- underpowered면 문서/대시보드가 이를 명시해야 한다. 통과처럼 표시하면 안 된다.

## Blocker 7: ML/SHAP/Uplift/CLV evidence gaps

증거:
- SHAP global artifacts는 있지만 local interpretation artifact가 없다.
- CLV validation은 `monetary_12m_proxy` target으로, feature `monetary`와 순환적이다.
- uplift comparison은 `s_learner`가 더 좋은데 output은 CLI default `t_learner`를 사용한다.

필요 조치:
- `run_train`에서 `shap_local_explanations.csv`와 대표 high-risk local explanation artifact를 저장한다.
- CLV는 observation window features와 future window revenue label을 분리하거나, proxy validation 한계를 명확히 하고 actual-vs-predicted 요구를 더 직접 충족한다.
- uplift learner default를 `auto`로 바꾸거나, best AUUC learner를 기본 선택한다.

## Blocker 8: repository hygiene

증거:
- `main`은 origin/main보다 2 commits ahead였다.
- tracked working diff는 clean이지만 `.audit/`이 untracked로 남아 있다.
- tracked generated residue로 `models/ml_churn_model.pkl.joblib`와 tracked `src/data/__pycache__/*.pyc`가 남아 있다.

필요 조치:
- `.audit/`을 ignore하거나 제거한다.
- tracked generated model/pycache는 후속 cleanup commit에서 index에서 제거하는 방안을 검토한다.
- push 전 `git status --short`가 clean 또는 의도된 ignored-only 상태여야 한다.

## 다음 사이클 계획

다음 구현 사이클은 이 문서를 기준으로 진행한다.

1. stale artifact 관리와 cohort fail-fast/checklist 강화.
2. dashboard all-customer churn predictions와 monitoring history wiring.
3. A/B detailed sample-size/power schema 보강.
4. SHAP local artifact, uplift auto learner selection, CLV validation 보강.
5. segmentation high-value actionable evidence 또는 absence report 구현.
6. artifact regeneration, focused tests, commit, push.
7. 새 6-agent 검증 결과는 `issue_final_v4.md`에 기록한다.

