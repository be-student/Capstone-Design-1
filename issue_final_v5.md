# issue_final_v5.md

## 목적

이번 문서는 `issue_final_v4.md`에서 v3 blocker가 모두 닫힌 뒤, 사용자 지침에 따라 `require.md` 전체를 6개 fresh-context agent로 다시 분해 분석한 require-first v5 사이클의 작업, 검증, 최종 판정을 기록한다.

시작 조건:
- 기준 문서: `issue_final_v4.md`
- 운영 원칙: 메인 컨텍스트는 오케스트레이션만 담당하고, 구현 수정은 fresh-context executor에게 위임한다.
- 사이클 목표: `issue_final_v4.md` 이후 남은 issue 문서 blocker가 없으므로, `require.md` 전체 요구사항을 기준으로 잔여 gap을 찾고 닫는다.

## require-first 작업 분해

| Lane | 범위 | 담당 agent | 판정 |
| --- | --- | --- | --- |
| 1 | 시뮬레이터, 코호트/여정, 피처 엔지니어링 | executor | PASS |
| 2 | ML/DL 이탈 모델, SHAP, Scoring API | executor | PASS |
| 3 | Uplift, CLV, Survival consistency | executor | PASS |
| 4 | 6세그먼트, 예산 최적화, 추천, A/B 테스트 | executor | PASS |
| 5 | 대시보드, 모니터링, Docker/runtime | executor | PASS |
| 6 | CLI/pipeline, 문서, traceability, hygiene | executor | PASS |
| 7 | MLflow tracking native crash 보정 | executor | PASS |
| 8 | verifier가 찾은 docs consistency blocker 보정 | executor | PASS |

## 닫은 gap

1. Simulator-to-feature evidence gap
   - 시뮬레이터 설정에는 `avg_session_minutes`와 `session_time_decay`가 있었지만 raw event에 `session_duration`이 없어 세션 시간 단축을 산출물로 증명하기 어려웠다.
   - `src/data/generator.py`가 방문 이벤트에 session duration evidence를 남기도록 보강했고, `src/features/feature_engineering.py`의 numeric missing/inf/outlier sanitation을 강화했다.

2. ML class imbalance / scoring semantics gap
   - LightGBM/XGBoost CV/tuning 경로에서 class imbalance weighting이 최종 학습 경로만큼 일관적이지 않았다.
   - `ScoringAPI`가 sklearn-style 2-column `predict_proba`에서 churn positive-class가 아닌 column 0을 사용할 수 있었다.
   - CV/tuning weighting과 scoring positive-class probability를 수정했다.

3. Uplift direction / Persuadables analysis gap
   - positive uplift가 treatment로 churn을 줄인다는 모델 계약과 AUUC/Qini sign 계산이 일관되지 않았다.
   - Persuadables targeting 기준을 모델 레벨에서 분석하는 helper가 부족했다.
   - AUUC/Qini churn-reduction 방향을 정리하고 Persuadables analysis helper를 추가했다.

4. A/B significance schema gap
   - `statistically_significant`가 p-value significance와 beneficial rollout gate를 혼동할 수 있었다.
   - `statistically_significant`는 p-value 기준, `is_significant`는 power와 beneficial direction까지 포함하는 rollout gate로 분리했다.

5. Dashboard required evidence fallback gap
   - cohort retention과 detailed A/B dashboard path에 required artifact가 없을 때 sample fallback 또는 기본 KPI shell이 보일 수 있었다.
   - required artifact missing/invalid 상태를 visible empty/error state로 노출하도록 수정했다.

6. Documentation / traceability drift
   - README, deployment, model docs, requirement traceability 문서에 stale evidence와 `--small` full-readiness ambiguity가 남아 있었다.
   - `docs/requirement_traceability.md` markdown table 구조 오류, `SMALL=true` 표현, `pipeline_state.json` 경로 불일치를 최종 verifier FAIL 이후 별도 docs-only executor가 닫았다.

7. MLflow native crash test stability
   - `tests/test_mlflow_tracking.py::test_ml_model_fit_with_tracker`가 native LightGBM/XGBoost path를 직접 실행해 CPU 환경에서 Python segfault를 낼 수 있었다.
   - test-scoped deterministic sklearn tree-model doubles로 MLflow metric logging과 CV 의미는 유지하면서 native crash surface를 제거했다.

## Fresh-context verifier rerun 결과

최종 verifier rerun 6개는 모두 PASS다.

| 관점 | 판정 | 핵심 evidence |
| --- | --- | --- |
| Simulator / cohort / journey / feature engineering | PASS | `259 passed`, full mode 20,000명, treatment/control 10,000/10,000, cohort M1/M3/M6/M12 exact milestones, feature store 20,000 x 38, 30+ feature docs. |
| ML / DL / SHAP / scoring / MLflow | PASS | `234 passed, 1 skipped`, class imbalance CV/final path, positive-class scoring, MLflow crash-safe tracker test, SHAP and ML/DL artifacts verified. |
| Uplift / CLV / survival | PASS | `148 passed`, churn-reduction AUUC/Qini direction, Persuadables helper, CLV actual-vs-predicted holdout, survival `.pkl` persistence fixed. |
| Segmentation / budget / recommendations / A/B | PASS | `589 passed`, 6 segments, priority score equality, 50/100/200 what-if, ROI, A/B p-value/power/schema verified. |
| Dashboard / monitoring / Docker runtime | PASS | `1029 passed`, `docker compose config --quiet` passed, no cohort/A-B sample fallback, real monitoring/performance artifacts loaded. |
| CLI / pipeline / docs / traceability / hygiene | PASS | `253 passed`, docs blockers fixed, checklist `29/29`, `full_submission_ready: true`, no tracked generated residue. |

## 메인 컨텍스트 최종 검증

- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest -q`
  - 결과: `2558 passed, 1 skipped, 66 warnings`.
- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt python src/main.py --mode all --quiet`
  - 결과: success.
- `docker compose config --quiet`
  - 결과: passed.
- `git diff --check`
  - 결과: passed.
- `results/required_artifacts_checklist.json`
  - 결과: `29 / 29`, `missing: []`, `full_submission_ready: true`, mirror hash valid.
- `data/raw/generation_summary.json`
  - 결과: full mode, 20,000 customers, treatment/control 10,000/10,000, churn rate 0.19995.

## 종합 판정

최종 판정: PASS.

`issue_final_v4.md` 이후 require-first v5 사이클에서 발견된 구현, 테스트 안정성, 대시보드 evidence, 문서/traceability gap은 모두 executor에게 위임되어 닫혔다. 최종 fresh-context verifier 6개가 모두 PASS했고, 메인 컨텍스트의 전체 pytest 및 full pipeline smoke도 통과했다.

다음 사이클은 사용자 지침대로 `issue_final_v5.md`를 기준으로 시작한다. v5에 더 이상 개선할 blocker가 없다고 판정되면, 다시 `require.md` 전체를 6개 fresh-context agent로 분해해 잔여 gap을 찾는 require-first 루프를 반복한다.
