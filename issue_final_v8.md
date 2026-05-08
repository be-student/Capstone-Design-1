# issue_final_v8.md

## 목적

이번 문서는 `issue_final_v7.md`에서 발견된 blocker를 닫고, `require.md` 전체를 다시 12개 조각으로 분해해 검증한 v8 반복 사이클의 최종 결과를 기록한다.

운영 원칙:
- 메인 컨텍스트는 오케스트레이션만 수행했다.
- 구현은 fresh-context worker에게 non-overlapping write scope로 위임했다.
- 검증은 fresh-context verifier에게 read-only로 위임했다.
- 최종 PASS는 모든 요구사항 조각이 같은 사이클에서 PASS일 때만 인정했다.

## 구현으로 닫은 blocker

1. 시뮬레이터 산출물 스키마
   - `data/raw/events.csv`와 `events.parquet`에 `session_duration`, `marketing_channel`, `marketing_response` evidence가 포함되도록 generator/orchestrator를 보강했다.
   - `src/data/make_dataset.py`를 추가해 데이터 생성부터 processed/feature-store까지 재현 가능한 진입점을 제공했다.

2. 코호트/여정 산출물
   - `results/cohort_churn_rate_differences.png`를 추가해 코호트별 이탈률 차이 시각화를 명확히 남겼다.
   - `results/journey_funnel.csv`에 stage timing, tenure, dropoff timing evidence를 추가했다.
   - `cohort_churn_rate_differences.png`와 required checklist가 `data/artifacts` mirror에 동기화되도록 보강했다.

3. A/B 테스트 교란 통제
   - random assignment 기반 교란 통제 rationale, covariate/persona balance-check, SMD threshold를 구현/문서화했다.
   - `results/ab_test_balance_check.csv/json`에 persisted evidence를 남겼다.

4. 모델 모니터링 성능 저하 alert
   - AUC/Precision/Recall 시간별 성능 저하를 threshold와 비교해 `performance_alerts`와 `performance_degradation`을 생성한다.
   - dashboard loader/view는 새 report field와 fallback을 처리한다.

5. GitHub Flow 문서화
   - `docs/development_workflow.md`를 추가하고 README에서 링크했다.
   - main 보호, feature branch, pull request review, CI, merge, release/tag 정책을 명시했다.

6. 모델 버전/MLflow evidence/경로 hygiene
   - fixed model artifact와 versioned artifact를 함께 유지하고 `models/model_artifacts_manifest.json`을 추가했다.
   - MLflow run evidence를 남기되 tracked evidence에 로컬 절대경로가 남지 않도록 scrub했다.

7. Dashboard logic separation
   - dashboard 순수 계산 helper를 `src/dashboard/calculations.py`로 분리했다.
   - app.py helper 직접 import test를 calculations module import로 전환했다.

8. Required artifact checklist freshness
   - checklist mirror helper를 보강해 `results/required_artifacts_checklist.json`와 `data/artifacts/required_artifacts_checklist.json`가 같은 bytes로 저장된다.
   - 최종 checklist는 `30/30`, `missing=[]`, hash mismatch 0개다.

## 12개 검증 조각 최종 판정

| 조각 | 범위 | 최종 판정 | 핵심 evidence |
| --- | --- | --- | --- |
| 1 | 최종 실행/산출물 계약 | PASS | `docker compose config --quiet`, CLI help, `320 passed`, checklist `30/30`, hash mismatch 0 |
| 2 | 고객 행동 시뮬레이터 | PASS | 20,000명, treatment/control 10,000/10,000, churn 0.19995, raw events schema 및 response 분포 확인 |
| 3 | 코호트 및 고객 여정 | PASS | M1/M3/M6/M12, churn-rate-difference PNG, mirror 동일, last30/pre-churn/journey timing evidence |
| 4 | 피처 엔지니어링 | PASS | 33 feature columns, RFM/behavior/session/sequence/time/journey features, null/inf 0, feature store 존재 |
| 5 | ML 이탈 모델 | PASS | LightGBM/XGBoost, class weight, 5-fold CV, AUC 1.0, SHAP global/local, top10, threshold analysis |
| 6 | DL 이탈 모델 | PASS | PyTorch LSTM/Transformer, padding/embedding, early stopping, ML/DL/ensemble 비교, CPU path, versioned DL artifact |
| 7 | Uplift Modeling | PASS | T/S Learner, CATE/uplift score 20,000명, 4분면, Qini curve, Persuadables analysis |
| 8 | CLV 예측 | PASS | ML CLV, 12개월 actual-vs-predicted, high-value top 20%, top-N/distribution reports |
| 9 | 고객 세그먼테이션 | PASS | 6 segments, segment summary, `priority_score = uplift_score * clv`, visualization |
| 10 | 전략/예산/A-B | PASS | LP budget optimization, what-if 50/100/200, ROI, p-value/CI/power, balance-check SMD evidence |
| 11 | 대시보드/모니터링 | PASS | Streamlit 8501, required views, manual refresh, PSI/KS, performance degradation alerts |
| 12 | 제약/문서/품질/보너스 | PASS | GitHub Flow docs, make_dataset, model versioning, MLflow evidence, no tracked local absolute paths, dashboard logic split |

## 대표 검증 명령

- `docker compose config --quiet`
- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt python src/main.py --help`
- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest -p no:cacheprovider tests/test_cli_entrypoint.py tests/test_main_cli.py tests/test_pipeline_runner.py tests/test_docker_setup.py` -> `320 passed`
- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest tests/test_pipeline_runner.py -k "artifact or checklist or cohort or generation_summary"` -> `10 passed`
- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt python -m pytest tests/test_dashboard.py tests/test_dashboard_helpers.py tests/test_streamlit_dashboard.py tests/test_model_monitoring_view.py` -> `380 passed`
- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt python -m pytest tests/test_churn_model.py tests/test_mlflow_tracking.py` -> `100 passed`
- `cmp results/required_artifacts_checklist.json data/artifacts/required_artifacts_checklist.json`
- `cmp results/cohort_churn_rate_differences.png data/artifacts/cohort_churn_rate_differences.png`

## 최종 판정

최종 판정: PASS.

`require.md` 전체 요구사항은 v8 기준으로 모두 구현되어 있고, 12개 fresh-context verifier 조각이 모두 PASS했다. 남은 blocker는 없다.
