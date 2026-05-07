# issue_final_v6.md

## 목적

이번 문서는 `issue_final_v5.md`를 기준으로 시작한 v6 require-first 반복 사이클의 작업, 검증, 최종 판정을 기록한다.

시작 조건:
- 기준 문서: `issue_final_v5.md`
- 운영 원칙: 메인 컨텍스트는 오케스트레이션만 담당하고, 구현 수정은 fresh-context executor에게 위임한다.
- 사이클 목표: `issue_final_v5.md`가 PASS 상태이므로, 다시 `require.md` 전체를 6개 fresh-context executor로 분해해 남은 gap이 있는지 확인하고 닫는다.

## require-first 작업 분해

| Lane | 범위 | 담당 agent | 판정 |
| --- | --- | --- | --- |
| 1 | 시뮬레이터, 코호트/여정, 피처 엔지니어링 | executor | PASS |
| 2 | ML/DL 이탈 모델, SHAP, Scoring API, MLflow | executor | PASS |
| 3 | Uplift, CLV, Survival consistency | executor | PASS/no changes |
| 4 | 6세그먼트, 예산 최적화, 추천, A/B 테스트 | executor | PASS/no changes |
| 5 | 대시보드, 모니터링, streaming, Docker/runtime | executor | PASS |
| 6 | CLI/pipeline, 문서, traceability, hygiene | executor | PASS |

## 닫은 gap

1. Persona purchase cadence gap
   - `config/simulator_config.yaml`에는 `purchase_frequency_monthly`가 정의되어 있었지만 purchase simulation에서 실질적으로 사용되지 않아 persona별 구매 주기가 충분히 모델링되지 않았다.
   - `src/data/generator.py`가 persona monthly purchase frequency를 daily purchase propensity에 반영하도록 보강했고, `tests/test_data_generator.py`에 높은 구매 빈도 설정이 더 많은 구매를 생성한다는 회귀 테스트를 추가했다.

2. DL sequence padding / scaling gap
   - real sequence padding row가 zero padding으로 만들어진 뒤 standardization을 거치며 non-zero 값으로 바뀔 수 있었다.
   - `src/models/sequence_utils.py`가 scaling 전 all-zero padding row mask를 보존하고 scaling 후 다시 0으로 복원하도록 수정했고, `tests/test_sequence_dataset.py`에 padding row 유지 테스트를 추가했다.

3. Docker Redis runtime config gap
   - dashboard Redis streaming health와 realtime scoring status가 YAML Redis host/port만 사용해 Docker Compose의 `REDIS_HOST=redis` 환경을 반영하지 못할 수 있었다.
   - `src/dashboard/system_health_view.py`에 `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB` env override를 반영하는 shared resolver를 추가했고, `src/dashboard/app.py`와 system health path가 같은 resolver를 사용하도록 정리했다.

4. Traceability stale evidence gap
   - `docs/requirement_traceability.md`가 v5 이후에도 v4/v2 evidence와 이전 `1339 passed` count를 일부 참조했다.
   - 현재 기준을 `issue_final_v5.md`와 v5 final evidence로 갱신했다.

## Fresh-context verifier 결과

최종 verifier 6개는 모두 PASS다.

| 관점 | 판정 | 핵심 evidence |
| --- | --- | --- |
| Simulator / cohort / journey / feature engineering | PASS | `260 passed`, purchase cadence probe에서 monthly frequency 0.5 대비 8.0 설정이 구매량과 구매 간격에 명확히 반영됨, full-mode 20,000명/treatment-control 10,000/10,000/churn 0.19995 유지. |
| ML / DL / SHAP / scoring / MLflow | PASS | `235 passed, 1 skipped`, sequence scaling 후 padding row가 0으로 보존되고 ML/DL/SHAP/scoring/MLflow regression suite 통과. |
| Uplift / CLV / survival | PASS | `148 passed`, v6 diff가 해당 lane 파일을 건드리지 않았고 uplift/CLV/survival 계약 유지. |
| Segmentation / budget / recommendations / A/B | PASS | `589 passed`, v6 diff가 해당 lane 파일을 건드리지 않았고 6세그먼트/예산/A-B 계약 유지. |
| Dashboard / monitoring / Docker runtime | PASS | `1030 passed`, `docker compose config --quiet` passed, Redis env override resolver가 dashboard scoring/system health에 공통 적용됨. |
| CLI / pipeline / docs / traceability / hygiene | PASS | `253 passed`, stale v2/v4/1339 traceability reference 없음, checklist `29/29`, no tracked generated residue. |

## 메인 컨텍스트 최종 검증

- `uv run --python /opt/homebrew/bin/python3.12 --with-requirements requirements.txt pytest -q`
  - 결과: `2561 passed, 1 skipped, 33 warnings`.
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

`issue_final_v5.md` 이후 require-first v6 사이클에서 발견된 simulator purchase cadence, sequence padding, Docker Redis config, traceability stale evidence gap은 모두 executor에게 위임되어 닫혔다. Fresh-context verifier 6개가 모두 PASS했고, 메인 컨텍스트의 전체 pytest 및 full pipeline smoke도 통과했다.

다음 사이클은 사용자 지침대로 `issue_final_v6.md`를 기준으로 시작한다. v6에 더 이상 개선할 blocker가 없다고 판정되면, 다시 `require.md` 전체를 6개 fresh-context agent로 분해해 잔여 gap을 찾는 require-first 루프를 반복한다.
