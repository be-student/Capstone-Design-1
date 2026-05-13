# F1 — Cross-cutting helpers

## Reasoning (어떤 사고로)

iter9 감사가 발견한 다음 결함들이 모두 단일 책임 helper의 부재 때문이었음. 각
페이지가 동일한 KPI를 직접 포맷하거나 직접 계산하면서 정의가 흩어졌고, 그 결과
같은 캠페인의 같은 지표가 페이지마다 다른 숫자로 노출됨:

- **a5 / Page 12**: `Customers Retained = 122.29548658078494` — 14자리 IEEE-754
  부동소수점이 KPI 카드에 그대로 새어 나옴. 정수 포맷 헬퍼와 입력 정화
  (NaN/None) 헬퍼가 없었기 때문. → 단일 진입점인 `format_count(integer=True)` /
  `customers_retained_int()` 두 곳 모두에서 차단.
- **a5 / Pages 05·09·12**: "Overall ROI"가 동일 캠페인에 대해 3.5x / 9.0x / 3.8x로
  서로 모순. 각 페이지가 분모를 다르게 잡고(예산 envelope vs 실제 지출 vs
  세그먼트 평균) 푸트노트도 없음. → `compute_overall_roi(scope_label=...)`로
  scope를 강제 선언하게 만들고, calculations.py에서는
  `roi_budget_envelope` / `roi_treated_only` / `roi_segment_average`로 계산
  계층까지 분리. `tooltip` 키에 실제 분자 ÷ 분모를 노출해 카드 옆에 footnote를
  걸 수 있게 함.
- **a4 / Pages 08·13c·14**: "Trend over time" 차트의 x축이 1.5ms 또는 0.1ms —
  단일 관측치를 시계열로 그림. → `drift_trend_guard(timeseries, min_points=5)`가
  관측 수와 시간 폭(<1시간)을 동시에 검사해 호출자에게 `(False, message)`를
  반환. 호출자는 차트 대신 `st.info(message)`를 그리도록 분기. 숫자 시퀀스가
  pandas에 의해 nanosecond epoch으로 오인되지 않도록 timestamp-like 입력일
  때만 시간 폭 검사를 수행.
- **a5 / a1 / Page 10**: `57,936,514,970 KRW`가 카드 폭을 넘어 `...`로 잘림. →
  `format_currency_krw()`가 `₩57.94B` / `₩192.2M` / `₩1.2K` 형태로 자릿수 압축.
  None / NaN / inf는 모두 `—`로 안전하게 떨어뜨림.

## Changes (무엇을)

### `src/dashboard/utils/dashboard_helpers.py`
- **추가**: `_is_missing(x)` — None / NaN / ±inf를 한 곳에서 판정하는 내부
  헬퍼. 모든 포맷터가 동일한 결손값 규칙을 공유.
- **확장**: `format_count(value, integer=True, suffix="")` — 기존 `(value: int)`
  단일 인자 시그니처를 유지(legacy 호출 26개 테스트 + app.py의 9개 호출 모두
  통과)하면서 `integer=False` (소수 1자리 고정점), `suffix` (단위 라벨),
  None/NaN 안전 분기를 추가. 122.295... 14자리 누출 차단.
- **확장**: `format_currency(value, currency="KRW")` — 기존 동작 유지, 결손값
  처리만 추가.
- **추가**: `format_currency_krw(x)` — B/M/K 자릿수 압축 + `₩` 기호. Page 10의
  ellipsis-truncation 차단.
- **확장**: `format_percentage(value, decimals=2)` — 결손값 처리 추가.
- **추가**: `compute_overall_roi(revenue_saved, cost_or_budget, scope_label)`
  — `{value, display, label, tooltip}` 사전을 반환해 KPI 카드와 footnote를
  한 번에 채울 수 있게 함. scope_label은 `"budget"` / `"treated"` /
  `"segment_avg"` 중 하나여야 하므로 호출자가 정의를 선언하지 않을 수 없음.
- **추가**: `drift_trend_guard(timeseries, min_points=5)` — `(ok, message)`
  튜플 반환. 관측 수 < min_points이거나 timestamp-like 입력의 시간 폭이
  3600초 미만이면 거부. pandas Series / DatetimeIndex / list[datetime] /
  list[str-iso] 모두를 인식하되 raw int 시퀀스는 timestamp로 오인하지 않음.

### `src/dashboard/calculations.py`
- **추가**: `roi_budget_envelope(revenue_saved, total_budget)` — Page 12용.
- **추가**: `roi_treated_only(revenue_saved, cost_spent)` — Page 09용.
- **추가**: `roi_segment_average(segment_rois)` — 0.0/`NaN` 세그먼트(`sleeping_dog`,
  `high_value_lost_cause` 등 의도적으로 비치료된 세그먼트)를 평균에서 제외.
- **추가**: `customers_retained_int(value)` — 14자리 부동소수점이 포맷 계층을
  우회해도 계산 계층에서 한 번 더 정수화하는 안전망.

## Closes (which iter9 issues)

- a5 P12 14-decimal float leak (`Customers Retained = 122.29548658078494`)
  → `format_count(integer=True)` + `customers_retained_int()`.
- a5 ROI 3-definition trap (Pages 05/09/12: 3.5x / 9.0x / 3.8x)
  → `compute_overall_roi(scope_label=...)` + `roi_budget_envelope` /
  `roi_treated_only` / `roi_segment_average`. 호출자가 scope를 명시하지
  않으면 카드 라벨이 분기되지 않음.
- a4 single-point drift "trend" (Pages 08/13c/14, 1.5ms x-axis)
  → `drift_trend_guard()` (n<5 또는 시간 폭 <1시간이면 차트 대신 info 배너).
- a5 / a1 currency truncation (Page 10 `57,936,514,970 ...`)
  → `format_currency_krw()` (B/M/K 자릿수 압축).

## Verification

- `python -m pytest tests/test_dashboard_helpers.py` — 26 passed (legacy
  call-sites preserved).
- 추가 인라인 어서션(파일 외부): format_count / format_currency_krw /
  compute_overall_roi의 4가지 스코프 / drift_trend_guard의 6가지 입력
  케이스(빈 리스트, 단일 datetime, ms-span Series, 시간 단위 Series, raw
  int 시퀀스, ISO 문자열 sub-hour) / ROI 계산 함수 4종 / customers_retained_int
  — 전부 통과.

## Out of scope (다른 에이전트가 호출함)

- app.py / monitoring_view.py / system_health_view.py / recommendations_view.py
  의 호출 사이트 변경은 F2~F5가 담당. F1은 헬퍼 표면만 제공.
- `utils/__init__.py`는 수정 금지 파일에 포함되어 있지 않지만 다른 에이전트의
  작업 충돌을 피하기 위해 건드리지 않음. 신규 헬퍼는 직접 경로
  `from src.dashboard.utils.dashboard_helpers import ...` 또는
  `from src.dashboard.calculations import ...`로 임포트 가능.
