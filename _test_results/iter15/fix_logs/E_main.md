# Agent E — main() shell + helpers i18n fix log

Iteration: iter15
Date: 2026-05-12
Scope: `src/dashboard/app.py` `main()` body + 2 module-level helpers, plus passthrough key registration in `src/dashboard/utils/dashboard_helpers.py`.

## Changes

### 1. `_show_loader_issue` (app.py ~L287)
Added a defensive `tr` import block at the top of the function. The `fallback` argument is now wrapped via `_tr(fallback)` so callers that pass the default English string get the localized version when `lang == "ko"`. A loader-supplied `issue` string is left untouched because those messages are dynamic and outside the i18n shell scope.

### 2. `_show_prediction_coverage` (app.py ~L301)
Same defensive `tr` block added. The `message` from `data_loader.get_prediction_coverage()` is now passed through `_tr()` (only when it is a `str`) before being handed to `st.success` / `st.warning`. Coverage messages are stable English templates from the loader, so the toggle works as soon as the main session adds Korean values for them.

### 3. Synthetic-data banner (app.py ~L5988)
Rewrote both branches of the `gen_mode == "small" or not group_passed` conditional. Static label fragments (`Synthetic data`, `mode`, `Group-size validation`, the long disclaimer, the PASSED/FAILED enum, the UNKNOWN sentinel) are routed through `_tr(..., _lang)`. The `n_customers` value and `gen_mode.upper()` interpolation are left raw so the count and mode token stay in their native form.

### 4. Sidebar info bullets (app.py ~L6017)
The `Churn Definition` / `Budget` bullet rows previously embedded English labels (`No purchase`, `No login`, `Operator`, `days`, `Total`) inline in an f-string. Each label is now wrapped via `_tr(...)` while numeric / formatted values (days count, currency string, operator value) flow through unchanged.

### 5. `dashboard_helpers.I18N_KO` (~L84)
Registered 14 new keys (`Synthetic data`, `UNKNOWN`, `unknown`, `mode`, `PASSED`, `FAILED`, the disclaimer sentence, `Group-size validation`, `All KPIs are simulator-generated.`, `No purchase`, `No login`, `Operator`, `days`, `Total`) as English passthrough so the toggle is a no-op for them until the main session consolidates Korean translations.

## Out-of-scope items intentionally left alone

- `_model_stamp_caption` was listed in the brief at line ~100-194 but in the actual file it is a nested closure inside `_render_predict_a_customer_tab` (L4394) and two sibling render functions (L4715, L5067). Those parent functions are owned by agents B/C; touching the closure means editing inside a render function. No edit performed.
- The pre-existing language toggle radio (`🌐 Language / 언어`), `Navigation` sidebar title, `Select Page` legend, page-name labels, section headers (`Churn Definition`, `Budget`, `Ensemble Weights`), and `🔄 Refresh Data` button were wrapped in a prior iteration. Left untouched.
- The `f"⚠️ ..."` warning text token (`PASSED` / `FAILED`) is now routed through `_tr` via a `_validation_label` local because using `_tr` inside an inline `{...}` expression of an f-string keeps the line readable. No other f-string interpolations were translated.

## Verification

- `python -c "import ast; ast.parse(open('src/dashboard/app.py', encoding='utf-8').read())"` — OK.
- `python -c "import ast; ast.parse(open('src/dashboard/utils/dashboard_helpers.py', encoding='utf-8').read())"` — OK.
- `from src.dashboard.utils.dashboard_helpers import tr; tr('Synthetic data','ko')` returns `"Synthetic data"` (passthrough as designed). `len(I18N_KO) == 44` (was 30 before, +14 new keys).
- No tests run in this fix; AST-parse is the gate per spec.
