# G2 fix log — src/dashboard/data_loader.py (iter13)

Date: 2026-05-12
Scope: removed every `_generate_sample_*` synthetic fallback path from
`DashboardDataLoader`. Real artifact missing → empty frame / empty dict
+ `is_real_artifact(name) == False` + a dashboard-visible issue. Adds
opt-in `DashboardArtifact` wrapper via `as_artifact=True`.

File: `src/dashboard/data_loader.py` (1912 -> 1790 lines)

## New top-level contract

Added a `DashboardArtifact` dataclass at module top (after the imports):

```python
@dataclass
class DashboardArtifact:
    data: Any
    is_real: bool
    source_path: Optional[str] = None
    reason: Optional[str] = None
    name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    def __bool__(self): return self.is_real
```

New instance helpers on `DashboardDataLoader`:

| Helper | Purpose |
|---|---|
| `self._is_real: Dict[str, bool]` | Tracks the most recent load_* outcome per artifact name. |
| `is_real_artifact(name) -> bool` | Public view-facing query. |
| `_mark_real(name)` | Set is_real=True + clear issue. |
| `_mark_missing(name, msg)` | Set is_real=False + record dashboard-visible issue. |
| `_missing_artifact_error(name)` | Builds the canonical `FileNotFoundError("X missing — run python -m src.main --mode all to produce real artifact. Dashboard will NOT render synthetic data.")`. |

`_required_csv` and `_required_json` now also update `self._is_real`
(True on success, False on every failure branch). All existing loaders
that route through these helpers (`load_predictions`,
`load_model_metrics`, `load_budget_results`,
`load_recommendations`, `load_uplift_results`, `load_clv_data`,
`load_feature_importance`, `load_model_performance_history`,
`load_ab_test_results`) automatically expose the new is_real signal
without any further change.

## load_* signature changes (all backward-compatible)

For each rewritten loader, the public signature gained one optional
keyword argument `as_artifact: bool = False`. With `as_artifact=False`
(default) the return type is exactly the same as before (DataFrame or
Dict), so existing callers in `app.py` and view modules keep working.

| Method | Old return | New return (default) | New `as_artifact=True` return |
|---|---|---|---|
| `load_survival_data` | `pd.DataFrame` (sample) | `pd.DataFrame` (segments-derived OR empty) | `DashboardArtifact` |
| `load_roc_data` | `Dict` (sample) | `Dict` (real OR empty `{}`) | `DashboardArtifact` |
| `load_confusion_matrices` | `Dict` (hardcoded fixture) | `Dict` (real OR empty `{}`) | `DashboardArtifact` |
| `load_survival_curves` | `Dict` (sample) | `Dict` (real OR empty `{}`) | `DashboardArtifact` |
| `load_scoring_history` | `pd.DataFrame` (n=200 sample) | `pd.DataFrame` (real OR empty) | `DashboardArtifact` |
| `load_scoring_throughput` | `pd.DataFrame` (sinusoidal sample) | `pd.DataFrame` (real OR empty) | `DashboardArtifact` |
| `load_retention_offers` | `pd.DataFrame` (n=50 sample) | `pd.DataFrame` (real OR empty) | `DashboardArtifact` |

Each loader now records the dashboard-visible reason via
`_mark_missing(...)` so a view can:

```python
df = loader.load_scoring_history()
if not loader.is_real_artifact("scoring_history"):
    st.error(loader.get_artifact_issue("scoring_history"))
else:
    render_chart(df)
```

…or use the wrapper form for explicit, typed branching:

```python
art = loader.load_scoring_history(as_artifact=True)
if art.is_real:
    render_chart(art.data)
else:
    st.error(art.reason)
```

## _generate_sample_* removals (20 methods)

Every method body was replaced with a single-line raise:

```python
def _generate_sample_<name>(self, ...):
    raise self._missing_artifact_error("<canonical_artifact_filename>")
```

| Method | Was producing | Now raises |
|---|---|---|
| `_generate_sample_predictions` | 500-row np.random.beta churn | `FileNotFoundError("churn_predictions.csv missing — run …")` |
| `_generate_sample_metrics` | hardcoded auc/precision/etc | `… ("model_metrics.json …")` |
| `_generate_sample_ab_results` | hardcoded retention-coupon test | `… ("ab_test_results.json …")` |
| `_generate_sample_budget_results` | 6-segment hardcoded budget | `… ("budget_results.csv …")` |
| `_generate_sample_survival_data` | 300-row np.random survival | `… ("survival_data.csv …")` |
| `_generate_sample_recommendations` | 50-row np.random recs | `… ("recommendations.csv …")` |
| `_generate_sample_uplift_results` | 200-row np.random uplift | `… ("uplift_results.csv …")` |
| `_generate_sample_clv_data` | 500-row np.random CLV | `… ("clv_data.csv …")` |
| `_generate_sample_cohort_data` | 200-customer np.random cohort | `… ("cohort_data.csv …")` |
| `_generate_sample_retention_matrix` | 6 cohort × 7 period matrix | `… ("cohort_retention_matrix.csv …")` |
| `_generate_sample_mlflow_runs` | 5-row np.random MLflow runs | `… ("model_performance_history.csv …")` |
| `_generate_sample_roc_data` | 100-point Beta-fake ROC | `… ("roc_data.json …")` |
| `_generate_sample_confusion_matrices` | `[[350,50],[80,120]]` fixture | `… ("confusion_matrices.json …")` |
| `_generate_sample_ab_detailed` | 3-experiment hardcoded fixture | `… ("ab_test_detailed.json …")` |
| `_generate_sample_survival_curves` | 6-segment np.random KM curves | `… ("survival_curves.json …")` |
| `_generate_sample_scoring_history` | 200-row np.random.beta(2,5) scoring | `… ("scoring_history.csv …")` |
| `_generate_sample_drift_history` | 30-row np.random drift | `… ("drift_history.csv …")` |
| `_generate_sample_scoring_throughput` | 48-point sinusoidal pattern | `… ("scoring_throughput.csv …")` |
| `_generate_sample_retention_offers` | 50-row np.random offers | `… ("retention_offers.csv …")` |
| `_generate_sample_feature_importance` | 15 hardcoded feature names + dirichlet | `… ("feature_importance.csv …")` |

The stubs are preserved (rather than deleted) so any straggling caller
elsewhere in the codebase fails fast with a clear actionable message
rather than silently rendering a sample dataset.

## Files touched

Only `src/dashboard/data_loader.py`. No changes to `app.py`, view
modules, tests, or pipeline code (per scope boundary; G1, G3 et al.
own those).

## Behaviour verification

Smoke test against current `results/` (no `confusion_matrices.json`,
`roc_data.json`, `survival_curves.json`, `scoring_history.csv`,
`scoring_throughput.csv`, `retention_offers.csv`,
`survival_data.csv` on disk):

```
load_roc_data:           dict       is_real=False
load_confusion_matrices: dict       is_real=False
load_survival_curves:    dict       is_real=False
load_scoring_history:    DataFrame  is_real=False  (empty, attrs set)
load_scoring_throughput: DataFrame  is_real=False  (empty, attrs set)
load_retention_offers:   DataFrame  is_real=False  (empty, attrs set)
load_survival_data:      DataFrame  is_real=False  (segments-derived, reason flagged)
art = load_roc_data(as_artifact=True)
  -> DashboardArtifact is_real=False reason="Required artifact missing: roc_data.json…"
```

The 3 tests in `tests/test_dashboard.py` that exercised the
`missing_monitoring_report`, `missing_required_cohort_and_ab`, and
`load_drift_history_from_monitoring_report` paths still pass.

`tests/test_dashboard.py::TestRealTimeScoringView` (4 tests) and
related tests that asserted `not df.empty` on the sample fallbacks now
fail because the fallbacks have been removed — these tests need to be
rewritten by the test-owning agent to either (a) drop the synthetic
artifacts into a tmp_path fixture, or (b) flip the assertion to assert
`df.empty and not loader.is_real_artifact(name)`.
