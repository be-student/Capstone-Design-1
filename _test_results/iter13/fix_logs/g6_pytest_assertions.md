# G6 Fix Log - Pytest assertions for dashboard no-fallback contract

## Iteration
iter13

## Agent
FIX AGENT G6 (pytest assertions)

## Goal
Add pytest tests that verify the dashboard reads ONLY pipeline-generated artifacts,
never `_generate_sample_*` fallbacks. Tests defensively skip when other agents'
APIs are not yet shipped, and clearly fail when a fixture leak is confirmed.

## Files created
- `tests/test_dashboard_no_fallback.py` (new file - the only file G6 is allowed to create)

## Files modified
None. G6 only writes the new test file and this fix log.

## Test classes added

### 1. `TestRealArtifactsExist` (parametrized, 7 cases)
Verifies the seven pipeline artifacts that G1 is expected to produce exist
under `results/` and are non-empty (>50 bytes):
- `confusion_matrices.json`
- `roc_data.json`
- `survival_data.csv`
- `survival_curves.json`
- `scoring_history.csv`
- `retention_offers.csv`
- `drift_history.csv`

**Catches**: G1 missed an artifact, or wrote an empty/header-only file.

### 2. `TestNoFixtureFallback` (5 tests)
For each loader (`confusion_matrices`, `survival_data`, `scoring_history`,
`retention_offers`, `drift_history`), invokes the loader with `as_artifact=True`
and asserts `art.is_real is True`. Wrapped in `try/except (TypeError, AttributeError)`
that emits `pytest.skip` so that this passes-as-skip until G2 ships
`DashboardArtifact` and the new keyword.

**Catches**: After G2 lands, any loader still silently returning fixture data
(is_real=False, reason set) becomes a hard fail with the exact reason printed.

### 3. `TestSampleGeneratorsRemoved` (parametrized, 7 cases)
For each of the 7 `_generate_sample_*` methods, the test passes if the method
is missing (best case, removed entirely). If still present, the method must
raise `FileNotFoundError`, `RuntimeError`, or `NotImplementedError` when called
with no real data - i.e. it must no longer silently return synthetic data.

**Catches**: A loader was edited to add `as_artifact` but its private
`_generate_sample_*` helper remained reachable and still returns fake data
on call.

### 4. `TestPage02NoFixtureOverride` (1 test)
Reads `results/model_metrics.json` for the ML model precision/recall, then
asserts they do NOT exactly match the hardcoded fixture values `0.7059 / 0.6000`
that were observed in iter12's audit on `src/dashboard/app.py:439-463`.
Skips if `model_metrics.json` is absent.

**Catches**: The Page 02 "headline metrics" KPI cards in `app.py` are still
hardcoded with the demo P/R rather than pulled from the real ML model
artifacts.

## Defensive design notes

- All tests use `pytest.skip(...)` when the upstream API (e.g. `as_artifact=True`,
  `DashboardArtifact`) is not yet implemented by parallel agents G2/G3. This
  means G6's test file never breaks the suite while iterations are in flight -
  it just shows "skipped" until the contract is ready.
- All tests use `pytest.fail(...)` (or plain `assert`) when a fixture leak is
  confirmed, with a precise message including filenames, line numbers, and
  values so the failure log gives the next agent a direct pointer.
- The path resolution uses `Path(__file__).resolve().parent.parent` so the file
  works from any cwd (Docker, local Windows, CI).
- `model_metrics.json` lookup tries both `ml_model.precision` and
  `ml_model.test.precision` to accommodate either schema G1 may have chosen.

## Verification (cannot run yet)

G6 cannot verify these pass yet because the upstream artifacts and the new
`as_artifact=True` API depend on G1/G2/G3 having committed. Once iter13's
parallel agents merge, run:

```
pytest tests/test_dashboard_no_fallback.py -v
```

Expected after all G1..G5 land:
- `TestRealArtifactsExist`: 7 passed
- `TestNoFixtureFallback`: 5 passed (or 5 skipped if G2 didn't ship)
- `TestSampleGeneratorsRemoved`: 7 passed
- `TestPage02NoFixtureOverride`: 1 passed (or 1 skipped if metrics file missing)

## Status
COMPLETE. Test file written. No production code touched. Safe to commit
independently regardless of order with G1..G5.
