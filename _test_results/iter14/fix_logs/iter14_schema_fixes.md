# iter14 schema-mismatch fixes

Solo FIX AGENT, 4 schema-mismatch regressions introduced in iter13 between
G1's artifact schemas and the dashboard reader code. Pipeline outputs are
treated as ground truth; only dashboard readers were modified.

## Files modified

- `src/dashboard/app.py`
- `src/dashboard/recommendations_view.py`
- `src/dashboard/data_loader.py`
- `src/dashboard/utils/dashboard_helpers.py`

---

## Defect 1 — Page 02 Confusion Matrix render IndexError (P0) — CLOSED

Root cause: `confusion_matrices.json` is a per-model dict keyed by model
name where each value is a `{tn, fp, fn, tp, n_samples, threshold, matrix}`
dict. The render code did `cm = np.array(matrix)` on that dict, producing
a 0-d object array, then `cm[0][0]` raised
`IndexError: too many indices for array: array is 0-dimensional`.

### `src/dashboard/app.py:194` — NEW helper `_extract_cm_cells`

Before: helper did not exist.

After:
```python
def _extract_cm_cells(cm):
    """Extract (tn, fp, fn, tp) from any confusion-matrix shape."""
    if isinstance(cm, dict):
        if "matrix" in cm and cm["matrix"] is not None:
            m = cm["matrix"]
            try:
                return int(m[0][0]), int(m[0][1]), int(m[1][0]), int(m[1][1])
            except (IndexError, TypeError, ValueError):
                pass
        try:
            return int(cm["tn"]), int(cm["fp"]), int(cm["fn"]), int(cm["tp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(...) from exc
    try:
        return int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
    except Exception as exc:
        raise ValueError(...) from exc
```

### `src/dashboard/app.py:~712` — heatmap render site

Before:
```python
for idx, (model_name, matrix) in enumerate(cm_data.items()):
    with cm_cols[idx]:
        cm = np.array(matrix)
        ...
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
```

After:
```python
for idx, (model_name, matrix) in enumerate(cm_data.items()):
    with cm_cols[idx]:
        tn, fp, fn, tp = _extract_cm_cells(matrix)
        cm = np.array([[tn, fp], [fn, tp]])
        ...
        total = tn + fp + fn + tp
```

### `src/dashboard/app.py:~712` — test-set-size aggregation

Before:
```python
for _matrix in (cm_data or {}).values():
    _cm_arr = np.array(_matrix)
    _total = float(_cm_arr.sum())
```

After:
```python
for _matrix in (cm_data or {}).values():
    _tn, _fp, _fn, _tp = _extract_cm_cells(_matrix)
    _total = float(_tn) + float(_fp) + float(_fn) + float(_tp)
```

---

## Defect 2 — Page 13 KeyError 'priority_rank' (P0) — CLOSED

Root cause: G1 `retention_offers.csv` ships a `priority_score` column
(higher = more important); reader sorted by non-existent `priority_rank`.

### `src/dashboard/app.py:4775`

Before: `filtered = filtered.sort_values("priority_rank")`

After:
```python
if "priority_score" in filtered.columns:
    filtered = filtered.sort_values("priority_score", ascending=False)
```

### `src/dashboard/app.py:~4920` — display column list

Before: `"priority_rank", "customer_id", ...`
After: `"priority_score", "customer_id", ...`

### `src/dashboard/recommendations_view.py:303`

Before: `elif not offers.empty and "priority_rank" in offers.columns:`
After:  `elif not offers.empty and "priority_score" in offers.columns:`

### `src/dashboard/recommendations_view.py:~766` — display column list

Before: `... "estimated_revenue_save_krw", "priority_rank", ...`
After:  `... "expected_revenue_saved_krw", "priority_score", ...`

### `src/dashboard/data_loader.py:1725, 1734`

Updated docstring (`priority_rank` → `priority_score`) and empty-frame
column template (same rename) so empty-artifact downstream guards
continue to pass.

---

## Defect 3 — Page 09 Cost-Benefit ₩0 / 0.00x ROI (P0) — CLOSED

Root cause: reader typo `estimated_revenue_save_krw` (missing `d`, wrong
word order). G1 emits `expected_revenue_saved_krw`. Sums silently
returned 0.0 → ROI 0.00x.

### `src/dashboard/recommendations_view.py` — 7 occurrences (replace_all)

Lines 579, 580, 611, 666, 669, 693, 698, 704, 765 — all
`estimated_revenue_save_krw` → `expected_revenue_saved_krw`.

### `src/dashboard/app.py` — 4 occurrences (replace_all)

Lines 4814, 4868, 4900, 4922 — all
`estimated_revenue_save_krw` → `expected_revenue_saved_krw`.

### `src/dashboard/data_loader.py` — 2 occurrences

Lines 1725, 1733 (docstring + empty-frame column template) — same rename.

Grep verification: no `estimated_revenue_save_krw` reference remains in
`src/dashboard/`.

---

## Defect 4 — Drift trend chart degenerate (P1) — CLOSED

Root cause: `drift_history.csv` rows all share one timestamp from a
single `run_monitor` invocation. Existing guard only rejected sub-hour
spans, but a single-batch timeseries can have span = 0 yet still pass
the `>=2` length check.

### `src/dashboard/utils/dashboard_helpers.py:~325` — new all-same-timestamp branch

Added new branch BEFORE the existing `<1 hour` check:
```python
if seconds < 5:
    return False, (
        "All drift checks come from one pipeline invocation "
        "(timestamps span <5s) — run `python -m src.main --mode monitor` "
        "multiple times to build trend history."
    )
if seconds < 3600:
    return False, f"Trend window is {seconds:.1f}s — too short ..."
```

Existing fallback path (`def drift_trend_guard` in `app.py:100`) is the
"shadow" definition used only when the canonical helper fails to import;
left unmodified.

---

## Syntax verification

```
> venv/Scripts/python.exe -c "import ast; ast.parse(...)"
all OK
```

All 4 modified files parse without SyntaxError.
