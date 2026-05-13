# Fix Log — Iter13 G5: Real-only Data Enforcement Helpers

**Agent:** FIX AGENT G5
**Date:** 2026-05-12
**Iter:** 13
**Scope:** Add helpers that other iter13 agents will use for the "real-only data" enforcement rollout.

---

## Files modified

- `src/dashboard/utils/dashboard_helpers.py`
- `src/dashboard/calculations.py`

No other files touched.

---

## Changes

### 1. `src/dashboard/utils/dashboard_helpers.py`

Appended a new section at the bottom of the file:

```
# =========================================================================
# Real-only artifact enforcement helpers (iter13)
# =========================================================================
```

Two new public functions:

#### `assert_real_or_error(st, artifact, label) -> bool`
Guard helper for Streamlit view code. Inspects `artifact.is_real`:

- `artifact is None` -> `st.error(...)` with wiring hint, return False.
- `artifact.is_real == False` -> `st.error(...)` showing label / source_path / reason and the regen command, return False.
- Otherwise (including legacy artifacts that lack an `is_real` attribute, treated as real for backwards compatibility) -> return True.

The error message includes:
- `**<label> unavailable** — real artifact missing.`
- `Source: `<source_path>``
- `Reason: <reason>`
- Regeneration hint: `Run \`python -m src.main --mode all\` to regenerate.`

#### `freshness_caption(artifact, default_label="Last refresh") -> str`
Renders a Markdown italic caption for use under charts/tables.

- `artifact is None` -> returns `""` (safe to unconditionally pass to `st.caption`).
- Has `computed_at` or `mtime` -> returns `"_<default_label>: <ts>_"`.
- Has neither timestamp -> returns `"_<default_label>: unknown_"` so the gap is visible to operators rather than silently disappearing.

Used by other iter13 agents to give the operator a one-glance read on which run produced the numbers.

### 2. `src/dashboard/calculations.py`

Appended a new section at the bottom of the file:

```
# =========================================================================
# Real-only KPI rendering helpers (iter13)
# =========================================================================
```

#### `safe_real_metric(value, fallback_indicator="—") -> str | numeric`
Last-line-of-defense renderer for values passed to `st.metric()`:

- `None` -> fallback indicator (default `"—"`).
- `float` NaN or +/- infinity -> fallback indicator.
- Anything else (finite numeric, pre-formatted strings, etc.) -> returned as-is.

Purpose: prevent fixture leak (zeros, NaN) from appearing as headline KPI numbers when a real artifact is missing. View code can call this defensively around every `st.metric(value=...)` even if upstream guards via `assert_real_or_error` are already in place.

---

## Verification

Wrote a transient smoke script under `_test_results/iter13/fix_logs/_g5_smoketest.py`, ran it under `PYTHONPATH=.` and removed it afterwards. Coverage:

- `safe_real_metric`: None, NaN, +inf, int, float, pre-formatted string, custom fallback indicator.
- `freshness_caption`: None artifact, artifact with no timestamp, artifact with `computed_at`, artifact with `mtime`.
- `assert_real_or_error`: None artifact, real artifact (is_real=True), fake artifact (is_real=False with reason+source_path), legacy artifact lacking `is_real` attribute.

Result: `ALL OK` — all assertions held.

---

## Notes for downstream agents

- `assert_real_or_error` takes the Streamlit module as its first argument (`st`) rather than importing it inside this module. This keeps the helper unit-testable without monkey-patching and avoids creating a hard import of `streamlit` from `dashboard_helpers`, which is also imported by non-dashboard callers.
- `freshness_caption` returns a Markdown italic so callers should pass the output to `st.caption(...)` (which renders Markdown) rather than `st.text(...)`.
- `safe_real_metric` lives in `calculations.py` per the spec; it does NOT do formatting, only sanitisation. Pair it with the existing `format_currency_krw` / `format_percentage` / `format_count` helpers from `dashboard_helpers.py` when display formatting is also needed.
