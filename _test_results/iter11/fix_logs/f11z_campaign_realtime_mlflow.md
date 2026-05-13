# F11-Z fix log — iter11 round 1

**Scope (read-only outside this list):** `src/dashboard/app.py`, only inside
the functions `render_retention_campaign`, `render_realtime_scoring`,
`_render_scoring_status_tab`, `_render_retention_offers_tab`,
`_render_monitoring_tab`, `render_mlflow_experiments`.

**Inputs consulted**
- `_test_results/iter10/verify_v4.md` (P14 issues #10, #11)
- `_test_results/iter10/verify_v5.md` (P12 taxonomy issue #9)
- `_test_results/iter10/verify_v6.md` (P13 issues #2, #4, #8, #9, #10)

**Defensive imports.** `drift_trend_guard`, `format_count`, and
`compute_overall_roi` are imported at module top with `try/except`
fallbacks (lines 47–115); no new top-level imports were needed.

---

## Defects closed

### 1. P12 (verify_v5 #9) — behavioral vs uplift taxonomy mix
**Function:** `render_retention_campaign` (line ~2706).

Added an `st.info(...)` crosswalk caption immediately above the
"1. Customer Lifetime Value Overview" subheader. The caption explicitly
names the two taxonomies, gives the behavioral⇄uplift mapping
(`dormant ≈ sleeping_dog / high_value_lost_cause`), and reassures the
reader that the same 20,000 customers are being viewed through two
lenses. This closes the iter10 audit blocker that a reader could not
trace a single customer between Section 1 and Sections 2–4 of the same
page.

### 2. P13 tab a (verify_v6 #2) — Oct 2024 throughput on a "real-time" page
**Function:** `_render_scoring_status_tab` (line ~3899).

Wrapped the throughput / latency rendering in a staleness check: when
the latest sample timestamp is older than 24 h, the section now emits
`st.warning("Historical sample data (Oct 2024) — replace with live
telemetry before production. Current refresh: <ts>")` *and* renames the
header to `"Scoring Throughput (historical sample)"`. When data is
fresh, the original header is preserved and a `Data window` caption is
still rendered for transparency.

### 3. P13 tab c (verify_v6 #8) — Scoring Volume = uniform 4 across 50 buckets
**Function:** `_render_monitoring_tab` (line ~4422).

Detect the degenerate-uniform case structurally: if the relative
standard deviation across hourly buckets is below 5% (and we have at
least 5 buckets), the bar chart is replaced with a single
`st.metric("Scoring Volume (last 24h, demo)", ...)` rollup plus a
`*synthetic uniform demo data — not real telemetry*` caption.
Non-degenerate data still renders the bar chart, but with a caption
warning the reader to interpret near-uniform output as fixture data.

### 4. P13 (verify_v6 #9) — clock mismatch between tabs
**Functions:** all three tab renderers.

Each tab now renders an explicit `st.caption("Last refresh: <utc-now>
· Model: …")` block under its top header. The throughput section adds
a `Data window: <latest sample>` caption, and the recent-scoring-
history section adds `Data window: <min> → <max>` so a buyer can see
each section's data anchor without inferring it from the chart axis.

### 5. P13 (verify_v6 #10) — model_version stamp absent
**Functions:** `_render_scoring_status_tab`, `_render_retention_offers_tab`,
`_render_monitoring_tab`.

Each tab carries its own inline resolver that tries
`data_loader.get_active_model()` first, falls back to
`config.ensemble.model_version` / `config.model.version` /
`config.mlflow.model_version`, and finally emits
`"Model: ensemble v? (version metadata missing)"` so the gap is
visible. The resolver is duplicated across tabs because the F11-Z
brief restricts edits to inside the listed functions and forbids
adding new module-level helpers.

### 6. P14 (verify_v4 #10) — degenerate sweep (3 runs, LR=0.1, training=1 s)
**Function:** `render_mlflow_experiments` (line ~4706).

Above the "Hyperparameter Analysis" section we now compute
`_lr_unique_count` and `_epochs_unique_count` from the runs dataframe;
if both are ≤1 the page renders an `st.info(...)` calling out the
single-config smoke test and pointing readers at `/docs` for grid
search. The `Learning Rate vs AUC` and `Epochs vs AUC` plots are
suppressed in the degenerate case and replaced with flat captions
naming the only LR / epochs value seen, so an analyst can no longer
interpret the 3-point cluster as a real sweep.

### 7. P14 (verify_v4 #11) — Experiment Timeline 0.1 ms x-axis
**Function:** `render_mlflow_experiments` (Run timeline section).

Wrapped the scatter in `drift_trend_guard(timestamps, min_points=5)`.
When the guard returns False the chart is replaced with an
`st.info(msg)` plus a `st.dataframe` showing the static run list
(`Run 1 / Run 2 / Run 3`) with key metrics. This mirrors the same
pattern already applied on Page 08 (`monitoring_view.py:467`) and on
Page 13 tab c — closing the only remaining drift-guard hole on Page
14.

---

## Verification

`python -m py_compile src/dashboard/app.py` ⇒ `OK` (syntax valid).

No edits outside the six listed functions; defensive imports for
`drift_trend_guard` / `format_count` / `compute_overall_roi` were
already present at module top from F1.

---

## 5-line summary

1. P12 taxonomy mix closed by an explicit behavioral⇄uplift crosswalk
   caption above the CLV Overview section in `render_retention_campaign`.
2. P13 tab a now warns when the throughput sample is >24 h old and
   renames the header to `Scoring Throughput (historical sample)`; a
   `Data window` + `Last refresh` caption is rendered on every tab.
3. P13 tab c Scoring Volume detects the degenerate-uniform case
   (relative std < 5%) and falls back to a single 24 h rollup with a
   `*synthetic uniform demo data*` caption.
4. All three Page 13 tabs now carry a `Model: …` stamp resolved from
   `data_loader.get_active_model()` → config → explicit
   `version metadata missing` placeholder.
5. P14 closes the two remaining iter10 NOT-FIXED items: degenerate
   sweep is announced via `st.info` and the LR/Epochs plots collapse to
   captions when only one value is logged; the Experiment Timeline now
   uses `drift_trend_guard(min_points=5)` and falls back to a Run-list
   table when fewer than 5 observations are available.
