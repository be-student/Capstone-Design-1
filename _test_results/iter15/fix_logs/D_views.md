# Agent D i18n fix log — monitoring / system_health / recommendations views

AST parse: OK (all 3 files)

## Pattern applied

Module-level closure helper `_tr` installed once per file, defensively
imported from `src.dashboard.utils.dashboard_helpers.get_lang` + `tr`.
Closure re-reads `get_lang()` at every call so module-level definition
remains correct across both public `render_*` functions and private
`_render_*` helpers — the smallest possible diff while keeping behavior
consistent for nested helpers.

```python
try:
    from src.dashboard.utils.dashboard_helpers import get_lang, tr

    def _tr(s: str) -> str:
        try:
            return tr(s, get_lang())
        except Exception:
            return s
except Exception:  # pragma: no cover - defensive fallback
    def _tr(s: str) -> str:
        return s
```

## File-by-file summary

### `src/dashboard/monitoring_view.py`
- Public entrypoint: `render_model_monitoring`
- Helpers wrapped: `_render_drift_section`, `_render_performance_section`,
  `_render_performance_alerts`, `_render_artifact_reason`,
  `_render_throughput_section`, `_classify_throughput_freshness`,
  `_render_config_section`.
- Wrapped surfaces: `st.header`/`subheader`, all `st.markdown("#### ...")`
  section headers, all `st.metric` labels, `st.info/warning/error/success/
  caption` messages, Plotly `update_layout(title=..., xaxis_title=...,
  yaxis_title=...)`, trace `name=` labels for legend, JSON-config bold
  markdown headers.
- Fallback messages such as "Real drift history missing", "No drift
  detection history available.", "SLO BREACH — error rate", "Historical
  ..." stale banner were translated.

### `src/dashboard/system_health_view.py`
- Public entrypoint: `render_system_health`
- Helpers wrapped: `_render_overall_health`, `_render_service_cards`,
  `_render_streaming_status`, `_render_mlflow_tracking`,
  `_render_model_health`, `_render_system_config`.
- Wrapped surfaces: header + description, status banner strings
  ("All Systems Operational" / "Degraded — Investigate Subsystems" /
  "System Issues Detected"), service card titles & captions (Redis
  Streaming / MLflow Tracking / ML Pipeline + their descriptions),
  metric labels (`Connected`, `Experiments`, `Total Runs`, ...),
  `mlflow_status_banner` output (passed through `_tr`), all section
  subheaders, full configuration expander label, Plotly chart titles
  + axis labels, dataframe column rename map (`Stream`, `Length`,
  `Consumers`, `Pending Messages`, `Group`, `Model`, `AUC`, `Precision`,
  `Recall`, `F1`).
- Two-state messages such as Redis "Yes (idle — no traffic)" and MLflow
  "No (showing cached runs)" are translated as full phrases.

### `src/dashboard/recommendations_view.py`
- Public entrypoint: `render_recommendations_view`
- Helpers wrapped: `_render_kpi_cards`, `_render_distribution_section`,
  `_render_uplift_analysis`, `_render_segment_breakdown`,
  `_render_cost_benefit_analysis`, `_render_recommendation_table`.
- Wrapped surfaces: header + lede, KPI labels (`Total Recommendations`,
  `Avg Predicted Uplift (all customers)`, `Top Action Type`,
  `High Priority`, `Offers Generated`, `Total Campaign Cost`,
  `Est. Revenue Saved`, `Avg Treated Uplift`), reconciliation `st.info`
  callout, distribution / uplift / segment-breakdown Plotly chart titles
  + axis labels + `labels={}` rename maps, ROI tile `roi_info["label"]`
  passed through `_tr`, filter widget labels (`Filter by Action Type`,
  `Minimum Priority Score`, `All`), table footer ("Showing N
  recommendations"), Top Priority / Detailed Retention Offers section
  headers.

## AST verification

```
C:/Users/yoonc/Capstone-Design-1/src/dashboard/monitoring_view.py: OK
C:/Users/yoonc/Capstone-Design-1/src/dashboard/system_health_view.py: OK
C:/Users/yoonc/Capstone-Design-1/src/dashboard/recommendations_view.py: OK
```

## Extracted keys

See `_test_results/iter15/keys/keys_D.json` (214 unique keys across the
3 view modules). Schema matches Agent B/C output (`agent`, `scope`,
`keys`, `files`).
