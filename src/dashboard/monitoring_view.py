"""
Model Monitoring Dashboard View.

Provides the render_model_monitoring function for the Streamlit dashboard,
showing drift detection, model performance metrics over time, and scoring
throughput/latency monitoring. Survival analysis is rendered on its own
dedicated page (Page 07) to avoid scope creep.

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# i18n (iter15 AGENT D) — defensive import so the page never crashes if
# helpers are missing. ``_tr`` is a module-level closure that re-reads the
# active language at call time, so it stays correct across helper functions.
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# F1 helper imports — guarded so monitoring_view does not crash if helpers
# from the parallel remediation are not yet present.
# -------------------------------------------------------------------------
try:
    from src.dashboard.utils.dashboard_helpers import drift_trend_guard
except Exception:  # pragma: no cover - defensive fallback
    def drift_trend_guard(timeseries: Any, min_points: int = 5):
        """Local fallback when dashboard_helpers.drift_trend_guard is missing.

        Mirrors the F1 helper contract: returns (ok, message). The fallback
        only enforces a minimum-observation count and is replaced with the
        canonical helper as soon as it ships.
        """
        try:
            n = 0 if timeseries is None else len(timeseries)
        except TypeError:
            try:
                n = sum(1 for _ in timeseries)
            except Exception:
                n = 0
        if n < min_points:
            return False, (
                f"Insufficient history — need ≥{min_points} observations, "
                f"have {n}."
            )
        return True, ""


# SaaS SLO target for scoring error rate. Surfaces a red callout when the
# observed error rate exceeds 0.1% (0.001), per the iter9 audit blocker.
ERROR_RATE_SLO_TARGET = 0.001


# -------------------------------------------------------------------------
# iter13 G4: defensive DashboardArtifact integration. G2 may not yet have
# shipped — we import optimistically and fall back to legacy behavior.
# -------------------------------------------------------------------------
try:  # pragma: no cover - defensive import
    from src.dashboard.data_loader import DashboardArtifact  # type: ignore
except Exception:  # pragma: no cover
    DashboardArtifact = None  # type: ignore[assignment]


def _load_artifact_safely(loader_callable, *args, **kwargs):
    """Try to call a data_loader method with ``as_artifact=True``.

    Returns a tuple ``(payload, artifact_or_none)`` where ``payload`` is the
    DataFrame/dict the legacy caller expected, and ``artifact_or_none`` is
    the DashboardArtifact wrapper when G2's loader signature supports it.
    Falls back to the legacy return shape on ``TypeError`` (old signature)
    or on any unexpected exception so the dashboard never crashes.
    """
    if loader_callable is None:
        return None, None
    try:
        result = loader_callable(*args, as_artifact=True, **kwargs)
    except TypeError:
        # Old signature — caller does not accept as_artifact.
        try:
            payload = loader_callable(*args, **kwargs)
        except Exception:
            return None, None
        return payload, None
    except Exception:
        return None, None

    # New signature returned a DashboardArtifact (or duck-typed equivalent).
    payload = getattr(result, "data", result)
    return payload, result


def render_model_monitoring(st_module, config: Dict, data_loader=None):
    """Render the Model Monitoring page.

    Shows:
    - Drift detection overview with alert status timeline (sparse-window
      guarded via ``drift_trend_guard``).
    - Model performance metrics tracked over training runs.
    - Scoring throughput and latency monitoring with explicit data-freshness
      labelling and a SaaS error-rate SLO badge.

    Survival analysis (Kaplan-Meier curves) is intentionally NOT rendered
    here — it is owned by the dedicated Survival Analysis page (Page 07).

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header(_tr("Model Monitoring"))

    if data_loader is None:
        from src.dashboard.app import get_data_loader
        data_loader = get_data_loader(config)

    # -----------------------------------------------------------------
    # Load all monitoring data. iter13 G4: probe each loader with
    # ``as_artifact=True`` so we can surface "real artifact missing"
    # warnings instead of silently rendering fixture-derived charts.
    # -----------------------------------------------------------------
    drift_history, drift_artifact = _load_artifact_safely(
        getattr(data_loader, "load_drift_history", None),
    )
    if drift_history is None:
        drift_history = pd.DataFrame()
    model_metrics = data_loader.load_model_metrics()
    scoring_throughput, throughput_artifact = _load_artifact_safely(
        getattr(data_loader, "load_scoring_throughput", None),
    )
    if scoring_throughput is None:
        scoring_throughput = pd.DataFrame()
    performance_history = _load_performance_history(data_loader)
    performance_alerts = _load_performance_alerts(data_loader)

    # =================================================================
    # Section 1: Drift Detection Overview
    # =================================================================
    st.subheader(_tr("Drift Detection Overview"))

    # iter13 G4: if G2's loader reports the underlying artifact is NOT real
    # (i.e. drift_history.csv missing and the loader returned a fixture or
    # an empty placeholder), surface an error and skip the charts entirely.
    if _artifact_marked_unreal(drift_artifact):
        st.error(_tr(
            "Real drift history missing — run the pipeline to populate "
            "`results/drift_history.csv` (or `monitoring_report.json`)."
        ))
        _render_artifact_reason(st, drift_artifact)
    elif drift_history.empty:
        st.warning(_tr("No drift detection history available."))
    else:
        _render_drift_section(st, config, drift_history, performance_alerts)

    # =================================================================
    # Section 2: Model Performance Metrics Over Time
    # =================================================================
    st.markdown("---")
    st.subheader(_tr("Model Performance Metrics Over Time"))

    _render_performance_section(
        st, model_metrics, performance_history, performance_alerts,
    )

    # =================================================================
    # Section 3: Scoring Throughput & Latency
    # =================================================================
    st.markdown("---")
    st.subheader(_tr("Scoring Throughput & Latency"))

    _render_throughput_section(st, scoring_throughput, throughput_artifact)

    # =================================================================
    # Section 4: Monitoring Configuration
    # =================================================================
    st.markdown("---")
    st.subheader(_tr("Monitoring Configuration"))

    _render_config_section(st, config)


# =========================================================================
# Internal rendering helpers
# =========================================================================


def _render_drift_section(
    st,
    config: Dict,
    drift_history: pd.DataFrame,
    performance_alerts: Optional[Dict[str, Any]] = None,
):
    """Render drift detection section with KPI cards, timeline, and charts.

    The drift status drives the page-level headline banner. When the latest
    drift level is RED we surface a ``Performance degradation: drift
    threshold breached`` callout up front, overriding any stale "no
    degradation" success banner that downstream sections might otherwise
    emit. This closes the iter9 banner-vs-KPI contradiction.
    """
    # KPI cards for drift status
    kd1, kd2, kd3, kd4 = st.columns(4)
    latest = drift_history.iloc[-1]
    total_checks = len(drift_history)
    red_count = int((drift_history["alert_level"] == "red").sum())
    yellow_count = int((drift_history["alert_level"] == "yellow").sum())
    latest_level = str(latest["alert_level"]).lower()

    kd1.metric(_tr("Total Checks"), f"{total_checks}")
    kd2.metric(_tr("Current Status"), latest_level.upper())
    kd3.metric(_tr("Red Alerts"), f"{red_count}")
    kd4.metric(_tr("Yellow Alerts"), f"{yellow_count}")

    # ------------------------------------------------------------------
    # Headline banner derived from drift status (closes defect 1).
    # ------------------------------------------------------------------
    perf_alerts = performance_alerts or {}
    model_type = perf_alerts.get("model_type", "ensemble")
    if latest_level == "red":
        st.error(
            f"{_tr('Performance degradation: drift threshold breached for')} "
            f"{model_type} (drift status = RED, red alerts = {red_count})."
        )
    elif latest_level == "yellow":
        st.warning(
            f"{_tr('Drift watchlist:')} {model_type} "
            f"{_tr('approaching thresholds')} "
            f"(drift status = YELLOW, yellow alerts = {yellow_count})."
        )

    # Alert timeline — only render as a trend if dense enough.
    st.markdown(f"#### {_tr('Drift Alert Timeline')}")
    ok_timeline, msg_timeline = drift_trend_guard(
        drift_history.get("timestamp", []),
    )
    if not ok_timeline:
        st.info(
            f"{_tr('Drift Alert Timeline')}: {msg_timeline} "
            f"{_tr('Showing latest snapshot only — a trend chart is suppressed to avoid a degenerate single-point line.')}"
        )
        # Emit the latest observation as a single KPI row so users still see
        # which level is current without a misleading "trend" line.
        if not drift_history.empty:
            snapshot_cols = [
                c for c in [
                    "timestamp", "alert_level", "num_drifted_features",
                ]
                if c in drift_history.columns
            ]
            st.dataframe(
                drift_history[snapshot_cols].tail(1),
                use_container_width=True,
            )
    else:
        alert_color_map = {
            "green": "#2ecc71",
            "yellow": "#f39c12",
            "red": "#e74c3c",
        }
        fig_timeline = go.Figure()
        for level, color in alert_color_map.items():
            mask = drift_history["alert_level"] == level
            if mask.any():
                subset = drift_history[mask]
                fig_timeline.add_trace(go.Scatter(
                    x=subset["timestamp"],
                    y=subset["num_drifted_features"],
                    mode="markers",
                    name=level.capitalize(),
                    marker=dict(color=color, size=10),
                ))
        fig_timeline.update_layout(
            title=_tr("Drift Alert Timeline"),
            xaxis_title=_tr("Date"),
            yaxis_title=_tr("Number of Drifted Features"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    # Per-chart "Last update" anchor (closes defect 3 partially: each chart
    # discloses its own freshness).
    last_drift_ts = _format_last_update(drift_history.get("timestamp"))
    st.caption(f"{_tr('Last update')}: {last_drift_ts}")

    # PSI and KS over time — gated by the same trend guard.
    st.markdown(f"#### {_tr('PSI & KS Statistics Over Time')}")
    col_psi, col_ks = st.columns(2)

    drift_cfg = config.get("drift_detection", {})
    psi_yellow = drift_cfg.get("yellow_threshold", 0.10)
    psi_red = drift_cfg.get("red_threshold", 0.25)
    ks_cfg = config.get("ks_drift_detection", {})
    ks_warn = ks_cfg.get("warning_threshold", 0.05)

    ok_psi, msg_psi = drift_trend_guard(drift_history.get("timestamp", []))

    with col_psi:
        if not ok_psi:
            st.info(f"{_tr('Mean PSI Over Time')}: {msg_psi}")
            latest_psi = (
                drift_history["psi_mean"].iloc[-1]
                if "psi_mean" in drift_history.columns
                and not drift_history.empty
                else float("nan")
            )
            st.metric(
                _tr("Latest PSI"),
                f"{latest_psi:.4f}" if pd.notna(latest_psi) else "—",
                help=(
                    f"Yellow ≥ {psi_yellow}, Red ≥ {psi_red}. Trend chart "
                    f"hidden until ≥5 observations are available."
                ),
            )
        else:
            fig_psi = go.Figure()
            fig_psi.add_trace(go.Scatter(
                x=drift_history["timestamp"],
                y=drift_history["psi_mean"],
                mode="lines+markers",
                name=_tr("Mean PSI"),
                line=dict(color="#3498db", width=2),
            ))
            fig_psi.add_hline(
                y=psi_yellow, line_dash="dash", line_color="#f39c12",
                annotation_text=f"Yellow ({psi_yellow})",
            )
            fig_psi.add_hline(
                y=psi_red, line_dash="dash", line_color="#e74c3c",
                annotation_text=f"Red ({psi_red})",
            )
            fig_psi.update_layout(
                title=_tr("Mean PSI Over Time"),
                xaxis_title=_tr("Date"),
                yaxis_title=_tr("PSI Value"),
            )
            st.plotly_chart(fig_psi, use_container_width=True)
        st.caption(f"{_tr('Last update')}: {last_drift_ts}")

    with col_ks:
        if not ok_psi:
            st.info(f"{_tr('Mean KS Statistic Over Time')}: {msg_psi}")
            latest_ks = (
                drift_history["ks_mean"].iloc[-1]
                if "ks_mean" in drift_history.columns
                and not drift_history.empty
                else float("nan")
            )
            st.metric(
                _tr("Latest KS"),
                f"{latest_ks:.4f}" if pd.notna(latest_ks) else "—",
                help=(
                    f"Warning ≥ {ks_warn}. Trend chart hidden until ≥5 "
                    f"observations are available."
                ),
            )
        else:
            fig_ks = go.Figure()
            fig_ks.add_trace(go.Scatter(
                x=drift_history["timestamp"],
                y=drift_history["ks_mean"],
                mode="lines+markers",
                name=_tr("Mean KS Statistic"),
                line=dict(color="#9b59b6", width=2),
            ))
            fig_ks.add_hline(
                y=ks_warn, line_dash="dash", line_color="#f39c12",
                annotation_text=f"Warning ({ks_warn})",
            )
            fig_ks.update_layout(
                title=_tr("Mean KS Statistic Over Time"),
                xaxis_title=_tr("Date"),
                yaxis_title=_tr("KS Statistic"),
            )
            st.plotly_chart(fig_ks, use_container_width=True)
        st.caption(f"{_tr('Last update')}: {last_drift_ts}")

    # Drift history table
    st.markdown(f"#### {_tr('Drift Detection Log')}")
    display_cols = [
        "timestamp", "alert_level", "num_drifted_features",
        "psi_mean", "ks_mean",
    ]
    avail_cols = [c for c in display_cols if c in drift_history.columns]
    st.dataframe(
        drift_history[avail_cols].sort_values(
            "timestamp", ascending=False,
        ),
        use_container_width=True,
    )


def _format_last_update(timestamps: Any) -> str:
    """Format the most recent timestamp from a series for a freshness label.

    Returns ``"unknown"`` when the input is empty or unparseable so the
    caller can still surface a clear ``Last update:`` caption without
    raising.
    """
    try:
        ts_series = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        ts_series = ts_series.dropna()
        if ts_series.empty:
            return "unknown"
        return ts_series.max().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "unknown"


def _load_performance_history(data_loader) -> pd.DataFrame:
    """Load model performance history through explicit dashboard loaders."""
    if hasattr(data_loader, "load_model_performance_history"):
        history = data_loader.load_model_performance_history()
        if not history.empty:
            return history

    metric_frames = []
    for loader_name in [
        "load_auc_history",
        "load_precision_history",
        "load_recall_history",
    ]:
        if hasattr(data_loader, loader_name):
            metric_frames.append(getattr(data_loader, loader_name)())
    metric_frames = [df for df in metric_frames if isinstance(df, pd.DataFrame) and not df.empty]
    if not metric_frames:
        return pd.DataFrame()

    merged = metric_frames[0]
    keys = [c for c in ["timestamp", "model_type"] if c in merged.columns]
    for frame in metric_frames[1:]:
        frame_keys = [c for c in ["timestamp", "model_type"] if c in frame.columns]
        join_keys = [c for c in keys if c in frame_keys]
        if join_keys:
            merged = merged.merge(frame, on=join_keys, how="outer")
        else:
            merged = pd.concat([merged, frame], ignore_index=True, sort=False)
    if "run_id" not in merged.columns:
        merged["run_id"] = [f"history_{i}" for i in range(len(merged))]
    if "training_time_s" not in merged.columns:
        merged["training_time_s"] = 1.0
    return merged


def _load_performance_alerts(data_loader) -> Dict[str, Any]:
    """Load model performance degradation alerts when the loader supports it."""
    if not hasattr(data_loader, "load_performance_alerts"):
        return {}
    try:
        alerts = data_loader.load_performance_alerts()
    except Exception as exc:
        logger.warning("Failed to load performance alerts: %s", exc)
        return {}
    return alerts if isinstance(alerts, dict) else {}


def _render_performance_section(
    st,
    model_metrics: Dict,
    mlflow_runs: pd.DataFrame,
    performance_alerts: Optional[Dict[str, Any]] = None,
):
    """Render model performance metrics section."""
    performance_alerts = performance_alerts or {}
    if not model_metrics and mlflow_runs.empty and not performance_alerts.get("metrics"):
        st.warning(_tr("No model performance metrics available."))
        return

    if model_metrics:
        st.markdown(f"#### {_tr('Current Model Performance')}")
        metric_names = [
            "auc", "precision", "recall", "f1_score", "accuracy",
        ]
        model_names = list(model_metrics.keys())
        colors = [
            "#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c",
        ]

        fig_perf = go.Figure()
        for i, model_name in enumerate(model_names):
            vals = [
                model_metrics[model_name].get(m, 0)
                for m in metric_names
            ]
            fig_perf.add_trace(go.Bar(
                name=model_name,
                x=metric_names,
                y=vals,
                marker_color=colors[i % len(colors)],
                text=[f"{v:.3f}" for v in vals],
                textposition="outside",
            ))
        fig_perf.update_layout(
            title=_tr("Model Performance Comparison"),
            xaxis_title=_tr("Metric"),
            yaxis_title=_tr("Score"),
            barmode="group",
            yaxis_range=[0, 1.05],
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        # Best model highlight
        best_model = max(
            model_metrics.keys(),
            key=lambda m: model_metrics[m].get("auc", 0),
        )
        best_auc = model_metrics[best_model].get("auc", 0)
        st.info(
            f"{_tr('Best model by AUC')}: **{best_model}** "
            f"(AUC = {best_auc:.4f})"
        )

    _render_performance_alerts(st, performance_alerts)

    # MLflow run history — defect 4: avoid rendering 3 sequential function
    # calls as a temporal series with a sub-millisecond x-axis. When fewer
    # than 5 runs are available, switch to an index-based x-axis ("Run 1 /
    # Run 2 / Run 3") so the chart never claims to be a longitudinal trend.
    if not mlflow_runs.empty:
        st.markdown(f"#### {_tr('Training Run History')}")
        n_runs = len(mlflow_runs)
        ok_runs, _ = drift_trend_guard(
            mlflow_runs.get("timestamp", []),
        )
        run_index = [f"Run {i + 1}" for i in range(n_runs)]
        if ok_runs and "timestamp" in mlflow_runs.columns:
            fig_runs = go.Figure()
            for metric in ["auc", "precision", "recall", "f1_score"]:
                if metric in mlflow_runs.columns:
                    fig_runs.add_trace(go.Scatter(
                        x=mlflow_runs["timestamp"],
                        y=mlflow_runs[metric],
                        mode="lines+markers",
                        name=metric.upper(),
                        text=mlflow_runs.get("model_type", None),
                    ))
            fig_runs.update_layout(
                title=_tr("Model Metrics Across Training Runs"),
                xaxis_title=_tr("Timestamp"),
                yaxis_title=_tr("Score"),
                hovermode="x unified",
            )
            st.plotly_chart(fig_runs, use_container_width=True)
        else:
            # Index-based bar chart — no implicit temporal claim.
            st.info(
                f"{_tr('Only')} {n_runs} {_tr('training run(s) available — showing an index-based comparison rather than a temporal trend.')}"
            )
            fig_runs = go.Figure()
            for metric in ["auc", "precision", "recall", "f1_score"]:
                if metric in mlflow_runs.columns:
                    fig_runs.add_trace(go.Bar(
                        x=run_index,
                        y=mlflow_runs[metric],
                        name=metric.upper(),
                    ))
            fig_runs.update_layout(
                title=_tr("Model Metrics Across Training Runs (per run)"),
                xaxis_title=_tr("Run"),
                yaxis_title=_tr("Score"),
                barmode="group",
                yaxis_range=[0, 1.05],
            )
            st.plotly_chart(fig_runs, use_container_width=True)

        st.markdown(f"#### {_tr('Run Details')}")
        run_display_cols = [
            c for c in [
                "run_id", "model_type", "auc", "precision",
                "recall", "f1_score", "accuracy", "training_time_s",
            ]
            if c in mlflow_runs.columns
        ]
        st.dataframe(
            mlflow_runs[run_display_cols], use_container_width=True,
        )


def _render_performance_alerts(st, performance_alerts: Dict[str, Any]):
    """Render threshold-based performance degradation alerts.

    The drift section already emits the page-level headline banner derived
    from drift status (defect 1). This function therefore renders only the
    metric-level table plus a status callout that is *consistent* with the
    drift headline — it never emits a green "no degradation" banner that
    could contradict a RED drift card sitting in the same viewport.
    """
    metrics = performance_alerts.get("metrics", {})
    if not metrics:
        return

    st.markdown(f"#### {_tr('Performance Degradation Alerts')}")
    degraded = bool(performance_alerts.get("performance_degradation"))
    status = str(performance_alerts.get("status", "ok"))
    model_type = performance_alerts.get("model_type", "model")
    if degraded:
        degraded_metrics = ", ".join(
            performance_alerts.get("degraded_metrics", [])
        )
        st.error(
            f"{_tr('Performance degradation detected for')} {model_type}: "
            f"{degraded_metrics}"
        )
    elif status == "warning":
        st.warning(
            f"{_tr('Performance metrics for')} {model_type} "
            f"{_tr('are approaching thresholds.')}"
        )
    else:
        # Caption (not success banner) so it cannot contradict a RED drift
        # headline. The authoritative status remains the drift banner.
        st.caption(
            f"{_tr('Per-metric thresholds: no degradation flagged for')} "
            f"{model_type}. "
            f"{_tr('(Authoritative health status comes from the Drift Detection Overview banner.)')}"
        )

    rows = []
    for metric, payload in metrics.items():
        rows.append({
            "metric": metric,
            "current": payload.get("current"),
            "baseline": payload.get("baseline"),
            "drop": payload.get("drop"),
            "threshold": payload.get("threshold"),
            "status": payload.get("status"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _artifact_marked_unreal(artifact: Any) -> bool:
    """Return True when ``artifact.is_real`` is explicitly False.

    Returning False on a missing artifact (None) preserves legacy behavior
    when G2's loader signature is not yet available. Only an explicit
    ``is_real=False`` triggers the new "real artifact missing" callout.
    """
    if artifact is None:
        return False
    is_real = getattr(artifact, "is_real", None)
    if is_real is None:
        return False
    return bool(is_real) is False


def _render_artifact_reason(st, artifact: Any) -> None:
    """Surface ``artifact.reason`` (if present) as a small caption."""
    if artifact is None:
        return
    reason = getattr(artifact, "reason", None)
    if reason:
        st.caption(f"{_tr('Reason')}: {reason}")


def _render_throughput_section(
    st,
    scoring_throughput: pd.DataFrame,
    artifact: Any = None,
):
    """Render scoring throughput and latency section.

    Adds three iter9 fixes:

    * Per-chart ``Last update:`` caption so operators can see the data
      epoch (defect 3 — drift charts and throughput charts had a 19-month
      time-anchor split with no freshness disclosure).
    * Historical-fixture banner when the most recent throughput timestamp
      is more than 24 hours old, so a 2024 fixture is never presented as
      live telemetry.
    * Red SLO badge when the observed average error rate exceeds the
      0.1% SaaS SLO target (defect 6).

    iter13 G4: when ``artifact.is_real`` is explicitly False, the loader
    is telling us that ``results/scoring_throughput.csv`` is not present
    (or only a fallback fixture is loaded). In that case surface a hard
    error and skip the synthetic charts entirely — closes the iter12
    audit finding for Page 08 throughput KPIs.
    """
    if _artifact_marked_unreal(artifact):
        st.error(_tr(
            "Real scoring throughput missing — run pipeline to populate "
            "`results/scoring_throughput.csv`."
        ))
        _render_artifact_reason(st, artifact)
        return

    if scoring_throughput.empty:
        st.info(_tr("No scoring throughput data available."))
        return

    last_throughput_ts = _format_last_update(
        scoring_throughput.get("timestamp"),
    )
    is_stale, stale_msg = _classify_throughput_freshness(scoring_throughput)
    if is_stale:
        st.warning(stale_msg)

    col_tp, col_lt = st.columns(2)

    with col_tp:
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(
            x=scoring_throughput["timestamp"],
            y=scoring_throughput["requests_per_minute"],
            mode="lines",
            name=_tr("Requests/min"),
            line=dict(color="#2ecc71", width=2),
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.1)",
        ))
        fig_tp.update_layout(
            title=_tr("Scoring Throughput"),
            xaxis_title=_tr("Time"),
            yaxis_title=_tr("Requests per Minute"),
        )
        st.plotly_chart(fig_tp, use_container_width=True)
        st.caption(f"{_tr('Last update')}: {last_throughput_ts}")

    with col_lt:
        fig_lt = go.Figure()
        fig_lt.add_trace(go.Scatter(
            x=scoring_throughput["timestamp"],
            y=scoring_throughput["avg_latency_ms"],
            mode="lines",
            name=_tr("Avg Latency (ms)"),
            line=dict(color="#e67e22", width=2),
        ))
        fig_lt.update_layout(
            title=_tr("Average Scoring Latency"),
            xaxis_title=_tr("Time"),
            yaxis_title=_tr("Latency (ms)"),
        )
        st.plotly_chart(fig_lt, use_container_width=True)
        st.caption(f"{_tr('Last update')}: {last_throughput_ts}")

    # Error rate
    if "error_rate" in scoring_throughput.columns:
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=scoring_throughput["timestamp"],
            y=scoring_throughput["error_rate"],
            mode="lines+markers",
            name=_tr("Error Rate"),
            line=dict(color="#e74c3c", width=2),
        ))
        fig_err.add_hline(
            y=ERROR_RATE_SLO_TARGET,
            line_dash="dash",
            line_color="#e74c3c",
            annotation_text=(
                f"SaaS SLO target ({ERROR_RATE_SLO_TARGET * 100:.1f}%)"
            ),
        )
        fig_err.update_layout(
            title=_tr("Scoring Error Rate"),
            xaxis_title=_tr("Time"),
            yaxis_title=_tr("Error Rate"),
            yaxis_tickformat=".2%",
        )
        st.plotly_chart(fig_err, use_container_width=True)
        st.caption(f"{_tr('Last update')}: {last_throughput_ts}")

    # Throughput summary metrics
    st.markdown(f"#### {_tr('Throughput Summary')}")
    tp_col1, tp_col2, tp_col3, tp_col4 = st.columns(4)
    tp_col1.metric(
        _tr("Avg Requests/min"),
        f"{scoring_throughput['requests_per_minute'].mean():.1f}",
    )
    tp_col2.metric(
        _tr("Peak Requests/min"),
        f"{scoring_throughput['requests_per_minute'].max():.1f}",
    )
    tp_col3.metric(
        _tr("Avg Latency"),
        f"{scoring_throughput['avg_latency_ms'].mean():.1f} ms",
    )
    avg_error_rate = (
        float(scoring_throughput["error_rate"].mean())
        if "error_rate" in scoring_throughput.columns
        else float("nan")
    )
    tp_col4.metric(
        _tr("Avg Error Rate"),
        f"{avg_error_rate:.4f}" if pd.notna(avg_error_rate) else "—",
    )

    # Defect 6 — SaaS SLO badge for error rate. Surface a red callout when
    # average error rate breaches the 0.1% target.
    if pd.notna(avg_error_rate):
        if avg_error_rate > ERROR_RATE_SLO_TARGET:
            multiplier = avg_error_rate / ERROR_RATE_SLO_TARGET
            st.error(
                f"{_tr('SLO BREACH — error rate')} {avg_error_rate * 100:.2f}% "
                f"{_tr('is')} {multiplier:.1f}× "
                f"{_tr('the SaaS SLO target of')} "
                f"{ERROR_RATE_SLO_TARGET * 100:.1f}%. "
                f"{_tr('Page on-call and open an incident.')}"
            )
        else:
            st.success(
                f"{_tr('Error rate')} {avg_error_rate * 100:.2f}% "
                f"{_tr('is within the')} "
                f"{ERROR_RATE_SLO_TARGET * 100:.1f}% "
                f"{_tr('SaaS SLO target.')}"
            )


def _classify_throughput_freshness(
    scoring_throughput: pd.DataFrame,
) -> tuple:
    """Return (is_stale, message) for the throughput timestamp column.

    Stale = most recent timestamp older than 24 hours. The banner makes
    explicit when a historical fixture (e.g. Oct 2024) is on screen so it
    is not mistaken for live telemetry.
    """
    if "timestamp" not in scoring_throughput.columns:
        return False, ""
    try:
        ts = pd.to_datetime(
            scoring_throughput["timestamp"], errors="coerce",
        ).dropna()
        if ts.empty:
            return False, ""
        latest = ts.max()
        now = pd.Timestamp(datetime.utcnow())
        # Both naive — compare directly; tz-aware values are coerced.
        try:
            age = now - latest.tz_localize(None) if latest.tzinfo else now - latest
        except Exception:
            age = now - latest
        age_hours = age.total_seconds() / 3600
        if age_hours > 24:
            return True, (
                f"{_tr('Historical')} ({latest.strftime('%b %Y')}) — "
                f"{_tr('replace with live telemetry before production. Latest sample is')} "
                f"{age_hours / 24:.0f} {_tr('day(s) old.')}"
            )
    except Exception:
        return False, ""
    return False, ""


def _render_config_section(st, config: Dict):
    """Render monitoring configuration section."""
    mon_cfg = config.get("monitoring", {})
    drift_cfg = config.get("drift_detection", {})
    ks_cfg = config.get("ks_drift_detection", {})

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.markdown(f"**{_tr('Drift Detection (PSI)')}**")
        st.json({
            "n_bins": drift_cfg.get("n_bins", 10),
            "binning_strategy": drift_cfg.get(
                "binning_strategy", "quantile",
            ),
            "yellow_threshold": drift_cfg.get(
                "yellow_threshold", 0.10,
            ),
            "red_threshold": drift_cfg.get("red_threshold", 0.25),
        })
    with col_cfg2:
        st.markdown(f"**{_tr('Drift Detection (KS)')}**")
        st.json({
            "warning_threshold": ks_cfg.get(
                "warning_threshold", 0.05,
            ),
            "drift_threshold": ks_cfg.get("drift_threshold", 0.01),
        })
        st.markdown(f"**{_tr('Monitoring Settings')}**")
        st.json({
            "alert_on_yellow": mon_cfg.get("alert_on_yellow", False),
            "alert_on_red": mon_cfg.get("alert_on_red", True),
            "log_to_mlflow": mon_cfg.get("log_to_mlflow", True),
        })
