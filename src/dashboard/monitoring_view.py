"""
Model Monitoring & Survival Analysis Dashboard View.

Provides the render_model_monitoring function for the Streamlit dashboard,
showing drift detection, model performance metrics over time, scoring
throughput/latency monitoring, and survival curve quick reference.

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def render_model_monitoring(st_module, config: Dict, data_loader=None):
    """Render model monitoring page with drift detection and survival analysis.

    Shows:
    - Drift detection overview with alert status timeline
    - Per-feature PSI/KS drift metrics over time
    - Model performance metrics tracked over time
    - Scoring throughput and latency monitoring
    - Survival curve overlays for quick visual reference

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("Model Monitoring & Survival Analysis")

    if data_loader is None:
        from src.dashboard.app import get_data_loader
        data_loader = get_data_loader(config)

    # -----------------------------------------------------------------
    # Load all monitoring data
    # -----------------------------------------------------------------
    drift_history = data_loader.load_drift_history()
    model_metrics = data_loader.load_model_metrics()
    scoring_throughput = data_loader.load_scoring_throughput()
    survival_curves = data_loader.load_survival_curves()
    survival_data = data_loader.load_survival_data()
    performance_history = _load_performance_history(data_loader)

    # =================================================================
    # Section 1: Drift Detection Overview
    # =================================================================
    st.subheader("Drift Detection Overview")

    if drift_history.empty:
        st.warning("No drift detection history available.")
    else:
        _render_drift_section(st, config, drift_history)

    # =================================================================
    # Section 2: Model Performance Metrics Over Time
    # =================================================================
    st.markdown("---")
    st.subheader("Model Performance Metrics Over Time")

    _render_performance_section(st, model_metrics, performance_history)

    # =================================================================
    # Section 3: Scoring Throughput & Latency
    # =================================================================
    st.markdown("---")
    st.subheader("Scoring Throughput & Latency")

    _render_throughput_section(st, scoring_throughput)

    # =================================================================
    # Section 4: Survival Curves Quick Reference
    # =================================================================
    st.markdown("---")
    st.subheader("Survival Curves (Quick Reference)")

    _render_survival_section(st, survival_curves, survival_data)

    # =================================================================
    # Section 5: Monitoring Configuration
    # =================================================================
    st.markdown("---")
    st.subheader("Monitoring Configuration")

    _render_config_section(st, config)


# =========================================================================
# Internal rendering helpers
# =========================================================================


def _render_drift_section(st, config: Dict, drift_history: pd.DataFrame):
    """Render drift detection section with KPI cards, timeline, and charts."""
    # KPI cards for drift status
    kd1, kd2, kd3, kd4 = st.columns(4)
    latest = drift_history.iloc[-1]
    total_checks = len(drift_history)
    red_count = int((drift_history["alert_level"] == "red").sum())
    yellow_count = int((drift_history["alert_level"] == "yellow").sum())

    kd1.metric("Total Checks", f"{total_checks}")
    kd2.metric("Current Status", latest["alert_level"].upper())
    kd3.metric("Red Alerts", f"{red_count}")
    kd4.metric("Yellow Alerts", f"{yellow_count}")

    # Alert timeline
    st.markdown("#### Drift Alert Timeline")
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
        title="Drift Alert Timeline",
        xaxis_title="Date",
        yaxis_title="Number of Drifted Features",
        hovermode="x unified",
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # PSI and KS over time
    st.markdown("#### PSI & KS Statistics Over Time")
    col_psi, col_ks = st.columns(2)

    with col_psi:
        fig_psi = go.Figure()
        fig_psi.add_trace(go.Scatter(
            x=drift_history["timestamp"],
            y=drift_history["psi_mean"],
            mode="lines+markers",
            name="Mean PSI",
            line=dict(color="#3498db", width=2),
        ))
        drift_cfg = config.get("drift_detection", {})
        psi_yellow = drift_cfg.get("yellow_threshold", 0.10)
        psi_red = drift_cfg.get("red_threshold", 0.25)
        fig_psi.add_hline(
            y=psi_yellow, line_dash="dash", line_color="#f39c12",
            annotation_text=f"Yellow ({psi_yellow})",
        )
        fig_psi.add_hline(
            y=psi_red, line_dash="dash", line_color="#e74c3c",
            annotation_text=f"Red ({psi_red})",
        )
        fig_psi.update_layout(
            title="Mean PSI Over Time",
            xaxis_title="Date",
            yaxis_title="PSI Value",
        )
        st.plotly_chart(fig_psi, use_container_width=True)

    with col_ks:
        fig_ks = go.Figure()
        fig_ks.add_trace(go.Scatter(
            x=drift_history["timestamp"],
            y=drift_history["ks_mean"],
            mode="lines+markers",
            name="Mean KS Statistic",
            line=dict(color="#9b59b6", width=2),
        ))
        ks_cfg = config.get("ks_drift_detection", {})
        ks_warn = ks_cfg.get("warning_threshold", 0.05)
        fig_ks.add_hline(
            y=ks_warn, line_dash="dash", line_color="#f39c12",
            annotation_text=f"Warning ({ks_warn})",
        )
        fig_ks.update_layout(
            title="Mean KS Statistic Over Time",
            xaxis_title="Date",
            yaxis_title="KS Statistic",
        )
        st.plotly_chart(fig_ks, use_container_width=True)

    # Drift history table
    st.markdown("#### Drift Detection Log")
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


def _render_performance_section(
    st, model_metrics: Dict, mlflow_runs: pd.DataFrame,
):
    """Render model performance metrics section."""
    if not model_metrics and mlflow_runs.empty:
        st.warning("No model performance metrics available.")
        return

    if model_metrics:
        st.markdown("#### Current Model Performance")
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
            title="Model Performance Comparison",
            xaxis_title="Metric",
            yaxis_title="Score",
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
            f"Best model by AUC: **{best_model}** "
            f"(AUC = {best_auc:.4f})"
        )

    # MLflow run history
    if not mlflow_runs.empty:
        st.markdown("#### Training Run History")
        if "timestamp" in mlflow_runs.columns:
            fig_runs = go.Figure()
            for metric in ["auc", "f1_score"]:
                if metric in mlflow_runs.columns:
                    fig_runs.add_trace(go.Scatter(
                        x=mlflow_runs["timestamp"],
                        y=mlflow_runs[metric],
                        mode="lines+markers",
                        name=metric.upper(),
                        text=mlflow_runs.get("model_type", None),
                    ))
            fig_runs.update_layout(
                title="Model Metrics Across Training Runs",
                xaxis_title="Timestamp",
                yaxis_title="Score",
                hovermode="x unified",
            )
            st.plotly_chart(fig_runs, use_container_width=True)

        st.markdown("#### Run Details")
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


def _render_throughput_section(st, scoring_throughput: pd.DataFrame):
    """Render scoring throughput and latency section."""
    if scoring_throughput.empty:
        st.info("No scoring throughput data available.")
        return

    col_tp, col_lt = st.columns(2)

    with col_tp:
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(
            x=scoring_throughput["timestamp"],
            y=scoring_throughput["requests_per_minute"],
            mode="lines",
            name="Requests/min",
            line=dict(color="#2ecc71", width=2),
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.1)",
        ))
        fig_tp.update_layout(
            title="Scoring Throughput",
            xaxis_title="Time",
            yaxis_title="Requests per Minute",
        )
        st.plotly_chart(fig_tp, use_container_width=True)

    with col_lt:
        fig_lt = go.Figure()
        fig_lt.add_trace(go.Scatter(
            x=scoring_throughput["timestamp"],
            y=scoring_throughput["avg_latency_ms"],
            mode="lines",
            name="Avg Latency (ms)",
            line=dict(color="#e67e22", width=2),
        ))
        fig_lt.update_layout(
            title="Average Scoring Latency",
            xaxis_title="Time",
            yaxis_title="Latency (ms)",
        )
        st.plotly_chart(fig_lt, use_container_width=True)

    # Error rate
    if "error_rate" in scoring_throughput.columns:
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=scoring_throughput["timestamp"],
            y=scoring_throughput["error_rate"],
            mode="lines+markers",
            name="Error Rate",
            line=dict(color="#e74c3c", width=2),
        ))
        fig_err.update_layout(
            title="Scoring Error Rate",
            xaxis_title="Time",
            yaxis_title="Error Rate",
            yaxis_tickformat=".2%",
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # Throughput summary metrics
    st.markdown("#### Throughput Summary")
    tp_col1, tp_col2, tp_col3, tp_col4 = st.columns(4)
    tp_col1.metric(
        "Avg Requests/min",
        f"{scoring_throughput['requests_per_minute'].mean():.1f}",
    )
    tp_col2.metric(
        "Peak Requests/min",
        f"{scoring_throughput['requests_per_minute'].max():.1f}",
    )
    tp_col3.metric(
        "Avg Latency",
        f"{scoring_throughput['avg_latency_ms'].mean():.1f} ms",
    )
    tp_col4.metric(
        "Avg Error Rate",
        f"{scoring_throughput['error_rate'].mean():.4f}",
    )


def _render_survival_section(
    st, survival_curves: Dict, survival_data: pd.DataFrame,
):
    """Render survival curves quick reference section."""
    if survival_curves:
        segment_colors = {
            "vip_loyal": "#2ecc71",
            "regular_loyal": "#3498db",
            "bargain_hunter": "#e67e22",
            "explorer": "#9b59b6",
            "dormant": "#e74c3c",
            "new_customer": "#f39c12",
        }

        fig_surv = go.Figure()
        for seg_name, curve_data in survival_curves.items():
            timeline = curve_data.get("timeline", [])
            surv_prob = curve_data.get("survival_prob", [])
            color = segment_colors.get(seg_name, "#95a5a6")
            fig_surv.add_trace(go.Scatter(
                x=timeline, y=surv_prob,
                mode="lines",
                name=seg_name,
                line=dict(color=color, width=2),
            ))

        fig_surv.add_hline(
            y=0.5, line_dash="dash", line_color="gray",
            annotation_text="50% Survival",
        )
        fig_surv.update_layout(
            title="Kaplan-Meier Survival Curves by Segment",
            xaxis_title="Days",
            yaxis_title="Survival Probability",
            yaxis_range=[0, 1.05],
            hovermode="x unified",
        )
        st.plotly_chart(fig_surv, use_container_width=True)

        # Median survival summary
        survival_summary = []
        for seg_name, curve_data in survival_curves.items():
            median_surv = curve_data.get("median_survival_days")
            surv_prob_list = curve_data.get("survival_prob", [])
            final_surv = surv_prob_list[-1] if surv_prob_list else 0
            survival_summary.append({
                "Segment": seg_name,
                "Median Survival (days)": (
                    median_surv if median_surv else ">360"
                ),
                "Final Survival Prob": f"{final_surv:.4f}",
                "Risk Level": (
                    "Low" if final_surv > 0.7
                    else "Medium" if final_surv > 0.5
                    else "High" if final_surv > 0.3
                    else "Critical"
                ),
            })
        st.dataframe(
            pd.DataFrame(survival_summary),
            use_container_width=True,
        )
    elif not survival_data.empty:
        st.info(
            "Survival curve data not available. "
            "See the Survival Analysis page for detailed views."
        )
    else:
        st.warning("No survival data available.")


def _render_config_section(st, config: Dict):
    """Render monitoring configuration section."""
    mon_cfg = config.get("monitoring", {})
    drift_cfg = config.get("drift_detection", {})
    ks_cfg = config.get("ks_drift_detection", {})

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.markdown("**Drift Detection (PSI)**")
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
        st.markdown("**Drift Detection (KS)**")
        st.json({
            "warning_threshold": ks_cfg.get(
                "warning_threshold", 0.05,
            ),
            "drift_threshold": ks_cfg.get("drift_threshold", 0.01),
        })
        st.markdown("**Monitoring Settings**")
        st.json({
            "alert_on_yellow": mon_cfg.get("alert_on_yellow", False),
            "alert_on_red": mon_cfg.get("alert_on_red", True),
            "log_to_mlflow": mon_cfg.get("log_to_mlflow", True),
        })
