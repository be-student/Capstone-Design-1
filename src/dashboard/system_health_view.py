"""
System Overview & Health Dashboard View.

Provides a unified health dashboard showing:
- System service status (Redis, MLflow, Pipeline)
- Streaming pipeline status with Redis stream metrics
- MLflow experiment tracking integration and run history
- Overall system health indicators and alerts

All configurable parameters are sourced from config/simulator_config.yaml.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Service status constants
STATUS_HEALTHY = "healthy"
STATUS_DEGRADED = "degraded"
STATUS_DOWN = "down"

STATUS_COLORS = {
    STATUS_HEALTHY: "#2ecc71",
    STATUS_DEGRADED: "#f39c12",
    STATUS_DOWN: "#e74c3c",
}

STATUS_ICONS = {
    STATUS_HEALTHY: "✅",
    STATUS_DEGRADED: "⚠️",
    STATUS_DOWN: "❌",
}


def check_redis_health(config: Dict) -> Dict[str, Any]:
    """Check Redis streaming pipeline health.

    Args:
        config: Configuration dictionary with redis section.

    Returns:
        Dict with status, connected, stream_lengths, consumer_groups,
        memory_used, uptime, and error details.
    """
    redis_config = config.get("redis", {})
    result = {
        "status": STATUS_DOWN,
        "connected": False,
        "host": redis_config.get("host", "localhost"),
        "port": redis_config.get("port", 6379),
        "stream_lengths": {},
        "consumer_groups": {},
        "memory_used_mb": 0,
        "uptime_seconds": 0,
        "error": None,
    }

    try:
        import redis as redis_lib
        r = redis_lib.Redis(
            host=result["host"],
            port=result["port"],
            db=redis_config.get("db", 0),
            socket_connect_timeout=2,
        )
        r.ping()
        result["connected"] = True
        result["status"] = STATUS_HEALTHY

        # Stream lengths
        req_stream = redis_config.get("stream_name", "scoring_requests")
        resp_stream = redis_config.get("response_stream", "scoring_responses")
        for stream_name in [req_stream, resp_stream]:
            try:
                if r.exists(stream_name):
                    result["stream_lengths"][stream_name] = r.xlen(stream_name)
                else:
                    result["stream_lengths"][stream_name] = 0
            except Exception:
                result["stream_lengths"][stream_name] = 0

        # Server info
        try:
            info = r.info()
            result["memory_used_mb"] = round(
                info.get("used_memory", 0) / (1024 * 1024), 2
            )
            result["uptime_seconds"] = info.get("uptime_in_seconds", 0)
        except Exception:
            pass

        # Consumer group info
        for stream_name in [req_stream, resp_stream]:
            try:
                if r.exists(stream_name):
                    groups = r.xinfo_groups(stream_name)
                    result["consumer_groups"][stream_name] = [
                        {
                            "name": g.get("name", b"").decode()
                            if isinstance(g.get("name", b""), bytes)
                            else str(g.get("name", "")),
                            "consumers": g.get("consumers", 0),
                            "pending": g.get("pending", 0),
                        }
                        for g in groups
                    ]
            except Exception:
                result["consumer_groups"][stream_name] = []

    except ImportError:
        result["error"] = "redis package not installed"
    except Exception as e:
        result["error"] = str(e)

    return result


def check_mlflow_health(config: Dict) -> Dict[str, Any]:
    """Check MLflow tracking server health.

    Args:
        config: Configuration dictionary with mlflow section.

    Returns:
        Dict with status, connected, tracking_uri, experiments,
        recent_runs, and error details.
    """
    mlflow_config = config.get("mlflow", {})
    tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow/mlflow.db")

    result = {
        "status": STATUS_DOWN,
        "connected": False,
        "tracking_uri": tracking_uri,
        "experiment_name": mlflow_config.get("experiment_name", "churn_prediction"),
        "experiments": [],
        "total_runs": 0,
        "recent_runs": [],
        "best_run": None,
        "error": None,
    }

    parsed_uri = urlparse(tracking_uri)

    if parsed_uri.scheme == "sqlite":
        db_path = Path(parsed_uri.path)
        result["connected"] = db_path.exists()
        result["status"] = STATUS_DEGRADED
        if db_path.exists():
            result["status"] = STATUS_HEALTHY
        else:
            result["error"] = f"MLflow tracking DB not found: {db_path}"
        return result

    if parsed_uri.scheme in {"http", "https"}:
        health_url = tracking_uri.rstrip("/") + "/health"
        try:
            import requests

            response = requests.get(health_url, timeout=1)
            result["connected"] = response.ok
            result["status"] = STATUS_HEALTHY if response.ok else STATUS_DEGRADED
            if not response.ok:
                result["error"] = f"MLflow health endpoint returned {response.status_code}"
        except ImportError:
            result["status"] = STATUS_DEGRADED
            result["error"] = "requests package not installed"
        except Exception as e:
            result["status"] = STATUS_DEGRADED
            result["error"] = f"MLflow server not reachable: {e}"
        return result

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        experiments = mlflow.search_experiments()

        if experiments:
            result["connected"] = True
            result["status"] = STATUS_HEALTHY
            result["experiments"] = [
                {
                    "name": e.name,
                    "id": e.experiment_id,
                    "lifecycle": e.lifecycle_stage,
                }
                for e in experiments
            ]

            # Get recent runs
            try:
                runs = mlflow.search_runs(
                    experiment_names=[result["experiment_name"]],
                    max_results=10,
                    order_by=["start_time DESC"],
                )
                if not runs.empty:
                    result["total_runs"] = len(runs)
                    result["recent_runs"] = runs.head(5).to_dict("records")

                    # Find best run by AUC
                    auc_col = None
                    for col in runs.columns:
                        if "auc" in col.lower():
                            auc_col = col
                            break
                    if auc_col and not runs[auc_col].isna().all():
                        best_idx = runs[auc_col].idxmax()
                        result["best_run"] = {
                            "run_id": runs.loc[best_idx, "run_id"],
                            "auc": float(runs.loc[best_idx, auc_col]),
                        }
            except Exception:
                pass
        else:
            result["connected"] = True
            result["status"] = STATUS_DEGRADED
            result["error"] = "No experiments found"

    except ImportError:
        result["error"] = "mlflow package not installed"
    except Exception as e:
        result["error"] = str(e)
        # Try artifacts-based fallback
        result["status"] = STATUS_DEGRADED
        result["error"] = f"MLflow server not reachable: {e}"

    return result


def check_pipeline_health(config: Dict) -> Dict[str, Any]:
    """Check ML pipeline health based on artifacts and state.

    Args:
        config: Configuration dictionary.

    Returns:
        Dict with status, models_available, last_training,
        and artifact details.
    """
    from pathlib import Path

    configured_artifacts_dir = Path(
        config.get("dashboard", {}).get("artifacts_dir", "data/artifacts")
    )
    artifact_candidates = [configured_artifacts_dir, Path("results")]
    artifacts_dir = next(
        (candidate for candidate in artifact_candidates if candidate.exists()),
        configured_artifacts_dir,
    )
    models_dir = Path("models")

    result = {
        "status": STATUS_DOWN,
        "artifacts_dir": str(artifacts_dir),
        "artifact_sources": [str(path) for path in artifact_candidates],
        "artifacts_exist": artifacts_dir.exists(),
        "artifact_count": 0,
        "models_available": [],
        "last_modified": None,
        "error": None,
    }

    try:
        if artifacts_dir.exists():
            artifact_files = list(artifacts_dir.glob("*"))
            result["artifact_count"] = len(artifact_files)

            if artifact_files:
                latest = max(artifact_files, key=lambda f: f.stat().st_mtime)
                result["last_modified"] = datetime.fromtimestamp(
                    latest.stat().st_mtime
                ).isoformat()

        if models_dir.exists():
            model_files = list(models_dir.glob("*.joblib")) + list(
                models_dir.glob("*.pt")
            )
            result["models_available"] = [f.name for f in model_files]

        if result["artifact_count"] > 0 or result["models_available"]:
            result["status"] = STATUS_HEALTHY
        elif result["artifacts_exist"]:
            result["status"] = STATUS_DEGRADED
            result["error"] = "Artifacts directory exists but is empty"
        else:
            result["error"] = "No artifacts or models found"

    except Exception as e:
        result["error"] = str(e)

    return result


def get_system_health_summary(config: Dict) -> Dict[str, Any]:
    """Get comprehensive system health summary.

    Args:
        config: Configuration dictionary.

    Returns:
        Dict with service statuses, overall health, and timestamp.
    """
    redis_health = check_redis_health(config)
    mlflow_health = check_mlflow_health(config)
    pipeline_health = check_pipeline_health(config)

    statuses = [
        redis_health["status"],
        mlflow_health["status"],
        pipeline_health["status"],
    ]

    if all(s == STATUS_HEALTHY for s in statuses):
        overall = STATUS_HEALTHY
    elif any(s == STATUS_DOWN for s in statuses):
        overall = STATUS_DEGRADED
    else:
        overall = STATUS_DEGRADED

    return {
        "overall_status": overall,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_health,
            "mlflow": mlflow_health,
            "pipeline": pipeline_health,
        },
    }


def render_system_health(st_module, config: Dict, data_loader=None):
    """Render system overview and health dashboard.

    Shows:
    - Overall system health status
    - Service health indicators (Redis, MLflow, Pipeline)
    - Streaming pipeline status with stream metrics
    - MLflow experiment tracking integration
    - Drift detection summary
    - Recent activity log

    Args:
        st_module: Streamlit module reference.
        config: Configuration dictionary.
        data_loader: Optional DashboardDataLoader instance.
    """
    st = st_module
    st.header("System Overview & Health")
    st.markdown(
        "Real-time health monitoring for all system components: "
        "streaming pipeline, ML tracking, and model serving."
    )

    if data_loader is None:
        from src.dashboard.app import get_data_loader
        data_loader = get_data_loader(config)

    # Gather health data
    health = get_system_health_summary(config)

    # ==================================================================
    # Section 1: Overall Health Status
    # ==================================================================
    _render_overall_health(st, health)

    # ==================================================================
    # Section 2: Service Status Cards
    # ==================================================================
    st.markdown("---")
    _render_service_cards(st, health)

    # ==================================================================
    # Section 3: Streaming Pipeline Status
    # ==================================================================
    st.markdown("---")
    _render_streaming_status(st, config, health, data_loader)

    # ==================================================================
    # Section 4: MLflow Experiment Tracking
    # ==================================================================
    st.markdown("---")
    _render_mlflow_tracking(st, config, health, data_loader)

    # ==================================================================
    # Section 5: Model Health & Drift Summary
    # ==================================================================
    st.markdown("---")
    _render_model_health(st, config, data_loader)

    # ==================================================================
    # Section 6: System Configuration
    # ==================================================================
    st.markdown("---")
    _render_system_config(st, config)


# =========================================================================
# Internal rendering helpers
# =========================================================================


def _render_overall_health(st, health: Dict):
    """Render overall system health banner."""
    overall = health["overall_status"]
    icon = STATUS_ICONS.get(overall, "❓")
    color = STATUS_COLORS.get(overall, "#95a5a6")
    timestamp = health.get("timestamp", "N/A")

    status_text = {
        STATUS_HEALTHY: "All Systems Operational",
        STATUS_DEGRADED: "Some Services Degraded",
        STATUS_DOWN: "System Issues Detected",
    }

    st.markdown(
        f"### {icon} System Status: **{status_text.get(overall, 'Unknown')}**"
    )
    st.caption(f"Last checked: {timestamp}")

    # Service count summary
    services = health.get("services", {})
    healthy_count = sum(
        1 for s in services.values() if s.get("status") == STATUS_HEALTHY
    )
    total_count = len(services)

    st.progress(
        healthy_count / max(total_count, 1),
        text=f"{healthy_count}/{total_count} services healthy",
    )


def _render_service_cards(st, health: Dict):
    """Render individual service health cards."""
    st.subheader("Service Health")

    services = health.get("services", {})
    cols = st.columns(3)

    service_display = [
        ("redis", "Redis Streaming", "Handles real-time scoring requests"),
        ("mlflow", "MLflow Tracking", "Experiment tracking & model registry"),
        ("pipeline", "ML Pipeline", "Model training & artifact storage"),
    ]

    for i, (key, name, desc) in enumerate(service_display):
        svc = services.get(key, {})
        status = svc.get("status", STATUS_DOWN)
        icon = STATUS_ICONS.get(status, "❓")
        error = svc.get("error")

        with cols[i]:
            if status == STATUS_HEALTHY:
                st.success(f"{icon} **{name}**")
            elif status == STATUS_DEGRADED:
                st.warning(f"{icon} **{name}**")
            else:
                st.error(f"{icon} **{name}**")

            st.caption(desc)

            if error:
                st.caption(f"⚠ {error}")

            # Service-specific details
            if key == "redis":
                st.metric(
                    "Connected",
                    "Yes" if svc.get("connected") else "No",
                )
                streams = svc.get("stream_lengths", {})
                for stream_name, length in streams.items():
                    short_name = stream_name.split("_")[-1]
                    st.metric(f"Stream ({short_name})", f"{length:,}")

            elif key == "mlflow":
                st.metric(
                    "Connected",
                    "Yes" if svc.get("connected") else "No",
                )
                exp_count = len(svc.get("experiments", []))
                st.metric("Experiments", exp_count)

            elif key == "pipeline":
                st.metric("Artifacts", svc.get("artifact_count", 0))
                models = svc.get("models_available", [])
                st.metric("Models", len(models))


def _render_streaming_status(
    st, config: Dict, health: Dict, data_loader,
):
    """Render streaming pipeline status section."""
    st.subheader("Streaming Pipeline Status")

    redis_health = health.get("services", {}).get("redis", {})
    redis_config = config.get("redis", {})

    # Stream configuration
    col_config, col_metrics = st.columns(2)

    with col_config:
        st.markdown("#### Configuration")
        st.json({
            "host": redis_config.get("host", "localhost"),
            "port": redis_config.get("port", 6379),
            "request_stream": redis_config.get("stream_name", "scoring_requests"),
            "response_stream": redis_config.get("response_stream", "scoring_responses"),
            "consumer_group": redis_config.get("consumer_group", "scoring_consumers"),
            "batch_size": redis_config.get("consumer_batch_size", 10),
            "max_stream_length": redis_config.get("stream_maxlen", 10000),
            "cache_ttl_seconds": redis_config.get("cache_ttl_seconds", 3600),
        })

    with col_metrics:
        st.markdown("#### Stream Metrics")
        streams = redis_health.get("stream_lengths", {})

        if streams:
            stream_df = pd.DataFrame([
                {"Stream": name, "Length": length}
                for name, length in streams.items()
            ])
            fig_streams = px.bar(
                stream_df, x="Stream", y="Length",
                title="Stream Lengths",
                color="Stream",
                text="Length",
            )
            fig_streams.update_traces(textposition="outside")
            st.plotly_chart(fig_streams, use_container_width=True)
        else:
            st.info(
                "Redis not connected. Stream metrics unavailable. "
                "Start Redis with `docker-compose up redis`."
            )

    # Consumer group details
    consumer_groups = redis_health.get("consumer_groups", {})
    if any(consumer_groups.values()):
        st.markdown("#### Consumer Groups")
        group_rows = []
        for stream, groups in consumer_groups.items():
            for g in groups:
                group_rows.append({
                    "Stream": stream,
                    "Group": g.get("name", "N/A"),
                    "Consumers": g.get("consumers", 0),
                    "Pending Messages": g.get("pending", 0),
                })
        if group_rows:
            st.dataframe(pd.DataFrame(group_rows), use_container_width=True)

    # Throughput chart from data loader
    throughput = data_loader.load_scoring_throughput()
    if not throughput.empty:
        st.markdown("#### Scoring Throughput (24h)")
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(
            x=throughput["timestamp"],
            y=throughput["requests_per_minute"],
            mode="lines",
            name="Requests/min",
            line=dict(color="#2ecc71", width=2),
            fill="tozeroy",
            fillcolor="rgba(46,204,113,0.1)",
        ))
        fig_tp.update_layout(
            xaxis_title="Time",
            yaxis_title="Requests per Minute",
            height=300,
        )
        st.plotly_chart(fig_tp, use_container_width=True)

        # Throughput summary
        tp1, tp2, tp3 = st.columns(3)
        tp1.metric(
            "Avg Throughput",
            f"{throughput['requests_per_minute'].mean():.1f} req/min",
        )
        tp2.metric(
            "Avg Latency",
            f"{throughput['avg_latency_ms'].mean():.1f} ms",
        )
        if "error_rate" in throughput.columns:
            tp3.metric(
                "Avg Error Rate",
                f"{throughput['error_rate'].mean():.4f}",
            )


def _render_mlflow_tracking(
    st, config: Dict, health: Dict, data_loader,
):
    """Render MLflow experiment tracking integration section."""
    st.subheader("MLflow Experiment Tracking")

    mlflow_health = health.get("services", {}).get("mlflow", {})
    mlflow_config = config.get("mlflow", {})

    # Connection status and config
    col_status, col_info = st.columns(2)

    with col_status:
        if mlflow_health.get("connected"):
            st.success("Connected to MLflow tracking server")
        else:
            st.warning(
                "MLflow server not available. "
                "Showing cached data from artifacts."
            )
        st.json({
            "tracking_uri": mlflow_health.get("tracking_uri", "N/A"),
            "experiment_name": mlflow_health.get("experiment_name", "N/A"),
            "log_models": mlflow_config.get("log_models", True),
            "log_artifacts": mlflow_config.get("log_artifacts", True),
        })

    with col_info:
        # Experiment list
        experiments = mlflow_health.get("experiments", [])
        if experiments:
            st.markdown("#### Registered Experiments")
            st.dataframe(
                pd.DataFrame(experiments),
                use_container_width=True,
            )
        else:
            st.info("No experiments found on MLflow server.")

    # MLflow run history from data loader
    mlflow_runs = data_loader.load_mlflow_runs()

    if not mlflow_runs.empty:
        st.markdown("#### Experiment Run History")

        # KPI cards from runs
        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Total Runs", len(mlflow_runs))

        if "auc" in mlflow_runs.columns:
            best_auc = mlflow_runs["auc"].max()
            kc2.metric("Best AUC", f"{best_auc:.4f}")

        if "model_type" in mlflow_runs.columns:
            best_model = mlflow_runs.loc[
                mlflow_runs["auc"].idxmax(), "model_type"
            ] if "auc" in mlflow_runs.columns else "N/A"
            kc3.metric("Best Model", best_model)

        if "training_time_s" in mlflow_runs.columns:
            total_time = mlflow_runs["training_time_s"].sum()
            kc4.metric("Total Train Time", f"{total_time:.0f}s")

        # Performance comparison chart
        col_perf, col_time = st.columns(2)

        with col_perf:
            if "model_type" in mlflow_runs.columns and "auc" in mlflow_runs.columns:
                fig_perf = px.bar(
                    mlflow_runs.sort_values("auc", ascending=False),
                    x="model_type",
                    y="auc",
                    color="model_type",
                    title="AUC by Model Type",
                    text=mlflow_runs.sort_values("auc", ascending=False)["auc"].apply(
                        lambda v: f"{v:.4f}"
                    ),
                )
                fig_perf.update_traces(textposition="outside")
                fig_perf.update_layout(
                    yaxis_range=[0, 1],
                    showlegend=False,
                )
                fig_perf.add_hline(
                    y=0.78, line_dash="dash", line_color="red",
                    annotation_text="Threshold (0.78)",
                )
                st.plotly_chart(fig_perf, use_container_width=True)

        with col_time:
            if "timestamp" in mlflow_runs.columns and "auc" in mlflow_runs.columns:
                fig_timeline = px.scatter(
                    mlflow_runs,
                    x="timestamp",
                    y="auc",
                    color="model_type" if "model_type" in mlflow_runs.columns else None,
                    size="training_time_s" if "training_time_s" in mlflow_runs.columns else None,
                    title="Model Performance Over Time",
                )
                st.plotly_chart(fig_timeline, use_container_width=True)

        # Run details table
        st.markdown("#### Run Details")
        display_cols = [
            c for c in [
                "run_id", "model_type", "auc", "precision", "recall",
                "f1_score", "training_time_s", "timestamp",
            ]
            if c in mlflow_runs.columns
        ]
        st.dataframe(mlflow_runs[display_cols], use_container_width=True)


def _render_model_health(st, config: Dict, data_loader):
    """Render model health and drift detection summary."""
    st.subheader("Model Health & Drift Detection")

    drift_history = data_loader.load_drift_history()
    model_metrics = data_loader.load_model_metrics()

    col_drift, col_perf = st.columns(2)

    with col_drift:
        if not drift_history.empty:
            latest = drift_history.iloc[-1]
            alert_level = latest.get("alert_level", "unknown")

            if alert_level == "green":
                st.success(f"Current Drift Status: **{alert_level.upper()}**")
            elif alert_level == "yellow":
                st.warning(f"Current Drift Status: **{alert_level.upper()}**")
            else:
                st.error(f"Current Drift Status: **{alert_level.upper()}**")

            # Recent drift trend
            fig_drift = go.Figure()
            fig_drift.add_trace(go.Scatter(
                x=drift_history["timestamp"],
                y=drift_history["psi_mean"],
                mode="lines+markers",
                name="Mean PSI",
                line=dict(color="#3498db", width=2),
            ))

            drift_cfg = config.get("drift_detection", {})
            fig_drift.add_hline(
                y=drift_cfg.get("yellow_threshold", 0.10),
                line_dash="dash", line_color="#f39c12",
            )
            fig_drift.add_hline(
                y=drift_cfg.get("red_threshold", 0.25),
                line_dash="dash", line_color="#e74c3c",
            )
            fig_drift.update_layout(
                title="PSI Drift Trend",
                xaxis_title="Date",
                yaxis_title="Mean PSI",
                height=300,
            )
            st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.info("No drift detection history available.")

    with col_perf:
        if model_metrics:
            st.markdown("#### Current Model Performance")
            metrics_rows = []
            for model_name, metrics in model_metrics.items():
                metrics_rows.append({
                    "Model": model_name,
                    "AUC": metrics.get("auc", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1": metrics.get("f1_score", 0),
                })
            metrics_df = pd.DataFrame(metrics_rows)
            st.dataframe(metrics_df, use_container_width=True)

            # Best model highlight
            best = max(
                model_metrics.keys(),
                key=lambda m: model_metrics[m].get("auc", 0),
            )
            st.info(
                f"Best model: **{best}** "
                f"(AUC = {model_metrics[best].get('auc', 0):.4f})"
            )
        else:
            st.info("No model performance metrics available.")


def _render_system_config(st, config: Dict):
    """Render system configuration summary."""
    st.subheader("System Configuration")

    with st.expander("Full Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Simulation**")
            sim = config.get("simulation", {})
            st.json({
                "n_customers": sim.get("n_customers", 20000),
                "horizon_months": sim.get("horizon_months", 12),
                "random_seed": sim.get("random_seed", 42),
            })

            st.markdown("**Budget**")
            budget = config.get("budget", {})
            st.json({
                "total_krw": budget.get("total_krw", 50000000),
                "currency": budget.get("currency", "KRW"),
            })

        with col2:
            st.markdown("**ML Model**")
            ml = config.get("ml_model", {})
            st.json({
                "n_splits": ml.get("n_splits", 5),
                "early_stopping_rounds": ml.get("early_stopping_rounds", 10),
            })

            st.markdown("**DL Model**")
            dl = config.get("dl_model", {})
            st.json({
                "architecture": dl.get("architecture", "transformer"),
                "hidden_size": dl.get("hidden_size", 64),
                "num_layers": dl.get("num_layers", 2),
                "sequence_window": dl.get("sequence_window", 6),
            })
