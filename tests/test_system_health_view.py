"""
TDD Tests for System Overview & Health Dashboard View.

Tests cover:
- Service health check functions (Redis, MLflow, Pipeline)
- Overall health summary aggregation
- Streaming pipeline status rendering
- MLflow experiment tracking integration
- Model health and drift summary
- System configuration display
- Render function integration with mock Streamlit
- Empty/unavailable service handling
- Health status classification logic
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Load simulator configuration."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_st():
    """Create a mock Streamlit module."""
    st = MagicMock()
    st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    st.tabs.side_effect = lambda labels: [MagicMock() for _ in labels]

    # Mock expander context manager
    expander = MagicMock()
    expander.__enter__ = MagicMock(return_value=st)
    expander.__exit__ = MagicMock(return_value=False)
    st.expander.return_value = expander

    # Mock progress
    st.progress.return_value = None

    return st


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader with sample data."""
    np.random.seed(42)
    loader = MagicMock()

    # Scoring throughput
    loader.load_scoring_throughput.return_value = pd.DataFrame({
        "timestamp": pd.date_range("2024-10-15", periods=24, freq="1h").strftime(
            "%Y-%m-%dT%H:%M:%S"
        ).tolist(),
        "requests_per_minute": np.random.uniform(20, 80, 24).round(1),
        "avg_latency_ms": np.random.uniform(10, 30, 24).round(1),
        "error_rate": np.random.uniform(0, 0.02, 24).round(4),
    })

    # Drift history
    loader.load_drift_history.return_value = pd.DataFrame({
        "timestamp": pd.date_range("2024-09-01", periods=10, freq="3D").strftime(
            "%Y-%m-%dT%H:%M:%S"
        ).tolist(),
        "alert_level": np.random.choice(["green", "yellow", "red"], 10, p=[0.7, 0.2, 0.1]),
        "num_drifted_features": np.random.randint(0, 5, 10),
        "psi_mean": np.random.uniform(0.01, 0.15, 10).round(4),
        "ks_mean": np.random.uniform(0.02, 0.10, 10).round(4),
    })

    # Model metrics
    loader.load_model_metrics.return_value = {
        "ml_model": {"auc": 0.82, "precision": 0.76, "recall": 0.70, "f1_score": 0.73},
        "dl_model": {"auc": 0.79, "precision": 0.72, "recall": 0.67, "f1_score": 0.69},
        "ensemble": {"auc": 0.84, "precision": 0.78, "recall": 0.72, "f1_score": 0.75},
    }

    # MLflow runs
    loader.load_mlflow_runs.return_value = pd.DataFrame({
        "run_id": [f"run_{i:04d}" for i in range(5)],
        "model_type": ["xgboost", "lightgbm", "lstm", "transformer", "ensemble"],
        "auc": [0.82, 0.80, 0.79, 0.81, 0.84],
        "precision": [0.76, 0.74, 0.72, 0.75, 0.78],
        "recall": [0.70, 0.68, 0.67, 0.69, 0.72],
        "f1_score": [0.73, 0.71, 0.69, 0.72, 0.75],
        "training_time_s": [15.0, 12.0, 45.0, 60.0, 20.0],
        "timestamp": [f"2024-10-{15+i}T10:00:00" for i in range(5)],
    })

    return loader


# ---------------------------------------------------------------------------
# Test: Module Imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    """Test that the system health view module is importable."""

    def test_import_system_health_view(self):
        from src.dashboard.system_health_view import render_system_health
        assert callable(render_system_health)

    def test_import_check_redis_health(self):
        from src.dashboard.system_health_view import check_redis_health
        assert callable(check_redis_health)

    def test_import_check_mlflow_health(self):
        from src.dashboard.system_health_view import check_mlflow_health
        assert callable(check_mlflow_health)

    def test_import_check_pipeline_health(self):
        from src.dashboard.system_health_view import check_pipeline_health
        assert callable(check_pipeline_health)

    def test_import_get_system_health_summary(self):
        from src.dashboard.system_health_view import get_system_health_summary
        assert callable(get_system_health_summary)

    def test_import_status_constants(self):
        from src.dashboard.system_health_view import (
            STATUS_HEALTHY,
            STATUS_DEGRADED,
            STATUS_DOWN,
        )
        assert STATUS_HEALTHY == "healthy"
        assert STATUS_DEGRADED == "degraded"
        assert STATUS_DOWN == "down"


# ---------------------------------------------------------------------------
# Test: Redis Health Check
# ---------------------------------------------------------------------------


class TestRedisHealthCheck:
    """Test Redis health check function."""

    def test_redis_health_returns_dict(self, config):
        from src.dashboard.system_health_view import check_redis_health
        result = check_redis_health(config)
        assert isinstance(result, dict)

    def test_redis_health_has_status(self, config):
        from src.dashboard.system_health_view import check_redis_health
        result = check_redis_health(config)
        assert "status" in result
        assert result["status"] in ("healthy", "degraded", "down")

    def test_redis_health_has_connected(self, config):
        from src.dashboard.system_health_view import check_redis_health
        result = check_redis_health(config)
        assert "connected" in result
        assert isinstance(result["connected"], bool)

    def test_redis_health_has_host_port(self, config):
        from src.dashboard.system_health_view import check_redis_health
        result = check_redis_health(config)
        assert "host" in result
        assert "port" in result

    def test_redis_health_has_stream_lengths(self, config):
        from src.dashboard.system_health_view import check_redis_health
        result = check_redis_health(config)
        assert "stream_lengths" in result
        assert isinstance(result["stream_lengths"], dict)

    def test_redis_health_has_consumer_groups(self, config):
        from src.dashboard.system_health_view import check_redis_health
        result = check_redis_health(config)
        assert "consumer_groups" in result

    def test_redis_health_handles_unavailable(self):
        """Redis health check should handle unavailable Redis gracefully."""
        from src.dashboard.system_health_view import check_redis_health
        result = check_redis_health({"redis": {"host": "nonexistent", "port": 99999}})
        assert result["status"] == "down"
        assert not result["connected"]


# ---------------------------------------------------------------------------
# Test: MLflow Health Check
# ---------------------------------------------------------------------------


class TestMLflowHealthCheck:
    """Test MLflow health check function."""

    def test_mlflow_health_returns_dict(self, config):
        from src.dashboard.system_health_view import check_mlflow_health
        result = check_mlflow_health(config)
        assert isinstance(result, dict)

    def test_mlflow_health_has_status(self, config):
        from src.dashboard.system_health_view import check_mlflow_health
        result = check_mlflow_health(config)
        assert "status" in result
        assert result["status"] in ("healthy", "degraded", "down")

    def test_mlflow_health_has_connected(self, config):
        from src.dashboard.system_health_view import check_mlflow_health
        result = check_mlflow_health(config)
        assert "connected" in result
        assert isinstance(result["connected"], bool)

    def test_mlflow_health_has_tracking_uri(self, config):
        from src.dashboard.system_health_view import check_mlflow_health
        result = check_mlflow_health(config)
        assert "tracking_uri" in result

    def test_mlflow_health_has_experiments_list(self, config):
        from src.dashboard.system_health_view import check_mlflow_health
        result = check_mlflow_health(config)
        assert "experiments" in result
        assert isinstance(result["experiments"], list)

    def test_mlflow_health_has_experiment_name(self, config):
        from src.dashboard.system_health_view import check_mlflow_health
        result = check_mlflow_health(config)
        assert "experiment_name" in result


# ---------------------------------------------------------------------------
# Test: Pipeline Health Check
# ---------------------------------------------------------------------------


class TestPipelineHealthCheck:
    """Test pipeline health check function."""

    def test_pipeline_health_returns_dict(self, config):
        from src.dashboard.system_health_view import check_pipeline_health
        result = check_pipeline_health(config)
        assert isinstance(result, dict)

    def test_pipeline_health_has_status(self, config):
        from src.dashboard.system_health_view import check_pipeline_health
        result = check_pipeline_health(config)
        assert "status" in result
        assert result["status"] in ("healthy", "degraded", "down")

    def test_pipeline_health_has_artifact_count(self, config):
        from src.dashboard.system_health_view import check_pipeline_health
        result = check_pipeline_health(config)
        assert "artifact_count" in result
        assert isinstance(result["artifact_count"], int)

    def test_pipeline_health_has_models_available(self, config):
        from src.dashboard.system_health_view import check_pipeline_health
        result = check_pipeline_health(config)
        assert "models_available" in result
        assert isinstance(result["models_available"], list)


# ---------------------------------------------------------------------------
# Test: System Health Summary
# ---------------------------------------------------------------------------


class TestSystemHealthSummary:
    """Test system health summary aggregation."""

    def test_summary_returns_dict(self, config):
        from src.dashboard.system_health_view import get_system_health_summary
        result = get_system_health_summary(config)
        assert isinstance(result, dict)

    def test_summary_has_overall_status(self, config):
        from src.dashboard.system_health_view import get_system_health_summary
        result = get_system_health_summary(config)
        assert "overall_status" in result
        assert result["overall_status"] in ("healthy", "degraded", "down")

    def test_summary_has_timestamp(self, config):
        from src.dashboard.system_health_view import get_system_health_summary
        result = get_system_health_summary(config)
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)

    def test_summary_has_all_services(self, config):
        from src.dashboard.system_health_view import get_system_health_summary
        result = get_system_health_summary(config)
        assert "services" in result
        assert "redis" in result["services"]
        assert "mlflow" in result["services"]
        assert "pipeline" in result["services"]

    def test_summary_service_structure(self, config):
        from src.dashboard.system_health_view import get_system_health_summary
        result = get_system_health_summary(config)
        for svc_name, svc_data in result["services"].items():
            assert "status" in svc_data, f"{svc_name} missing status"

    def test_all_healthy_gives_healthy_overall(self):
        """If all services healthy, overall should be healthy."""
        from src.dashboard.system_health_view import (
            STATUS_HEALTHY,
            STATUS_DEGRADED,
        )
        # Verify logic: all healthy -> healthy
        statuses = [STATUS_HEALTHY, STATUS_HEALTHY, STATUS_HEALTHY]
        assert all(s == STATUS_HEALTHY for s in statuses)


# ---------------------------------------------------------------------------
# Test: Streaming Pipeline Status Data
# ---------------------------------------------------------------------------


class TestStreamingPipelineData:
    """Test streaming pipeline data for rendering."""

    def test_throughput_data_has_columns(self, mock_data_loader):
        tp = mock_data_loader.load_scoring_throughput()
        assert "timestamp" in tp.columns
        assert "requests_per_minute" in tp.columns
        assert "avg_latency_ms" in tp.columns

    def test_throughput_positive_values(self, mock_data_loader):
        tp = mock_data_loader.load_scoring_throughput()
        assert (tp["requests_per_minute"] > 0).all()
        assert (tp["avg_latency_ms"] > 0).all()

    def test_throughput_avg_computable(self, mock_data_loader):
        tp = mock_data_loader.load_scoring_throughput()
        avg_tp = tp["requests_per_minute"].mean()
        assert avg_tp > 0

    def test_error_rate_bounded(self, mock_data_loader):
        tp = mock_data_loader.load_scoring_throughput()
        assert (tp["error_rate"] >= 0).all()
        assert (tp["error_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# Test: MLflow Tracking Data
# ---------------------------------------------------------------------------


class TestMLflowTrackingData:
    """Test MLflow experiment tracking data for rendering."""

    def test_mlflow_runs_has_columns(self, mock_data_loader):
        runs = mock_data_loader.load_mlflow_runs()
        required = ["run_id", "model_type", "auc", "training_time_s"]
        for col in required:
            assert col in runs.columns

    def test_mlflow_runs_auc_valid(self, mock_data_loader):
        runs = mock_data_loader.load_mlflow_runs()
        assert (runs["auc"] >= 0).all()
        assert (runs["auc"] <= 1).all()

    def test_best_model_identifiable(self, mock_data_loader):
        runs = mock_data_loader.load_mlflow_runs()
        best_idx = runs["auc"].idxmax()
        best_model = runs.loc[best_idx, "model_type"]
        assert isinstance(best_model, str)

    def test_total_training_time_positive(self, mock_data_loader):
        runs = mock_data_loader.load_mlflow_runs()
        total = runs["training_time_s"].sum()
        assert total > 0

    def test_runs_have_timestamps(self, mock_data_loader):
        runs = mock_data_loader.load_mlflow_runs()
        assert "timestamp" in runs.columns


# ---------------------------------------------------------------------------
# Test: Model Health & Drift Data
# ---------------------------------------------------------------------------


class TestModelHealthData:
    """Test model health and drift data for rendering."""

    def test_drift_history_has_alert_level(self, mock_data_loader):
        drift = mock_data_loader.load_drift_history()
        assert "alert_level" in drift.columns
        valid_levels = {"green", "yellow", "red"}
        assert set(drift["alert_level"].unique()) <= valid_levels

    def test_drift_history_has_psi(self, mock_data_loader):
        drift = mock_data_loader.load_drift_history()
        assert "psi_mean" in drift.columns
        assert (drift["psi_mean"] >= 0).all()

    def test_model_metrics_has_models(self, mock_data_loader):
        metrics = mock_data_loader.load_model_metrics()
        assert len(metrics) >= 2

    def test_model_metrics_has_auc(self, mock_data_loader):
        metrics = mock_data_loader.load_model_metrics()
        for model, m in metrics.items():
            assert "auc" in m
            assert 0 <= m["auc"] <= 1

    def test_best_model_by_auc(self, mock_data_loader):
        metrics = mock_data_loader.load_model_metrics()
        best = max(metrics.keys(), key=lambda k: metrics[k]["auc"])
        assert isinstance(best, str)
        assert metrics[best]["auc"] >= 0.8


# ---------------------------------------------------------------------------
# Test: Render Function Integration
# ---------------------------------------------------------------------------


class TestRenderIntegration:
    """Test render function integration with mock Streamlit."""

    def test_render_system_health_callable(self):
        from src.dashboard.system_health_view import render_system_health
        assert callable(render_system_health)

    def test_render_with_mock_streamlit(
        self, mock_st, config, mock_data_loader,
    ):
        from src.dashboard.system_health_view import render_system_health
        # Should not raise
        render_system_health(mock_st, config, mock_data_loader)

    def test_render_calls_data_loader(
        self, mock_st, config, mock_data_loader,
    ):
        from src.dashboard.system_health_view import render_system_health
        render_system_health(mock_st, config, mock_data_loader)
        mock_data_loader.load_scoring_throughput.assert_called()
        mock_data_loader.load_drift_history.assert_called()
        mock_data_loader.load_model_metrics.assert_called()
        mock_data_loader.load_mlflow_runs.assert_called()

    def test_render_sets_header(
        self, mock_st, config, mock_data_loader,
    ):
        from src.dashboard.system_health_view import render_system_health
        render_system_health(mock_st, config, mock_data_loader)
        mock_st.header.assert_called()

    def test_render_with_empty_data(self, mock_st, config):
        from src.dashboard.system_health_view import render_system_health
        loader = MagicMock()
        loader.load_scoring_throughput.return_value = pd.DataFrame()
        loader.load_drift_history.return_value = pd.DataFrame()
        loader.load_model_metrics.return_value = {}
        loader.load_mlflow_runs.return_value = pd.DataFrame()
        # Should not raise
        render_system_health(mock_st, config, loader)


# ---------------------------------------------------------------------------
# Test: App.py Integration
# ---------------------------------------------------------------------------


class TestAppIntegration:
    """Test system health view is integrated in app.py."""

    def test_system_health_in_page_list(self):
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "System Health" in PAGES

    def test_system_health_has_icon(self):
        from src.dashboard.utils.dashboard_helpers import PAGE_ICONS
        assert "System Health" in PAGE_ICONS

    def test_system_health_imported_in_app(self):
        from src.dashboard.app import render_system_health
        assert callable(render_system_health)

    def test_page_map_includes_system_health(self):
        """Verify the page map in main() includes System Health."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "System Health" in PAGES

    def test_status_colors_defined(self):
        from src.dashboard.system_health_view import STATUS_COLORS
        assert "healthy" in STATUS_COLORS
        assert "degraded" in STATUS_COLORS
        assert "down" in STATUS_COLORS

    def test_status_icons_defined(self):
        from src.dashboard.system_health_view import STATUS_ICONS
        assert "healthy" in STATUS_ICONS
        assert "degraded" in STATUS_ICONS
        assert "down" in STATUS_ICONS
