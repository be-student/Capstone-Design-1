"""
TDD tests for Model Monitoring Service.

Tests cover:
- ModelMonitoringService initialization from config
- Running PSI drift checks on scoring data
- Running KS drift checks on scoring data
- Combined drift orchestration (PSI + KS)
- MLflow logging of drift results
- Alert triggering when drift exceeds thresholds
- Configurable thresholds from YAML config
- Alert callback mechanism
- Monitoring report generation
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from src.monitoring.monitoring_service import (
    ModelMonitoringService,
    MonitoringResult,
    AlertLevel,
    evaluate_performance_degradation,
)


@pytest.fixture
def monitoring_config():
    """Monitoring config matching simulator_config.yaml structure."""
    return {
        "drift_detection": {
            "n_bins": 10,
            "binning_strategy": "quantile",
            "yellow_threshold": 0.10,
            "red_threshold": 0.25,
            "epsilon": 1e-6,
        },
        "ks_drift_detection": {
            "warning_threshold": 0.05,
            "drift_threshold": 0.01,
        },
        "mlflow": {
            "tracking_uri": "sqlite:///test_mlflow.db",
            "artifact_location": "test_artifacts",
            "experiment_name": "churn_prediction",
        },
        "monitoring": {
            "enabled": True,
            "check_interval_seconds": 3600,
            "alert_on_yellow": False,
            "alert_on_red": True,
            "log_to_mlflow": True,
        },
    }


@pytest.fixture
def sample_reference_data():
    """Reference (training) data for drift detection."""
    rng = np.random.RandomState(42)
    n = 5000
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.exponential(2, n),
        "feature_c": rng.uniform(0, 10, n),
    })


@pytest.fixture
def sample_production_stable(sample_reference_data):
    """Production data with no drift."""
    rng = np.random.RandomState(99)
    n = 3000
    return pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.exponential(2, n),
        "feature_c": rng.uniform(0, 10, n),
    })


@pytest.fixture
def sample_production_drifted():
    """Production data with significant drift."""
    rng = np.random.RandomState(99)
    n = 3000
    return pd.DataFrame({
        "feature_a": rng.normal(5, 3, n),       # Large shift
        "feature_b": rng.exponential(20, n),     # Large shift
        "feature_c": rng.uniform(50, 100, n),    # Large shift
    })


class TestAlertLevel:
    """Test AlertLevel enum."""

    def test_alert_levels_exist(self):
        """AlertLevel should have GREEN, YELLOW, RED levels."""
        assert AlertLevel.GREEN.value == "green"
        assert AlertLevel.YELLOW.value == "yellow"
        assert AlertLevel.RED.value == "red"


class TestMonitoringResult:
    """Test MonitoringResult data class."""

    def test_creation(self):
        """MonitoringResult should store PSI and KS results."""
        result = MonitoringResult(
            psi_report={"feature_psi": {"f1": 0.05}},
            ks_report={"alerts": {"f1": {"p_value": 0.5}}},
            overall_alert_level=AlertLevel.GREEN,
            drifted_features=[],
            timestamp="2024-01-01T00:00:00",
        )
        assert result.overall_alert_level == AlertLevel.GREEN
        assert len(result.drifted_features) == 0

    def test_to_dict(self):
        """MonitoringResult.to_dict() should be JSON-serializable."""
        result = MonitoringResult(
            psi_report={"feature_psi": {"f1": 0.3}},
            ks_report={"alerts": {"f1": {"p_value": 0.001}}},
            overall_alert_level=AlertLevel.RED,
            drifted_features=["f1"],
            timestamp="2024-01-01T00:00:00",
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["overall_alert_level"] == "red"
        json.dumps(d)  # Should not raise

    def test_has_drift_property(self):
        """has_drift should be True when drifted_features is non-empty."""
        result_no_drift = MonitoringResult(
            psi_report={}, ks_report={},
            overall_alert_level=AlertLevel.GREEN,
            drifted_features=[],
            timestamp="2024-01-01T00:00:00",
        )
        assert result_no_drift.has_drift is False

        result_drift = MonitoringResult(
            psi_report={}, ks_report={},
            overall_alert_level=AlertLevel.RED,
            drifted_features=["f1"],
            timestamp="2024-01-01T00:00:00",
        )
        assert result_drift.has_drift is True

    def test_performance_degradation_serialized(self):
        """MonitoringResult should include performance degradation payload."""
        result = MonitoringResult(
            psi_report={},
            ks_report={},
            overall_alert_level=AlertLevel.RED,
            drifted_features=[],
            timestamp="2024-01-01T00:00:00",
            performance_alerts={
                "performance_degradation": True,
                "metrics": {
                    "auc": {
                        "current": 0.86,
                        "baseline": 0.91,
                        "drop": 0.05,
                        "threshold": 0.03,
                        "status": "degraded",
                    }
                },
            },
        )
        payload = result.to_dict()
        assert payload["performance_degradation"] is True
        assert payload["performance_alerts"]["metrics"]["auc"]["drop"] == 0.05


class TestPerformanceDegradationEvaluation:
    """Test threshold-based model performance degradation alerts."""

    def test_degradation_alert_for_auc_precision_recall_drop(self):
        """Latest metrics below baseline by threshold should alert."""
        history = pd.DataFrame({
            "timestamp": [
                "2026-05-01T00:00:00Z",
                "2026-05-02T00:00:00Z",
                "2026-05-03T00:00:00Z",
            ],
            "model": ["ensemble", "ensemble", "ensemble"],
            "auc": [0.91, 0.90, 0.86],
            "precision": [0.82, 0.81, 0.75],
            "recall": [0.73, 0.72, 0.66],
        })

        alerts = evaluate_performance_degradation(
            history,
            thresholds={"auc": 0.03, "precision": 0.05, "recall": 0.05},
        )

        assert alerts["performance_degradation"] is True
        assert alerts["alert_level"] == "red"
        assert set(alerts["degraded_metrics"]) == {
            "auc", "precision", "recall",
        }
        assert alerts["metrics"]["auc"]["current"] == pytest.approx(0.86)
        assert alerts["metrics"]["auc"]["baseline"] == pytest.approx(0.91)
        assert alerts["metrics"]["auc"]["drop"] == pytest.approx(0.05)
        assert alerts["metrics"]["auc"]["threshold"] == pytest.approx(0.03)
        assert alerts["metrics"]["auc"]["status"] == "degraded"

    def test_no_alert_when_metrics_stay_within_threshold(self):
        """Small metric movement should stay OK."""
        history = pd.DataFrame({
            "timestamp": [
                "2026-05-01T00:00:00Z",
                "2026-05-02T00:00:00Z",
            ],
            "model_type": ["ensemble", "ensemble"],
            "auc": [0.91, 0.90],
            "precision": [0.82, 0.80],
            "recall": [0.73, 0.72],
        })

        alerts = evaluate_performance_degradation(history)

        assert alerts["performance_degradation"] is False
        assert alerts["status"] == "ok"
        assert alerts["metrics"]["auc"]["status"] == "ok"

    def test_prefers_ensemble_when_multiple_models_exist(self):
        """Evaluation should select ensemble history by default."""
        history = pd.DataFrame({
            "timestamp": [
                "2026-05-01T00:00:00Z",
                "2026-05-02T00:00:00Z",
                "2026-05-01T00:00:00Z",
                "2026-05-02T00:00:00Z",
            ],
            "model": ["ml_model", "ml_model", "ensemble", "ensemble"],
            "auc": [0.99, 0.99, 0.95, 0.90],
            "precision": [0.99, 0.99, 0.84, 0.78],
            "recall": [0.99, 0.99, 0.75, 0.69],
        })

        alerts = evaluate_performance_degradation(history)

        assert alerts["model_type"] == "ensemble"
        assert alerts["performance_degradation"] is True


class TestModelMonitoringServiceInit:
    """Test ModelMonitoringService initialization."""

    def test_init_from_config(self, monitoring_config):
        """Should initialize from config dict."""
        service = ModelMonitoringService(monitoring_config)
        assert service is not None
        assert service.config == monitoring_config

    def test_init_creates_psi_detector(self, monitoring_config):
        """Should create a PSI DriftDetector."""
        service = ModelMonitoringService(monitoring_config)
        assert service.psi_detector is not None

    def test_init_creates_ks_detector(self, monitoring_config):
        """Should create a KSDriftDetector."""
        service = ModelMonitoringService(monitoring_config)
        assert service.ks_detector is not None

    def test_default_monitoring_settings(self):
        """Should use defaults if monitoring section is missing."""
        config = {
            "drift_detection": {"n_bins": 10},
            "ks_drift_detection": {"warning_threshold": 0.05},
        }
        service = ModelMonitoringService(config)
        assert service.alert_on_red is True


class TestModelMonitoringServiceFit:
    """Test fitting the monitoring service with reference data."""

    def test_fit_reference_data(
        self, monitoring_config, sample_reference_data
    ):
        """fit() should store reference data in both detectors."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        assert service.is_fitted is True

    def test_fit_returns_self(
        self, monitoring_config, sample_reference_data
    ):
        """fit() should return self for method chaining."""
        service = ModelMonitoringService(monitoring_config)
        result = service.fit(sample_reference_data)
        assert result is service

    def test_check_without_fit_raises(
        self, monitoring_config, sample_production_stable
    ):
        """check() without fit() should raise RuntimeError."""
        service = ModelMonitoringService(monitoring_config)
        with pytest.raises(RuntimeError, match="fit"):
            service.check(sample_production_stable)


class TestModelMonitoringServiceCheck:
    """Test drift checking functionality."""

    def test_check_stable_data_returns_green(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """Stable data should return GREEN alert level."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        result = service.check(sample_production_stable)
        assert isinstance(result, MonitoringResult)
        assert result.overall_alert_level == AlertLevel.GREEN

    def test_check_drifted_data_returns_red(
        self, monitoring_config, sample_reference_data, sample_production_drifted
    ):
        """Significantly drifted data should return RED alert level."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        result = service.check(sample_production_drifted)
        assert result.overall_alert_level == AlertLevel.RED
        assert len(result.drifted_features) > 0

    def test_check_returns_monitoring_result(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """check() should return a MonitoringResult."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        result = service.check(sample_production_stable)
        assert isinstance(result, MonitoringResult)
        assert result.psi_report is not None
        assert result.ks_report is not None

    def test_check_includes_timestamp(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """MonitoringResult should include a timestamp."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        result = service.check(sample_production_stable)
        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_check_psi_report_has_feature_psi(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """PSI report should contain feature_psi values."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        result = service.check(sample_production_stable)
        assert "feature_psi" in result.psi_report

    def test_check_ks_report_has_alerts(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """KS report should contain alerts."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        result = service.check(sample_production_stable)
        assert "alerts" in result.ks_report


class TestModelMonitoringServiceMLflow:
    """Test MLflow integration for logging drift results."""

    @patch("src.monitoring.monitoring_service.mlflow")
    def test_log_to_mlflow_called_on_check(
        self, mock_mlflow,
        monitoring_config, sample_reference_data, sample_production_stable
    ):
        """check() should log results to MLflow when configured."""
        monitoring_config["monitoring"]["log_to_mlflow"] = True
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        service.check(sample_production_stable)
        # Should have started and ended a run
        assert mock_mlflow.start_run.called
        assert mock_mlflow.end_run.called

    @patch("src.monitoring.monitoring_service.mlflow")
    def test_mlflow_logs_psi_metrics(
        self, mock_mlflow,
        monitoring_config, sample_reference_data, sample_production_stable
    ):
        """Should log PSI values as MLflow metrics."""
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        service.check(sample_production_stable)
        # Should have logged at least one metric
        assert mock_mlflow.log_metric.called or mock_mlflow.log_metrics.called

    @patch("src.monitoring.monitoring_service.mlflow")
    def test_mlflow_not_called_when_disabled(
        self, mock_mlflow,
        monitoring_config, sample_reference_data, sample_production_stable
    ):
        """Should not log to MLflow when log_to_mlflow is False."""
        monitoring_config["monitoring"]["log_to_mlflow"] = False
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        service.check(sample_production_stable)
        assert not mock_mlflow.start_run.called

    @patch("src.monitoring.monitoring_service.mlflow")
    def test_mlflow_logs_alert_level(
        self, mock_mlflow,
        monitoring_config, sample_reference_data, sample_production_drifted
    ):
        """Should log the overall alert level as an MLflow tag."""
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock()
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        service.check(sample_production_drifted)
        # Check that set_tag was called with alert level
        tag_calls = [c for c in mock_mlflow.set_tag.call_args_list
                     if c[0][0] == "drift_alert_level"]
        assert len(tag_calls) > 0


class TestModelMonitoringServiceAlerts:
    """Test alert triggering mechanism."""

    def test_alert_callback_called_on_red(
        self, monitoring_config, sample_reference_data, sample_production_drifted
    ):
        """Alert callback should be called when drift is RED."""
        callback = MagicMock()
        monitoring_config["monitoring"]["alert_on_red"] = True
        service = ModelMonitoringService(monitoring_config)
        service.register_alert_callback(callback)
        service.fit(sample_reference_data)
        service.check(sample_production_drifted)
        assert callback.called

    def test_alert_callback_not_called_on_green(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """Alert callback should NOT be called when no drift (GREEN)."""
        callback = MagicMock()
        service = ModelMonitoringService(monitoring_config)
        service.register_alert_callback(callback)
        service.fit(sample_reference_data)
        service.check(sample_production_stable)
        assert not callback.called

    def test_alert_callback_receives_result(
        self, monitoring_config, sample_reference_data, sample_production_drifted
    ):
        """Callback should receive the MonitoringResult."""
        callback = MagicMock()
        service = ModelMonitoringService(monitoring_config)
        service.register_alert_callback(callback)
        service.fit(sample_reference_data)
        service.check(sample_production_drifted)
        args = callback.call_args[0]
        assert isinstance(args[0], MonitoringResult)

    def test_multiple_callbacks(
        self, monitoring_config, sample_reference_data, sample_production_drifted
    ):
        """Multiple registered callbacks should all be called."""
        cb1 = MagicMock()
        cb2 = MagicMock()
        service = ModelMonitoringService(monitoring_config)
        service.register_alert_callback(cb1)
        service.register_alert_callback(cb2)
        service.fit(sample_reference_data)
        service.check(sample_production_drifted)
        assert cb1.called
        assert cb2.called

    def test_alert_on_yellow_configurable(
        self, monitoring_config, sample_reference_data
    ):
        """When alert_on_yellow is True, yellow-level drift should trigger alerts."""
        # Create production data with moderate drift (PSI between 0.1 and 0.25)
        rng = np.random.RandomState(123)
        n = 3000
        production_moderate = pd.DataFrame({
            "feature_a": rng.normal(0.5, 1.2, n),
            "feature_b": rng.exponential(3, n),
            "feature_c": rng.uniform(0, 10, n),
        })
        monitoring_config["monitoring"]["alert_on_yellow"] = True
        callback = MagicMock()
        service = ModelMonitoringService(monitoring_config)
        service.register_alert_callback(callback)
        service.fit(sample_reference_data)
        result = service.check(production_moderate)
        # If any feature is yellow or red, callback should fire
        if result.overall_alert_level in (AlertLevel.YELLOW, AlertLevel.RED):
            assert callback.called

    def test_performance_degradation_triggers_alert(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """Performance degradation should alert even when feature drift is absent."""
        monitoring_config["monitoring"]["log_to_mlflow"] = False
        callback = MagicMock()
        service = ModelMonitoringService(monitoring_config)
        service.register_alert_callback(callback)
        service.fit(sample_reference_data)
        performance_history = pd.DataFrame({
            "timestamp": [
                "2026-05-01T00:00:00Z",
                "2026-05-02T00:00:00Z",
            ],
            "model": ["ensemble", "ensemble"],
            "auc": [0.92, 0.86],
            "precision": [0.83, 0.76],
            "recall": [0.74, 0.67],
        })

        result = service.check(
            sample_production_stable,
            performance_history=performance_history,
        )

        assert result.has_performance_degradation is True
        assert result.overall_alert_level == AlertLevel.RED
        assert callback.called


class TestModelMonitoringServiceHistory:
    """Test monitoring history tracking."""

    def test_history_tracked(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """check() results should be appended to history."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        service.check(sample_production_stable)
        service.check(sample_production_stable)
        assert len(service.history) == 2

    def test_get_history(
        self, monitoring_config, sample_reference_data, sample_production_stable
    ):
        """get_history() should return list of MonitoringResult."""
        service = ModelMonitoringService(monitoring_config)
        service.fit(sample_reference_data)
        service.check(sample_production_stable)
        history = service.get_history()
        assert len(history) == 1
        assert isinstance(history[0], MonitoringResult)
