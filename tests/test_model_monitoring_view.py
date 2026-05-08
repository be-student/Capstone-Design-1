"""
TDD Tests for Model Monitoring & Survival Analysis Dashboard View.

Tests cover:
- render_model_monitoring function callable with mock Streamlit module
- Drift detection section renders KPI cards, timeline, PSI/KS charts
- Model performance section renders comparison chart and best model info
- Scoring throughput section renders throughput and latency charts
- Survival curves section renders KM curves and summary table
- Monitoring configuration section renders PSI/KS config
- Empty/missing data handling (graceful degradation)
- Integration with DashboardDataLoader
- Page is registered in PAGES list and page_map
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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
    """Load simulator configuration from YAML."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_st():
    """Create a mock Streamlit module with all required methods."""
    st = MagicMock()

    def make_cols(n):
        cols = [MagicMock() for _ in range(n)]
        for c in cols:
            c.__enter__ = MagicMock(return_value=c)
            c.__exit__ = MagicMock(return_value=False)
        return cols

    st.columns.side_effect = make_cols
    return st


@pytest.fixture
def data_loader(config):
    """Create a DashboardDataLoader with sample data."""
    from src.dashboard.data_loader import DashboardDataLoader
    return DashboardDataLoader(config)


@pytest.fixture
def sample_drift_history():
    """Create sample drift history data."""
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-09-01", periods=n, freq="1D")
            .strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        "alert_level": np.random.choice(
            ["green", "yellow", "red"], n, p=[0.6, 0.3, 0.1],
        ),
        "num_drifted_features": np.random.randint(0, 5, n),
        "psi_mean": np.random.uniform(0.01, 0.20, n).round(4),
        "ks_mean": np.random.uniform(0.02, 0.10, n).round(4),
    })


@pytest.fixture
def sample_model_metrics():
    """Create sample model metrics."""
    return {
        "ml_model": {
            "auc": 0.82, "precision": 0.76,
            "recall": 0.70, "f1_score": 0.73, "accuracy": 0.81,
        },
        "dl_model": {
            "auc": 0.79, "precision": 0.72,
            "recall": 0.67, "f1_score": 0.69, "accuracy": 0.78,
        },
        "ensemble": {
            "auc": 0.84, "precision": 0.78,
            "recall": 0.72, "f1_score": 0.75, "accuracy": 0.83,
        },
    }


@pytest.fixture
def sample_survival_curves():
    """Create sample survival curves."""
    return {
        "vip_loyal": {
            "timeline": list(range(0, 361, 30)),
            "survival_prob": [1.0, 0.98, 0.96, 0.94, 0.92,
                              0.90, 0.88, 0.86, 0.84, 0.82,
                              0.80, 0.78, 0.76],
            "ci_lower": [1.0, 0.96, 0.94, 0.92, 0.90,
                         0.88, 0.86, 0.84, 0.82, 0.80,
                         0.78, 0.76, 0.74],
            "ci_upper": [1.0, 1.0, 0.98, 0.96, 0.94,
                         0.92, 0.90, 0.88, 0.86, 0.84,
                         0.82, 0.80, 0.78],
            "median_survival_days": None,
        },
        "dormant": {
            "timeline": list(range(0, 361, 30)),
            "survival_prob": [1.0, 0.80, 0.65, 0.50, 0.40,
                              0.32, 0.25, 0.20, 0.16, 0.13,
                              0.10, 0.08, 0.06],
            "ci_lower": [1.0, 0.78, 0.63, 0.48, 0.38,
                         0.30, 0.23, 0.18, 0.14, 0.11,
                         0.08, 0.06, 0.04],
            "ci_upper": [1.0, 0.82, 0.67, 0.52, 0.42,
                         0.34, 0.27, 0.22, 0.18, 0.15,
                         0.12, 0.10, 0.08],
            "median_survival_days": 90,
        },
    }


# ---------------------------------------------------------------------------
# Test: Page Registration
# ---------------------------------------------------------------------------


class TestPageRegistration:
    """Verify Model Monitoring is registered in the navigation."""

    def test_model_monitoring_in_pages_list(self):
        """Model Monitoring must appear in the PAGES list."""
        from src.dashboard.utils.dashboard_helpers import PAGES
        assert "Model Monitoring" in PAGES

    def test_model_monitoring_has_icon(self):
        """Model Monitoring must have an icon defined."""
        from src.dashboard.utils.dashboard_helpers import PAGE_ICONS
        assert "Model Monitoring" in PAGE_ICONS
        assert len(PAGE_ICONS["Model Monitoring"]) > 0

    def test_model_monitoring_in_page_list_function(self):
        """get_page_list() must include Model Monitoring."""
        from src.dashboard.utils.dashboard_helpers import get_page_list
        assert "Model Monitoring" in get_page_list()

    def test_app_exports_monitoring_view_renderer(self):
        from src.dashboard.app import render_model_monitoring as app_renderer
        from src.dashboard.monitoring_view import (
            render_model_monitoring as view_renderer,
        )
        assert app_renderer is not None
        assert callable(app_renderer)
        assert app_renderer.__name__ == view_renderer.__name__


# ---------------------------------------------------------------------------
# Test: render_model_monitoring callable
# ---------------------------------------------------------------------------


class TestRenderModelMonitoring:
    """Verify render_model_monitoring executes without error."""

    def test_renders_with_data_loader(self, mock_st, config, data_loader):
        """render_model_monitoring should execute with a data_loader."""
        from src.dashboard.monitoring_view import render_model_monitoring
        render_model_monitoring(mock_st, config, data_loader)
        mock_st.header.assert_called_once_with(
            "Model Monitoring & Survival Analysis"
        )

    def test_renders_subheaders(self, mock_st, config, data_loader):
        """Should render all major section subheaders."""
        from src.dashboard.monitoring_view import render_model_monitoring
        render_model_monitoring(mock_st, config, data_loader)
        subheader_calls = [
            c.args[0] for c in mock_st.subheader.call_args_list
        ]
        assert "Drift Detection Overview" in subheader_calls
        assert "Model Performance Metrics Over Time" in subheader_calls
        assert "Scoring Throughput & Latency" in subheader_calls
        assert "Survival Curves (Quick Reference)" in subheader_calls
        assert "Monitoring Configuration" in subheader_calls

    def test_renders_plotly_charts(self, mock_st, config, data_loader):
        """Should render multiple plotly charts."""
        from src.dashboard.monitoring_view import render_model_monitoring
        render_model_monitoring(mock_st, config, data_loader)
        assert mock_st.plotly_chart.call_count >= 3

    def test_uses_model_performance_history_loader(self, mock_st, config):
        """Monitoring page should use real performance history, not only MLflow fallback."""
        from src.dashboard.monitoring_view import render_model_monitoring

        loader = MagicMock()
        loader.load_drift_history.return_value = pd.DataFrame()
        loader.load_model_metrics.return_value = {
            "ensemble": {
                "auc": 0.91,
                "precision": 0.82,
                "recall": 0.73,
                "f1_score": 0.77,
                "accuracy": 0.88,
            }
        }
        loader.load_scoring_throughput.return_value = pd.DataFrame()
        loader.load_survival_curves.return_value = {}
        loader.load_survival_data.return_value = pd.DataFrame()
        loader.load_model_performance_history.return_value = pd.DataFrame({
            "timestamp": ["2026-05-07T00:00:00Z"],
            "run_id": ["history_0"],
            "model_type": ["ensemble"],
            "auc": [0.91],
            "precision": [0.82],
            "recall": [0.73],
            "f1_score": [0.77],
            "accuracy": [0.88],
            "training_time_s": [1.0],
        })

        render_model_monitoring(mock_st, config, loader)

        loader.load_model_performance_history.assert_called_once()
        assert not loader.load_mlflow_runs.called

    def test_uses_performance_alerts_loader(self, mock_st, config):
        """Monitoring page should read performance degradation alerts."""
        from src.dashboard.monitoring_view import render_model_monitoring

        loader = MagicMock()
        loader.load_drift_history.return_value = pd.DataFrame()
        loader.load_model_metrics.return_value = {}
        loader.load_scoring_throughput.return_value = pd.DataFrame()
        loader.load_survival_curves.return_value = {}
        loader.load_survival_data.return_value = pd.DataFrame()
        loader.load_model_performance_history.return_value = pd.DataFrame({
            "timestamp": ["2026-05-07T00:00:00Z"],
            "run_id": ["history_0"],
            "model_type": ["ensemble"],
            "auc": [0.91],
            "precision": [0.82],
            "recall": [0.73],
            "f1_score": [0.77],
            "accuracy": [0.88],
            "training_time_s": [1.0],
        })
        loader.load_performance_alerts.return_value = {
            "performance_degradation": True,
            "status": "degraded",
            "model_type": "ensemble",
            "degraded_metrics": ["auc"],
            "metrics": {
                "auc": {
                    "current": 0.86,
                    "baseline": 0.91,
                    "drop": 0.05,
                    "threshold": 0.03,
                    "status": "degraded",
                }
            },
        }

        render_model_monitoring(mock_st, config, loader)

        loader.load_performance_alerts.assert_called_once()
        mock_st.error.assert_called()


# ---------------------------------------------------------------------------
# Test: Drift Detection Section
# ---------------------------------------------------------------------------


class TestDriftDetectionSection:
    """Verify drift detection rendering."""

    def test_drift_kpi_metrics_rendered(
        self, mock_st, config, sample_drift_history,
    ):
        """KPI cards for drift should show Total Checks, Status, etc."""
        from src.dashboard.monitoring_view import _render_drift_section
        _render_drift_section(mock_st, config, sample_drift_history)

        # columns() should have been called to create KPI row
        mock_st.columns.assert_called()

    def test_drift_timeline_chart(
        self, mock_st, config, sample_drift_history,
    ):
        """Drift alert timeline should be rendered as plotly chart."""
        from src.dashboard.monitoring_view import _render_drift_section
        _render_drift_section(mock_st, config, sample_drift_history)
        assert mock_st.plotly_chart.call_count >= 1

    def test_drift_psi_ks_charts(
        self, mock_st, config, sample_drift_history,
    ):
        """PSI and KS time series charts should be rendered."""
        from src.dashboard.monitoring_view import _render_drift_section
        _render_drift_section(mock_st, config, sample_drift_history)
        # Timeline + PSI + KS = at least 3 plotly_chart calls
        # (PSI and KS are rendered in column context managers)
        total_plotly_calls = mock_st.plotly_chart.call_count
        cols = mock_st.columns.return_value
        col_plotly_calls = sum(
            c.plotly_chart.call_count for c in cols
        )
        assert total_plotly_calls + col_plotly_calls >= 2

    def test_drift_log_table(
        self, mock_st, config, sample_drift_history,
    ):
        """Drift detection log should be displayed as dataframe."""
        from src.dashboard.monitoring_view import _render_drift_section
        _render_drift_section(mock_st, config, sample_drift_history)
        assert mock_st.dataframe.call_count >= 1


# ---------------------------------------------------------------------------
# Test: Model Performance Section
# ---------------------------------------------------------------------------


class TestModelPerformanceSection:
    """Verify model performance metrics rendering."""

    def test_performance_bar_chart(
        self, mock_st, sample_model_metrics,
    ):
        """Should render model performance comparison bar chart."""
        from src.dashboard.monitoring_view import (
            _render_performance_section,
        )
        mlflow_runs = pd.DataFrame()
        _render_performance_section(
            mock_st, sample_model_metrics, mlflow_runs,
        )
        assert mock_st.plotly_chart.call_count >= 1

    def test_best_model_info(self, mock_st, sample_model_metrics):
        """Should display best model by AUC."""
        from src.dashboard.monitoring_view import (
            _render_performance_section,
        )
        mlflow_runs = pd.DataFrame()
        _render_performance_section(
            mock_st, sample_model_metrics, mlflow_runs,
        )
        mock_st.info.assert_called_once()
        info_text = mock_st.info.call_args[0][0]
        assert "ensemble" in info_text
        assert "0.8400" in info_text

    def test_with_mlflow_runs(self, mock_st, sample_model_metrics):
        """Should render training run history when mlflow_runs present."""
        from src.dashboard.monitoring_view import (
            _render_performance_section,
        )
        mlflow_runs = pd.DataFrame({
            "run_id": ["r1", "r2"],
            "model_type": ["xgboost", "lstm"],
            "auc": [0.82, 0.79],
            "precision": [0.76, 0.72],
            "recall": [0.70, 0.67],
            "f1_score": [0.73, 0.69],
            "accuracy": [0.81, 0.78],
            "training_time_s": [10.5, 45.2],
            "timestamp": [
                "2024-10-15T10:00:00",
                "2024-10-16T10:00:00",
            ],
        })
        _render_performance_section(
            mock_st, sample_model_metrics, mlflow_runs,
        )
        # Should have run details table
        assert mock_st.dataframe.call_count >= 1

    def test_empty_metrics(self, mock_st):
        """Should handle empty model metrics gracefully."""
        from src.dashboard.monitoring_view import (
            _render_performance_section,
        )
        _render_performance_section(mock_st, {}, pd.DataFrame())
        # Should not crash; no plotly chart rendered
        assert mock_st.plotly_chart.call_count == 0

    def test_performance_degradation_alert_table(self, mock_st, sample_model_metrics):
        """Performance alert payload should render as a threshold table."""
        from src.dashboard.monitoring_view import (
            _render_performance_section,
        )
        alerts = {
            "performance_degradation": True,
            "status": "degraded",
            "model_type": "ensemble",
            "degraded_metrics": ["auc", "precision", "recall"],
            "metrics": {
                "auc": {
                    "current": 0.86,
                    "baseline": 0.91,
                    "drop": 0.05,
                    "threshold": 0.03,
                    "status": "degraded",
                },
                "precision": {
                    "current": 0.75,
                    "baseline": 0.82,
                    "drop": 0.07,
                    "threshold": 0.05,
                    "status": "degraded",
                },
                "recall": {
                    "current": 0.66,
                    "baseline": 0.73,
                    "drop": 0.07,
                    "threshold": 0.05,
                    "status": "degraded",
                },
            },
        }

        _render_performance_section(
            mock_st,
            sample_model_metrics,
            pd.DataFrame(),
            alerts,
        )

        mock_st.error.assert_called()
        assert mock_st.dataframe.call_count >= 1


# ---------------------------------------------------------------------------
# Test: Scoring Throughput Section
# ---------------------------------------------------------------------------


class TestScoringThroughputSection:
    """Verify scoring throughput rendering."""

    def test_throughput_charts(self, mock_st):
        """Should render throughput and latency charts."""
        from src.dashboard.monitoring_view import (
            _render_throughput_section,
        )
        throughput = pd.DataFrame({
            "timestamp": pd.date_range(
                "2024-10-15", periods=10, freq="30min"
            ).strftime("%Y-%m-%dT%H:%M:%S").tolist(),
            "requests_per_minute": np.random.uniform(30, 80, 10),
            "avg_latency_ms": np.random.uniform(10, 30, 10),
            "error_rate": np.random.uniform(0, 0.02, 10),
        })
        _render_throughput_section(mock_st, throughput)
        # columns called for throughput/latency split
        mock_st.columns.assert_called()

    def test_empty_throughput(self, mock_st):
        """Should show info message when no throughput data."""
        from src.dashboard.monitoring_view import (
            _render_throughput_section,
        )
        _render_throughput_section(mock_st, pd.DataFrame())
        mock_st.info.assert_called_once_with(
            "No scoring throughput data available."
        )


# ---------------------------------------------------------------------------
# Test: Survival Curves Section
# ---------------------------------------------------------------------------


class TestSurvivalCurvesSection:
    """Verify survival curves rendering."""

    def test_survival_curves_rendered(
        self, mock_st, sample_survival_curves,
    ):
        """Should render KM survival curves chart."""
        from src.dashboard.monitoring_view import (
            _render_survival_section,
        )
        _render_survival_section(
            mock_st, sample_survival_curves, pd.DataFrame(),
        )
        assert mock_st.plotly_chart.call_count >= 1

    def test_survival_summary_table(
        self, mock_st, sample_survival_curves,
    ):
        """Should render survival summary dataframe."""
        from src.dashboard.monitoring_view import (
            _render_survival_section,
        )
        _render_survival_section(
            mock_st, sample_survival_curves, pd.DataFrame(),
        )
        assert mock_st.dataframe.call_count >= 1

    def test_survival_summary_risk_levels(
        self, mock_st, sample_survival_curves,
    ):
        """Summary table should contain risk level classification."""
        from src.dashboard.monitoring_view import (
            _render_survival_section,
        )
        _render_survival_section(
            mock_st, sample_survival_curves, pd.DataFrame(),
        )
        # Get the DataFrame passed to st.dataframe
        df_call = mock_st.dataframe.call_args
        df = df_call[0][0] if df_call[0] else df_call[1].get("data")
        assert "Risk Level" in df.columns
        assert "Segment" in df.columns

    def test_empty_survival_data(self, mock_st):
        """Should show warning when no survival data at all."""
        from src.dashboard.monitoring_view import (
            _render_survival_section,
        )
        _render_survival_section(mock_st, {}, pd.DataFrame())
        mock_st.warning.assert_called()

    def test_no_curves_but_survival_data(self, mock_st):
        """Should show info when curves missing but data exists."""
        from src.dashboard.monitoring_view import (
            _render_survival_section,
        )
        survival_data = pd.DataFrame({
            "customer_id": ["C1"],
            "duration_days": [100],
        })
        _render_survival_section(mock_st, {}, survival_data)
        mock_st.info.assert_called()


# ---------------------------------------------------------------------------
# Test: Configuration Section
# ---------------------------------------------------------------------------


class TestConfigSection:
    """Verify monitoring configuration display."""

    def test_config_renders_json(self, mock_st, config):
        """Should render PSI and KS config as JSON."""
        from src.dashboard.monitoring_view import (
            _render_config_section,
        )
        _render_config_section(mock_st, config)
        # columns should be called for layout
        mock_st.columns.assert_called()

    def test_config_with_empty_config(self, mock_st):
        """Should handle empty config gracefully with defaults."""
        from src.dashboard.monitoring_view import (
            _render_config_section,
        )
        _render_config_section(mock_st, {})
        mock_st.columns.assert_called()


# ---------------------------------------------------------------------------
# Test: Data Loader Integration
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:
    """Verify integration with DashboardDataLoader."""

    def test_data_loader_loads_drift_history(self, data_loader):
        """DashboardDataLoader should return drift history DataFrame."""
        drift = data_loader.load_drift_history()
        assert isinstance(drift, pd.DataFrame)
        assert not drift.empty
        assert "alert_level" in drift.columns
        assert "psi_mean" in drift.columns
        assert "ks_mean" in drift.columns

    def test_data_loader_loads_scoring_throughput(self, data_loader):
        """DashboardDataLoader should return scoring throughput DataFrame."""
        throughput = data_loader.load_scoring_throughput()
        assert isinstance(throughput, pd.DataFrame)
        assert not throughput.empty
        assert "requests_per_minute" in throughput.columns
        assert "avg_latency_ms" in throughput.columns

    def test_data_loader_loads_survival_curves(self, data_loader):
        """DashboardDataLoader should return survival curves dict."""
        curves = data_loader.load_survival_curves()
        assert isinstance(curves, dict)
        assert len(curves) > 0
        for seg_name, curve_data in curves.items():
            assert "timeline" in curve_data
            assert "survival_prob" in curve_data
            assert len(curve_data["timeline"]) == len(
                curve_data["survival_prob"]
            )

    def test_data_loader_loads_model_metrics(self, data_loader):
        """DashboardDataLoader should return model metrics dict."""
        metrics = data_loader.load_model_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        for model_name, model_metrics in metrics.items():
            assert "auc" in model_metrics
            assert 0 <= model_metrics["auc"] <= 1

    def test_data_loader_loads_mlflow_runs(self, data_loader):
        """DashboardDataLoader should return mlflow runs DataFrame."""
        runs = data_loader.load_mlflow_runs()
        assert isinstance(runs, pd.DataFrame)
        assert not runs.empty
        assert "model_type" in runs.columns
        assert "auc" in runs.columns

    def test_drift_history_alert_levels_valid(self, data_loader):
        """All alert levels should be green, yellow, or red."""
        drift = data_loader.load_drift_history()
        valid_levels = {"green", "yellow", "red"}
        assert set(drift["alert_level"].unique()).issubset(valid_levels)

    def test_data_loader_loads_performance_alerts(self, data_loader):
        """DashboardDataLoader should expose degradation alert fields."""
        alerts = data_loader.load_performance_alerts()
        assert isinstance(alerts, dict)
        assert "performance_degradation" in alerts
        assert "metrics" in alerts
        for metric in ["auc", "precision", "recall"]:
            assert metric in alerts["metrics"]
            metric_alert = alerts["metrics"][metric]
            for field in ["current", "baseline", "drop", "threshold", "status"]:
                assert field in metric_alert

    def test_data_loader_derives_alerts_for_legacy_report(
        self, tmp_path, config
    ):
        """Old monitoring reports should use history CSV as alert fallback."""
        from src.dashboard.data_loader import DashboardDataLoader

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "monitoring_report.json").write_text(
            '{"timestamp": "2026-05-07T00:00:00Z", '
            '"psi_report": {"feature_alerts": {}}, '
            '"ks_report": {"feature_alerts": {}}, '
            '"performance": {"latest": []}}'
        )
        pd.DataFrame({
            "timestamp": [
                "2026-05-01T00:00:00Z",
                "2026-05-02T00:00:00Z",
            ],
            "model": ["ensemble", "ensemble"],
            "auc": [0.91, 0.86],
            "precision": [0.82, 0.75],
            "recall": [0.73, 0.66],
        }).to_csv(results_dir / "model_performance_history.csv", index=False)

        cfg = dict(config)
        cfg["dashboard"] = {"results_dir": str(results_dir)}
        loader = DashboardDataLoader(cfg)
        alerts = loader.load_performance_alerts()

        assert alerts["performance_degradation"] is True
        assert alerts["metrics"]["auc"]["status"] == "degraded"

    def test_survival_curves_monotonically_decreasing(self, data_loader):
        """Survival probabilities should generally decrease over time."""
        curves = data_loader.load_survival_curves()
        for seg_name, curve_data in curves.items():
            probs = curve_data["survival_prob"]
            # First value should be 1.0 (or close)
            assert probs[0] >= 0.99, (
                f"{seg_name}: first survival prob should be ~1.0"
            )
            # Last value should be less than or equal to first
            assert probs[-1] <= probs[0], (
                f"{seg_name}: survival should decrease over time"
            )


# ---------------------------------------------------------------------------
# Test: Full Page Render End-to-End
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end rendering tests."""

    def test_full_render_no_crash(self, mock_st, config, data_loader):
        """Full render should complete without exceptions."""
        from src.dashboard.monitoring_view import render_model_monitoring
        render_model_monitoring(mock_st, config, data_loader)

    def test_full_render_calls_header(
        self, mock_st, config, data_loader,
    ):
        """Should call st.header exactly once."""
        from src.dashboard.monitoring_view import render_model_monitoring
        render_model_monitoring(mock_st, config, data_loader)
        mock_st.header.assert_called_once()

    def test_full_render_has_separators(
        self, mock_st, config, data_loader,
    ):
        """Should have section separators (markdown ---)."""
        from src.dashboard.monitoring_view import render_model_monitoring
        render_model_monitoring(mock_st, config, data_loader)
        separator_calls = [
            c for c in mock_st.markdown.call_args_list
            if c.args and c.args[0] == "---"
        ]
        assert len(separator_calls) >= 3

    def test_import_from_app(self):
        """render_model_monitoring should be importable from app module."""
        from src.dashboard.app import render_model_monitoring as rmm
        assert callable(rmm)
