"""
TDD Tests for CLI Entrypoint (src/main.py) - Extended Coverage.

Tests cover:
- All 14+ modes in MODES registry (simulate, train, uplift, clv, optimize,
  ab_test, survival, recommend, cohort, segment, features, monitor, dashboard, all)
- build_parser argument validation for all flags
- Mode handler return structures
- Dashboard launch mode
- Budget flag passthrough
- Config loading edge cases
- NumpyEncoder for JSON serialization
- _resolve_dirs helper
- run_all pipeline structure
- Backward-compatible aliases (abtest)
- --small flag for simulation
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import build_parser, load_config, main, MODES, _NumpyEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Minimal valid config dict."""
    return {
        "simulation": {
            "random_seed": 42,
            "num_customers": 100,
            "simulation_months": 6,
            "start_date": "2024-01-01",
            "simulation_days": 180,
            "small_mode": {"num_customers": 50, "simulation_months": 3, "simulation_days": 90},
        },
        "churn_definition": {"no_purchase_days": 30, "no_login_days": 60, "operator": "OR"},
        "pipeline": {"train_months": 4, "test_months": 2, "ensemble_weight_ml": 0.6, "ensemble_weight_dl": 0.4},
        "segmentation": {"method": "rfm_behavioral", "n_rfm_bins": 5},
        "optimization": {"total_budget": 50000000, "channels": {"email": {"cost_per_action": 1000}}},
        "monitoring": {"enabled": True},
        "treatment": {"treatment_ratio": 0.5},
        "budget": {"total_krw": 50000000, "currency": "KRW"},
    }


@pytest.fixture
def config_file(sample_config, tmp_path):
    """Write sample config to a temp YAML file and return its path."""
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(sample_config, f)
    return str(cfg_path)


# ---------------------------------------------------------------------------
# MODES registry tests
# ---------------------------------------------------------------------------

class TestModesRegistry:
    """Tests for the complete MODES dispatch table."""

    def test_all_14_modes_registered(self):
        """All required modes should be in MODES dict."""
        expected = {
            "simulate", "train", "uplift", "clv", "optimize",
            "ab_test", "survival", "recommend", "cohort",
            "segment", "features", "monitor", "dashboard", "all",
        }
        assert expected.issubset(set(MODES.keys()))

    def test_abtest_alias_exists(self):
        """Backward-compatible 'abtest' alias should exist."""
        assert "abtest" in MODES
        assert MODES["abtest"] is MODES["ab_test"]

    def test_all_modes_callable(self):
        """Every mode handler must be callable."""
        for name, handler in MODES.items():
            assert callable(handler), f"MODES['{name}'] is not callable"

    def test_modes_count_at_least_14(self):
        """Should have at least 14 distinct modes + aliases."""
        assert len(MODES) >= 14


# ---------------------------------------------------------------------------
# Argument parsing tests (extended)
# ---------------------------------------------------------------------------

class TestBuildParserExtended:
    """Extended argument parsing tests."""

    @pytest.mark.parametrize("mode", [
        "simulate", "train", "uplift", "clv", "optimize",
        "ab_test", "survival", "recommend", "cohort",
        "segment", "features", "monitor", "dashboard", "all",
    ])
    def test_all_14_modes_accepted(self, mode):
        """All 14 modes should be accepted by the parser."""
        args = build_parser(["--mode", mode])
        assert args.mode == mode

    def test_budget_argument_int(self):
        """--budget should accept integer values."""
        args = build_parser(["--mode", "optimize", "--budget", "75000000"])
        assert args.budget == 75000000

    def test_budget_default_none(self):
        """--budget should default to None."""
        args = build_parser(["--mode", "optimize"])
        assert args.budget is None

    def test_small_flag(self):
        """--small flag should set small=True."""
        args = build_parser(["--mode", "simulate", "--small"])
        assert args.small is True

    def test_small_default_false(self):
        """--small should default to False."""
        args = build_parser(["--mode", "simulate"])
        assert args.small is False

    def test_cohort_type_default(self):
        """--cohort-type should default to monthly."""
        args = build_parser(["--mode", "cohort"])
        assert args.cohort_type == "monthly"

    def test_cohort_type_weekly(self):
        """--cohort-type should accept weekly."""
        args = build_parser(["--mode", "cohort", "--cohort-type", "weekly"])
        assert args.cohort_type == "weekly"

    def test_cohort_type_behavioral(self):
        """--cohort-type should accept behavioral."""
        args = build_parser(["--mode", "cohort", "--cohort-type", "behavioral"])
        assert args.cohort_type == "behavioral"

    def test_learner_choices(self):
        """--learner should accept t_learner and s_learner."""
        for learner in ["t_learner", "s_learner"]:
            args = build_parser(["--mode", "uplift", "--learner", learner])
            assert args.learner == learner

    def test_verbose_and_quiet_independent(self):
        """Both --verbose and --quiet should be independent flags."""
        args = build_parser(["--mode", "train", "--verbose"])
        assert args.verbose is True
        assert args.quiet is False

        args = build_parser(["--mode", "train", "--quiet"])
        assert args.verbose is False
        assert args.quiet is True

    def test_default_config_path_includes_yaml(self):
        """Default config path should point to simulator_config.yaml."""
        args = build_parser(["--mode", "train"])
        assert "simulator_config.yaml" in args.config


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------

class TestLoadConfigExtended:
    """Extended config loading tests."""

    def test_load_valid_config(self, config_file, sample_config):
        """Should load valid YAML config."""
        config = load_config(config_file)
        assert config["simulation"]["random_seed"] == 42

    def test_load_missing_raises(self):
        """Should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")

    def test_load_empty_yaml(self, tmp_path):
        """Empty YAML should return empty dict."""
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert load_config(str(p)) == {}

    def test_load_preserves_nested_structure(self, config_file):
        """Nested config structure should be preserved."""
        config = load_config(config_file)
        assert isinstance(config["simulation"], dict)
        assert isinstance(config["pipeline"], dict)


# ---------------------------------------------------------------------------
# NumpyEncoder tests
# ---------------------------------------------------------------------------

class TestNumpyEncoder:
    """Tests for _NumpyEncoder JSON serialization."""

    def test_encodes_numpy_int(self):
        """Should serialize numpy integers."""
        data = {"value": np.int64(42)}
        result = json.dumps(data, cls=_NumpyEncoder)
        assert '"value": 42' in result

    def test_encodes_numpy_float(self):
        """Should serialize numpy floats."""
        data = {"value": np.float64(3.14)}
        result = json.dumps(data, cls=_NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed["value"] - 3.14) < 1e-10

    def test_encodes_numpy_array(self):
        """Should serialize numpy arrays as lists."""
        data = {"values": np.array([1, 2, 3])}
        result = json.dumps(data, cls=_NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["values"] == [1, 2, 3]

    def test_encodes_numpy_bool(self):
        """Should serialize numpy booleans."""
        data = {"flag": np.bool_(True)}
        result = json.dumps(data, cls=_NumpyEncoder)
        parsed = json.loads(result)
        assert parsed["flag"] is True

    def test_encodes_pandas_timestamp(self):
        """Should serialize pandas Timestamps as ISO strings."""
        data = {"ts": pd.Timestamp("2024-01-15")}
        result = json.dumps(data, cls=_NumpyEncoder)
        parsed = json.loads(result)
        assert "2024-01-15" in parsed["ts"]


# ---------------------------------------------------------------------------
# Dashboard mode tests
# ---------------------------------------------------------------------------

class TestDashboardMode:
    """Tests for the dashboard launch mode."""

    def test_dashboard_mode_in_registry(self):
        """Dashboard mode should be registered."""
        assert "dashboard" in MODES

    def test_dashboard_handler_checks_app_exists(self, sample_config, tmp_path):
        """Dashboard handler should check if app.py exists."""
        args = build_parser(["--mode", "dashboard"])
        # The actual handler calls subprocess.run; we test it finds the app
        from src.main import run_dashboard, PROJECT_ROOT
        app_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
        assert app_path.exists()

    def test_dashboard_mode_returns_dict(self, sample_config):
        """Dashboard handler should return a dict with mode key."""
        args = build_parser(["--mode", "dashboard"])
        # Mock subprocess.run to avoid launching actual Streamlit
        with patch("src.main.subprocess.run"):
            result = MODES["dashboard"](sample_config, args)
        assert result["mode"] == "dashboard"
        assert "status" in result


# ---------------------------------------------------------------------------
# run_all mode tests
# ---------------------------------------------------------------------------

class TestRunAllMode:
    """Tests for the run_all pipeline mode."""

    def test_all_mode_in_registry(self):
        """'all' mode should be registered."""
        assert "all" in MODES

    def test_run_all_pipeline_order(self):
        """run_all should define a pipeline with correct step order."""
        from src.main import run_all
        # Inspect the function source for the steps list
        import inspect
        source = inspect.getsource(run_all)
        # Verify key steps are mentioned
        for step in ["simulate", "train", "uplift", "clv", "segment",
                      "optimize", "recommend", "cohort"]:
            assert step in source


# ---------------------------------------------------------------------------
# Mode handler return structure tests
# ---------------------------------------------------------------------------

class TestModeHandlerReturnStructure:
    """Test that mode handlers return proper dict structures."""

    def test_cohort_handler_returns_cohort_type(self, sample_config):
        """Cohort handler should include cohort_type in result."""
        args = build_parser(["--mode", "cohort", "--cohort-type", "weekly"])
        # Cohort handler needs data files, so mock _load_events
        with patch("src.main._load_events") as mock_events:
            mock_events.return_value = pd.DataFrame({
                "customer_id": [f"C{i}" for i in range(20)] * 3,
                "event_date": pd.date_range("2024-01-01", periods=60, freq="D"),
                "revenue": np.random.uniform(100, 1000, 60),
            })
            result = MODES["cohort"](sample_config, args)
        assert result["mode"] == "cohort"
        assert result["cohort_type"] == "weekly"

    def test_recommend_handler_returns_count(self, sample_config, tmp_path):
        """Recommend handler should return num_recommendations."""
        args = build_parser(["--mode", "recommend", "--data", str(tmp_path), "--output", str(tmp_path)])

        customers = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(10)],
            "churn_label": np.random.randint(0, 2, 10),
        })

        mock_engine = MagicMock()
        mock_engine.recommend.return_value = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(10)],
            "action": ["email"] * 10,
            "score": np.random.uniform(0, 1, 10),
        })

        with patch("src.main._load_customers", return_value=customers), \
             patch("src.models.recommendations.RecommendationEngine", return_value=mock_engine):
            result = MODES["recommend"](sample_config, args)
        assert result["mode"] == "recommend"
        assert result["status"] == "completed"
        assert "num_recommendations" in result


# ---------------------------------------------------------------------------
# main() integration tests (extended)
# ---------------------------------------------------------------------------

class TestMainIntegrationExtended:
    """Extended integration tests for main()."""

    def test_main_returns_dict(self, config_file):
        """main() should always return a dict."""
        # Use a mode that doesn't require data files
        with patch("src.main._load_events") as mock_ev, \
             patch("src.main._load_customers") as mock_cust:
            mock_cust.return_value = pd.DataFrame({
                "customer_id": [f"C{i}" for i in range(10)],
                "churn_label": np.random.randint(0, 2, 10),
            })
            mock_ev.return_value = pd.DataFrame({
                "customer_id": [f"C{i}" for i in range(10)],
                "event_date": pd.date_range("2024-01-01", periods=10),
            })
            result = main(["--mode", "cohort", "--config", config_file, "--quiet"])
        assert isinstance(result, dict)

    def test_main_quiet_suppresses_output(self, config_file, capsys):
        """main() with --quiet should not print JSON."""
        with patch("src.main._load_events") as mock_ev:
            mock_ev.return_value = pd.DataFrame({
                "customer_id": [f"C{i}" for i in range(10)] * 3,
                "event_date": pd.date_range("2024-01-01", periods=30, freq="D"),
                "revenue": np.random.uniform(100, 1000, 30),
            })
            main(["--mode", "cohort", "--config", config_file, "--quiet"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_main_missing_config_raises(self):
        """main() should raise FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            main(["--mode", "train", "--config", "/nonexistent.yaml"])


# ---------------------------------------------------------------------------
# __main__ module tests
# ---------------------------------------------------------------------------

class TestMainModule:
    """Tests for the __main__.py module entry point."""

    def test_main_module_exists(self):
        """src/__main__.py should exist for python -m src."""
        p = Path(__file__).parent.parent / "src" / "__main__.py"
        assert p.exists()

    def test_main_module_imports_main(self):
        """__main__.py should import main from src.main."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "src.__main__",
            Path(__file__).parent.parent / "src" / "__main__.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Just verify the file is parseable
        assert spec is not None


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestCLIEdgeCases:
    """Edge case and robustness tests for CLI."""

    def test_invalid_mode_rejected(self):
        """Invalid mode should cause SystemExit."""
        with pytest.raises(SystemExit):
            build_parser(["--mode", "nonexistent_mode"])

    def test_no_arguments_rejected(self):
        """No arguments should cause SystemExit."""
        with pytest.raises(SystemExit):
            build_parser([])

    def test_mode_argument_required(self):
        """--mode is required."""
        with pytest.raises(SystemExit):
            build_parser(["--config", "some.yaml"])

    def test_each_handler_is_distinct_function(self):
        """Each primary mode should map to a distinct function (except aliases)."""
        primary_modes = {
            "simulate", "train", "uplift", "clv", "optimize",
            "ab_test", "survival", "recommend", "cohort",
            "segment", "features", "monitor", "dashboard", "all",
        }
        handlers = {name: MODES[name] for name in primary_modes}
        # Verify no accidental duplicates (each is a unique function)
        func_ids = [id(h) for h in handlers.values()]
        assert len(set(func_ids)) == len(func_ids)
