"""
Tests for CLI entrypoint (src/main.py).

TDD tests covering:
- Argument parsing for all modes (simulate, train, uplift, clv, optimize,
  ab_test, survival, recommend, cohort, segment, features, monitor,
  dashboard, all, abtest alias)
- Config loading
- Mode dispatch registry
- --budget, --small, --learner, --cohort-type flags
- JSON serialization helpers (_NumpyEncoder, _save_json)
- Data loading helpers
- Error handling
- Verbose/quiet flags
- Mode handler signatures
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import (
    MODES,
    _NumpyEncoder,
    _feature_cols,
    _load_customers,
    _load_events,
    _resolve_dirs,
    _save_json,
    build_parser,
    load_config,
    main,
)


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
            "simulation_days": 180,
            "start_date": "2024-01-01",
            "small_mode": {
                "num_customers": 50,
                "simulation_months": 3,
                "simulation_days": 90,
            },
        },
        "churn_definition": {"no_purchase_days": 30, "no_login_days": 60},
        "pipeline": {
            "train_months": 4,
            "test_months": 2,
            "ensemble_weight_ml": 0.6,
            "ensemble_weight_dl": 0.4,
        },
        "segmentation": {"method": "rfm_behavioral", "n_rfm_bins": 5},
        "budget": {"total_krw": 50_000_000},
        "optimization": {
            "total_budget": 50_000_000,
            "channels": {"email": {"cost_per_action": 1000}},
        },
        "monitoring": {"enabled": True},
        "treatment": {"treatment_ratio": 0.5},
        "drift_detection": {"n_bins": 10, "yellow_threshold": 0.1, "red_threshold": 0.25},
        "ks_drift_detection": {"warning_threshold": 0.05, "drift_threshold": 0.01},
    }


@pytest.fixture
def config_file(sample_config, tmp_path):
    """Write sample config to a temp YAML file and return its path."""
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(sample_config, f)
    return str(cfg_path)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with minimal CSV files."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)

    n = 50
    rng = np.random.default_rng(42)

    customers = pd.DataFrame({
        "customer_id": range(n),
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "persona": rng.choice(["vip_loyal", "regular_loyal", "bargain_hunter",
                                "new_customer", "dormant", "high_value_at_risk"], n),
        "churn_label": rng.choice([0, 1], n, p=[0.8, 0.2]),
        "treatment_group": rng.choice(["treatment", "control"], n),
    })
    customers.to_csv(data_dir / "customers.csv", index=False)

    events = pd.DataFrame({
        "customer_id": rng.integers(0, n, 200),
        "event_date": pd.date_range("2024-01-01", periods=200, freq="4h"),
        "event_type": rng.choice(["page_view", "search", "purchase", "add_to_cart"], 200),
        "revenue": rng.uniform(0, 100000, 200),
    })
    events.to_csv(data_dir / "events.csv", index=False)

    return str(data_dir)


@pytest.fixture
def mock_args(config_file, tmp_data_dir):
    """Create a mock Namespace with all required CLI args."""
    return argparse.Namespace(
        mode="train",
        config=config_file,
        data=tmp_data_dir,
        output=None,
        budget=None,
        small=False,
        learner="t_learner",
        cohort_type="monthly",
        verbose=False,
        quiet=True,
    )


# ===========================================================================
# Tests: Config loading
# ===========================================================================

class TestLoadConfig:
    """Tests for load_config utility."""

    def test_load_valid_yaml(self, config_file):
        config = load_config(config_file)
        assert isinstance(config, dict)
        assert config["simulation"]["random_seed"] == 42

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        assert load_config(str(p)) == {}


# ===========================================================================
# Tests: Argument parsing
# ===========================================================================

class TestBuildParser:
    """Tests for CLI argument parsing."""

    @pytest.mark.parametrize("mode", [
        "simulate", "train", "uplift", "clv", "optimize",
        "ab_test", "survival", "recommend", "cohort",
        "segment", "features", "monitor", "dashboard", "all",
        "abtest",
    ])
    def test_all_modes_accepted(self, mode):
        args = build_parser(["--mode", mode])
        assert args.mode == mode

    def test_invalid_mode_rejected(self):
        with pytest.raises(SystemExit):
            build_parser(["--mode", "nonexistent"])

    def test_missing_mode_rejected(self):
        with pytest.raises(SystemExit):
            build_parser([])

    def test_default_config_path(self):
        args = build_parser(["--mode", "train"])
        assert "simulator_config.yaml" in args.config

    def test_custom_config_path(self, config_file):
        args = build_parser(["--mode", "train", "--config", config_file])
        assert args.config == config_file

    def test_data_argument(self):
        args = build_parser(["--mode", "train", "--data", "/some/data"])
        assert args.data == "/some/data"

    def test_output_argument(self):
        args = build_parser(["--mode", "train", "--output", "/some/output"])
        assert args.output == "/some/output"

    def test_budget_argument(self):
        args = build_parser(["--mode", "optimize", "--budget", "100000000"])
        assert args.budget == 100_000_000

    def test_budget_default_none(self):
        args = build_parser(["--mode", "optimize"])
        assert args.budget is None

    def test_small_flag(self):
        args = build_parser(["--mode", "simulate", "--small"])
        assert args.small is True

    def test_small_default_false(self):
        args = build_parser(["--mode", "simulate"])
        assert args.small is False

    def test_learner_argument(self):
        args = build_parser(["--mode", "uplift", "--learner", "s_learner"])
        assert args.learner == "s_learner"

    def test_learner_default(self):
        args = build_parser(["--mode", "uplift"])
        assert args.learner == "t_learner"

    def test_cohort_type_argument(self):
        args = build_parser(["--mode", "cohort", "--cohort-type", "weekly"])
        assert args.cohort_type == "weekly"

    def test_cohort_type_default(self):
        args = build_parser(["--mode", "cohort"])
        assert args.cohort_type == "monthly"

    def test_verbose_flag(self):
        args = build_parser(["--mode", "train", "-v"])
        assert args.verbose is True

    def test_quiet_flag(self):
        args = build_parser(["--mode", "train", "-q"])
        assert args.quiet is True

    def test_default_data_none(self):
        args = build_parser(["--mode", "train"])
        assert args.data is None

    def test_default_output_none(self):
        args = build_parser(["--mode", "train"])
        assert args.output is None


# ===========================================================================
# Tests: MODES registry
# ===========================================================================

class TestModesRegistry:
    """Tests for the MODES dispatch table."""

    def test_all_required_modes_registered(self):
        required = {
            "simulate", "train", "uplift", "clv", "optimize",
            "ab_test", "survival", "recommend", "cohort",
            "segment", "features", "monitor", "dashboard", "all",
        }
        assert required.issubset(set(MODES.keys()))

    def test_abtest_alias_exists(self):
        assert "abtest" in MODES

    def test_abtest_alias_same_handler(self):
        assert MODES["abtest"] is MODES["ab_test"]

    def test_all_modes_callable(self):
        for name, handler in MODES.items():
            assert callable(handler), f"'{name}' handler is not callable"

    def test_mode_count_at_least_15(self):
        # 14 modes + 1 alias
        assert len(MODES) >= 15


# ===========================================================================
# Tests: JSON encoder
# ===========================================================================

class TestNumpyEncoder:
    """Tests for _NumpyEncoder."""

    def test_numpy_int64(self):
        result = json.dumps({"v": np.int64(42)}, cls=_NumpyEncoder)
        assert json.loads(result)["v"] == 42

    def test_numpy_float64(self):
        result = json.dumps({"v": np.float64(3.14)}, cls=_NumpyEncoder)
        assert abs(json.loads(result)["v"] - 3.14) < 1e-6

    def test_numpy_array(self):
        result = json.dumps({"v": np.array([1, 2, 3])}, cls=_NumpyEncoder)
        assert json.loads(result)["v"] == [1, 2, 3]

    def test_numpy_bool(self):
        result = json.dumps({"v": np.bool_(True)}, cls=_NumpyEncoder)
        assert json.loads(result)["v"] is True

    def test_pandas_timestamp(self):
        ts = pd.Timestamp("2024-01-15")
        result = json.dumps({"v": ts}, cls=_NumpyEncoder)
        assert "2024-01-15" in json.loads(result)["v"]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            json.dumps({"v": object()}, cls=_NumpyEncoder)


# ===========================================================================
# Tests: _save_json
# ===========================================================================

class TestSaveJson:
    """Tests for _save_json helper."""

    def test_save_creates_file(self, tmp_path):
        p = tmp_path / "test.json"
        _save_json({"a": 1}, p)
        assert p.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "nested" / "dir" / "out.json"
        _save_json({"b": 2}, p)
        assert p.exists()

    def test_save_roundtrip(self, tmp_path):
        p = tmp_path / "rt.json"
        data = {"x": 1, "y": np.float64(2.5), "z": np.array([10, 20])}
        _save_json(data, p)
        loaded = json.loads(p.read_text())
        assert loaded["x"] == 1
        assert loaded["y"] == 2.5
        assert loaded["z"] == [10, 20]


# ===========================================================================
# Tests: Helper functions
# ===========================================================================

class TestFeatureCols:
    """Tests for _feature_cols."""

    def test_excludes_meta_columns(self):
        df = pd.DataFrame({
            "customer_id": [1], "churn_label": [0],
            "reference_date": ["2024-01-01"],
            "treatment_group": ["treatment"],
            "signup_date": ["2024-01-01"],
            "feat_a": [1.0], "feat_b": [2.0],
        })
        cols = _feature_cols(df)
        assert "feat_a" in cols
        assert "feat_b" in cols
        for meta in ("customer_id", "churn_label", "reference_date",
                     "treatment_group", "signup_date"):
            assert meta not in cols

    def test_empty_df(self):
        df = pd.DataFrame({"customer_id": []})
        assert _feature_cols(df) == []


class TestResolveDirs:
    """Tests for _resolve_dirs."""

    def test_creates_data_dir(self, tmp_path):
        args = argparse.Namespace(data=str(tmp_path / "d"), output=None)
        d, _, _ = _resolve_dirs(args)
        assert d.exists()

    def test_output_override(self, tmp_path):
        args = argparse.Namespace(data=None, output=str(tmp_path / "o"))
        _, r, m = _resolve_dirs(args)
        assert "results" in str(r)
        assert "models" in str(m)


# ===========================================================================
# Tests: Data loading
# ===========================================================================

class TestDataLoading:
    """Tests for _load_customers and _load_events."""

    def test_load_customers_csv(self, tmp_data_dir):
        df = _load_customers(Path(tmp_data_dir))
        assert len(df) > 0
        assert "customer_id" in df.columns

    def test_load_events_csv(self, tmp_data_dir):
        df = _load_events(Path(tmp_data_dir))
        assert len(df) > 0
        assert "event_type" in df.columns

    def test_load_customers_parquet_priority(self, tmp_data_dir):
        d = Path(tmp_data_dir)
        csv_df = pd.read_csv(d / "customers.csv")
        csv_df.to_parquet(d / "customers.parquet", index=False)
        df = _load_customers(d)
        assert len(df) == len(csv_df)

    def test_load_customers_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="simulate"):
            _load_customers(tmp_path / "empty")

    def test_load_events_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="simulate"):
            _load_events(tmp_path / "empty")


# ===========================================================================
# Tests: Mode handlers (mocked)
# ===========================================================================

class TestRunSimulate:
    """Tests for run_simulate handler."""

    def test_returns_completed(self, sample_config, mock_args):
        mock_args.mode = "simulate"
        with patch("src.data.SimulatorOrchestrator") as M:
            inst = MagicMock()
            inst.run.return_value = {
                "summary": {"num_customers": 100, "num_events": 1000, "churn_rate": 0.2}
            }
            M.return_value = inst
            from src.main import run_simulate
            r = run_simulate(sample_config, mock_args)
            assert r["status"] == "completed"
            assert r["mode"] == "simulate"

    def test_small_mode_overrides_config(self, sample_config, mock_args):
        mock_args.small = True
        with patch("src.data.SimulatorOrchestrator") as M:
            inst = MagicMock()
            inst.run.return_value = {
                "summary": {"num_customers": 50, "num_events": 500, "churn_rate": 0.2}
            }
            M.return_value = inst
            from src.main import run_simulate
            run_simulate(sample_config, mock_args)
            assert sample_config["simulation"]["num_customers"] == 50


class TestRunOptimize:
    """Tests for run_optimize handler."""

    def test_budget_flag_overrides(self, sample_config, mock_args):
        mock_args.budget = 100_000_000
        with patch("src.models.budget_optimizer.BudgetOptimizer") as M:
            inst = MagicMock()
            inst.optimize.return_value = pd.DataFrame({
                "customer_id": [0], "allocated_budget": [1000],
            })
            M.return_value = inst
            from src.main import run_optimize
            r = run_optimize(sample_config, mock_args)
            assert r["total_budget"] == 100_000_000

    def test_default_budget_from_config(self, sample_config, mock_args):
        mock_args.budget = None
        with patch("src.models.budget_optimizer.BudgetOptimizer") as M:
            inst = MagicMock()
            inst.optimize.return_value = pd.DataFrame({
                "customer_id": [0], "allocated_budget": [1000],
            })
            M.return_value = inst
            from src.main import run_optimize
            r = run_optimize(sample_config, mock_args)
            assert r["total_budget"] == 50_000_000


class TestRunDashboard:
    """Tests for run_dashboard handler."""

    def test_missing_app_returns_error(self, sample_config, mock_args):
        with patch("src.main.PROJECT_ROOT", Path("/nonexistent")):
            from src.main import run_dashboard
            r = run_dashboard(sample_config, mock_args)
            assert r["status"] == "error"


# ===========================================================================
# Tests: Handler signature consistency
# ===========================================================================

class TestModeHandlerSignatures:
    """All mode handlers must accept (config, args)."""

    @pytest.mark.parametrize("mode_name", [
        "simulate", "train", "uplift", "clv", "optimize",
        "ab_test", "survival", "recommend", "cohort",
        "segment", "features", "monitor", "dashboard", "all",
    ])
    def test_handler_accepts_config_and_args(self, mode_name):
        import inspect
        handler = MODES[mode_name]
        sig = inspect.signature(handler)
        params = list(sig.parameters.keys())
        assert len(params) == 2
        assert params[0] == "config"
        assert params[1] == "args"


# ===========================================================================
# Tests: main() entry point
# ===========================================================================

class TestMainFunction:
    """Tests for main() dispatch."""

    def _patch_modes(self, mode_name, return_value):
        """Helper to patch a specific handler in the MODES dict."""
        original = MODES[mode_name]
        mock_handler = MagicMock(return_value=return_value)
        MODES[mode_name] = mock_handler
        return mock_handler, original

    def test_dispatches_to_handler(self, config_file, tmp_data_dir):
        mock_handler, orig = self._patch_modes(
            "simulate", {"mode": "simulate", "status": "completed"})
        try:
            r = main(["--mode", "simulate", "--config", config_file,
                       "--data", tmp_data_dir, "-q"])
            mock_handler.assert_called_once()
            assert r["mode"] == "simulate"
        finally:
            MODES["simulate"] = orig

    def test_missing_config_raises(self):
        with pytest.raises(FileNotFoundError):
            main(["--mode", "train", "--config", "/no/such.yaml", "-q"])

    def test_invalid_mode_exits(self, config_file):
        with pytest.raises(SystemExit):
            main(["--mode", "nonexistent", "--config", config_file])

    def test_missing_mode_exits(self):
        with pytest.raises(SystemExit):
            main([])

    def test_quiet_suppresses_output(self, config_file, tmp_data_dir, capsys):
        mock_h, orig = self._patch_modes(
            "simulate", {"mode": "simulate", "status": "completed"})
        try:
            main(["--mode", "simulate", "--config", config_file,
                  "--data", tmp_data_dir, "-q"])
            captured = capsys.readouterr()
            assert captured.out == ""
        finally:
            MODES["simulate"] = orig

    def test_non_quiet_prints_json(self, config_file, tmp_data_dir, capsys):
        mock_h, orig = self._patch_modes(
            "simulate", {"mode": "simulate", "status": "completed"})
        try:
            main(["--mode", "simulate", "--config", config_file,
                  "--data", tmp_data_dir])
            captured = capsys.readouterr()
            output = json.loads(captured.out)
            assert output["mode"] == "simulate"
        finally:
            MODES["simulate"] = orig

    def test_budget_passed_through(self, config_file, tmp_data_dir):
        mock_h, orig = self._patch_modes(
            "optimize", {"mode": "optimize", "status": "completed"})
        try:
            main(["--mode", "optimize", "--config", config_file,
                  "--data", tmp_data_dir, "--budget", "75000000", "-q"])
            call_args = mock_h.call_args[0][1]
            assert call_args.budget == 75_000_000
        finally:
            MODES["optimize"] = orig

    def test_abtest_alias_dispatches(self, config_file, tmp_data_dir):
        mock_h, orig = self._patch_modes(
            "abtest", {"mode": "ab_test", "status": "completed"})
        try:
            r = main(["--mode", "abtest", "--config", config_file,
                       "--data", tmp_data_dir, "-q"])
            mock_h.assert_called_once()
        finally:
            MODES["abtest"] = orig

    def test_all_mode_dispatches(self, config_file, tmp_data_dir):
        mock_h, orig = self._patch_modes(
            "all", {"mode": "all", "status": "completed"})
        try:
            r = main(["--mode", "all", "--config", config_file,
                       "--data", tmp_data_dir, "-q"])
            assert r["mode"] == "all"
        finally:
            MODES["all"] = orig
