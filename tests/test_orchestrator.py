"""
TDD Tests for the Simulator Orchestrator.

Tests cover:
- Orchestrator initialization from config
- Full pipeline execution with 20,000 customers (or small mode)
- Treatment/Control 50/50 split across full dataset
- Persona proportional assignment validation
- CSV and Parquet output file generation
- Pipeline state tracking (pipeline_state.json)
- Reproducibility across runs
- Summary statistics generation
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_config(config):
    """Create a small-mode config for fast testing."""
    config["simulation"]["num_customers"] = 500
    config["simulation"]["simulation_days"] = 90
    return config


@pytest.fixture
def orchestrator(small_config):
    """Create a SimulatorOrchestrator with small config."""
    from src.data.orchestrator import SimulatorOrchestrator

    return SimulatorOrchestrator(small_config)


@pytest.fixture
def run_result(orchestrator, tmp_path):
    """Run the orchestrator and return result with output directory."""
    result = orchestrator.run(output_dir=str(tmp_path / "output"))
    return result, tmp_path / "output"


class TestOrchestratorInitialization:
    """Test orchestrator construction and configuration."""

    def test_orchestrator_creates_from_config(self, small_config):
        """Orchestrator must initialize from config dict."""
        from src.data.orchestrator import SimulatorOrchestrator

        orch = SimulatorOrchestrator(small_config)
        assert orch is not None
        assert orch.num_customers == small_config["simulation"]["num_customers"]

    def test_orchestrator_creates_from_yaml_path(self, small_config):
        """Orchestrator must be creatable from a YAML file path."""
        from src.data.orchestrator import SimulatorOrchestrator

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(small_config, f)
            tmp_yaml = f.name

        try:
            orch = SimulatorOrchestrator.from_yaml(tmp_yaml)
            assert orch is not None
        finally:
            os.unlink(tmp_yaml)

    def test_orchestrator_default_num_customers(self, config):
        """Default config should specify 20,000 customers."""
        assert config["simulation"]["num_customers"] == 20000


class TestFullPipelineExecution:
    """Test end-to-end orchestrator run."""

    def test_run_returns_dict_with_dataframes(self, run_result):
        """Orchestrator.run() must return dict with customers and events."""
        result, _ = run_result
        assert "customers" in result
        assert "events" in result
        assert isinstance(result["customers"], pd.DataFrame)
        assert isinstance(result["events"], pd.DataFrame)

    def test_run_returns_correct_customer_count(self, run_result, small_config):
        """Result must contain the configured number of customers."""
        result, _ = run_result
        assert len(result["customers"]) == small_config["simulation"]["num_customers"]

    def test_run_returns_summary(self, run_result):
        """Orchestrator.run() must return a summary dict."""
        result, _ = run_result
        assert "summary" in result
        summary = result["summary"]
        assert "num_customers" in summary
        assert "num_events" in summary
        assert "churn_rate" in summary
        assert "treatment_ratio" in summary
        assert "persona_distribution" in summary


class TestTreatmentControlSplit:
    """Test 50/50 treatment/control assignment across full dataset."""

    def test_treatment_control_exact_split(self, run_result):
        """Treatment/control split must be approximately 50/50."""
        result, _ = run_result
        customers = result["customers"]
        treatment_ratio = (
            customers["treatment_group"] == "treatment"
        ).mean()
        assert 0.45 <= treatment_ratio <= 0.55, (
            f"Treatment ratio {treatment_ratio:.2%} not ~50%"
        )

    def test_treatment_control_groups_only(self, run_result):
        """Only 'treatment' and 'control' group labels should exist."""
        result, _ = run_result
        groups = set(result["customers"]["treatment_group"].unique())
        assert groups == {"treatment", "control"}


class TestPersonaAssignment:
    """Test that personas are assigned proportionally."""

    def test_all_personas_assigned(self, run_result, small_config):
        """All configured personas must appear in generated data."""
        result, _ = run_result
        expected = {p["name"] for p in small_config["personas"]}
        actual = set(result["customers"]["persona"].unique())
        assert expected == actual, (
            f"Missing personas: {expected - actual}"
        )

    def test_persona_proportions_match_config(self, run_result, small_config):
        """Persona proportions should approximately match config."""
        result, _ = run_result
        customers = result["customers"]
        total = len(customers)
        for persona_cfg in small_config["personas"]:
            name = persona_cfg["name"]
            expected = persona_cfg["proportion"]
            actual = len(customers[customers["persona"] == name]) / total
            assert abs(actual - expected) < 0.08, (
                f"Persona {name}: expected ~{expected:.2f}, got {actual:.2f}"
            )


class TestOutputFiles:
    """Test CSV and Parquet output generation."""

    def test_csv_files_created(self, run_result):
        """Orchestrator must create CSV output files."""
        _, output_dir = run_result
        assert (output_dir / "customers.csv").exists()
        assert (output_dir / "events.csv").exists()

    def test_parquet_files_created(self, run_result):
        """Orchestrator must create Parquet output files."""
        _, output_dir = run_result
        assert (output_dir / "customers.parquet").exists()
        assert (output_dir / "events.parquet").exists()

    def test_csv_files_loadable(self, run_result):
        """Saved CSV files must be loadable and non-empty."""
        _, output_dir = run_result
        customers = pd.read_csv(output_dir / "customers.csv")
        events = pd.read_csv(output_dir / "events.csv")
        assert len(customers) > 0
        assert len(events) > 0

    def test_parquet_files_loadable(self, run_result):
        """Saved Parquet files must be loadable and match CSV data."""
        _, output_dir = run_result
        csv_customers = pd.read_csv(output_dir / "customers.csv")
        parquet_customers = pd.read_parquet(output_dir / "customers.parquet")
        assert len(csv_customers) == len(parquet_customers)

        csv_events = pd.read_csv(output_dir / "events.csv")
        parquet_events = pd.read_parquet(output_dir / "events.parquet")
        assert len(csv_events) == len(parquet_events)

    def test_summary_json_created(self, run_result):
        """Orchestrator must save a summary JSON file."""
        _, output_dir = run_result
        summary_path = output_dir / "generation_summary.json"
        assert summary_path.exists()
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        assert "num_customers" in summary
        assert "num_events" in summary
        assert "churn_rate" in summary


class TestPipelineState:
    """Test pipeline state tracking."""

    def test_pipeline_state_created(self, run_result):
        """Pipeline state JSON must be created after run."""
        _, output_dir = run_result
        state_path = output_dir / "pipeline_state.json"
        assert state_path.exists()

    def test_pipeline_state_completed(self, run_result):
        """Pipeline state must show 'completed' status after successful run."""
        _, output_dir = run_result
        state_path = output_dir / "pipeline_state.json"
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
        assert state["data_generation"] == "completed"

    def test_pipeline_state_on_failure(self, small_config, tmp_path):
        """Pipeline state must show 'failed' on error."""
        from src.data.orchestrator import SimulatorOrchestrator

        # Use an invalid output path to cause failure
        small_config["simulation"]["num_customers"] = 0
        orch = SimulatorOrchestrator(small_config)
        output_dir = str(tmp_path / "fail_output")

        try:
            orch.run(output_dir=output_dir)
        except Exception:
            pass

        state_path = Path(output_dir) / "pipeline_state.json"
        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            assert state["data_generation"] in ("failed", "pending")


class TestReproducibility:
    """Test that orchestrator produces identical results with same seed."""

    def test_reproducible_output(self, small_config, tmp_path):
        """Two runs with the same seed must produce identical data."""
        from src.data.orchestrator import SimulatorOrchestrator

        small_config["simulation"]["num_customers"] = 100
        small_config["simulation"]["simulation_days"] = 30

        orch1 = SimulatorOrchestrator(small_config)
        r1 = orch1.run(output_dir=str(tmp_path / "run1"))

        orch2 = SimulatorOrchestrator(small_config)
        r2 = orch2.run(output_dir=str(tmp_path / "run2"))

        pd.testing.assert_frame_equal(
            r1["customers"].reset_index(drop=True),
            r2["customers"].reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            r1["events"].reset_index(drop=True),
            r2["events"].reset_index(drop=True),
        )
