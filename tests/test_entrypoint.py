"""Tests for the orchestration entrypoint script (scripts/run_pipeline_and_dashboard.py).

Verifies that the entrypoint correctly:
- Parses CLI arguments and environment variables
- Runs pipeline before dashboard
- Handles skip-pipeline and pipeline-only modes
- Verifies pipeline outputs
- Reads pipeline state
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the orchestration module
from scripts.run_pipeline_and_dashboard import (
    check_pipeline_state,
    parse_args,
    verify_pipeline_outputs,
)


# ===========================================================================
# Tests for parse_args
# ===========================================================================
class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_args(self):
        """Default arguments should have reasonable values."""
        args = parse_args([])
        assert args.skip_pipeline is False
        assert args.pipeline_only is False
        assert args.small is False
        assert args.verbose is False
        assert args.quiet is False
        assert args.port == 8501
        assert args.max_retries >= 1

    def test_skip_pipeline_flag(self):
        """--skip-pipeline should set skip_pipeline to True."""
        args = parse_args(["--skip-pipeline"])
        assert args.skip_pipeline is True

    def test_pipeline_only_flag(self):
        """--pipeline-only should set pipeline_only to True."""
        args = parse_args(["--pipeline-only"])
        assert args.pipeline_only is True

    def test_small_flag(self):
        """--small should set small to True."""
        args = parse_args(["--small"])
        assert args.small is True

    def test_verbose_flag(self):
        """-v/--verbose should set verbose to True."""
        args = parse_args(["-v"])
        assert args.verbose is True
        args2 = parse_args(["--verbose"])
        assert args2.verbose is True

    def test_custom_port(self):
        """--port should set custom Streamlit port."""
        args = parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_custom_max_retries(self):
        """--max-retries should set retry count."""
        args = parse_args(["--max-retries", "3"])
        assert args.max_retries == 3

    def test_config_path(self):
        """--config should set config path."""
        args = parse_args(["--config", "/tmp/test.yaml"])
        assert args.config == "/tmp/test.yaml"

    def test_env_skip_pipeline(self):
        """SKIP_PIPELINE env var should override default."""
        with patch.dict(os.environ, {"SKIP_PIPELINE": "true"}):
            args = parse_args([])
            assert args.skip_pipeline is True

    def test_env_pipeline_only(self):
        """PIPELINE_ONLY env var should override default."""
        with patch.dict(os.environ, {"PIPELINE_ONLY": "true"}):
            args = parse_args([])
            assert args.pipeline_only is True


# ===========================================================================
# Tests for verify_pipeline_outputs
# ===========================================================================
class TestVerifyPipelineOutputs:
    """Tests for pipeline output verification."""

    def test_returns_dict(self, tmp_path):
        """verify_pipeline_outputs should return a dictionary."""
        with patch(
            "scripts.run_pipeline_and_dashboard.PROJECT_ROOT", tmp_path
        ):
            (tmp_path / "data" / "raw").mkdir(parents=True)
            (tmp_path / "models").mkdir(parents=True)
            (tmp_path / "results").mkdir(parents=True)
            result = verify_pipeline_outputs()
            assert isinstance(result, dict)

    def test_empty_dirs_return_false(self, tmp_path):
        """Empty output directories should return False."""
        with patch(
            "scripts.run_pipeline_and_dashboard.PROJECT_ROOT", tmp_path
        ):
            (tmp_path / "data" / "raw").mkdir(parents=True)
            (tmp_path / "models").mkdir(parents=True)
            (tmp_path / "results").mkdir(parents=True)
            result = verify_pipeline_outputs()
            assert result["data"] is False
            assert result["models"] is False
            assert result["results"] is False

    def test_populated_dirs_return_true(self, tmp_path):
        """Populated output directories should return True."""
        with patch(
            "scripts.run_pipeline_and_dashboard.PROJECT_ROOT", tmp_path
        ):
            data_dir = tmp_path / "data" / "raw"
            data_dir.mkdir(parents=True)
            (data_dir / "customers.csv").write_text("id,name\n1,Alice")

            models_dir = tmp_path / "models"
            models_dir.mkdir(parents=True)
            (models_dir / "model.pkl").write_text("model")

            results_dir = tmp_path / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "metrics.json").write_text("{}")

            result = verify_pipeline_outputs()
            assert result["data"] is True
            assert result["models"] is True
            assert result["results"] is True


# ===========================================================================
# Tests for check_pipeline_state
# ===========================================================================
class TestCheckPipelineState:
    """Tests for pipeline state reading."""

    def test_no_state_file_returns_none(self, tmp_path):
        """Missing state file should return None."""
        state_file = tmp_path / "pipeline_state.json"
        with patch(
            "scripts.run_pipeline_and_dashboard.PIPELINE_STATE_FILE",
            state_file,
        ):
            result = check_pipeline_state()
            assert result is None

    def test_valid_state_file_returns_dict(self, tmp_path):
        """Valid state file should return parsed dictionary."""
        state_file = tmp_path / "pipeline_state.json"
        state = {
            "steps": {
                "data_generation": {"status": "completed"},
                "ml_model_training": {"status": "pending"},
            }
        }
        state_file.write_text(json.dumps(state))
        with patch(
            "scripts.run_pipeline_and_dashboard.PIPELINE_STATE_FILE",
            state_file,
        ):
            result = check_pipeline_state()
            assert result is not None
            assert "steps" in result

    def test_corrupt_state_file_returns_none(self, tmp_path):
        """Corrupt state file should return None."""
        state_file = tmp_path / "pipeline_state.json"
        state_file.write_text("{invalid json")
        with patch(
            "scripts.run_pipeline_and_dashboard.PIPELINE_STATE_FILE",
            state_file,
        ):
            result = check_pipeline_state()
            assert result is None


# ===========================================================================
# Tests for entrypoint.sh (bash script)
# ===========================================================================
class TestBashEntrypoint:
    """Tests for the bash entrypoint script."""

    def test_script_is_executable(self):
        """entrypoint.sh should be executable."""
        script = PROJECT_ROOT / "scripts" / "entrypoint.sh"
        assert script.exists(), "entrypoint.sh not found"
        assert os.access(script, os.X_OK), "entrypoint.sh not executable"

    def test_script_syntax_valid(self):
        """entrypoint.sh should pass bash syntax check."""
        script = PROJECT_ROOT / "scripts" / "entrypoint.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"


# ===========================================================================
# Tests for Docker Compose configuration
# ===========================================================================
class TestDockerComposeConfig:
    """Tests for docker-compose.yml orchestration settings."""

    @pytest.fixture
    def compose_config(self):
        """Load docker-compose.yml."""
        import yaml
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_dashboard_depends_on_pipeline(self, compose_config):
        """Dashboard should depend on pipeline completion."""
        dashboard = compose_config["services"]["dashboard"]
        deps = dashboard.get("depends_on", {})
        assert "pipeline" in deps, (
            "Dashboard must depend on pipeline service"
        )

    def test_pipeline_depends_on_mlflow_and_redis(self, compose_config):
        """Pipeline should depend on mlflow and redis."""
        pipeline = compose_config["services"]["pipeline"]
        deps = pipeline.get("depends_on", {})
        assert "mlflow" in deps
        assert "redis" in deps

    def test_four_services_defined(self, compose_config):
        """docker-compose.yml should define exactly 4 services."""
        services = compose_config["services"]
        assert len(services) == 4
        assert set(services.keys()) == {
            "mlflow", "redis", "pipeline", "dashboard"
        }

    def test_dashboard_has_shared_volumes(self, compose_config):
        """Dashboard should mount shared data volumes."""
        dashboard = compose_config["services"]["dashboard"]
        volumes = dashboard.get("volumes", [])
        volume_strs = [str(v) for v in volumes]
        # Check essential shared mounts
        assert any("data" in v for v in volume_strs)
        assert any("models" in v for v in volume_strs)
        assert any("results" in v for v in volume_strs)
