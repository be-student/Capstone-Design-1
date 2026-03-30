"""Tests for PipelineRunner checkpoint/resume logic."""

import json
import argparse
import pytest
from unittest.mock import MagicMock

from src.pipeline.runner import PipelineRunner, PIPELINE_STEP_ORDER


@pytest.fixture
def runner(tmp_path, config_dict):
    """Create a PipelineRunner with temp state file."""
    state_path = str(tmp_path / "pipeline_state.json")
    return PipelineRunner(
        config=config_dict,
        state_path=state_path,
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def config_dict():
    """Minimal config dict for testing."""
    return {
        "simulation": {"random_seed": 42},
        "pipeline": {
            "train_months": 10,
            "test_months": 2,
            "ensemble_weight_ml": 0.6,
            "ensemble_weight_dl": 0.4,
        },
    }


class TestPipelineRunnerInit:
    """Test PipelineRunner initialization."""

    def test_instantiation(self, runner):
        """PipelineRunner can be instantiated."""
        assert runner is not None

    def test_has_step_order(self, runner):
        """PipelineRunner exposes step order."""
        steps = runner.get_step_order()
        assert isinstance(steps, list)
        assert len(steps) >= 13

    def test_step_order_starts_with_data_generation(self, runner):
        """First step must be data_generation."""
        assert runner.get_step_order()[0] == "data_generation"

    def test_step_order_ends_with_mlflow_logging(self, runner):
        """Last step must be mlflow_logging."""
        assert runner.get_step_order()[-1] == "mlflow_logging"


class TestRunStep:
    """Test run_step with checkpoint wrapping."""

    def test_run_step_creates_pending_then_completed(self, runner, tmp_path):
        """run_step saves pending checkpoint, then completed on success."""
        ok_handler = MagicMock(return_value={"status": "completed"})
        runner.register_step("test_step", ok_handler)

        result = runner.run_step("test_step")
        assert result == {"status": "completed"}

        state = runner.get_state()
        assert state["stages"]["test_step"]["status"] == "completed"
        assert "duration_seconds" in state["stages"]["test_step"]["metadata"]

    def test_run_step_marks_failed_on_error(self, runner):
        """run_step marks stage failed if handler raises."""
        fail_handler = MagicMock(side_effect=ValueError("boom"))
        runner.register_step("fail_step", fail_handler)

        with pytest.raises(RuntimeError, match="boom"):
            runner.run_step("fail_step")

        state = runner.get_state()
        assert state["stages"]["fail_step"]["status"] == "failed"
        assert "boom" in state["stages"]["fail_step"]["metadata"]["error"]

    def test_run_step_unknown_raises(self, runner):
        """run_step raises ValueError for unknown step."""
        with pytest.raises(ValueError, match="Unknown pipeline step"):
            runner.run_step("nonexistent_step")


class TestResume:
    """Test resume from checkpoint."""

    def test_resume_skips_completed(self, runner):
        """resume() skips stages marked completed."""
        handler1 = MagicMock(return_value={"status": "ok"})
        handler2 = MagicMock(return_value={"status": "ok"})

        # Override step order to just 2 steps
        runner._step_order = ["step_a", "step_b"]
        runner._step_handlers = {
            "step_a": handler1,
            "step_b": handler2,
        }

        # Mark step_a as already completed
        runner._state.mark_complete("step_a")

        result = runner.resume()
        assert result["status"] == "completed"

        # step_a handler should NOT have been called
        handler1.assert_not_called()
        # step_b handler SHOULD have been called
        handler2.assert_called_once()

    def test_resume_all_completed(self, runner):
        """resume() returns immediately when all stages are done."""
        runner._step_order = ["step_a", "step_b"]
        runner._step_handlers = {"step_a": MagicMock(), "step_b": MagicMock()}
        runner._state.mark_complete("step_a")
        runner._state.mark_complete("step_b")

        result = runner.resume()
        assert result["status"] == "completed"

    def test_resume_retries_failed(self, runner):
        """resume() retries stages that previously failed."""
        handler = MagicMock(return_value={"status": "ok"})
        runner._step_order = ["step_a"]
        runner._step_handlers = {"step_a": handler}
        runner._state.mark_failed("step_a", error="previous error")

        result = runner.resume()
        assert result["status"] == "completed"
        handler.assert_called_once()


class TestRunForce:
    """Test run with force reset."""

    def test_run_force_resets_state(self, runner):
        """run(force=True) clears all checkpoints and re-runs."""
        handler = MagicMock(return_value={"status": "ok"})
        runner._step_order = ["step_a"]
        runner._step_handlers = {"step_a": handler}
        runner._state.mark_complete("step_a")

        result = runner.run(force=True)
        assert result["status"] == "completed"
        # Should have re-run since force=True
        handler.assert_called_once()


class TestSaveGetState:
    """Test state persistence methods."""

    def test_get_state_empty_initially(self, runner):
        """get_state returns empty stages initially."""
        state = runner.get_state()
        assert state["stages"] == {}

    def test_save_state_persists(self, runner):
        """save_state writes to disk."""
        runner.save_state({
            "stages": {"test": {"status": "completed", "timestamp": "now"}}
        })
        state = runner.get_state()
        assert "test" in state["stages"]

    def test_save_state_no_args_creates_file(self, runner, tmp_path):
        """save_state() with no args ensures file exists."""
        runner.save_state()
        state = runner.get_state()
        assert "stages" in state


class TestSeedTracking:
    """Test that pipeline state tracks seed for reproducibility."""

    def test_resume_records_seed(self, runner):
        """resume() records the random seed in state."""
        runner._step_order = []
        runner._step_handlers = {}

        runner.resume()
        state = runner.get_state()
        assert state["seed"] == 42


class TestCanonicalStepOrder:
    """Test PIPELINE_STEP_ORDER constant."""

    def test_has_14_steps(self):
        """Canonical order has 14 steps."""
        assert len(PIPELINE_STEP_ORDER) == 14

    def test_first_is_data_generation(self):
        assert PIPELINE_STEP_ORDER[0] == "data_generation"

    def test_last_is_mlflow_logging(self):
        assert PIPELINE_STEP_ORDER[-1] == "mlflow_logging"

    def test_training_after_features(self):
        fe_idx = PIPELINE_STEP_ORDER.index("feature_engineering")
        ml_idx = PIPELINE_STEP_ORDER.index("ml_model_training")
        assert ml_idx > fe_idx

    def test_ensemble_after_dl(self):
        dl_idx = PIPELINE_STEP_ORDER.index("dl_model_training")
        ens_idx = PIPELINE_STEP_ORDER.index("ensemble_creation")
        assert ens_idx > dl_idx
