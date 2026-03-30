"""Tests for PipelineState class (TDD - written first)."""

import json
import os
import tempfile
from datetime import datetime

import pytest

from src.pipeline.pipeline_state import PipelineState


class TestPipelineStateInit:
    """Test PipelineState initialization."""

    def test_init_creates_state_file_path(self, tmp_path):
        """PipelineState stores the path to pipeline_state.json."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        assert ps.state_path == str(state_path)

    def test_init_default_path(self, tmp_path):
        """PipelineState uses default path when none given."""
        ps = PipelineState()
        assert ps.state_path.endswith("pipeline_state.json")

    def test_init_with_custom_path(self, tmp_path):
        """PipelineState accepts a custom state file path."""
        custom = str(tmp_path / "custom_state.json")
        ps = PipelineState(custom)
        assert ps.state_path == custom


class TestSaveCheckpoint:
    """Test save_checkpoint method."""

    def test_save_checkpoint_creates_file(self, tmp_path):
        """save_checkpoint creates pipeline_state.json if missing."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("data_generation", "completed")
        assert state_path.exists()

    def test_save_checkpoint_stores_stage_name(self, tmp_path):
        """Checkpoint contains the stage name."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("data_generation", "completed")
        data = json.loads(state_path.read_text())
        assert "data_generation" in data["stages"]
        assert data["stages"]["data_generation"]["status"] == "completed"

    def test_save_checkpoint_stores_timestamp(self, tmp_path):
        """Checkpoint includes an ISO-format timestamp."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("feature_engineering", "pending")
        data = json.loads(state_path.read_text())
        ts = data["stages"]["feature_engineering"]["timestamp"]
        # Should parse as valid ISO timestamp
        datetime.fromisoformat(ts)

    def test_save_checkpoint_with_metadata(self, tmp_path):
        """Checkpoint can store arbitrary metadata dict."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        meta = {"num_rows": 1000, "auc": 0.85}
        ps.save_checkpoint("ml_training", "completed", metadata=meta)
        data = json.loads(state_path.read_text())
        assert data["stages"]["ml_training"]["metadata"] == meta

    def test_save_checkpoint_preserves_previous_stages(self, tmp_path):
        """Saving a new stage preserves previously saved stages."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("data_generation", "completed")
        ps.save_checkpoint("feature_engineering", "completed")
        data = json.loads(state_path.read_text())
        assert "data_generation" in data["stages"]
        assert "feature_engineering" in data["stages"]

    def test_save_checkpoint_overwrites_same_stage(self, tmp_path):
        """Re-saving a stage overwrites its previous entry."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("ml_training", "pending")
        ps.save_checkpoint("ml_training", "completed", metadata={"auc": 0.80})
        data = json.loads(state_path.read_text())
        assert data["stages"]["ml_training"]["status"] == "completed"
        assert data["stages"]["ml_training"]["metadata"]["auc"] == 0.80


class TestLoadState:
    """Test load_state method."""

    def test_load_state_returns_dict(self, tmp_path):
        """load_state returns a dictionary."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("data_generation", "completed")
        state = ps.load_state()
        assert isinstance(state, dict)
        assert "stages" in state

    def test_load_state_empty_when_no_file(self, tmp_path):
        """load_state returns empty stages when file doesn't exist."""
        state_path = tmp_path / "nonexistent.json"
        ps = PipelineState(str(state_path))
        state = ps.load_state()
        assert state == {"stages": {}}

    def test_load_state_handles_corrupt_json(self, tmp_path):
        """load_state returns empty on corrupt JSON."""
        state_path = tmp_path / "pipeline_state.json"
        state_path.write_text("NOT VALID JSON{{{")
        ps = PipelineState(str(state_path))
        state = ps.load_state()
        assert state == {"stages": {}}


class TestMarkComplete:
    """Test mark_complete method."""

    def test_mark_complete_sets_status(self, tmp_path):
        """mark_complete sets stage status to 'completed'."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("ml_training", "pending")
        ps.mark_complete("ml_training")
        data = json.loads(state_path.read_text())
        assert data["stages"]["ml_training"]["status"] == "completed"

    def test_mark_complete_with_metadata(self, tmp_path):
        """mark_complete can attach metadata."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("ml_training", "pending")
        ps.mark_complete("ml_training", metadata={"auc": 0.82})
        data = json.loads(state_path.read_text())
        assert data["stages"]["ml_training"]["metadata"]["auc"] == 0.82


class TestMarkFailed:
    """Test mark_failed method."""

    def test_mark_failed_sets_status(self, tmp_path):
        """mark_failed sets stage status to 'failed'."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("dl_training", "pending")
        ps.mark_failed("dl_training", error="OOM error")
        data = json.loads(state_path.read_text())
        assert data["stages"]["dl_training"]["status"] == "failed"

    def test_mark_failed_stores_error(self, tmp_path):
        """mark_failed stores the error message in metadata."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.mark_failed("dl_training", error="OOM error")
        data = json.loads(state_path.read_text())
        assert data["stages"]["dl_training"]["metadata"]["error"] == "OOM error"


class TestGetNextPending:
    """Test get_next_pending method."""

    def test_get_next_pending_returns_first_pending(self, tmp_path):
        """get_next_pending returns the first stage with 'pending' status."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        stages = [
            "data_generation",
            "feature_engineering",
            "ml_training",
            "dl_training",
            "ensemble",
        ]
        ps.save_checkpoint("data_generation", "completed")
        ps.save_checkpoint("feature_engineering", "pending")
        ps.save_checkpoint("ml_training", "pending")
        result = ps.get_next_pending(stages)
        assert result == "feature_engineering"

    def test_get_next_pending_returns_none_when_all_complete(self, tmp_path):
        """get_next_pending returns None when all stages are completed."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        stages = ["data_generation", "feature_engineering"]
        ps.save_checkpoint("data_generation", "completed")
        ps.save_checkpoint("feature_engineering", "completed")
        result = ps.get_next_pending(stages)
        assert result is None

    def test_get_next_pending_returns_missing_stage(self, tmp_path):
        """get_next_pending returns stages not yet in state file."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        stages = ["data_generation", "feature_engineering"]
        ps.save_checkpoint("data_generation", "completed")
        # feature_engineering not saved yet → treated as pending
        result = ps.get_next_pending(stages)
        assert result == "feature_engineering"

    def test_get_next_pending_skips_failed(self, tmp_path):
        """get_next_pending does NOT skip failed stages (returns them)."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        stages = ["data_generation", "feature_engineering"]
        ps.save_checkpoint("data_generation", "failed")
        result = ps.get_next_pending(stages)
        assert result == "data_generation"


class TestGetStageStatus:
    """Test get_stage_status method."""

    def test_get_stage_status_returns_status(self, tmp_path):
        """get_stage_status returns the status string for a stage."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("ml_training", "completed")
        assert ps.get_stage_status("ml_training") == "completed"

    def test_get_stage_status_unknown_returns_none(self, tmp_path):
        """get_stage_status returns None for unknown stage."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        assert ps.get_stage_status("nonexistent") is None


class TestReset:
    """Test reset method."""

    def test_reset_clears_all_stages(self, tmp_path):
        """reset clears all stage data."""
        state_path = tmp_path / "pipeline_state.json"
        ps = PipelineState(str(state_path))
        ps.save_checkpoint("data_generation", "completed")
        ps.save_checkpoint("ml_training", "completed")
        ps.reset()
        state = ps.load_state()
        assert state["stages"] == {}
