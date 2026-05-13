"""Pipeline state management via pipeline_state.json.

Provides checkpoint-based pipeline state tracking so that long-running
pipelines can resume from the last successful stage after a failure.

Usage:
    ps = PipelineState("pipeline_state.json")
    ps.save_checkpoint("data_generation", "pending")
    # ... run stage ...
    ps.mark_complete("data_generation", metadata={"rows": 20000})
    # On failure:
    ps.mark_failed("ml_training", error="OOM")
    # Resume:
    next_stage = ps.get_next_pending(PIPELINE_STAGES)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = "pipeline_state.json"


class PipelineState:
    """Manages reading/writing pipeline_state.json for checkpoint tracking.

    Each pipeline stage is recorded with:
        - status: 'pending', 'completed', or 'failed'
        - timestamp: ISO-format datetime of last update
        - metadata: arbitrary dict (e.g., metrics, error messages)

    Attributes:
        state_path: Absolute or relative path to the JSON state file.
    """

    def __init__(self, state_path: Optional[str] = None) -> None:
        """Initialize PipelineState.

        Args:
            state_path: Path to pipeline_state.json. Defaults to
                        'pipeline_state.json' in the current directory.
        """
        self.state_path = state_path or DEFAULT_STATE_PATH

    def load_state(self) -> Dict[str, Any]:
        """Load the current pipeline state from disk.

        Returns:
            Dictionary with 'stages' key mapping stage names to their
            status, timestamp, and metadata. Returns empty stages dict
            if file is missing or corrupt.
        """
        if not os.path.exists(self.state_path):
            return {"stages": {}}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "stages" not in data:
                data["stages"] = {}
            return data
        except (json.JSONDecodeError, IOError):
            logger.warning(
                "Corrupt or unreadable state file: %s", self.state_path
            )
            return {"stages": {}}

    def _save_state(self, state: Dict[str, Any]) -> None:
        """Write state dictionary to disk.

        Creates parent directories if needed.

        Args:
            state: Full state dictionary to persist.
        """
        parent = os.path.dirname(self.state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def save_checkpoint(
        self,
        stage_name: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a checkpoint for a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage (e.g., 'data_generation').
            status: Status string ('pending', 'completed', 'failed').
            metadata: Optional dictionary of extra information.
        """
        state = self.load_state()
        state["stages"][stage_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save_state(state)
        logger.info("Checkpoint saved: %s → %s", stage_name, status)

    def mark_complete(
        self,
        stage_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a pipeline stage as completed.

        Args:
            stage_name: Name of the pipeline stage.
            metadata: Optional metadata to attach.
        """
        self.save_checkpoint(stage_name, "completed", metadata=metadata)

    def mark_failed(
        self,
        stage_name: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a pipeline stage as failed.

        Args:
            stage_name: Name of the pipeline stage.
            error: Error message string.
            metadata: Optional additional metadata.
        """
        meta = metadata or {}
        if error:
            meta["error"] = error
        self.save_checkpoint(stage_name, "failed", metadata=meta)

    def get_stage_status(self, stage_name: str) -> Optional[str]:
        """Get the status of a specific stage.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            Status string or None if stage not found.
        """
        state = self.load_state()
        stage = state["stages"].get(stage_name)
        if stage is None:
            return None
        return stage["status"]

    def get_next_pending(self, stage_order: List[str]) -> Optional[str]:
        """Get the next stage that needs to be executed.

        Iterates through the ordered list of stages and returns the
        first one that is not 'completed'. Stages that are 'failed'
        or missing from the state file are considered pending.

        Args:
            stage_order: Ordered list of stage names.

        Returns:
            Name of the next pending/failed stage, or None if all complete.
        """
        state = self.load_state()
        for stage_name in stage_order:
            stage_data = state["stages"].get(stage_name)
            if stage_data is None or stage_data["status"] != "completed":
                return stage_name
        return None

    def reset(self) -> None:
        """Reset pipeline state, clearing all stage data."""
        self._save_state({"stages": {}})
        logger.info("Pipeline state reset.")
