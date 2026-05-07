"""Pipeline runner with checkpoint/resume logic.

Wraps pipeline stage execution with PipelineState checkpointing so that
the pipeline can resume from the last successful stage on restart.

Usage:
    runner = PipelineRunner(config, state_path="pipeline_state.json")
    runner.run(args)            # Run all stages with checkpointing
    runner.resume(args)         # Resume from last successful stage
    runner.run_step("train", args)  # Run a single named step
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.pipeline.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

# Canonical pipeline step order (16 stages)
PIPELINE_STEP_ORDER: List[str] = [
    "data_generation",
    "preprocessing",
    "feature_engineering",
    "ml_model_training",
    "dl_model_training",
    "ensemble_creation",
    "uplift_modeling",
    "clv_prediction",
    "customer_segmentation",
    "budget_optimization",
    "recommendations",
    "cohort_analysis",
    "ab_testing",
    "survival_analysis",
    "scoring_api_setup",
    "mlflow_logging",
]


class PipelineRunner:
    """Orchestrates pipeline execution with checkpoint/resume support.

    Each pipeline stage is wrapped with checkpoint logic:
    1. Before execution: save checkpoint with status='pending'
    2. After successful execution: mark_complete with metadata
    3. On failure: mark_failed with error details

    On restart, ``resume()`` skips already-completed stages and continues
    from the first non-completed stage.

    Parameters
    ----------
    config : dict
        Full YAML configuration dictionary.
    state_path : str, optional
        Path to the pipeline_state.json checkpoint file.
    output_dir : str, optional
        Base output directory for results.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        state_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir or "results"
        self._state = PipelineState(state_path or "pipeline_state.json")
        self._step_handlers: Dict[str, Callable] = {}
        self._step_order = list(PIPELINE_STEP_ORDER)

        # Register default step handlers (lazy-bound to main.py functions)
        self._register_default_handlers()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_step_order(self) -> List[str]:
        """Return the ordered list of pipeline steps.

        Returns
        -------
        list of str
            Step names in execution order.
        """
        return list(self._step_order)

    def get_state(self) -> Dict[str, Any]:
        """Return the current pipeline state dictionary.

        Returns
        -------
        dict
            Full state including all stage statuses.
        """
        return self._state.load_state()

    def save_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """Persist pipeline state to disk.

        Parameters
        ----------
        state : dict, optional
            If given, merge stage info into current state. Otherwise
            just ensure the current state is written.
        """
        if state is not None:
            current = self._state.load_state()
            if "stages" in state:
                current["stages"].update(state["stages"])
            self._state._save_state(current)
        else:
            # Touch state file to ensure it exists
            current = self._state.load_state()
            self._state._save_state(current)

    def run_step(
        self,
        step_name: str,
        args: Optional[argparse.Namespace] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a single pipeline step with checkpoint wrapping.

        Parameters
        ----------
        step_name : str
            Name of the pipeline step to run.
        args : argparse.Namespace, optional
            CLI arguments to pass to the step handler.
        **kwargs
            Additional keyword arguments for the handler.

        Returns
        -------
        dict
            Result dictionary from the step handler.

        Raises
        ------
        ValueError
            If step_name is not recognised.
        RuntimeError
            If the step handler raises an exception.
        """
        if step_name not in self._step_handlers:
            raise ValueError(
                f"Unknown pipeline step: '{step_name}'. "
                f"Valid steps: {list(self._step_handlers.keys())}"
            )

        handler = self._step_handlers[step_name]
        if args is None:
            args = self._default_args()

        # --- PRE checkpoint ---
        logger.info("Checkpoint: %s → pending", step_name)
        self._state.save_checkpoint(step_name, "pending")

        start_time = time.time()
        try:
            result = handler(self.config, args)
            elapsed = time.time() - start_time

            # --- POST checkpoint (success) ---
            metadata = {
                "duration_seconds": round(elapsed, 2),
                "completed_at": datetime.now().isoformat(),
            }
            if isinstance(result, dict):
                metadata["result_summary"] = {
                    k: v for k, v in result.items()
                    if isinstance(v, (str, int, float, bool))
                }
            self._state.mark_complete(step_name, metadata=metadata)
            logger.info(
                "Checkpoint: %s → completed (%.1fs)", step_name, elapsed
            )
            return result

        except Exception as exc:
            elapsed = time.time() - start_time
            # --- POST checkpoint (failure) ---
            self._state.mark_failed(
                step_name,
                error=str(exc),
                metadata={
                    "duration_seconds": round(elapsed, 2),
                    "failed_at": datetime.now().isoformat(),
                },
            )
            logger.error(
                "Checkpoint: %s → failed (%.1fs): %s", step_name, elapsed, exc
            )
            raise RuntimeError(
                f"Pipeline step '{step_name}' failed: {exc}"
            ) from exc

    def run(
        self,
        args: Optional[argparse.Namespace] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Run the full pipeline from the beginning.

        Parameters
        ----------
        args : argparse.Namespace, optional
            CLI arguments.
        force : bool
            If True, reset state and re-run all stages. Otherwise
            behave like ``resume()`` (skip completed stages).

        Returns
        -------
        dict
            Aggregated results from all pipeline stages.
        """
        if force:
            self._state.reset()
            logger.info("Pipeline state reset — running from scratch.")

        return self.resume(args)

    def resume(
        self,
        args: Optional[argparse.Namespace] = None,
    ) -> Dict[str, Any]:
        """Resume the pipeline from the last successful stage.

        Skips any stage with status='completed' and starts execution
        from the first non-completed stage.

        Parameters
        ----------
        args : argparse.Namespace, optional
            CLI arguments.

        Returns
        -------
        dict
            Aggregated results from all executed stages.
        """
        if args is None:
            args = self._default_args()

        # Record seed for reproducibility
        seed = self.config.get("simulation", {}).get("random_seed", 42)
        state = self._state.load_state()
        state["seed"] = seed
        self._state._save_state(state)

        all_results: Dict[str, Any] = {"mode": "all"}
        total = len(self._step_order)

        # Find resume point
        next_step = self._state.get_next_pending(self._step_order)
        if next_step is None:
            logger.info("All pipeline stages already completed!")
            all_results["status"] = "completed"
            return all_results

        start_idx = self._step_order.index(next_step)
        if start_idx > 0:
            logger.info(
                "Resuming from step '%s' (skipping %d completed stages)",
                next_step, start_idx,
            )

        logger.info("=" * 60)
        logger.info(
            "Running pipeline (%d/%d steps, starting at '%s')",
            total - start_idx, total, next_step,
        )
        logger.info("=" * 60)

        for i, step_name in enumerate(self._step_order):
            step_num = i + 1
            status = self._state.get_stage_status(step_name)

            if status == "completed":
                logger.info(
                    "--- Step %d/%d: %s [SKIPPED - already completed] ---",
                    step_num, total, step_name,
                )
                continue

            logger.info(
                "--- Step %d/%d: %s ---", step_num, total, step_name
            )

            try:
                result = self.run_step(step_name, args)
                all_results[step_name] = result
                logger.info("[OK] %s", step_name)
            except (RuntimeError, Exception) as exc:
                logger.error("[FAIL] %s: %s", step_name, exc)
                import traceback
                traceback.print_exc()
                all_results[step_name] = {
                    "status": "failed",
                    "error": str(exc),
                }
                # Continue to next steps instead of crashing
                logger.warning(
                    "Continuing pipeline despite failure in '%s'",
                    step_name,
                )

        failed_steps = [k for k, v in all_results.items()
                        if isinstance(v, dict) and v.get("status") == "failed"]
        if failed_steps:
            all_results["status"] = "completed_with_errors"
            logger.warning("=" * 60)
            logger.warning(
                "Pipeline completed with %d failed step(s): %s",
                len(failed_steps), ", ".join(failed_steps),
            )
            logger.warning("=" * 60)
        else:
            all_results["status"] = "completed"
            logger.info("=" * 60)
            logger.info("Full pipeline completed!")
            logger.info("=" * 60)
        return all_results

    def register_step(
        self,
        step_name: str,
        handler: Callable,
        position: Optional[int] = None,
    ) -> None:
        """Register (or replace) a pipeline step handler.

        Parameters
        ----------
        step_name : str
            Name of the step.
        handler : callable
            Function(config, args) -> dict to execute.
        position : int, optional
            Insert position in step order (if new step).
        """
        self._step_handlers[step_name] = handler
        if step_name not in self._step_order:
            if position is not None:
                self._step_order.insert(position, step_name)
            else:
                self._step_order.append(step_name)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_default_handlers(self) -> None:
        """Register default handlers mapped to main.py run_* functions.

        Uses lazy imports to avoid circular dependencies.
        """
        # Map canonical step names to main.py mode handlers
        step_to_mode = {
            "data_generation": "simulate",
            "preprocessing": "features",
            "feature_engineering": "features",
            "ml_model_training": "train",
            "dl_model_training": "train",
            "ensemble_creation": "train",
            "uplift_modeling": "uplift",
            "clv_prediction": "clv",
            "customer_segmentation": "segment",
            "budget_optimization": "optimize",
            "ab_testing": "ab_test",
            "survival_analysis": "survival",
            "recommendations": "recommend",
            "cohort_analysis": "cohort",
            "scoring_api_setup": "monitor",
            "mlflow_logging": "monitor",
        }

        for step_name, mode_name in step_to_mode.items():
            self._step_handlers[step_name] = _make_lazy_handler(mode_name)

    @staticmethod
    def _default_args() -> argparse.Namespace:
        """Create a default argparse.Namespace for pipeline execution."""
        return argparse.Namespace(
            mode="all",
            config="config/simulator_config.yaml",
            data=None,
            output=None,
            budget=None,
            small=False,
            learner="auto",
            cohort_type="monthly",
            verbose=False,
            quiet=False,
        )


def _make_lazy_handler(mode_name: str) -> Callable:
    """Create a lazy handler that imports from main on first call.

    Parameters
    ----------
    mode_name : str
        The mode key from main.MODES dict.

    Returns
    -------
    callable
        A function(config, args) -> dict.
    """
    def _handler(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
        from src.main import MODES
        fn = MODES.get(mode_name)
        if fn is None:
            raise ValueError(f"Mode '{mode_name}' not found in src.main.MODES")
        return fn(config, args)
    return _handler
