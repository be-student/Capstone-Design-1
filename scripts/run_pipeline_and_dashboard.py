#!/usr/bin/env python3
"""
Orchestration script that runs the full ML pipeline to completion,
then launches the Streamlit dashboard.

This script is the Python-native entrypoint for the churn prediction system.
It provides more granular control than the bash entrypoint, including:

- Pipeline checkpoint/resume via pipeline_state.json
- Configurable retry logic
- Output verification before dashboard launch
- Graceful signal handling

Usage:
    python scripts/run_pipeline_and_dashboard.py
    python scripts/run_pipeline_and_dashboard.py --skip-pipeline
    python scripts/run_pipeline_and_dashboard.py --pipeline-only
    python scripts/run_pipeline_and_dashboard.py --small -v

Environment Variables:
    SKIP_PIPELINE       Skip pipeline, launch dashboard directly
    PIPELINE_ONLY       Run pipeline only, no dashboard
    STREAMLIT_PORT      Dashboard port (default: 8501)
    PIPELINE_MAX_RETRIES  Max pipeline retry attempts (default: 1)
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
DEFAULT_STREAMLIT_HOST = os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
DEFAULT_MAX_RETRIES = int(os.getenv("PIPELINE_MAX_RETRIES", "1"))
PIPELINE_STATE_FILE = PROJECT_ROOT / "data" / "pipeline_state.json"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse orchestration CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="run_pipeline_and_dashboard",
        description="Run ML pipeline then launch Streamlit dashboard",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        default=os.getenv("SKIP_PIPELINE", "false").lower() == "true",
        help="Skip pipeline execution, launch dashboard directly",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        default=os.getenv("PIPELINE_ONLY", "false").lower() == "true",
        help="Run pipeline only, do not launch dashboard",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        default=False,
        help="Use small dataset for faster testing",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max pipeline retry attempts (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_STREAMLIT_PORT,
        help=f"Streamlit dashboard port (default: {DEFAULT_STREAMLIT_PORT})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging for pipeline",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Suppress pipeline output (WARNING only)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "simulator_config.yaml"),
        help="Path to YAML config file",
    )
    return parser.parse_args(argv)


def verify_pipeline_outputs() -> Dict[str, bool]:
    """Check that pipeline generated expected outputs.

    Returns
    -------
    dict
        Mapping of output category to whether files exist.
    """
    checks = {
        "data": PROJECT_ROOT / "data" / "raw",
        "models": PROJECT_ROOT / "models",
        "results": PROJECT_ROOT / "results",
    }
    status = {}
    for name, path in checks.items():
        exists = path.is_dir() and any(path.iterdir())
        status[name] = exists
        if exists:
            file_count = sum(1 for _ in path.rglob("*") if _.is_file())
            logger.info("  [OK] %s: %d files in %s", name, file_count, path)
        else:
            logger.warning("  [MISSING] %s: %s", name, path)
    return status


def check_pipeline_state() -> Optional[Dict]:
    """Read pipeline_state.json if it exists.

    Returns
    -------
    dict or None
        Pipeline state dictionary, or None if file doesn't exist.
    """
    if PIPELINE_STATE_FILE.exists():
        try:
            with open(PIPELINE_STATE_FILE) as f:
                state = json.load(f)
            logger.info("Pipeline state found: %s", PIPELINE_STATE_FILE)
            return state
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read pipeline state: %s", exc)
    return None


def run_pipeline(args: argparse.Namespace) -> bool:
    """Execute the ML pipeline via src.main --mode all.

    Uses subprocess to run the pipeline, supporting checkpoint/resume.
    Returns True on success, False on failure.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    bool
        True if pipeline completed successfully.
    """
    logger.info("=" * 60)
    logger.info("  STARTING ML PIPELINE")
    logger.info("=" * 60)

    cmd = [
        sys.executable, "-m", "src.main",
        "--mode", "all",
        "--config", args.config,
    ]

    if args.small:
        cmd.append("--small")
    if args.verbose:
        cmd.append("-v")
    if args.quiet:
        cmd.append("-q")

    for attempt in range(1, args.max_retries + 1):
        logger.info("Pipeline attempt %d/%d", attempt, args.max_retries)
        logger.info("Command: %s", " ".join(cmd))

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
                check=True,
            )
            elapsed = time.time() - start_time
            logger.info(
                "Pipeline completed successfully in %.1f seconds", elapsed
            )
            return True

        except subprocess.CalledProcessError as exc:
            elapsed = time.time() - start_time
            logger.error(
                "Pipeline failed (exit code %d) after %.1f seconds "
                "(attempt %d/%d)",
                exc.returncode, elapsed, attempt, args.max_retries,
            )

            if attempt < args.max_retries:
                wait_secs = 5
                logger.info(
                    "Retrying in %d seconds (checkpoint/resume enabled)...",
                    wait_secs,
                )
                time.sleep(wait_secs)

    logger.error("Pipeline failed after %d attempt(s)", args.max_retries)
    return False


def launch_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard.

    This function replaces the current process with Streamlit
    (via os.execvp) so that the dashboard runs as PID 1 in Docker.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    logger.info("=" * 60)
    logger.info("  LAUNCHING STREAMLIT DASHBOARD")
    logger.info("  Port: %d  |  Host: %s", args.port, DEFAULT_STREAMLIT_HOST)
    logger.info("=" * 60)

    dashboard_cmd = [
        "streamlit", "run", "src/dashboard/app.py",
        f"--server.port={args.port}",
        f"--server.address={DEFAULT_STREAMLIT_HOST}",
        "--server.headless=true",
    ]

    # Use os.execvp to replace this process with streamlit
    # This ensures proper signal handling in Docker containers
    os.execvp(dashboard_cmd[0], dashboard_cmd)


def main(argv: Optional[List[str]] = None) -> int:
    """Main orchestration entrypoint.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure).
    """
    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("  Churn Prediction System - Orchestrator")
    logger.info("=" * 60)
    logger.info("  skip_pipeline:  %s", args.skip_pipeline)
    logger.info("  pipeline_only:  %s", args.pipeline_only)
    logger.info("  small:          %s", args.small)
    logger.info("  config:         %s", args.config)
    logger.info("=" * 60)

    # Ensure output directories exist
    for d in ["data/raw", "results", "models"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    # --- Pipeline Phase ---
    if args.skip_pipeline:
        logger.info("Skipping pipeline (--skip-pipeline)")
        state = check_pipeline_state()
        if state:
            logger.info("Previous pipeline state available")
        output_status = verify_pipeline_outputs()
        if not all(output_status.values()):
            logger.warning(
                "Some pipeline outputs are missing. "
                "Dashboard may show incomplete data."
            )
    else:
        success = run_pipeline(args)
        if not success:
            logger.error("Pipeline failed. Aborting.")
            return 1

        logger.info("Verifying pipeline outputs...")
        output_status = verify_pipeline_outputs()
        if not all(output_status.values()):
            logger.warning("Some pipeline outputs may be missing")

    # --- Dashboard Phase ---
    if args.pipeline_only:
        logger.info("Pipeline-only mode. Exiting without dashboard.")
        return 0

    # launch_dashboard uses os.execvp - does not return
    launch_dashboard(args)

    # Should not reach here (execvp replaces process)
    return 0


if __name__ == "__main__":
    sys.exit(main())
