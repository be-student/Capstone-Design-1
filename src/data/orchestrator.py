"""
Simulator Orchestrator for E-Commerce Customer Churn Prediction.

Orchestrates the full data generation pipeline:
1. Loads configuration from YAML
2. Generates customer profiles with persona proportional assignment
3. Assigns Treatment/Control groups (50/50 split)
4. Runs the event generator for each customer
5. Labels churn based on configurable rules
6. Outputs CSV and Parquet files
7. Tracks pipeline state via pipeline_state.json
8. Produces generation summary statistics

Usage:
    orch = SimulatorOrchestrator.from_yaml("config/simulator_config.yaml")
    result = orch.run(output_dir="data/raw")

    # Or with in-memory config:
    orch = SimulatorOrchestrator(config_dict)
    result = orch.run(output_dir="data/raw")
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from src.data.generator import CustomerDataGenerator

logger = logging.getLogger(__name__)


class SimulatorOrchestrator:
    """Orchestrate full customer data generation pipeline.

    Wraps CustomerDataGenerator with pipeline state management,
    dual-format output (CSV + Parquet), and summary statistics.

    Attributes:
        config: Full configuration dictionary.
        num_customers: Number of customers to generate.
        generator: Underlying CustomerDataGenerator instance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize orchestrator from configuration dictionary.

        Args:
            config: Configuration dictionary (from simulator_config.yaml).
        """
        self.config = config
        self.num_customers = config["simulation"]["num_customers"]
        self.generator = CustomerDataGenerator(config)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SimulatorOrchestrator":
        """Create orchestrator from a YAML configuration file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            SimulatorOrchestrator instance.
        """
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run(self, output_dir: str) -> Dict[str, Any]:
        """Execute the full data generation pipeline.

        Steps:
            1. Update pipeline state to 'pending'
            2. Generate customer profiles and events
            3. Compute summary statistics
            4. Save CSV and Parquet files
            5. Save summary JSON
            6. Update pipeline state to 'completed'

        On failure, pipeline state is set to 'failed'.

        Args:
            output_dir: Directory to save output files.

        Returns:
            Dictionary containing:
                - 'customers': Customer profiles DataFrame
                - 'events': Event logs DataFrame
                - 'summary': Summary statistics dict

        Raises:
            Exception: Re-raises any generation errors after
                       recording 'failed' state.
        """
        os.makedirs(output_dir, exist_ok=True)
        self._update_pipeline_state(output_dir, "pending")

        try:
            start_time = time.time()
            logger.info(
                "Starting data generation for %d customers...",
                self.num_customers,
            )

            # Generate data
            data = self.generator.generate()
            customers_df = data["customers"]
            events_df = data["events"]

            # Compute summary
            elapsed = time.time() - start_time
            summary = self._compute_summary(customers_df, events_df, elapsed)

            # Save outputs
            self._save_outputs(customers_df, events_df, summary, output_dir)

            # Update state
            self._update_pipeline_state(output_dir, "completed")

            logger.info(
                "Data generation completed in %.1fs: %d customers, "
                "%d events, churn_rate=%.2f%%",
                elapsed,
                summary["num_customers"],
                summary["num_events"],
                summary["churn_rate"] * 100,
            )

            return {
                "customers": customers_df,
                "events": events_df,
                "summary": summary,
            }

        except Exception as e:
            logger.error("Data generation failed: %s", str(e))
            self._update_pipeline_state(output_dir, "failed", str(e))
            raise

    def _compute_summary(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
        elapsed_seconds: float,
    ) -> Dict[str, Any]:
        """Compute summary statistics for the generated data.

        Args:
            customers_df: Customer profiles DataFrame.
            events_df: Event logs DataFrame.
            elapsed_seconds: Time taken for generation.

        Returns:
            Summary statistics dictionary.
        """
        num_customers = len(customers_df)
        num_events = len(events_df)
        churn_rate = float(customers_df["churn_label"].mean())

        treatment_count = int(
            (customers_df["treatment_group"] == "treatment").sum()
        )
        treatment_ratio = treatment_count / num_customers

        # Persona distribution
        persona_dist = (
            customers_df["persona"]
            .value_counts(normalize=True)
            .to_dict()
        )

        # Event type distribution
        event_dist = (
            events_df["event_type"]
            .value_counts()
            .to_dict()
        ) if len(events_df) > 0 else {}

        # Convert numpy types for JSON serialization
        event_dist = {k: int(v) for k, v in event_dist.items()}
        persona_dist = {k: float(v) for k, v in persona_dist.items()}

        return {
            "num_customers": num_customers,
            "num_events": num_events,
            "churn_rate": churn_rate,
            "treatment_ratio": treatment_ratio,
            "treatment_count": treatment_count,
            "control_count": num_customers - treatment_count,
            "persona_distribution": persona_dist,
            "event_type_distribution": event_dist,
            "generation_time_seconds": round(elapsed_seconds, 2),
            "generated_at": datetime.now().isoformat(),
            "random_seed": self.config["simulation"]["random_seed"],
        }

    def _save_outputs(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
        summary: Dict[str, Any],
        output_dir: str,
    ) -> None:
        """Save generated data as CSV, Parquet, and summary JSON.

        Args:
            customers_df: Customer profiles DataFrame.
            events_df: Event logs DataFrame.
            summary: Summary statistics dictionary.
            output_dir: Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)

        # CSV output
        customers_df.to_csv(
            os.path.join(output_dir, "customers.csv"), index=False
        )
        events_df.to_csv(
            os.path.join(output_dir, "events.csv"), index=False
        )

        # Parquet output
        customers_df.to_parquet(
            os.path.join(output_dir, "customers.parquet"), index=False
        )
        events_df.to_parquet(
            os.path.join(output_dir, "events.parquet"), index=False
        )

        # Summary JSON
        with open(os.path.join(output_dir, "generation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("Outputs saved to %s", output_dir)

    def _update_pipeline_state(
        self,
        output_dir: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update pipeline state JSON file.

        Args:
            output_dir: Output directory containing pipeline_state.json.
            status: Status string ('pending', 'completed', 'failed').
            error_message: Optional error message for failed state.
        """
        os.makedirs(output_dir, exist_ok=True)
        state_path = os.path.join(output_dir, "pipeline_state.json")

        # Load existing state or create new
        state: Dict[str, Any] = {}
        if os.path.exists(state_path):
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, IOError):
                state = {}

        state["data_generation"] = status
        state["data_generation_updated_at"] = datetime.now().isoformat()
        if error_message:
            state["data_generation_error"] = error_message

        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
