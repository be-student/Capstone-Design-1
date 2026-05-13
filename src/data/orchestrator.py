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
        with open(yaml_path, "r", encoding="utf-8") as f:
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
            self._enforce_generation_requirements(summary)

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
        validation = self._build_validation_summary(customers_df, events_df)

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
            "generation_mode": validation["mode"],
            "validation": validation,
        }

    def _build_validation_summary(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Summarize generator requirement checks for reporting and enforcement."""
        target_cfg = self.config["target_churn_rate"]
        treatment_cfg = self.config["treatment"]
        mode = self.generator.infer_generation_mode()
        churn_rate = float(customers_df["churn_label"].mean())
        treatment_count = int((customers_df["treatment_group"] == "treatment").sum())
        control_count = int((customers_df["treatment_group"] == "control").sum())
        min_group_size = int(treatment_cfg.get("min_group_size", 0))

        group_size_check = {
            "required_min_per_group": min_group_size,
            "treatment_count": treatment_count,
            "control_count": control_count,
            "passed": treatment_count >= min_group_size and control_count >= min_group_size,
        }
        target_churn_check = {
            "target_min": float(target_cfg["min"]),
            "target_max": float(target_cfg["max"]),
            "actual": churn_rate,
            "passed": target_cfg["min"] <= churn_rate <= target_cfg["max"],
        }

        warnings = []
        if mode == "small":
            warnings.append(
                "Small mode summary only: treatment/control 10000-min validation is skipped."
            )
        elif not group_size_check["passed"]:
            warnings.append("Treatment/control minimum group size requirement failed.")

        if not target_churn_check["passed"]:
            warnings.append(
                "Generated churn rate is outside the configured 15%-25% target band."
            )
        event_metadata_check = self._build_event_metadata_check(
            customers_df,
            events_df,
        )
        if not event_metadata_check["event_schema_check"]["passed"]:
            warnings.append("Generated events are missing required metadata columns.")
        if not event_metadata_check["session_duration_check"]["passed"]:
            warnings.append(
                "Generated events do not contain positive session_duration values "
                "for all visit-session events."
            )
        if not event_metadata_check["marketing_response_check"]["passed"]:
            warnings.append(
                "Generated events do not contain a meaningful marketing_response "
                "distribution."
            )

        return {
            "mode": mode,
            "group_size_check": group_size_check,
            "target_churn_check": target_churn_check,
            **event_metadata_check,
            "warnings": warnings,
        }

    def _build_event_metadata_check(
        self,
        customers_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Validate event metadata needed to audit session and marketing behavior."""
        required_columns = CustomerDataGenerator.EVENT_COLUMNS
        missing_columns = [
            column for column in required_columns
            if column not in events_df.columns
        ]
        event_schema_check = {
            "required_columns": required_columns,
            "missing_columns": missing_columns,
            "passed": not missing_columns,
        }

        session_duration_check = {
            "checked_event_count": 0,
            "positive_count": 0,
            "passed": False,
        }
        marketing_response_check = {
            "allowed_values": sorted(CustomerDataGenerator.MARKETING_RESPONSES),
            "response_counts": {},
            "persona_response_counts": {},
            "responses_observed": [],
            "response_events_have_channel": False,
            "passed": False,
        }

        if missing_columns:
            return {
                "event_schema_check": event_schema_check,
                "session_duration_check": session_duration_check,
                "marketing_response_check": marketing_response_check,
            }

        session_events = events_df[
            events_df["event_type"].isin(CustomerDataGenerator.SESSION_EVENT_TYPES)
        ]
        session_values = pd.to_numeric(
            session_events["session_duration"],
            errors="coerce",
        )
        positive_sessions = session_values.notna() & (session_values > 0)
        session_duration_check = {
            "checked_event_count": int(len(session_events)),
            "positive_count": int(positive_sessions.sum()),
            "passed": bool(len(session_events) > 0 and positive_sessions.all()),
        }

        response_events = events_df[events_df["marketing_response"].notna()].copy()
        response_values = set(response_events["marketing_response"].astype(str))
        allowed_values = CustomerDataGenerator.MARKETING_RESPONSES
        response_counts = (
            response_events["marketing_response"].astype(str).value_counts().to_dict()
        )
        response_counts = {key: int(value) for key, value in response_counts.items()}

        persona_response_counts: Dict[str, Dict[str, int]] = {}
        if not response_events.empty and "persona" in customers_df.columns:
            customer_persona = customers_df.set_index("customer_id")["persona"]
            response_events["persona"] = response_events["customer_id"].map(
                customer_persona
            )
            counts = (
                response_events
                .groupby(["persona", "marketing_response"])
                .size()
                .unstack(fill_value=0)
            )
            persona_response_counts = {
                str(persona): {
                    str(response): int(count)
                    for response, count in row.items()
                }
                for persona, row in counts.iterrows()
            }

        expected_responses = allowed_values
        response_events_have_channel = bool(
            not response_events.empty
            and response_events["marketing_channel"].notna().all()
        )
        marketing_response_check = {
            "allowed_values": sorted(allowed_values),
            "response_counts": response_counts,
            "persona_response_counts": persona_response_counts,
            "responses_observed": sorted(response_values),
            "response_events_have_channel": response_events_have_channel,
            "passed": bool(
                response_values.issubset(allowed_values)
                and expected_responses.issubset(response_values)
                and response_events_have_channel
                and len(persona_response_counts) >= 2
            ),
        }

        return {
            "event_schema_check": event_schema_check,
            "session_duration_check": session_duration_check,
            "marketing_response_check": marketing_response_check,
        }

    def _enforce_generation_requirements(self, summary: Dict[str, Any]) -> None:
        """Fail full-mode runs when hard simulator requirements are not met."""
        validation = summary["validation"]
        if validation["mode"] != "full":
            return

        failures = []
        if not validation["group_size_check"]["passed"]:
            failures.append("treatment/control minimum group size")
        if not validation["target_churn_check"]["passed"]:
            failures.append("target churn range")
        if not validation["event_schema_check"]["passed"]:
            failures.append("event metadata schema")
        if not validation["session_duration_check"]["passed"]:
            failures.append("session duration metadata")
        if not validation["marketing_response_check"]["passed"]:
            failures.append("marketing response metadata")

        if failures:
            raise ValueError(
                "Generated data failed full-mode validation: " + ", ".join(failures)
            )

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
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, IOError):
                state = {}

        state["data_generation"] = status
        state["data_generation_updated_at"] = datetime.now().isoformat()
        if error_message:
            state["data_generation_error"] = error_message

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
