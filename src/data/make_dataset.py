"""
Reproducible dataset build entrypoint.

Runs the simulator, writes raw customer/event artifacts, computes the processed
feature matrix, and persists the file-based feature store.

Usage:
    python3 src/data/make_dataset.py
    python3 src/data/make_dataset.py --small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.orchestrator import SimulatorOrchestrator  # noqa: E402
from src.features import FeatureEngineer  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_FEATURE_STORE_DIR = PROJECT_ROOT / "data" / "feature_store"


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load simulator configuration from YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_generation_overrides(
    config: Dict[str, Any],
    *,
    small: bool = False,
    num_customers: Optional[int] = None,
    simulation_days: Optional[int] = None,
    simulation_months: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply CLI sizing overrides without mutating the caller's config."""
    updated = json.loads(json.dumps(config))
    sim_cfg = updated["simulation"]

    if small:
        small_cfg = sim_cfg.get("small_mode", {})
        sim_cfg["num_customers"] = int(
            small_cfg.get("num_customers", sim_cfg["num_customers"])
        )
        sim_cfg["simulation_months"] = int(
            small_cfg.get("simulation_months", sim_cfg.get("simulation_months", 6))
        )
        sim_cfg["simulation_days"] = int(
            small_cfg.get("simulation_days", sim_cfg.get("simulation_days", 180))
        )

    if num_customers is not None:
        sim_cfg["num_customers"] = int(num_customers)
    if simulation_days is not None:
        sim_cfg["simulation_days"] = int(simulation_days)
    if simulation_months is not None:
        sim_cfg["simulation_months"] = int(simulation_months)
        if simulation_days is None:
            sim_cfg["simulation_days"] = int(simulation_months) * 30

    return updated


def default_reference_date(config: Dict[str, Any]) -> str:
    """Return the same reference date convention used by the main pipeline."""
    sim_cfg = config.get("simulation", {})
    start = pd.Timestamp(sim_cfg.get("start_date", "2024-01-01"))
    sim_days = int(sim_cfg.get("simulation_days", 365))
    return str((start + pd.Timedelta(days=sim_days)).date())


def save_processed_features(features: pd.DataFrame, processed_dir: Path) -> None:
    """Persist the processed feature matrix as CSV and Parquet."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_dir / "features.csv", index=False)
    features.to_parquet(processed_dir / "features.parquet", index=False)


def run_make_dataset(
    *,
    config: Dict[str, Any],
    raw_dir: Path,
    processed_dir: Path,
    feature_store_dir: Path,
    reference_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Run simulation and feature generation, returning an artifact summary."""
    orchestrator = SimulatorOrchestrator(config)
    generation = orchestrator.run(output_dir=str(raw_dir))

    customers = generation["customers"]
    events = generation["events"]
    ref_date = reference_date or default_reference_date(config)

    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.compute_all_features(
        customers,
        events,
        reference_date=ref_date,
    )
    save_processed_features(features, processed_dir)
    feature_engineer.save_to_feature_store(features, str(feature_store_dir))

    summary = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "feature_store_dir": str(feature_store_dir),
        "reference_date": ref_date,
        "num_customers": int(len(customers)),
        "num_events": int(len(events)),
        "num_feature_rows": int(len(features)),
        "num_feature_columns": int(len(features.columns)),
        "generation_summary": generation["summary"],
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    with open(processed_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Generate raw simulator data and processed feature datasets.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--feature-store-dir",
        type=Path,
        default=DEFAULT_FEATURE_STORE_DIR,
    )
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--num-customers", type=int, default=None)
    parser.add_argument("--simulation-days", type=int, default=None)
    parser.add_argument("--simulation-months", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    config = load_config(args.config)
    config = apply_generation_overrides(
        config,
        small=args.small,
        num_customers=args.num_customers,
        simulation_days=args.simulation_days,
        simulation_months=args.simulation_months,
    )
    summary = run_make_dataset(
        config=config,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        feature_store_dir=args.feature_store_dir,
        reference_date=args.reference_date,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
