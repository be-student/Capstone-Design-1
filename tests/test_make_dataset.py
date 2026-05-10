import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


@pytest.fixture
def config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_apply_generation_overrides_small_mode(config):
    from src.data.make_dataset import apply_generation_overrides

    updated = apply_generation_overrides(config, small=True)
    small_cfg = config["simulation"]["small_mode"]

    assert updated["simulation"]["num_customers"] == small_cfg["num_customers"]
    assert updated["simulation"]["simulation_days"] == small_cfg["simulation_days"]
    assert config["simulation"]["num_customers"] == 20000


def test_make_dataset_writes_raw_processed_and_feature_store(config, tmp_path):
    from src.data.make_dataset import apply_generation_overrides, run_make_dataset

    config = apply_generation_overrides(
        config,
        num_customers=120,
        simulation_days=90,
    )
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    feature_store_dir = tmp_path / "feature_store"

    summary = run_make_dataset(
        config=config,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        feature_store_dir=feature_store_dir,
    )

    assert (raw_dir / "customers.csv").exists()
    assert (raw_dir / "events.csv").exists()
    assert (raw_dir / "events.parquet").exists()
    assert (raw_dir / "generation_summary.json").exists()
    assert (processed_dir / "features.csv").exists()
    assert (processed_dir / "features.parquet").exists()
    assert (processed_dir / "dataset_summary.json").exists()
    assert (feature_store_dir / "features.csv").exists()
    assert (feature_store_dir / "features.parquet").exists()

    events = pd.read_csv(raw_dir / "events.csv", nrows=200)
    assert {
        "session_duration", "marketing_channel", "marketing_response",
    }.issubset(events.columns)

    features = pd.read_csv(processed_dir / "features.csv")
    assert len(features) == summary["num_customers"] == 120
    assert summary["num_feature_rows"] == 120
