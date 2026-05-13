import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


@pytest.fixture
def small_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["simulation"]["num_customers"] = 200
    config["simulation"]["simulation_days"] = 90
    return config


def test_generation_summary_validates_event_metadata(small_config, tmp_path):
    from src.data.orchestrator import SimulatorOrchestrator

    result = SimulatorOrchestrator(small_config).run(str(tmp_path / "raw"))
    validation = result["summary"]["validation"]

    assert validation["event_schema_check"]["missing_columns"] == []
    assert validation["event_schema_check"]["passed"] is True
    assert validation["session_duration_check"]["passed"] is True
    assert validation["session_duration_check"]["checked_event_count"] > 0

    marketing_check = validation["marketing_response_check"]
    assert marketing_check["response_events_have_channel"] is True
    assert set(marketing_check["responses_observed"]) == {
        "conversion", "no_response", "adverse",
    }
    assert marketing_check["passed"] is True
    assert len(marketing_check["persona_response_counts"]) >= 2


def test_event_metadata_check_rejects_legacy_raw_schema(small_config):
    from src.data.orchestrator import SimulatorOrchestrator

    orchestrator = SimulatorOrchestrator(small_config)
    customers = pd.DataFrame({
        "customer_id": ["C1"],
        "persona": ["vip_loyal"],
        "treatment_group": ["treatment"],
    })
    legacy_events = pd.DataFrame({
        "customer_id": ["C1"],
        "event_type": ["page_view"],
        "event_date": ["2024-01-01"],
        "timestamp": ["2024-01-01 10:00:00"],
        "amount": [None],
    })

    check = orchestrator._build_event_metadata_check(customers, legacy_events)

    assert check["event_schema_check"]["passed"] is False
    assert "session_duration" in check["event_schema_check"]["missing_columns"]
    assert "marketing_channel" in check["event_schema_check"]["missing_columns"]
    assert "marketing_response" in check["event_schema_check"]["missing_columns"]
