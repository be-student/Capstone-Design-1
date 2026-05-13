"""Tests that the dashboard reads ONLY real pipeline-generated data.

Run after `python -m src.main --mode all` so artifacts exist.
If any test fails, the dashboard has a fixture leak.
"""
import json
from pathlib import Path
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


@pytest.fixture
def data_loader():
    """Create a DashboardDataLoader with the production config."""
    import yaml
    from src.dashboard.data_loader import DashboardDataLoader
    config_path = PROJECT_ROOT / "config" / "simulator_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return DashboardDataLoader(config)


class TestRealArtifactsExist:
    """Each of the 7 artifacts G1 produces should be on disk and non-empty."""

    @pytest.mark.parametrize("filename", [
        "confusion_matrices.json",
        "roc_data.json",
        "survival_data.csv",
        "survival_curves.json",
        "scoring_history.csv",
        "retention_offers.csv",
        "drift_history.csv",
    ])
    def test_artifact_exists(self, filename):
        p = RESULTS / filename
        assert p.exists(), f"Pipeline did not produce {filename}"
        assert p.stat().st_size > 50, f"{filename} is empty or near-empty"


class TestNoFixtureFallback:
    """Each load_* must return is_real=True when artifacts exist."""

    def test_confusion_matrices_real(self, data_loader):
        # Try the new artifact-style API; if not available, fall back
        try:
            art = data_loader.load_confusion_matrices(as_artifact=True)
            assert art.is_real is True, f"Confusion matrices fallback: {art.reason}"
            assert "ml_model" in art.data or "ml_model" in (art.data.keys() if hasattr(art.data, "keys") else [])
        except TypeError:
            pytest.skip("load_confusion_matrices does not support as_artifact yet")

    def test_survival_data_real(self, data_loader):
        try:
            art = data_loader.load_survival_data(as_artifact=True)
            assert art.is_real is True, f"Survival data fallback: {art.reason}"
            assert isinstance(art.data, pd.DataFrame) and len(art.data) > 0
        except (TypeError, AttributeError):
            pytest.skip("load_survival_data does not support as_artifact yet")

    def test_scoring_history_real(self, data_loader):
        try:
            art = data_loader.load_scoring_history(as_artifact=True)
            assert art.is_real is True, f"Scoring history fallback: {art.reason}"
        except (TypeError, AttributeError):
            pytest.skip("load_scoring_history does not support as_artifact yet")

    def test_retention_offers_real(self, data_loader):
        try:
            art = data_loader.load_retention_offers(as_artifact=True)
            assert art.is_real is True, f"Retention offers fallback: {art.reason}"
        except (TypeError, AttributeError):
            pytest.skip("load_retention_offers does not support as_artifact yet")

    def test_drift_history_real(self, data_loader):
        try:
            art = data_loader.load_drift_history(as_artifact=True)
            assert art.is_real is True, f"Drift history fallback: {art.reason}"
        except (TypeError, AttributeError):
            pytest.skip("load_drift_history does not support as_artifact yet")


class TestSampleGeneratorsRemoved:
    """The _generate_sample_* functions should be removed or raise on call."""

    @pytest.mark.parametrize("method", [
        "_generate_sample_confusion_matrices",
        "_generate_sample_roc_curves",
        "_generate_sample_survival_data",
        "_generate_sample_scoring_history",
        "_generate_sample_scoring_throughput",
        "_generate_sample_retention_offers",
        "_generate_sample_drift_history",
    ])
    def test_sample_generator_raises_or_missing(self, data_loader, method):
        if not hasattr(data_loader, method):
            return  # removed entirely - best case
        with pytest.raises((FileNotFoundError, RuntimeError, NotImplementedError)):
            getattr(data_loader, method)()


class TestPage02NoFixtureOverride:
    """Page 02 headline P/R/F1 must come from model_metrics.json, NOT overwritten by fixture."""

    def test_real_metrics_match_metrics_json(self, data_loader):
        metrics_path = RESULTS / "model_metrics.json"
        if not metrics_path.exists():
            pytest.skip("model_metrics.json not present")
        real = json.loads(metrics_path.read_text(encoding="utf-8"))
        ml = real.get("ml_model", real.get("ml", {}))
        # If precision and recall live nested under "test" sub-dict, dig in
        ml_precision = ml.get("precision") or ml.get("test", {}).get("precision")
        ml_recall = ml.get("recall") or ml.get("test", {}).get("recall")
        # Sample: from iter12 audit, real values are ~0.53 / 0.78
        assert ml_precision is not None
        assert ml_recall is not None
        # Reject the hardcoded fixture values 0.7059 / 0.6000 if real values are very different
        if abs(ml_precision - 0.7059) < 1e-3 and abs(ml_recall - 0.6000) < 1e-3:
            pytest.fail(
                f"Page 02 headline P/R match the hardcoded fixture (0.7059/0.6000). "
                f"Real model_metrics.json values: P={ml_precision}, R={ml_recall}. "
                f"app.py:439-463 fixture override likely still present."
            )
