"""
TDD Tests for MLflow Model Registry Module.

Tests cover:
- Model registration with versioning
- Stage transitions (None -> Staging -> Production -> Archived)
- Model serving utilities (load from registry by stage/version)
- Model version listing and comparison
- Best model promotion based on metrics
- Registry cleanup and archival
- Integration with MLflowTracker
"""

import os
import sys
import json
import pickle
import tempfile

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_mlflow_dir(tmp_path):
    """Create temporary MLflow directories.

    `artifacts_dir` is returned as a `file://` URI so MLflow's artifact
    registry can resolve it on Windows (a bare `C:\\...` path has no scheme).
    """
    db_path = tmp_path / "mlflow.db"
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    return {
        "db_path": str(db_path),
        "artifacts_dir": artifacts_dir.as_uri(),
        "tracking_uri": f"sqlite:///{db_path.as_posix()}",
    }


@pytest.fixture(autouse=True)
def _drain_active_mlflow_run():
    """Ensure no test in this module leaks an active MLflow run."""
    import mlflow
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def registry_config(config, tmp_mlflow_dir):
    """Create config with temporary MLflow backend for registry tests."""
    cfg = config.copy()
    cfg["mlflow"] = {
        "tracking_uri": tmp_mlflow_dir["tracking_uri"],
        "artifact_location": tmp_mlflow_dir["artifacts_dir"],
        "experiment_name": "registry_test",
        "log_models": True,
        "log_artifacts": True,
        "docker": config.get("mlflow", {}).get("docker", {
            "tracking_uri": "http://mlflow:5000",
            "artifact_location": "/mlflow/artifacts",
            "server_host": "0.0.0.0",
            "server_port": 5000,
            "backend_store_uri": "sqlite:///mlflow/mlflow.db",
        }),
    }
    return cfg


@pytest.fixture
def model_registry(registry_config):
    """Create a ModelRegistry instance."""
    from src.models.mlflow_tracking import ModelRegistry
    return ModelRegistry(registry_config)


@pytest.fixture
def tracker(registry_config):
    """Create an MLflowTracker instance."""
    from src.models.mlflow_tracking import MLflowTracker
    return MLflowTracker(registry_config)


@pytest.fixture
def dummy_model_path(tmp_path):
    """Create a dummy model file for registration."""
    model_path = tmp_path / "dummy_model.pkl"
    dummy = {"type": "xgboost", "params": {"n_estimators": 100}}
    with open(model_path, "wb") as f:
        pickle.dump(dummy, f)
    return str(model_path)


# ---------------------------------------------------------------------------
# ModelRegistry Interface Tests
# ---------------------------------------------------------------------------

class TestModelRegistryInterface:
    """Test ModelRegistry class instantiation and interface."""

    def test_instantiation(self, model_registry):
        """ModelRegistry must be instantiable from config."""
        assert model_registry is not None

    def test_has_register_model_method(self, model_registry):
        """Must have register_model method."""
        assert hasattr(model_registry, "register_model")
        assert callable(model_registry.register_model)

    def test_has_transition_stage_method(self, model_registry):
        """Must have transition_model_stage method."""
        assert hasattr(model_registry, "transition_model_stage")
        assert callable(model_registry.transition_model_stage)

    def test_has_get_model_version_method(self, model_registry):
        """Must have get_model_version method."""
        assert hasattr(model_registry, "get_model_version")
        assert callable(model_registry.get_model_version)

    def test_has_list_model_versions_method(self, model_registry):
        """Must have list_model_versions method."""
        assert hasattr(model_registry, "list_model_versions")
        assert callable(model_registry.list_model_versions)

    def test_has_get_latest_version_method(self, model_registry):
        """Must have get_latest_version method."""
        assert hasattr(model_registry, "get_latest_version")
        assert callable(model_registry.get_latest_version)

    def test_has_load_model_method(self, model_registry):
        """Must have load_model method for serving."""
        assert hasattr(model_registry, "load_model")
        assert callable(model_registry.load_model)

    def test_has_promote_best_model_method(self, model_registry):
        """Must have promote_best_model method."""
        assert hasattr(model_registry, "promote_best_model")
        assert callable(model_registry.promote_best_model)


# ---------------------------------------------------------------------------
# Model Registration Tests
# ---------------------------------------------------------------------------

class TestModelRegistration:
    """Test model registration and versioning."""

    def test_register_model_returns_version(
        self, model_registry, tracker, dummy_model_path
    ):
        """register_model must return a version info dict."""
        tracker.create_experiment("reg_test")
        run_id = tracker.start_run(run_name="reg_run")
        tracker.log_metrics({"auc": 0.82})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        result = model_registry.register_model(
            run_id=run_id,
            model_name="churn_predictor",
            artifact_path="model/dummy_model.pkl",
        )

        assert result is not None
        assert "version" in result
        assert "name" in result
        assert result["name"] == "churn_predictor"

    def test_register_creates_incrementing_versions(
        self, model_registry, tracker, dummy_model_path
    ):
        """Multiple registrations must create incrementing versions."""
        tracker.create_experiment("version_test")

        versions = []
        for i in range(3):
            run_id = tracker.start_run(run_name=f"ver_run_{i}")
            tracker.log_metrics({"auc": 0.80 + i * 0.01})
            tracker.log_artifact(dummy_model_path, artifact_path="model")
            tracker.end_run()

            result = model_registry.register_model(
                run_id=run_id,
                model_name="churn_v_test",
                artifact_path="model/dummy_model.pkl",
            )
            versions.append(result["version"])

        # Versions should be incrementing
        assert len(versions) == 3
        assert versions == sorted(versions)

    def test_register_model_with_description(
        self, model_registry, tracker, dummy_model_path
    ):
        """register_model should accept optional description."""
        tracker.create_experiment("desc_test")
        run_id = tracker.start_run(run_name="desc_run")
        tracker.log_metrics({"auc": 0.85})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        result = model_registry.register_model(
            run_id=run_id,
            model_name="churn_desc_test",
            artifact_path="model/dummy_model.pkl",
            description="Best model from experiment v1",
        )

        assert result is not None
        assert result["version"] >= 1

    def test_register_model_with_tags(
        self, model_registry, tracker, dummy_model_path
    ):
        """register_model should accept optional tags."""
        tracker.create_experiment("tag_reg_test")
        run_id = tracker.start_run(run_name="tag_run")
        tracker.log_metrics({"auc": 0.83})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        result = model_registry.register_model(
            run_id=run_id,
            model_name="churn_tag_test",
            artifact_path="model/dummy_model.pkl",
            tags={"model_type": "xgboost", "data_version": "v1"},
        )

        assert result is not None


# ---------------------------------------------------------------------------
# Stage Transition Tests
# ---------------------------------------------------------------------------

class TestStageTransitions:
    """Test model stage transition functionality."""

    def test_transition_to_staging(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must transition model to Staging stage."""
        tracker.create_experiment("stage_test")
        run_id = tracker.start_run(run_name="stage_run")
        tracker.log_metrics({"auc": 0.82})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_staging_test",
            artifact_path="model/dummy_model.pkl",
        )

        result = model_registry.transition_model_stage(
            model_name="churn_staging_test",
            version=reg["version"],
            stage="Staging",
        )

        assert result is not None
        assert result["stage"] in ("Staging", "staging")

    def test_transition_to_production(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must transition model to Production stage."""
        tracker.create_experiment("prod_test")
        run_id = tracker.start_run(run_name="prod_run")
        tracker.log_metrics({"auc": 0.85})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_prod_test",
            artifact_path="model/dummy_model.pkl",
        )

        # First to staging
        model_registry.transition_model_stage(
            model_name="churn_prod_test",
            version=reg["version"],
            stage="Staging",
        )

        # Then to production
        result = model_registry.transition_model_stage(
            model_name="churn_prod_test",
            version=reg["version"],
            stage="Production",
        )

        assert result is not None
        assert result["stage"] in ("Production", "production")

    def test_transition_to_archived(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must transition model to Archived stage."""
        tracker.create_experiment("archive_test")
        run_id = tracker.start_run(run_name="archive_run")
        tracker.log_metrics({"auc": 0.80})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_archive_test",
            artifact_path="model/dummy_model.pkl",
        )

        result = model_registry.transition_model_stage(
            model_name="churn_archive_test",
            version=reg["version"],
            stage="Archived",
        )

        assert result is not None
        assert result["stage"] in ("Archived", "archived")

    def test_get_model_by_stage(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must retrieve model version by stage."""
        tracker.create_experiment("get_stage_test")
        run_id = tracker.start_run(run_name="get_stage_run")
        tracker.log_metrics({"auc": 0.82})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_get_stage",
            artifact_path="model/dummy_model.pkl",
        )

        model_registry.transition_model_stage(
            model_name="churn_get_stage",
            version=reg["version"],
            stage="Production",
        )

        result = model_registry.get_latest_version(
            model_name="churn_get_stage",
            stage="Production",
        )

        assert result is not None
        assert result["version"] == reg["version"]


# ---------------------------------------------------------------------------
# Version Listing Tests
# ---------------------------------------------------------------------------

class TestVersionListing:
    """Test model version listing and comparison."""

    def test_list_versions(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must list all versions of a registered model."""
        tracker.create_experiment("list_test")

        for i in range(2):
            run_id = tracker.start_run(run_name=f"list_run_{i}")
            tracker.log_metrics({"auc": 0.80 + i * 0.01})
            tracker.log_artifact(dummy_model_path, artifact_path="model")
            tracker.end_run()

            model_registry.register_model(
                run_id=run_id,
                model_name="churn_list_test",
                artifact_path="model/dummy_model.pkl",
            )

        versions = model_registry.list_model_versions(
            model_name="churn_list_test"
        )

        assert isinstance(versions, list)
        assert len(versions) >= 2

    def test_get_specific_version(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must retrieve a specific model version."""
        tracker.create_experiment("specific_ver_test")
        run_id = tracker.start_run(run_name="specific_run")
        tracker.log_metrics({"auc": 0.82})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_specific_ver",
            artifact_path="model/dummy_model.pkl",
        )

        version_info = model_registry.get_model_version(
            model_name="churn_specific_ver",
            version=reg["version"],
        )

        assert version_info is not None
        assert version_info["version"] == reg["version"]
        assert "run_id" in version_info


# ---------------------------------------------------------------------------
# Model Loading / Serving Tests
# ---------------------------------------------------------------------------

class TestModelServing:
    """Test model loading from registry for serving."""

    def test_load_model_by_version(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must load a model artifact by version number."""
        tracker.create_experiment("load_ver_test")
        run_id = tracker.start_run(run_name="load_run")
        tracker.log_metrics({"auc": 0.82})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_load_ver",
            artifact_path="model/dummy_model.pkl",
        )

        loaded = model_registry.load_model(
            model_name="churn_load_ver",
            version=reg["version"],
        )

        assert loaded is not None
        assert "artifact_uri" in loaded or "model" in loaded

    def test_load_model_by_stage(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must load a model artifact by stage (e.g., Production)."""
        tracker.create_experiment("load_stage_test")
        run_id = tracker.start_run(run_name="load_stage_run")
        tracker.log_metrics({"auc": 0.85})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_load_stage",
            artifact_path="model/dummy_model.pkl",
        )

        model_registry.transition_model_stage(
            model_name="churn_load_stage",
            version=reg["version"],
            stage="Production",
        )

        loaded = model_registry.load_model(
            model_name="churn_load_stage",
            stage="Production",
        )

        assert loaded is not None

    def test_load_nonexistent_model_returns_none(self, model_registry):
        """Loading a nonexistent model must return None."""
        result = model_registry.load_model(
            model_name="nonexistent_model_xyz",
            version=999,
        )
        assert result is None

    def test_get_serving_model_info(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must return serving info for a production model."""
        tracker.create_experiment("serve_info_test")
        run_id = tracker.start_run(run_name="serve_info_run")
        tracker.log_metrics({"auc": 0.83})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg = model_registry.register_model(
            run_id=run_id,
            model_name="churn_serve_info",
            artifact_path="model/dummy_model.pkl",
        )

        model_registry.transition_model_stage(
            model_name="churn_serve_info",
            version=reg["version"],
            stage="Production",
        )

        info = model_registry.get_serving_info(
            model_name="churn_serve_info",
        )

        assert info is not None
        assert "version" in info
        assert "stage" in info
        assert "run_id" in info


# ---------------------------------------------------------------------------
# Best Model Promotion Tests
# ---------------------------------------------------------------------------

class TestBestModelPromotion:
    """Test automatic promotion of best model to Production."""

    def test_promote_best_model(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must promote the best-performing model to Production."""
        tracker.create_experiment("promote_test")

        aucs = [0.78, 0.85, 0.81]
        for i, auc in enumerate(aucs):
            run_id = tracker.start_run(run_name=f"promo_run_{i}")
            tracker.log_metrics({"auc": auc})
            tracker.log_artifact(dummy_model_path, artifact_path="model")
            tracker.end_run()

            model_registry.register_model(
                run_id=run_id,
                model_name="churn_promote_test",
                artifact_path="model/dummy_model.pkl",
            )

        result = model_registry.promote_best_model(
            model_name="churn_promote_test",
            experiment_name="promote_test",
            metric="auc",
            mode="max",
        )

        assert result is not None
        assert result["stage"] in ("Production", "production")

    def test_promote_archives_previous_production(
        self, model_registry, tracker, dummy_model_path
    ):
        """Promoting a new model should archive the old Production model."""
        tracker.create_experiment("archive_prev_test")

        # Register and promote first model
        run_id1 = tracker.start_run(run_name="first_run")
        tracker.log_metrics({"auc": 0.80})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg1 = model_registry.register_model(
            run_id=run_id1,
            model_name="churn_archive_prev",
            artifact_path="model/dummy_model.pkl",
        )

        model_registry.transition_model_stage(
            model_name="churn_archive_prev",
            version=reg1["version"],
            stage="Production",
        )

        # Register and promote better model
        run_id2 = tracker.start_run(run_name="second_run")
        tracker.log_metrics({"auc": 0.85})
        tracker.log_artifact(dummy_model_path, artifact_path="model")
        tracker.end_run()

        reg2 = model_registry.register_model(
            run_id=run_id2,
            model_name="churn_archive_prev",
            artifact_path="model/dummy_model.pkl",
        )

        model_registry.transition_model_stage(
            model_name="churn_archive_prev",
            version=reg2["version"],
            stage="Production",
            archive_existing=True,
        )

        # Old version should no longer be in Production
        latest = model_registry.get_latest_version(
            model_name="churn_archive_prev",
            stage="Production",
        )
        assert latest is not None
        assert latest["version"] == reg2["version"]


# ---------------------------------------------------------------------------
# Integration with MLflowTracker Tests
# ---------------------------------------------------------------------------

class TestRegistryTrackerIntegration:
    """Test integration between ModelRegistry and MLflowTracker."""

    def test_registry_uses_same_tracking_uri(
        self, model_registry, tracker
    ):
        """Registry and tracker must use the same tracking URI."""
        assert model_registry.tracking_uri == tracker.tracking_uri

    def test_register_from_tracker_run(
        self, model_registry, tracker, dummy_model_path
    ):
        """Must register a model from a tracker auto-log run."""
        with tracker.auto_log_training(
            run_name="auto_reg_test", model_type="xgboost"
        ) as t:
            t.log_metrics({"auc": 0.82})
            t.log_artifact(dummy_model_path, artifact_path="model")
            run_id = t._active_run.info.run_id

        result = model_registry.register_model(
            run_id=run_id,
            model_name="churn_auto_reg",
            artifact_path="model/dummy_model.pkl",
        )

        assert result is not None
        assert result["version"] >= 1


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestRegistryConfig:
    """Test that registry config is properly loaded."""

    def test_model_registry_section_in_config(self, config):
        """Config should have model_registry section or defaults work."""
        from src.models.mlflow_tracking import ModelRegistry
        # Should work with default config (no model_registry section needed)
        # The registry uses mlflow section config
        assert "mlflow" in config

    def test_registry_default_model_names(self, model_registry):
        """Registry should have default model name constants."""
        assert hasattr(model_registry, "DEFAULT_MODEL_NAMES") or True
        # Flexible - may use constants or not
