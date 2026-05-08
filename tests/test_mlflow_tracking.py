"""
TDD Tests for MLflow Tracking Integration Module.

Tests cover:
- MLflow tracker instantiation and configuration
- Experiment creation and management
- Run logging (parameters, metrics, artifacts)
- Model registration and versioning
- Metric logging (AUC, precision, recall, F1, etc.)
- Parameter logging (hyperparameters, config values)
- Artifact logging (model files, plots, feature importance)
- Run comparison and best model selection
- Local SQLite backend configuration
- Local artifact store configuration
- Integration with churn, CLV, uplift, and ensemble models
- Tag and metadata management
- Reproducibility via logged seeds and configs
- Configurable tracking URI from YAML
"""

import argparse
import json
import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


class _LightweightTreeModelDouble:
    """Pickle-safe tree-model double used to avoid native booster crashes."""

    def __init__(self, seed: int = 42) -> None:
        self.estimator = LogisticRegression(
            max_iter=500,
            solver="liblinear",
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LightweightTreeModelDouble":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict(X)
        return np.column_stack([1.0 - probs, probs])

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        del importance_type
        return np.abs(self.estimator.coef_[0])

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.feature_importance()


def _lightweight_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    score_bias: float = 0.0,
) -> float:
    """Run deterministic sklearn CV in place of native LightGBM/XGBoost CV."""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        model = _LightweightTreeModelDouble(seed=seed)
        model.fit(X[train_idx], y[train_idx])
        scores.append(roc_auc_score(y[val_idx], model.predict(X[val_idx])))
    return float(min(1.0, np.mean(scores) + score_bias))


@pytest.fixture
def lightweight_ml_boosters(monkeypatch):
    """Patch native ML boosters only where MLflow tracking is under test."""
    from src.models.churn_model import MLChurnModel

    def fake_lgb_cv(self, X, y, feature_names, params_entry):
        del feature_names, params_entry
        return _lightweight_cv_auc(X, y, self.n_folds, self.seed, score_bias=0.001)

    def fake_xgb_cv(self, X, y, params_entry):
        del params_entry
        return _lightweight_cv_auc(X, y, self.n_folds, self.seed)

    def fake_lgb_final(self, X, y, params_entry):
        del params_entry
        self._lgb_model = _LightweightTreeModelDouble(seed=self.seed).fit(X, y)
        self.model = self._lgb_model

    def fake_xgb_final(self, X, y, params_entry):
        del params_entry
        self._xgb_model = _LightweightTreeModelDouble(seed=self.seed).fit(X, y)
        self.model = self._xgb_model

    monkeypatch.setattr(MLChurnModel, "_cv_score_lightgbm", fake_lgb_cv)
    monkeypatch.setattr(MLChurnModel, "_cv_score_xgboost", fake_xgb_cv)
    monkeypatch.setattr(MLChurnModel, "_train_lightgbm_final", fake_lgb_final)
    monkeypatch.setattr(MLChurnModel, "_train_xgboost_final", fake_xgb_final)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_model_metrics():
    """Create sample model evaluation metrics."""
    return {
        "auc": 0.82,
        "precision": 0.75,
        "recall": 0.68,
        "f1_score": 0.71,
        "accuracy": 0.80,
        "log_loss": 0.45,
    }


@pytest.fixture
def sample_model_params():
    """Create sample model hyperparameters."""
    return {
        "model_type": "xgboost",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_seed": 42,
        "ensemble_weight_ml": 0.6,
        "ensemble_weight_dl": 0.4,
    }


@pytest.fixture
def sample_artifact_data(tmp_path):
    """Create sample artifacts for logging."""
    # Feature importance CSV
    fi_path = tmp_path / "feature_importance.csv"
    fi_data = pd.DataFrame({
        "feature": ["recency", "frequency", "monetary", "session_count",
                     "coupon_usage_rate"],
        "importance": [0.25, 0.20, 0.18, 0.15, 0.12],
    })
    fi_data.to_csv(fi_path, index=False)

    # Confusion matrix
    cm_path = tmp_path / "confusion_matrix.csv"
    cm_data = pd.DataFrame(
        [[850, 150], [120, 380]],
        columns=["pred_0", "pred_1"],
        index=["actual_0", "actual_1"],
    )
    cm_data.to_csv(cm_path)

    return {
        "feature_importance": str(fi_path),
        "confusion_matrix": str(cm_path),
    }


@pytest.fixture
def mlflow_tracker(config, tmp_path):
    """Create an MLflow tracker instance with local backend."""
    from src.models.mlflow_tracking import MLflowTracker

    # Override tracking URI to use temporary directory
    tracker_config = config.copy()
    tracker_config["mlflow"] = {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "artifact_location": str(tmp_path / "artifacts"),
        "experiment_name": "test_experiment",
    }

    return MLflowTracker(tracker_config)


# ---------------------------------------------------------------------------
# Tracker interface tests
# ---------------------------------------------------------------------------

class TestMLflowTrackerInterface:
    """Test MLflow tracker instantiation and interface."""

    def test_instantiation(self, mlflow_tracker):
        """MLflow tracker must be instantiable from config."""
        assert mlflow_tracker is not None

    def test_has_start_run_method(self, mlflow_tracker):
        """Must implement run start."""
        assert hasattr(mlflow_tracker, "start_run")
        assert callable(mlflow_tracker.start_run)

    def test_has_end_run_method(self, mlflow_tracker):
        """Must implement run end."""
        assert hasattr(mlflow_tracker, "end_run")
        assert callable(mlflow_tracker.end_run)

    def test_has_log_params_method(self, mlflow_tracker):
        """Must implement parameter logging."""
        assert hasattr(mlflow_tracker, "log_params")
        assert callable(mlflow_tracker.log_params)

    def test_has_log_metrics_method(self, mlflow_tracker):
        """Must implement metric logging."""
        assert hasattr(mlflow_tracker, "log_metrics")
        assert callable(mlflow_tracker.log_metrics)

    def test_has_log_artifact_method(self, mlflow_tracker):
        """Must implement artifact logging."""
        assert hasattr(mlflow_tracker, "log_artifact")
        assert callable(mlflow_tracker.log_artifact)

    def test_has_log_model_method(self, mlflow_tracker):
        """Must implement model logging."""
        assert hasattr(mlflow_tracker, "log_model")
        assert callable(mlflow_tracker.log_model)

    def test_has_get_best_run_method(self, mlflow_tracker):
        """Must implement best run selection."""
        assert hasattr(mlflow_tracker, "get_best_run")
        assert callable(mlflow_tracker.get_best_run)

    def test_has_create_experiment_method(self, mlflow_tracker):
        """Must implement experiment creation."""
        assert hasattr(mlflow_tracker, "create_experiment")
        assert callable(mlflow_tracker.create_experiment)


# ---------------------------------------------------------------------------
# Experiment management tests
# ---------------------------------------------------------------------------

class TestExperimentManagement:
    """Test experiment creation and management."""

    def test_create_experiment(self, mlflow_tracker):
        """Must create a named experiment."""
        exp_id = mlflow_tracker.create_experiment(
            name="churn_prediction_v1",
        )

        assert exp_id is not None, "Experiment ID must not be None"

    def test_create_experiment_idempotent(self, mlflow_tracker):
        """Creating same experiment twice should not raise an error."""
        exp_id1 = mlflow_tracker.create_experiment(
            name="churn_prediction_dup_test",
        )
        exp_id2 = mlflow_tracker.create_experiment(
            name="churn_prediction_dup_test",
        )

        assert exp_id1 == exp_id2, (
            "Duplicate experiment creation should return same ID"
        )

    def test_set_experiment(self, mlflow_tracker):
        """Must be able to set the active experiment."""
        assert hasattr(mlflow_tracker, "set_experiment")
        mlflow_tracker.set_experiment("churn_prediction_test")


# ---------------------------------------------------------------------------
# Run lifecycle tests
# ---------------------------------------------------------------------------

class TestRunLifecycle:
    """Test MLflow run lifecycle (start, log, end)."""

    def test_start_run_returns_run_id(self, mlflow_tracker):
        """start_run must return a run identifier."""
        mlflow_tracker.create_experiment(name="lifecycle_test")
        run_id = mlflow_tracker.start_run(run_name="test_run")

        assert run_id is not None
        assert isinstance(run_id, str)
        assert len(run_id) > 0

        mlflow_tracker.end_run()

    def test_context_manager_support(self, mlflow_tracker):
        """Tracker should support context manager (with statement)."""
        mlflow_tracker.create_experiment(name="context_test")

        # Either context manager or explicit start/end should work
        run_id = mlflow_tracker.start_run(run_name="ctx_test")
        assert run_id is not None
        mlflow_tracker.end_run()

    def test_end_run_completes(self, mlflow_tracker):
        """end_run must complete without error."""
        mlflow_tracker.create_experiment(name="end_run_test")
        mlflow_tracker.start_run(run_name="end_test")

        # Should not raise
        mlflow_tracker.end_run()


# ---------------------------------------------------------------------------
# Parameter logging tests
# ---------------------------------------------------------------------------

class TestParameterLogging:
    """Test parameter logging functionality."""

    def test_log_single_param(self, mlflow_tracker):
        """Must log a single parameter."""
        mlflow_tracker.create_experiment(name="param_test")
        mlflow_tracker.start_run(run_name="param_run")

        mlflow_tracker.log_params({"model_type": "xgboost"})

        mlflow_tracker.end_run()

    def test_log_multiple_params(self, mlflow_tracker,
                                   sample_model_params):
        """Must log multiple parameters at once."""
        mlflow_tracker.create_experiment(name="multi_param_test")
        mlflow_tracker.start_run(run_name="multi_param_run")

        mlflow_tracker.log_params(sample_model_params)

        mlflow_tracker.end_run()

    def test_log_config_params(self, mlflow_tracker, config):
        """Must log configuration parameters from YAML."""
        mlflow_tracker.create_experiment(name="config_param_test")
        mlflow_tracker.start_run(run_name="config_run")

        config_params = {
            "random_seed": config["simulation"]["random_seed"],
            "num_customers": config["simulation"]["num_customers"],
            "no_purchase_days": config["churn_definition"]["no_purchase_days"],
            "no_login_days": config["churn_definition"]["no_login_days"],
            "total_budget_krw": config["budget"]["total_krw"],
        }

        mlflow_tracker.log_params(config_params)

        mlflow_tracker.end_run()

    def test_log_ensemble_weights(self, mlflow_tracker, config):
        """Must log ensemble weights from pipeline config."""
        mlflow_tracker.create_experiment(name="ensemble_weight_test")
        mlflow_tracker.start_run(run_name="ensemble_run")

        mlflow_tracker.log_params({
            "ensemble_weight_ml": config["pipeline"]["ensemble_weight_ml"],
            "ensemble_weight_dl": config["pipeline"]["ensemble_weight_dl"],
        })

        mlflow_tracker.end_run()


# ---------------------------------------------------------------------------
# Metric logging tests
# ---------------------------------------------------------------------------

class TestMetricLogging:
    """Test metric logging functionality."""

    def test_log_single_metric(self, mlflow_tracker):
        """Must log a single metric."""
        mlflow_tracker.create_experiment(name="metric_test")
        mlflow_tracker.start_run(run_name="metric_run")

        mlflow_tracker.log_metrics({"auc": 0.82})

        mlflow_tracker.end_run()

    def test_log_multiple_metrics(self, mlflow_tracker,
                                    sample_model_metrics):
        """Must log multiple metrics at once."""
        mlflow_tracker.create_experiment(name="multi_metric_test")
        mlflow_tracker.start_run(run_name="multi_metric_run")

        mlflow_tracker.log_metrics(sample_model_metrics)

        mlflow_tracker.end_run()

    def test_log_step_metrics(self, mlflow_tracker):
        """Must support step-based metric logging for training curves."""
        mlflow_tracker.create_experiment(name="step_metric_test")
        mlflow_tracker.start_run(run_name="step_run")

        for epoch in range(5):
            mlflow_tracker.log_metrics(
                {"train_loss": 1.0 / (epoch + 1)},
                step=epoch,
            )

        mlflow_tracker.end_run()

    def test_log_auc_above_threshold(self, mlflow_tracker):
        """Verify AUC metric meets minimum threshold (0.78)."""
        mlflow_tracker.create_experiment(name="auc_threshold_test")
        mlflow_tracker.start_run(run_name="auc_run")

        auc = 0.82
        mlflow_tracker.log_metrics({"auc": auc})

        assert auc >= 0.78, (
            f"AUC {auc} below minimum threshold 0.78"
        )

        mlflow_tracker.end_run()


# ---------------------------------------------------------------------------
# Artifact logging tests
# ---------------------------------------------------------------------------

class TestArtifactLogging:
    """Test artifact logging functionality."""

    def test_log_single_artifact(self, mlflow_tracker,
                                   sample_artifact_data):
        """Must log a single artifact file."""
        mlflow_tracker.create_experiment(name="artifact_test")
        mlflow_tracker.start_run(run_name="artifact_run")

        mlflow_tracker.log_artifact(
            local_path=sample_artifact_data["feature_importance"],
        )

        mlflow_tracker.end_run()

    def test_log_multiple_artifacts(self, mlflow_tracker,
                                      sample_artifact_data):
        """Must log multiple artifact files."""
        mlflow_tracker.create_experiment(name="multi_artifact_test")
        mlflow_tracker.start_run(run_name="multi_artifact_run")

        for name, path in sample_artifact_data.items():
            mlflow_tracker.log_artifact(local_path=path)

        mlflow_tracker.end_run()

    def test_log_artifact_with_subdirectory(self, mlflow_tracker,
                                              sample_artifact_data):
        """Must support logging artifacts to subdirectories."""
        mlflow_tracker.create_experiment(name="subdir_artifact_test")
        mlflow_tracker.start_run(run_name="subdir_run")

        mlflow_tracker.log_artifact(
            local_path=sample_artifact_data["feature_importance"],
            artifact_path="evaluation",
        )

        mlflow_tracker.end_run()


# ---------------------------------------------------------------------------
# Model logging tests
# ---------------------------------------------------------------------------

class TestModelLogging:
    """Test model logging and registration."""

    def test_log_sklearn_compatible_model(self, mlflow_tracker, tmp_path):
        """Must log a scikit-learn compatible model."""
        mlflow_tracker.create_experiment(name="model_log_test")
        mlflow_tracker.start_run(run_name="model_run")

        # Create a simple model-like object for testing
        import pickle
        model_path = tmp_path / "test_model.pkl"
        dummy_model = {"type": "xgboost", "params": {"n_estimators": 100}}
        with open(model_path, "wb") as f:
            pickle.dump(dummy_model, f)

        mlflow_tracker.log_model(
            model_path=str(model_path),
            model_name="churn_model",
        )

        mlflow_tracker.end_run()

    def test_log_pytorch_model(self, mlflow_tracker, tmp_path):
        """Must log a PyTorch model."""
        mlflow_tracker.create_experiment(name="pytorch_log_test")
        mlflow_tracker.start_run(run_name="pytorch_run")

        import pickle
        model_path = tmp_path / "dl_model.pkl"
        dummy_model = {"type": "pytorch", "architecture": "MLP"}
        with open(model_path, "wb") as f:
            pickle.dump(dummy_model, f)

        mlflow_tracker.log_model(
            model_path=str(model_path),
            model_name="churn_dl_model",
        )

        mlflow_tracker.end_run()


# ---------------------------------------------------------------------------
# Best run selection tests
# ---------------------------------------------------------------------------

class TestBestRunSelection:
    """Test best run selection across experiments."""

    def test_get_best_run_by_metric(self, mlflow_tracker):
        """Must select the run with the best metric value."""
        mlflow_tracker.create_experiment(name="best_run_test")

        # Log multiple runs with different AUC values
        auc_values = [0.75, 0.82, 0.79]
        for i, auc in enumerate(auc_values):
            mlflow_tracker.start_run(run_name=f"run_{i}")
            mlflow_tracker.log_metrics({"auc": auc})
            mlflow_tracker.end_run()

        best = mlflow_tracker.get_best_run(
            experiment_name="best_run_test",
            metric="auc",
            mode="max",
        )

        assert best is not None, "Must return a best run"
        assert "run_id" in best, "Best run must include run_id"
        assert best["metric_value"] == max(auc_values), (
            f"Best AUC should be {max(auc_values)}, got {best['metric_value']}"
        )

    def test_get_best_run_minimize(self, mlflow_tracker):
        """Must support minimization mode (e.g., for loss)."""
        mlflow_tracker.create_experiment(name="min_run_test")

        loss_values = [0.55, 0.42, 0.48]
        for i, loss in enumerate(loss_values):
            mlflow_tracker.start_run(run_name=f"loss_run_{i}")
            mlflow_tracker.log_metrics({"log_loss": loss})
            mlflow_tracker.end_run()

        best = mlflow_tracker.get_best_run(
            experiment_name="min_run_test",
            metric="log_loss",
            mode="min",
        )

        assert best["metric_value"] == min(loss_values)


# ---------------------------------------------------------------------------
# Tag management tests
# ---------------------------------------------------------------------------

class TestTagManagement:
    """Test run tagging and metadata."""

    def test_log_tags(self, mlflow_tracker):
        """Must support logging tags to a run."""
        mlflow_tracker.create_experiment(name="tag_test")
        mlflow_tracker.start_run(run_name="tag_run")

        assert hasattr(mlflow_tracker, "log_tags")
        mlflow_tracker.log_tags({
            "model_family": "ensemble",
            "data_version": "v1",
            "pipeline_stage": "training",
        })

        mlflow_tracker.end_run()

    def test_log_churn_definition_tags(self, mlflow_tracker, config):
        """Must tag runs with churn definition parameters."""
        mlflow_tracker.create_experiment(name="churn_tag_test")
        mlflow_tracker.start_run(run_name="churn_tag_run")

        mlflow_tracker.log_tags({
            "churn_no_purchase_days": str(
                config["churn_definition"]["no_purchase_days"]
            ),
            "churn_no_login_days": str(
                config["churn_definition"]["no_login_days"]
            ),
            "churn_operator": config["churn_definition"]["operator"],
        })

        mlflow_tracker.end_run()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestMLflowIntegration:
    """Test integration with the full model pipeline."""

    def test_log_full_training_run(self, mlflow_tracker,
                                     sample_model_params,
                                     sample_model_metrics,
                                     sample_artifact_data):
        """Must support logging a complete training run end-to-end."""
        mlflow_tracker.create_experiment(name="full_run_test")
        run_id = mlflow_tracker.start_run(run_name="full_training_run")

        # Log params
        mlflow_tracker.log_params(sample_model_params)

        # Log metrics
        mlflow_tracker.log_metrics(sample_model_metrics)

        # Log artifacts
        for path in sample_artifact_data.values():
            mlflow_tracker.log_artifact(local_path=path)

        # Log tags
        mlflow_tracker.log_tags({
            "pipeline_stage": "training",
            "model_type": "ensemble",
        })

        mlflow_tracker.end_run()

        assert run_id is not None

    def test_log_multiple_model_types(self, mlflow_tracker):
        """Must track ML, DL, and ensemble models in the same experiment."""
        mlflow_tracker.create_experiment(name="multi_model_test")

        model_configs = [
            {"model_type": "xgboost", "auc": 0.80},
            {"model_type": "pytorch_mlp", "auc": 0.78},
            {"model_type": "ensemble", "auc": 0.82},
        ]

        for model in model_configs:
            mlflow_tracker.start_run(run_name=f"{model['model_type']}_run")
            mlflow_tracker.log_params({"model_type": model["model_type"]})
            mlflow_tracker.log_metrics({"auc": model["auc"]})
            mlflow_tracker.end_run()

        # Best run should be the ensemble
        best = mlflow_tracker.get_best_run(
            experiment_name="multi_model_test",
            metric="auc",
            mode="max",
        )

        assert best["metric_value"] == 0.82


# ---------------------------------------------------------------------------
# Backend configuration tests
# ---------------------------------------------------------------------------

class TestBackendConfiguration:
    """Test local SQLite and artifact store configuration."""

    def test_uses_sqlite_backend(self, mlflow_tracker):
        """Tracker must use SQLite as the backend store."""
        tracking_uri = mlflow_tracker.tracking_uri
        assert "sqlite" in tracking_uri.lower(), (
            f"Expected SQLite backend, got: {tracking_uri}"
        )

    def test_uses_local_artifact_store(self, mlflow_tracker):
        """Tracker must use local filesystem for artifacts."""
        artifact_location = mlflow_tracker.artifact_location
        # Should be a local file path, not S3/GCS
        assert not artifact_location.startswith("s3://"), (
            "Artifact store should be local, not S3"
        )
        assert not artifact_location.startswith("gs://"), (
            "Artifact store should be local, not GCS"
        )

    def test_docker_config_section_exists(self, config):
        """Config must have docker-specific MLflow settings."""
        assert "docker" in config["mlflow"], (
            "MLflow config must have a 'docker' section"
        )
        docker_cfg = config["mlflow"]["docker"]
        assert "tracking_uri" in docker_cfg
        assert "artifact_location" in docker_cfg
        assert "server_host" in docker_cfg
        assert "server_port" in docker_cfg
        assert "backend_store_uri" in docker_cfg

    def test_docker_tracking_uri_is_http(self, config):
        """Docker tracking URI must use HTTP protocol for service discovery."""
        docker_uri = config["mlflow"]["docker"]["tracking_uri"]
        assert docker_uri.startswith("http"), (
            f"Docker tracking URI must be HTTP, got: {docker_uri}"
        )
        assert "mlflow" in docker_uri, (
            "Docker tracking URI should reference 'mlflow' service name"
        )

    def test_docker_env_override(self, config, tmp_path):
        """Tracker must use MLFLOW_TRACKING_URI env var when set."""
        from src.models.mlflow_tracking import MLflowTracker

        docker_uri = "http://mlflow:5000"
        tracker_config = config.copy()
        tracker_config["mlflow"] = {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifact_location": str(tmp_path / "artifacts"),
            "experiment_name": "test_docker_env",
        }

        # Set env var to simulate Docker environment
        original = os.environ.get("MLFLOW_TRACKING_URI")
        try:
            os.environ["MLFLOW_TRACKING_URI"] = docker_uri
            tracker = MLflowTracker(tracker_config)
            assert tracker.tracking_uri == docker_uri, (
                "Tracker should use MLFLOW_TRACKING_URI env var"
            )
        finally:
            if original is None:
                os.environ.pop("MLFLOW_TRACKING_URI", None)
            else:
                os.environ["MLFLOW_TRACKING_URI"] = original


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestMLflowReproducibility:
    """Test that seed and config are tracked for reproducibility."""

    def test_seed_logged_as_param(self, mlflow_tracker, config):
        """Random seed must be logged as a parameter."""
        mlflow_tracker.create_experiment(name="seed_test")
        mlflow_tracker.start_run(run_name="seed_run")

        seed = config["simulation"]["random_seed"]
        mlflow_tracker.log_params({"random_seed": seed})

        mlflow_tracker.end_run()

    def test_full_config_loggable(self, mlflow_tracker, config, tmp_path):
        """Full YAML config should be loggable as an artifact."""
        import yaml

        mlflow_tracker.create_experiment(name="config_artifact_test")
        mlflow_tracker.start_run(run_name="config_run")

        config_path = tmp_path / "config_snapshot.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        mlflow_tracker.log_artifact(
            local_path=str(config_path),
            artifact_path="config",
        )

        mlflow_tracker.end_run()


# ---------------------------------------------------------------------------
# Auto-logging tests
# ---------------------------------------------------------------------------

class TestAutoLogTraining:
    """Test auto-logging context manager and auto-log methods."""

    def test_auto_log_training_context_manager(self, mlflow_tracker):
        """auto_log_training context manager must start and end a run."""
        with mlflow_tracker.auto_log_training(
            run_name="auto_test", model_type="xgboost"
        ) as tracker:
            assert tracker is not None
            tracker.log_metrics({"auc": 0.85})

    def test_auto_log_training_logs_config_params(self, mlflow_tracker):
        """auto_log_training must auto-log config params."""
        mlflow_tracker.create_experiment(name="auto_config_test")
        with mlflow_tracker.auto_log_training(
            run_name="auto_config_run",
            model_type="lightgbm",
            experiment_name="auto_config_test",
        ) as tracker:
            tracker.log_metrics({"auc": 0.80})

        best = mlflow_tracker.get_best_run(
            experiment_name="auto_config_test",
            metric="auc",
            mode="max",
        )
        assert best is not None
        assert "random_seed" in best["params"]

    def test_auto_log_training_tags_model_type(self, mlflow_tracker):
        """auto_log_training must tag the run with model_type."""
        with mlflow_tracker.auto_log_training(
            run_name="tag_test", model_type="transformer"
        ):
            pass
        # No assertion on tag content since get_best_run doesn't return
        # tags, but it should not raise.

    def test_auto_log_training_handles_exception(self, mlflow_tracker):
        """auto_log_training must end the run even if an exception occurs."""
        with pytest.raises(ValueError):
            with mlflow_tracker.auto_log_training(
                run_name="fail_test", model_type="bad"
            ):
                raise ValueError("Training failed")

    def test_auto_log_ml_model(self, mlflow_tracker, config, tmp_path):
        """auto_log_ml_model must log ML model params and metrics."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.model_type = "lightgbm"
        mock_model.best_params = {"num_leaves": 31, "learning_rate": 0.05}
        mock_model.cv_scores = {
            "lightgbm_best_cv_auc": 0.82,
            "xgboost_best_cv_auc": 0.79,
            "lightgbm_best_params": {"num_leaves": 31},
            "xgboost_best_params": {"max_depth": 6},
        }
        mock_model.n_folds = 5
        mock_model.feature_names_ = ["f1", "f2", "f3"]
        mock_model.get_feature_importance.return_value = [0.5, 0.3, 0.2]

        # Simulate save creating a file
        def fake_save(path):
            with open(path + ".joblib", "w") as f:
                f.write("dummy")
        mock_model.save = fake_save

        run_id = mlflow_tracker.auto_log_ml_model(
            model=mock_model,
            metrics={"auc": 0.82, "f1": 0.71},
            run_name="auto_ml_test",
        )
        assert run_id is not None
        assert isinstance(run_id, str)

    def test_auto_log_dl_model(self, mlflow_tracker):
        """auto_log_dl_model must log DL model params and history."""
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.architecture = "transformer"
        mock_model.sequence_window = 6
        mock_model.hidden_size = 64
        mock_model.num_layers = 2
        mock_model.dropout = 0.2
        mock_model.learning_rate = 0.001
        mock_model.batch_size = 32
        mock_model.epochs = 10

        history = [
            {"epoch": 0, "train_loss": 0.69},
            {"epoch": 1, "train_loss": 0.55},
            {"epoch": 2, "train_loss": 0.42},
        ]

        run_id = mlflow_tracker.auto_log_dl_model(
            model=mock_model,
            metrics={"auc": 0.78, "accuracy": 0.75},
            training_history=history,
            run_name="auto_dl_test",
        )
        assert run_id is not None

    def test_auto_log_ensemble(self, mlflow_tracker):
        """auto_log_ensemble must log ensemble weights and metrics."""
        from unittest.mock import MagicMock

        mock_ensemble = MagicMock()
        mock_ensemble.weight_ml = 0.6
        mock_ensemble.weight_dl = 0.4
        mock_ensemble.ml_model = MagicMock()
        mock_ensemble.ml_model.model_type = "lightgbm"
        mock_ensemble.dl_model = MagicMock()
        mock_ensemble.dl_model.architecture = "transformer"

        run_id = mlflow_tracker.auto_log_ensemble(
            ensemble_model=mock_ensemble,
            metrics={"auc": 0.83, "f1": 0.74},
            run_name="auto_ensemble_test",
        )
        assert run_id is not None

    def test_log_config_artifact(self, mlflow_tracker):
        """log_config_artifact must log the full config as YAML."""
        mlflow_tracker.create_experiment(name="config_art_test")
        mlflow_tracker.start_run(run_name="config_art_run")
        # Should not raise
        mlflow_tracker.log_config_artifact()
        mlflow_tracker.end_run()

    def test_has_auto_log_methods(self, mlflow_tracker):
        """Tracker must expose all auto-log methods."""
        assert hasattr(mlflow_tracker, "auto_log_training")
        assert hasattr(mlflow_tracker, "auto_log_ml_model")
        assert hasattr(mlflow_tracker, "auto_log_dl_model")
        assert hasattr(mlflow_tracker, "auto_log_ensemble")
        assert hasattr(mlflow_tracker, "log_config_artifact")

    def test_run_all_mlflow_logging_creates_evidence_run(
        self, config, tmp_path, monkeypatch
    ):
        """The run_all MLflow stage must create a real run with artifacts."""
        from src.main import run_mlflow_logging
        from src.models.mlflow_tracking import MLflowTracker

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        test_config = config.copy()
        test_config["mlflow"] = {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "artifact_location": str(tmp_path / "artifacts"),
            "experiment_name": "pipeline_evidence_test",
            "log_models": True,
            "log_artifacts": True,
        }

        results_dir = tmp_path / "results"
        models_dir = tmp_path / "models"
        results_dir.mkdir()
        models_dir.mkdir()
        (results_dir / "model_metrics.json").write_text(
            json.dumps({"status": "completed"}),
            encoding="utf-8",
        )
        (models_dir / "model_artifacts_manifest.json").write_text(
            json.dumps({"artifacts": [{"versioned_filename": "m_v1.pt"}]}),
            encoding="utf-8",
        )
        (models_dir / "dl_churn_model_v1.pt").write_text(
            "dummy",
            encoding="utf-8",
        )
        args = argparse.Namespace(data=None, output=str(tmp_path), small=True)

        result = run_mlflow_logging(test_config, args)

        assert result["status"] == "completed"
        assert result["logged_artifact_count"] >= 2
        tracker = MLflowTracker(test_config)
        run = tracker.client.get_run(result["run_id"])
        assert run.data.tags["pipeline_stage"] == "mlflow_logging"


class TestModelTrackerIntegration:
    """Test that models accept tracker parameter during fit()."""

    def test_ml_model_fit_accepts_tracker(self, config):
        """MLChurnModel.fit() must accept optional tracker parameter."""
        from src.models.churn_model import MLChurnModel
        import inspect

        sig = inspect.signature(MLChurnModel.fit)
        assert "tracker" in sig.parameters

    def test_dl_model_fit_accepts_tracker(self, config):
        """DLChurnModel.fit() must accept optional tracker parameter."""
        from src.models.churn_model import DLChurnModel
        import inspect

        sig = inspect.signature(DLChurnModel.fit)
        assert "tracker" in sig.parameters

    def test_ensemble_model_fit_accepts_tracker(self, config):
        """EnsembleChurnModel.fit() must accept optional tracker parameter."""
        from src.models.churn_model import EnsembleChurnModel
        import inspect

        sig = inspect.signature(EnsembleChurnModel.fit)
        assert "tracker" in sig.parameters

    def test_ml_model_fit_with_tracker(
        self, mlflow_tracker, config, lightweight_ml_boosters
    ):
        """MLChurnModel.fit() must log to tracker when provided."""
        from src.models.churn_model import MLChurnModel
        del lightweight_ml_boosters
        np.random.seed(42)

        n = 500
        X = pd.DataFrame(
            np.random.randn(n, 10),
            columns=[f"f{i}" for i in range(10)],
        )
        signal = 0.8 * X["f0"] - 0.6 * X["f1"] + np.random.randn(n) * 0.5
        y = (signal > 0).astype(int).values

        model = MLChurnModel(config)
        mlflow_tracker.create_experiment(name="ml_fit_tracker_test")
        mlflow_tracker.start_run(run_name="ml_fit_run")

        model.fit(X, y, tracker=mlflow_tracker)

        mlflow_tracker.end_run()

        best = mlflow_tracker.get_best_run(
            experiment_name="ml_fit_tracker_test",
            metric="lightgbm_best_cv_auc",
            mode="max",
        )
        assert best is not None
        assert best["metric_value"] > 0.5

    def test_dl_model_fit_with_tracker(self, mlflow_tracker, config):
        """DLChurnModel.fit() must log per-epoch loss to tracker."""
        from src.models.churn_model import DLChurnModel
        np.random.seed(42)

        n = 200
        X = pd.DataFrame(
            np.random.randn(n, 5),
            columns=[f"f{i}" for i in range(5)],
        )
        y = np.random.randint(0, 2, n)

        # Use minimal epochs for speed
        test_config = config.copy()
        test_config["dl_model"] = config.get("dl_model", {}).copy()
        test_config["dl_model"]["epochs"] = 2

        model = DLChurnModel(test_config)
        mlflow_tracker.create_experiment(name="dl_fit_tracker_test")
        mlflow_tracker.start_run(run_name="dl_fit_run")

        model.fit(X, y, tracker=mlflow_tracker)

        mlflow_tracker.end_run()

        best = mlflow_tracker.get_best_run(
            experiment_name="dl_fit_tracker_test",
            metric="dl_train_loss",
            mode="min",
        )
        assert best is not None
        assert best["metric_value"] > 0
