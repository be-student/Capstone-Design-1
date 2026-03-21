"""
MLflow Tracking Integration Module.

Provides a wrapper around MLflow for experiment tracking, parameter/metric
logging, artifact management, and model registration. Uses local SQLite
backend and filesystem artifact store (no PostgreSQL required).

Supports auto-logging of model parameters, metrics, and artifacts during
training via the ``auto_log_training`` context manager and direct
integration hooks for MLChurnModel, DLChurnModel, and EnsembleChurnModel.

All configurable parameters are read from the YAML config dictionary.
"""

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking wrapper with auto-logging support.

    Manages experiment lifecycle including run creation, parameter/metric
    logging, artifact storage, and best-run selection. Uses SQLite as
    the backend store and local filesystem for artifacts.

    Provides ``auto_log_training`` context manager for automatic logging
    of model parameters, metrics, and artifacts during training.

    Args:
        config: Configuration dictionary with 'mlflow' section containing
            tracking_uri, artifact_location, and experiment_name.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize MLflow tracker from config.

        Automatically detects Docker environment via MLFLOW_TRACKING_URI
        environment variable. When running inside Docker containers, uses
        the HTTP tracking server; otherwise uses local SQLite backend.

        Args:
            config: Configuration dictionary with mlflow settings.
        """
        self.config = config
        mlflow_cfg = config.get("mlflow", {})

        # Docker environment detection: env var takes precedence
        env_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if env_tracking_uri:
            self.tracking_uri = env_tracking_uri
        else:
            self.tracking_uri = mlflow_cfg.get(
                "tracking_uri", "sqlite:///mlflow/mlflow.db"
            )

        self.artifact_location = mlflow_cfg.get(
            "artifact_location", "mlflow/artifacts"
        )
        self.default_experiment_name = mlflow_cfg.get(
            "experiment_name", "churn_prediction"
        )

        # Auto-logging flags from config
        self.log_models = mlflow_cfg.get("log_models", True)
        self.log_artifacts = mlflow_cfg.get("log_artifacts", True)

        # Docker config section for reference
        self.docker_config = mlflow_cfg.get("docker", {})

        # Ensure artifact directory exists (only for local filesystem)
        artifact_dir = self.artifact_location
        if (not artifact_dir.startswith("s3://")
                and not artifact_dir.startswith("gs://")
                and not artifact_dir.startswith("http")):
            os.makedirs(artifact_dir, exist_ok=True)

        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

        self._active_run = None
        self._current_experiment_name = None

    def create_experiment(self, name: str) -> str:
        """Create or retrieve an MLflow experiment.

        Args:
            name: Experiment name.

        Returns:
            Experiment ID string.
        """
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            exp_id = experiment.experiment_id
        else:
            exp_id = self.client.create_experiment(
                name,
                artifact_location=self.artifact_location,
            )
        self._current_experiment_name = name
        mlflow.set_experiment(name)
        return str(exp_id)

    def set_experiment(self, name: str) -> None:
        """Set the active experiment by name.

        Args:
            name: Experiment name. Created if it doesn't exist.
        """
        self.create_experiment(name)

    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run.

        Args:
            run_name: Optional human-readable run name.

        Returns:
            Run ID string.
        """
        if self._active_run is not None:
            mlflow.end_run()

        self._active_run = mlflow.start_run(run_name=run_name)
        return self._active_run.info.run_id

    def end_run(self) -> None:
        """End the current active MLflow run."""
        if self._active_run is not None:
            mlflow.end_run()
            self._active_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the active run.

        Args:
            params: Dictionary of parameter name-value pairs.
        """
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics to the active run.

        Args:
            metrics: Dictionary of metric name-value pairs.
            step: Optional step number for time-series metrics.
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """Log an artifact file to the active run.

        Args:
            local_path: Local filesystem path to the artifact.
            artifact_path: Optional subdirectory in the artifact store.
        """
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(
        self,
        model_path: str,
        model_name: str = "model",
    ) -> None:
        """Log a model file as an artifact.

        Args:
            model_path: Local path to the model file.
            model_name: Name for the model artifact subdirectory.
        """
        mlflow.log_artifact(model_path, artifact_path=model_name)

    def log_tags(self, tags: Dict[str, str]) -> None:
        """Log tags to the active run.

        Args:
            tags: Dictionary of tag name-value pairs.
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        mode: str = "max",
    ) -> Optional[Dict[str, Any]]:
        """Find the run with the best metric value in an experiment.

        Args:
            experiment_name: Name of the experiment to search.
            metric: Metric name to optimize.
            mode: 'max' for maximization, 'min' for minimization.

        Returns:
            Dictionary with 'run_id' and 'metric_value', or None if
            no runs found.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        order = "DESC" if mode == "max" else "ASC"
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if not runs:
            return None

        best_run = runs[0]
        metric_value = best_run.data.metrics.get(metric)

        return {
            "run_id": best_run.info.run_id,
            "metric_value": metric_value,
            "params": best_run.data.params,
            "metrics": best_run.data.metrics,
        }

    # ------------------------------------------------------------------
    # Auto-logging support
    # ------------------------------------------------------------------

    @contextmanager
    def auto_log_training(
        self,
        run_name: str,
        model_type: str = "unknown",
        experiment_name: Optional[str] = None,
    ) -> Generator["MLflowTracker", None, None]:
        """Context manager for automatic training run logging.

        Automatically creates an experiment (if needed), starts a run,
        and logs common config parameters. On exit, logs the run status
        and ends the run. If an exception occurs it is re-raised after
        the run is terminated with a ``FAILED`` tag.

        Usage::

            with tracker.auto_log_training("ml_run", "xgboost") as t:
                model.fit(X, y)
                t.log_metrics({"auc": auc})

        Args:
            run_name: Human-readable run name.
            model_type: Model type tag (e.g. 'xgboost', 'transformer').
            experiment_name: Experiment name. Defaults to the config value.

        Yields:
            The tracker instance with an active run.
        """
        exp_name = experiment_name or self.default_experiment_name
        self.create_experiment(exp_name)
        self.start_run(run_name=run_name)

        try:
            # Auto-log common parameters from config
            self._auto_log_config_params(model_type)
            yield self
            self.log_tags({"run_status": "completed"})
        except Exception:
            self.log_tags({"run_status": "failed"})
            raise
        finally:
            self.end_run()

    def _auto_log_config_params(self, model_type: str) -> None:
        """Log common config parameters automatically.

        Extracts and logs simulation seed, churn definition, pipeline
        weights, and model-specific parameters from the config dict.

        Args:
            model_type: Model type string for tagging.
        """
        # Tags
        self.log_tags({
            "model_type": model_type,
            "pipeline_stage": "training",
        })

        # Simulation params
        sim_cfg = self.config.get("simulation", {})
        params: Dict[str, Any] = {
            "random_seed": sim_cfg.get("random_seed", 42),
        }

        # Churn definition params
        churn_cfg = self.config.get("churn_definition", {})
        if churn_cfg:
            params["churn_no_purchase_days"] = churn_cfg.get(
                "no_purchase_days", 30
            )
            params["churn_no_login_days"] = churn_cfg.get(
                "no_login_days", 60
            )
            params["churn_operator"] = churn_cfg.get("operator", "OR")

        # Pipeline weights
        pipeline_cfg = self.config.get("pipeline", {})
        if pipeline_cfg:
            params["ensemble_weight_ml"] = pipeline_cfg.get(
                "ensemble_weight_ml", 0.6
            )
            params["ensemble_weight_dl"] = pipeline_cfg.get(
                "ensemble_weight_dl", 0.4
            )

        self.log_params(params)

    def auto_log_ml_model(
        self,
        model: Any,
        metrics: Dict[str, float],
        run_name: str = "ml_training",
        experiment_name: Optional[str] = None,
    ) -> str:
        """Auto-log an ML model's training results to MLflow.

        Logs model parameters (best_params, model_type, cv_scores),
        evaluation metrics, feature importance as artifact, and the
        serialised model file.

        Args:
            model: A trained MLChurnModel instance.
            metrics: Evaluation metrics dict (e.g. {'auc': 0.82}).
            run_name: MLflow run name.
            experiment_name: Experiment name override.

        Returns:
            The MLflow run ID.
        """
        exp_name = experiment_name or self.default_experiment_name
        self.create_experiment(exp_name)
        run_id = self.start_run(run_name=run_name)

        try:
            # Auto-log config params
            self._auto_log_config_params(
                model_type=getattr(model, "model_type", "ml")
            )

            # Log model-specific params
            model_params: Dict[str, Any] = {}
            if hasattr(model, "best_params") and model.best_params:
                for k, v in model.best_params.items():
                    model_params[f"best_{k}"] = v
            if hasattr(model, "model_type"):
                model_params["selected_model_type"] = model.model_type
            if hasattr(model, "n_folds"):
                model_params["n_folds"] = model.n_folds

            self.log_params(model_params)

            # Log CV scores as metrics
            if hasattr(model, "cv_scores") and model.cv_scores:
                cv_metrics = {}
                for k, v in model.cv_scores.items():
                    if isinstance(v, (int, float)):
                        cv_metrics[k] = float(v)
                if cv_metrics:
                    self.log_metrics(cv_metrics)

            # Log evaluation metrics
            self.log_metrics(metrics)

            # Log feature importance as artifact
            if (self.log_artifacts
                    and hasattr(model, "get_feature_importance")):
                try:
                    importance = model.get_feature_importance()
                    feature_names = getattr(model, "feature_names_", None)
                    if importance is not None:
                        self._log_feature_importance(
                            importance, feature_names
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to log feature importance: {e}"
                    )

            # Log model file as artifact
            if self.log_models and hasattr(model, "save"):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = os.path.join(tmpdir, "ml_model")
                        model.save(model_path)
                        # Find the saved file
                        for f in os.listdir(tmpdir):
                            fpath = os.path.join(tmpdir, f)
                            if os.path.isfile(fpath):
                                self.log_model(fpath, "ml_model")
                except Exception as e:
                    logger.warning(f"Failed to log model artifact: {e}")

            self.log_tags({"run_status": "completed"})
        except Exception:
            self.log_tags({"run_status": "failed"})
            raise
        finally:
            self.end_run()

        return run_id

    def auto_log_dl_model(
        self,
        model: Any,
        metrics: Dict[str, float],
        training_history: Optional[list] = None,
        run_name: str = "dl_training",
        experiment_name: Optional[str] = None,
    ) -> str:
        """Auto-log a DL model's training results to MLflow.

        Logs architecture parameters, evaluation metrics, per-epoch
        training history, and the serialised model file.

        Args:
            model: A trained DLChurnModel instance.
            metrics: Evaluation metrics dict.
            training_history: List of per-epoch metric dicts.
            run_name: MLflow run name.
            experiment_name: Experiment name override.

        Returns:
            The MLflow run ID.
        """
        exp_name = experiment_name or self.default_experiment_name
        self.create_experiment(exp_name)
        run_id = self.start_run(run_name=run_name)

        try:
            arch = getattr(model, "architecture", "dl")
            self._auto_log_config_params(model_type=arch)

            # Log DL-specific params
            dl_params: Dict[str, Any] = {
                "architecture": arch,
                "sequence_window": getattr(model, "sequence_window", 6),
                "hidden_size": getattr(model, "hidden_size", 64),
                "num_layers": getattr(model, "num_layers", 2),
                "dropout": getattr(model, "dropout", 0.2),
                "learning_rate": getattr(model, "learning_rate", 0.001),
                "batch_size": getattr(model, "batch_size", 32),
                "epochs": getattr(model, "epochs", 10),
            }
            self.log_params(dl_params)

            # Log evaluation metrics
            self.log_metrics(metrics)

            # Log training history as step metrics
            if training_history:
                for entry in training_history:
                    step = entry.get("epoch", 0)
                    step_metrics = {
                        k: float(v)
                        for k, v in entry.items()
                        if k != "epoch" and isinstance(v, (int, float))
                    }
                    if step_metrics:
                        self.log_metrics(step_metrics, step=step)

            # Log model file as artifact
            if self.log_models and hasattr(model, "save"):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        model_path = os.path.join(tmpdir, "dl_model.pt")
                        model.save(model_path)
                        self.log_model(model_path, "dl_model")
                except Exception as e:
                    logger.warning(f"Failed to log DL model artifact: {e}")

            self.log_tags({"run_status": "completed"})
        except Exception:
            self.log_tags({"run_status": "failed"})
            raise
        finally:
            self.end_run()

        return run_id

    def auto_log_ensemble(
        self,
        ensemble_model: Any,
        metrics: Dict[str, float],
        run_name: str = "ensemble_training",
        experiment_name: Optional[str] = None,
    ) -> str:
        """Auto-log an ensemble model's training results to MLflow.

        Logs ensemble weights, sub-model types, and evaluation metrics.

        Args:
            ensemble_model: A trained EnsembleChurnModel instance.
            metrics: Evaluation metrics dict.
            run_name: MLflow run name.
            experiment_name: Experiment name override.

        Returns:
            The MLflow run ID.
        """
        exp_name = experiment_name or self.default_experiment_name
        self.create_experiment(exp_name)
        run_id = self.start_run(run_name=run_name)

        try:
            self._auto_log_config_params(model_type="ensemble")

            # Ensemble-specific params
            ens_params: Dict[str, Any] = {
                "weight_ml": getattr(ensemble_model, "weight_ml", 0.6),
                "weight_dl": getattr(ensemble_model, "weight_dl", 0.4),
            }
            if hasattr(ensemble_model, "ml_model") and ensemble_model.ml_model:
                ens_params["ml_model_type"] = getattr(
                    ensemble_model.ml_model, "model_type", "unknown"
                )
            if hasattr(ensemble_model, "dl_model") and ensemble_model.dl_model:
                ens_params["dl_architecture"] = getattr(
                    ensemble_model.dl_model, "architecture", "unknown"
                )
            self.log_params(ens_params)

            # Log metrics
            self.log_metrics(metrics)

            self.log_tags({"run_status": "completed"})
        except Exception:
            self.log_tags({"run_status": "failed"})
            raise
        finally:
            self.end_run()

        return run_id

    def _log_feature_importance(
        self,
        importance: Any,
        feature_names: Optional[list] = None,
    ) -> None:
        """Log feature importance as a CSV artifact.

        Args:
            importance: Array-like of importance scores.
            feature_names: Optional list of feature names.
        """
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            fi_path = os.path.join(tmpdir, "feature_importance.csv")
            n_features = len(importance)
            if feature_names and len(feature_names) == n_features:
                names = feature_names
            else:
                names = [f"feature_{i}" for i in range(n_features)]

            lines = ["feature,importance"]
            imp_arr = np.array(importance)
            sorted_idx = np.argsort(imp_arr)[::-1]
            for idx in sorted_idx:
                lines.append(f"{names[idx]},{imp_arr[idx]:.6f}")

            with open(fi_path, "w") as f:
                f.write("\n".join(lines))

            self.log_artifact(fi_path, artifact_path="evaluation")

    def log_config_artifact(self) -> None:
        """Log the full config dictionary as a YAML artifact.

        Saves a snapshot of the current configuration to the active run
        for reproducibility.
        """
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not available; skipping config artifact")
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config_snapshot.yaml")
            with open(config_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            self.log_artifact(config_path, artifact_path="config")
