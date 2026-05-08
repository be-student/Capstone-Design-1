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

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT_NAMES = ("Users", "home", "private")
TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_ARTIFACT_NAMES = {
    "meta.yaml",
    "mlflow.user",
    "models_dir",
    "results_dir",
}


def _is_local_absolute_path(value: str) -> bool:
    """Return True for common workstation absolute path forms.

    Recognizes POSIX roots (``/Users``, ``/home``, ``/private``) and Windows
    drive roots (``C:\\Users``, ``D:\\Users``, ...).
    """
    parts = Path(value).parts
    if len(parts) < 2 or parts[1] not in LOCAL_ROOT_NAMES:
        return False
    if parts[0] == os.sep:
        return True
    return bool(os.path.splitdrive(value)[0])


def _collapse_absolute_path(value: str) -> str:
    """Collapse local absolute paths to evidence-safe relative paths."""
    path = Path(value)
    parts = path.parts
    for marker in ("models", "results", "data", "mlruns", "docs", "src", "tests"):
        if marker in parts:
            return Path(*parts[parts.index(marker):]).as_posix()
    return path.name


def _sanitize_text_evidence(text: str) -> str:
    """Remove machine-local absolute paths and usernames from text evidence."""
    root = PROJECT_ROOT.resolve(strict=False).as_posix()
    sanitized = text.replace(f"file://{root}/", "")
    sanitized = sanitized.replace(f"file://{root}", ".")
    sanitized = sanitized.replace(f"{root}/", "")
    sanitized = sanitized.replace(root, ".")

    local_user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if local_user:
        sanitized = sanitized.replace(local_user, "local-user")
    local_path_pattern = re.compile(
        r"(?:file://)?/(?:"
        + "|".join(re.escape(name) for name in LOCAL_ROOT_NAMES)
        + r")/[^\s\"',}]+"
    )
    sanitized = local_path_pattern.sub(
        lambda match: _collapse_absolute_path(
            match.group(0).removeprefix("file://")
        ),
        sanitized,
    )
    return sanitized


def _sanitize_mlflow_value(value: Any) -> Any:
    """Sanitize MLflow params/tags without changing numeric values."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str):
        return value

    sanitized = _sanitize_text_evidence(value)
    if sanitized != value:
        return sanitized

    if _is_local_absolute_path(value):
        return _collapse_absolute_path(value)
    if value.startswith("file://"):
        file_path = value[len("file://"):]
        if _is_local_absolute_path(file_path):
            return _collapse_absolute_path(file_path)
    return sanitized


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
        if (artifact_dir
                and not artifact_dir.startswith("s3://")
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
            if self.artifact_location:
                exp_id = self.client.create_experiment(
                    name,
                    artifact_location=self.artifact_location,
                )
            else:
                exp_id = self.client.create_experiment(name)
        self._current_experiment_name = name
        mlflow.set_experiment(name)
        self._scrub_local_file_store_evidence()
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
        mlflow.set_tag("mlflow.user", "local-user")
        return self._active_run.info.run_id

    def end_run(self) -> None:
        """End the current active MLflow run."""
        if self._active_run is not None:
            mlflow.end_run()
            self._active_run = None
            self._scrub_local_file_store_evidence()

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the active run.

        Args:
            params: Dictionary of parameter name-value pairs.
        """
        for key, value in params.items():
            mlflow.log_param(key, _sanitize_mlflow_value(value))

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
        safe_path, tmp_ctx = self._prepare_text_artifact_for_logging(local_path)
        try:
            mlflow.log_artifact(safe_path, artifact_path=artifact_path)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

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
            mlflow.set_tag(key, _sanitize_mlflow_value(value))

    def _prepare_text_artifact_for_logging(
        self,
        local_path: str,
    ) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
        """Return a sanitized temp artifact path when text evidence needs it."""
        source = Path(local_path)
        if (
            not source.is_file()
            or (
                source.suffix.lower() not in TEXT_ARTIFACT_SUFFIXES
                and source.name not in TEXT_ARTIFACT_NAMES
            )
        ):
            return local_path, None
        try:
            if source.stat().st_size > 5_000_000:
                return local_path, None
            original = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return local_path, None

        sanitized = _sanitize_text_evidence(original)
        if sanitized == original:
            return local_path, None

        tmp_ctx = tempfile.TemporaryDirectory()
        target = Path(tmp_ctx.name) / source.name
        target.write_text(sanitized, encoding="utf-8")
        return str(target), tmp_ctx

    def _scrub_local_file_store_evidence(self) -> None:
        """Best-effort scrub of repo-tracked local MLflow text evidence."""
        mlruns_dir = PROJECT_ROOT / "mlruns"
        if not mlruns_dir.exists():
            return

        for path in mlruns_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                if path.stat().st_size > 5_000_000:
                    continue
                original = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            sanitized = _sanitize_text_evidence(original)
            if sanitized != original:
                path.write_text(sanitized, encoding="utf-8")

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


class ModelRegistry:
    """MLflow Model Registry wrapper for versioning and stage management.

    Provides model registration, version tracking, stage transitions
    (None → Staging → Production → Archived), and model serving utilities.
    Uses the same tracking URI and backend as MLflowTracker.

    Args:
        config: Configuration dictionary with 'mlflow' section.
    """

    # Default registered model names used by the pipeline
    DEFAULT_MODEL_NAMES = {
        "ml": "churn_ml_model",
        "dl": "churn_dl_model",
        "ensemble": "churn_ensemble_model",
    }

    VALID_STAGES = ("None", "Staging", "Production", "Archived")

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model registry.

        Args:
            config: Configuration dictionary with mlflow settings.
        """
        self.config = config
        mlflow_cfg = config.get("mlflow", {})

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

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Register a model artifact from an MLflow run.

        Creates or updates a registered model and adds a new version
        linked to the specified run and artifact path.

        Args:
            run_id: MLflow run ID containing the model artifact.
            model_name: Name for the registered model.
            artifact_path: Relative path to the model artifact within
                the run's artifact store.
            description: Optional description for the model version.
            tags: Optional dict of tags to set on the model version.

        Returns:
            Dict with 'name', 'version', 'run_id', 'stage', 'source'.
        """
        # Ensure registered model exists
        try:
            self.client.get_registered_model(model_name)
        except Exception:
            self.client.create_registered_model(
                model_name,
                description=description or f"Registered model: {model_name}",
            )

        # Build the artifact URI
        source = f"runs:/{run_id}/{artifact_path}"

        # Create a new model version
        mv = self.client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
            description=description,
        )

        # Set tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    model_name, mv.version, key, value
                )

        logger.info(
            "Registered model '%s' version %s from run %s",
            model_name, mv.version, run_id,
        )

        return {
            "name": mv.name,
            "version": int(mv.version),
            "run_id": mv.run_id,
            "stage": mv.current_stage,
            "source": mv.source,
        }

    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = False,
    ) -> Dict[str, Any]:
        """Transition a model version to a new stage.

        Supports transitions to: None, Staging, Production, Archived.
        Optionally archives existing models in the target stage.

        Args:
            model_name: Registered model name.
            version: Model version number.
            stage: Target stage ('Staging', 'Production', 'Archived').
            archive_existing: If True, archive any existing model
                versions in the target stage before transitioning.

        Returns:
            Dict with 'name', 'version', 'stage'.
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}'. Must be one of {self.VALID_STAGES}"
            )

        # Archive existing versions in the target stage if requested
        if archive_existing and stage in ("Staging", "Production"):
            self._archive_existing_in_stage(model_name, stage)

        mv = self.client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )

        logger.info(
            "Transitioned model '%s' v%s to stage '%s'",
            model_name, version, stage,
        )

        return {
            "name": mv.name,
            "version": int(mv.version),
            "stage": mv.current_stage,
        }

    def _archive_existing_in_stage(
        self, model_name: str, stage: str
    ) -> None:
        """Archive all model versions currently in the given stage.

        Args:
            model_name: Registered model name.
            stage: Stage to clear (e.g. 'Production').
        """
        try:
            versions = self.client.get_latest_versions(
                model_name, stages=[stage]
            )
            for v in versions:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                )
                logger.info(
                    "Archived model '%s' v%s (was in '%s')",
                    model_name, v.version, stage,
                )
        except Exception as e:
            logger.warning("Failed to archive existing versions: %s", e)

    def get_model_version(
        self,
        model_name: str,
        version: int,
    ) -> Optional[Dict[str, Any]]:
        """Get details of a specific model version.

        Args:
            model_name: Registered model name.
            version: Version number.

        Returns:
            Dict with version details or None if not found.
        """
        try:
            mv = self.client.get_model_version(model_name, str(version))
            return {
                "name": mv.name,
                "version": int(mv.version),
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "source": mv.source,
                "status": mv.status,
                "description": mv.description,
            }
        except Exception:
            return None

    def list_model_versions(
        self,
        model_name: str,
        stage: Optional[str] = None,
    ) -> list:
        """List all versions of a registered model.

        Args:
            model_name: Registered model name.
            stage: Optional stage filter.

        Returns:
            List of version info dicts.
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
            else:
                # Search all versions
                filter_str = f"name='{model_name}'"
                versions = self.client.search_model_versions(filter_str)

            return [
                {
                    "name": v.name,
                    "version": int(v.version),
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                }
                for v in versions
            ]
        except Exception:
            return []

    def get_latest_version(
        self,
        model_name: str,
        stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the latest version of a model, optionally filtered by stage.

        Args:
            model_name: Registered model name.
            stage: Optional stage filter (e.g. 'Production').

        Returns:
            Dict with latest version info or None.
        """
        try:
            stages = [stage] if stage else None
            versions = self.client.get_latest_versions(
                model_name, stages=stages
            )
            if not versions:
                return None

            latest = max(versions, key=lambda v: int(v.version))
            return {
                "name": latest.name,
                "version": int(latest.version),
                "stage": latest.current_stage,
                "run_id": latest.run_id,
                "source": latest.source,
            }
        except Exception:
            return None

    def load_model(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load model artifact info from the registry.

        Returns the artifact URI and metadata needed to load the actual
        model object. For full deserialization, use the returned
        'artifact_uri' with joblib.load() or torch.load().

        Args:
            model_name: Registered model name.
            version: Specific version to load.
            stage: Stage to load from (e.g. 'Production').

        Returns:
            Dict with 'artifact_uri', 'version', 'stage', 'run_id',
            or None if model not found.
        """
        try:
            if version is not None:
                mv = self.client.get_model_version(
                    model_name, str(version)
                )
            elif stage is not None:
                versions = self.client.get_latest_versions(
                    model_name, stages=[stage]
                )
                if not versions:
                    return None
                mv = versions[0]
            else:
                return None

            # Build artifact URI from run info
            run = self.client.get_run(mv.run_id)
            artifact_uri = run.info.artifact_uri

            return {
                "model": None,  # Placeholder for deserialized model
                "artifact_uri": artifact_uri,
                "source": mv.source,
                "version": int(mv.version),
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "name": mv.name,
            }
        except Exception as e:
            logger.warning("Failed to load model '%s': %s", model_name, e)
            return None

    def get_serving_info(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[Dict[str, Any]]:
        """Get serving information for the current production model.

        Returns metadata about the model currently in the specified
        stage, suitable for configuring a scoring API endpoint.

        Args:
            model_name: Registered model name.
            stage: Stage to query (default: 'Production').

        Returns:
            Dict with version, stage, run_id, source, or None.
        """
        latest = self.get_latest_version(model_name, stage=stage)
        if latest is None:
            return None

        return {
            "name": model_name,
            "version": latest["version"],
            "stage": latest["stage"],
            "run_id": latest["run_id"],
            "source": latest.get("source", ""),
        }

    def promote_best_model(
        self,
        model_name: str,
        experiment_name: str,
        metric: str,
        mode: str = "max",
    ) -> Optional[Dict[str, Any]]:
        """Find the best-performing run and promote its model to Production.

        Searches all versions of the model, finds the one whose run has
        the best metric value, and transitions it to Production. Archives
        any previously-promoted model version.

        Args:
            model_name: Registered model name.
            experiment_name: Experiment name to search for best run.
            metric: Metric name to optimize.
            mode: 'max' for maximization, 'min' for minimization.

        Returns:
            Dict with promoted version info, or None if no versions found.
        """
        # Find the best run in the experiment
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning("Experiment '%s' not found", experiment_name)
            return None

        order = "DESC" if mode == "max" else "ASC"
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if not runs:
            logger.warning("No runs found in experiment '%s'", experiment_name)
            return None

        best_run_id = runs[0].info.run_id

        # Find the model version linked to this run
        versions = self.list_model_versions(model_name)
        target_version = None
        for v in versions:
            if v["run_id"] == best_run_id:
                target_version = v["version"]
                break

        if target_version is None:
            logger.warning(
                "No version of '%s' linked to best run %s",
                model_name, best_run_id,
            )
            return None

        # Promote to Production (archive existing)
        result = self.transition_model_stage(
            model_name=model_name,
            version=target_version,
            stage="Production",
            archive_existing=True,
        )

        logger.info(
            "Promoted model '%s' v%s to Production (metric %s=%s)",
            model_name, target_version, metric,
            runs[0].data.metrics.get(metric),
        )

        return result
