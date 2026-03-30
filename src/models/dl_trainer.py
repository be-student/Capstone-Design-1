"""
DL Training and Evaluation Pipeline.

Provides a complete training pipeline for deep learning churn prediction
models with:
- Early stopping with configurable patience and min_delta
- Model selection between LSTM and Transformer architectures
- MLflow integration for experiment tracking
- Training/validation split for monitoring
- Comprehensive evaluation metrics (AUC, precision, recall, F1)
- Model checkpointing and best-model restoration

All configurable parameters are read from the YAML config dictionary.
"""

import copy
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from src.models.churn_model import (
    DLChurnModel,
    LSTMChurnNetwork,
    TransformerChurnNetwork,
)


class EarlyStopping:
    """Early stopping monitor for training loops.

    Tracks a monitored metric and signals when training should stop
    if no improvement is observed for ``patience`` consecutive epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        monitor: Metric name being monitored (for logging).
        mode: 'min' for loss-like metrics, 'max' for accuracy-like.
        restore_best_weights: Whether to restore model to best state.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        monitor: str = "val_loss",
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        """Initialize early stopping monitor."""
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.should_stop: bool = False
        self.best_state_dict: Optional[dict] = None
        self.history: List[float] = []

    def _is_improvement(self, current: float) -> bool:
        """Check if current score is an improvement over the best.

        Args:
            current: Current metric value.

        Returns:
            True if current is an improvement.
        """
        if self.best_score is None:
            return True

        if self.mode == "min":
            return current < (self.best_score - self.min_delta)
        else:
            return current > (self.best_score + self.min_delta)

    def step(
        self, score: float, model: Optional[nn.Module] = None, epoch: int = 0
    ) -> bool:
        """Update early stopping state with new score.

        Args:
            score: Current metric value.
            model: PyTorch model (for saving best state).
            epoch: Current epoch number.

        Returns:
            True if training should stop.
        """
        self.history.append(score)

        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def restore(self, model: nn.Module) -> None:
        """Restore model to best weights.

        Args:
            model: PyTorch model to restore.
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


class DLTrainer:
    """Deep learning training and evaluation pipeline.

    Manages the full lifecycle of DL model training including:
    - Architecture selection (LSTM vs Transformer)
    - Training with early stopping
    - Validation monitoring
    - MLflow experiment tracking
    - Model evaluation with comprehensive metrics
    - Model checkpointing

    Args:
        config: Configuration dictionary with dl_model, simulation,
            and optionally mlflow sections.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize DL trainer from config.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.seed = config.get("simulation", {}).get("random_seed", 42)

        dl_cfg = config.get("dl_model", {})
        self.architecture = dl_cfg.get("architecture", "transformer")
        self.sequence_window = dl_cfg.get("sequence_window", 6)
        self.hidden_size = dl_cfg.get("hidden_size", 64)
        self.num_layers = dl_cfg.get("num_layers", 2)
        self.num_attention_heads = dl_cfg.get("num_attention_heads", 4)
        self.dim_feedforward = dl_cfg.get("dim_feedforward", 128)
        self.dropout = dl_cfg.get("dropout", 0.2)
        self.learning_rate = dl_cfg.get("learning_rate", 0.001)
        self.batch_size = dl_cfg.get("batch_size", 32)
        self.epochs = dl_cfg.get("epochs", 10)

        # Early stopping config
        es_cfg = dl_cfg.get("early_stopping", {})
        self.early_stopping_enabled = es_cfg.get("enabled", True)
        self.es_patience = es_cfg.get("patience", 5)
        self.es_min_delta = es_cfg.get("min_delta", 0.001)
        self.es_monitor = es_cfg.get("monitor", "val_loss")
        self.es_restore_best = es_cfg.get("restore_best_weights", True)

        self.device = torch.device("cpu")
        self.model: Optional[nn.Module] = None
        self.dl_model: Optional[DLChurnModel] = None
        self.training_history: List[Dict[str, float]] = []
        self.best_architecture: Optional[str] = None
        self.evaluation_results: Optional[Dict[str, float]] = None

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_network(self, input_size: int, architecture: str) -> nn.Module:
        """Build a neural network for the specified architecture.

        Args:
            input_size: Number of input features.
            architecture: 'lstm' or 'transformer'.

        Returns:
            PyTorch Module.
        """
        if architecture == "transformer":
            return TransformerChurnNetwork(
                input_size=input_size,
                d_model=self.hidden_size,
                nhead=self.num_attention_heads,
                num_encoder_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                sequence_window=self.sequence_window,
            )
        else:
            return LSTMChurnNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

    def _prepare_tensor(
        self,
        X: pd.DataFrame,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Convert DataFrame to 3D tensor for sequence models.

        Args:
            X: Feature DataFrame.
            feature_mean: Pre-computed feature means (None to compute).
            feature_std: Pre-computed feature stds (None to compute).

        Returns:
            Tuple of (tensor, feature_mean, feature_std).
        """
        values = X.values.astype(np.float32)

        if feature_mean is None:
            feature_mean = values.mean(axis=0)
        if feature_std is None:
            feature_std = values.std(axis=0) + 1e-8

        values = (values - feature_mean) / feature_std

        n_samples = values.shape[0]
        sequences = np.tile(
            values[:, np.newaxis, :], (1, self.sequence_window, 1)
        )
        for t in range(self.sequence_window):
            scale = (t + 1) / self.sequence_window
            sequences[:, t, :] *= scale

        return (
            torch.tensor(sequences, dtype=torch.float32),
            feature_mean,
            feature_std,
        )

    def train_single_architecture(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        architecture: Optional[str] = None,
        tracker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train a single DL architecture with early stopping.

        Args:
            X_train: Training features DataFrame.
            y_train: Training labels.
            X_val: Optional validation features. If None, splits from train.
            y_val: Optional validation labels.
            architecture: 'lstm' or 'transformer'. Defaults to config value.
            tracker: Optional MLflowTracker instance for logging.

        Returns:
            Dictionary with 'model', 'history', 'best_epoch',
            'val_loss', and 'architecture'.
        """
        self._set_seed()

        if architecture is None:
            architecture = self.architecture

        input_size = X_train.shape[1]

        # Create validation split if not provided
        if X_val is None or y_val is None:
            val_ratio = 0.2
            n_val = max(1, int(len(X_train) * val_ratio))
            X_val_df = X_train.iloc[-n_val:]
            y_val_arr = y_train[-n_val:]
            X_train_df = X_train.iloc[:-n_val]
            y_train_arr = y_train[:-n_val]
        else:
            X_train_df = X_train
            y_train_arr = y_train
            X_val_df = X_val
            y_val_arr = y_val

        # Prepare tensors
        X_train_t, feat_mean, feat_std = self._prepare_tensor(X_train_df)
        X_val_t, _, _ = self._prepare_tensor(X_val_df, feat_mean, feat_std)
        y_train_t = torch.tensor(y_train_arr.astype(np.float32))
        y_val_t = torch.tensor(y_val_arr.astype(np.float32))

        # Build network
        network = self._build_network(input_size, architecture).to(self.device)

        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Optimizer and loss
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Early stopping
        early_stopper = EarlyStopping(
            patience=self.es_patience,
            min_delta=self.es_min_delta,
            monitor=self.es_monitor,
            mode="min",
            restore_best_weights=self.es_restore_best,
        )

        history = []

        # Training loop
        for epoch in range(self.epochs):
            # Train phase
            network.train()
            train_loss_sum = 0.0
            train_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = network(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            train_loss = train_loss_sum / max(train_batches, 1)

            # Validation phase
            network.eval()
            with torch.no_grad():
                val_logits = network(X_val_t.to(self.device))
                val_loss = criterion(val_logits, y_val_t.to(self.device)).item()
                val_probs = torch.sigmoid(val_logits).cpu().numpy()

                # Compute validation AUC if possible
                val_auc = 0.0
                if len(np.unique(y_val_arr)) > 1:
                    val_auc = roc_auc_score(y_val_arr, val_probs)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
            }
            history.append(epoch_metrics)

            # Log to MLflow if tracker provided
            if tracker is not None:
                tracker.log_metrics(
                    {
                        f"{architecture}_train_loss": train_loss,
                        f"{architecture}_val_loss": val_loss,
                        f"{architecture}_val_auc": val_auc,
                    },
                    step=epoch,
                )

            # Early stopping check
            if self.early_stopping_enabled:
                if early_stopper.step(val_loss, network, epoch):
                    break

        # Restore best weights
        if self.early_stopping_enabled and self.es_restore_best:
            early_stopper.restore(network)

        network.eval()

        return {
            "model": network,
            "history": history,
            "best_epoch": early_stopper.best_epoch,
            "val_loss": early_stopper.best_score if early_stopper.best_score is not None else val_loss,
            "architecture": architecture,
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "input_size": input_size,
        }

    def select_best_architecture(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        architectures: Optional[List[str]] = None,
        tracker: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train both LSTM and Transformer and select the best.

        Trains each architecture independently and selects the one
        with the lowest validation loss.

        Args:
            X_train: Training features DataFrame.
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.
            architectures: List of architectures to try.
                Defaults to ['lstm', 'transformer'].
            tracker: Optional MLflowTracker for logging.

        Returns:
            Dictionary with best model results and comparison info.
        """
        if architectures is None:
            architectures = ["lstm", "transformer"]

        results = {}

        for arch in architectures:
            self._set_seed()  # Reset seed for fair comparison
            result = self.train_single_architecture(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                architecture=arch,
                tracker=tracker,
            )
            results[arch] = result

        # Select best by validation loss
        best_arch = min(results, key=lambda k: results[k]["val_loss"])
        self.best_architecture = best_arch

        if tracker is not None:
            tracker.log_params({"best_architecture": best_arch})
            for arch, result in results.items():
                tracker.log_metrics({
                    f"{arch}_best_val_loss": result["val_loss"],
                    f"{arch}_best_epoch": result["best_epoch"],
                })

        best_result = results[best_arch]
        best_result["all_results"] = results
        best_result["comparison"] = {
            arch: {
                "val_loss": r["val_loss"],
                "best_epoch": r["best_epoch"],
            }
            for arch, r in results.items()
        }

        return best_result

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        tracker: Optional[Any] = None,
        select_architecture: bool = False,
    ) -> Dict[str, Any]:
        """Full training and evaluation pipeline.

        Trains the DL model, evaluates on test set, and returns
        comprehensive metrics. Optionally selects between architectures.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_test: Test features.
            y_test: Test labels.
            tracker: Optional MLflowTracker.
            select_architecture: If True, compare LSTM and Transformer.

        Returns:
            Dictionary with trained DLChurnModel, evaluation metrics,
            training history, and architecture info.
        """
        if select_architecture:
            result = self.select_best_architecture(
                X_train=X_train,
                y_train=y_train,
                tracker=tracker,
            )
        else:
            result = self.train_single_architecture(
                X_train=X_train,
                y_train=y_train,
                architecture=self.architecture,
                tracker=tracker,
            )

        # Build a DLChurnModel wrapper with the trained network
        dl_config = self.config.copy()
        dl_config["dl_model"] = dl_config.get("dl_model", {}).copy()
        dl_config["dl_model"]["architecture"] = result["architecture"]

        dl_model = DLChurnModel(dl_config)
        dl_model.model = result["model"]
        dl_model.network = result["model"]
        dl_model.input_size_ = result["input_size"]
        dl_model._feature_mean = result["feature_mean"]
        dl_model._feature_std = result["feature_std"]

        # Evaluate on test set
        test_probs = dl_model.predict_proba(X_test)
        eval_metrics = self._compute_metrics(y_test, test_probs)
        self.evaluation_results = eval_metrics

        # Log evaluation metrics
        if tracker is not None:
            tracker.log_metrics({
                f"test_{k}": v for k, v in eval_metrics.items()
            })
            tracker.log_params({
                "architecture": result["architecture"],
                "epochs_trained": len(result["history"]),
                "best_epoch": result["best_epoch"],
                "early_stopping_enabled": str(self.early_stopping_enabled),
                "early_stopping_patience": self.es_patience,
            })

        self.dl_model = dl_model
        self.model = result["model"]
        self.training_history = result["history"]

        return {
            "dl_model": dl_model,
            "network": result["model"],
            "evaluation": eval_metrics,
            "history": result["history"],
            "architecture": result["architecture"],
            "best_epoch": result["best_epoch"],
        }

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics.

        Args:
            y_true: Ground truth binary labels.
            y_proba: Predicted probabilities.
            threshold: Classification threshold.

        Returns:
            Dictionary with auc, precision, recall, f1, accuracy, log_loss.
        """
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_true, y_proba)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "log_loss": float(log_loss(y_true, y_proba)),
        }

        return metrics

    def save_model(self, path: str) -> None:
        """Save the trained DL model to disk.

        Args:
            path: File path for saving.
        """
        if self.dl_model is not None:
            self.dl_model.save(path)

    def load_model(self, path: str) -> DLChurnModel:
        """Load a DL model from disk.

        Args:
            path: File path to load from.

        Returns:
            Loaded DLChurnModel instance.
        """
        self.dl_model = DLChurnModel.load(path)
        self.model = self.dl_model.model
        return self.dl_model
