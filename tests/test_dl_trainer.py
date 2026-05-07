"""
TDD Tests for DL Training and Evaluation Pipeline.

Tests cover:
- DLTrainer instantiation and configuration
- Early stopping mechanism (patience, min_delta, restore best weights)
- Single architecture training (LSTM and Transformer)
- Architecture selection (model comparison)
- MLflow integration during training
- Evaluation metrics computation (AUC, precision, recall, F1)
- Training history tracking
- Model save/load through trainer
- Reproducibility with same seed
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

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
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def small_config(config):
    """Config with reduced epochs for fast testing."""
    cfg = config.copy()
    cfg["dl_model"] = cfg.get("dl_model", {}).copy()
    cfg["dl_model"]["epochs"] = 5
    cfg["dl_model"]["batch_size"] = 32
    cfg["dl_model"]["early_stopping"] = {
        "enabled": True,
        "patience": 3,
        "min_delta": 0.001,
        "monitor": "val_loss",
        "restore_best_weights": True,
    }
    return cfg


@pytest.fixture
def sample_data():
    """Create synthetic data with learnable signal for DL training tests."""
    np.random.seed(42)
    n = 500
    n_features = 20

    X = np.random.randn(n, n_features).astype(np.float32)
    signal = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.5 * X[:, 2]
        - 0.4 * X[:, 3]
        + np.random.randn(n) * 0.3
    )
    prob = 1 / (1 + np.exp(-signal))
    y = (prob > 0.5).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    # Split into train/test (80/20)
    split_idx = int(n * 0.8)
    X_train = df.iloc[:split_idx].reset_index(drop=True)
    X_test = df.iloc[split_idx:].reset_index(drop=True)
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Early Stopping Tests
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    """Test early stopping mechanism."""

    def test_early_stopping_instantiation(self):
        """Early stopping must instantiate with default params."""
        from src.models.dl_trainer import EarlyStopping

        es = EarlyStopping()
        assert es is not None
        assert es.patience == 5
        assert es.should_stop is False

    def test_early_stopping_with_config(self, small_config):
        """Early stopping must use config parameters."""
        from src.models.dl_trainer import EarlyStopping

        es_cfg = small_config["dl_model"]["early_stopping"]
        es = EarlyStopping(
            patience=es_cfg["patience"],
            min_delta=es_cfg["min_delta"],
            monitor=es_cfg["monitor"],
            restore_best_weights=es_cfg["restore_best_weights"],
        )
        assert es.patience == 3
        assert es.min_delta == 0.001

    def test_early_stopping_tracks_improvement(self):
        """Early stopping must recognize improving scores."""
        from src.models.dl_trainer import EarlyStopping

        es = EarlyStopping(patience=3, mode="min")
        # Decreasing loss = improvement
        assert not es.step(1.0, epoch=0)
        assert not es.step(0.9, epoch=1)
        assert not es.step(0.8, epoch=2)
        assert es.counter == 0

    def test_early_stopping_triggers_after_patience(self):
        """Early stopping must trigger after patience epochs without improvement."""
        from src.models.dl_trainer import EarlyStopping

        es = EarlyStopping(patience=2, mode="min")
        es.step(0.5, epoch=0)  # best
        es.step(0.6, epoch=1)  # no improvement
        result = es.step(0.7, epoch=2)  # no improvement, patience=2 reached
        assert result is True
        assert es.should_stop is True

    def test_early_stopping_records_best_epoch(self):
        """Early stopping must record the best epoch."""
        from src.models.dl_trainer import EarlyStopping

        es = EarlyStopping(patience=5, mode="min")
        es.step(1.0, epoch=0)
        es.step(0.5, epoch=1)
        es.step(0.8, epoch=2)
        es.step(0.9, epoch=3)
        assert es.best_epoch == 1

    def test_early_stopping_restores_best_weights(self):
        """Early stopping must restore model to best state."""
        import torch.nn as nn
        from src.models.dl_trainer import EarlyStopping

        model = nn.Linear(10, 1)
        es = EarlyStopping(patience=2, mode="min", restore_best_weights=True)

        # Epoch 0: best
        es.step(0.5, model=model, epoch=0)
        best_weight = model.weight.data.clone()

        # Change weights and report worse loss
        with torch.no_grad():
            model.weight.data.fill_(999.0)
        es.step(0.8, model=model, epoch=1)
        es.step(0.9, model=model, epoch=2)  # triggers stop

        # Restore
        es.restore(model)
        assert torch.allclose(model.weight.data, best_weight)

    def test_early_stopping_history(self):
        """Early stopping must track score history."""
        from src.models.dl_trainer import EarlyStopping

        es = EarlyStopping(patience=5, mode="min")
        scores = [1.0, 0.9, 0.8, 0.85, 0.82]
        for i, s in enumerate(scores):
            es.step(s, epoch=i)
        assert es.history == scores


# Need torch for the restore test
import torch


# ---------------------------------------------------------------------------
# DLTrainer Tests
# ---------------------------------------------------------------------------

class TestDLTrainerInstantiation:
    """Test DLTrainer instantiation and configuration."""

    def test_trainer_instantiation(self, small_config):
        """DLTrainer must instantiate from config."""
        from src.models.dl_trainer import DLTrainer

        trainer = DLTrainer(small_config)
        assert trainer is not None

    def test_trainer_reads_config(self, small_config):
        """DLTrainer must read configuration parameters."""
        from src.models.dl_trainer import DLTrainer

        trainer = DLTrainer(small_config)
        assert trainer.epochs == 5
        assert trainer.early_stopping_enabled is True
        assert trainer.es_patience == 3
        assert trainer.architecture in ("lstm", "transformer")

    def test_trainer_has_train_method(self, small_config):
        """DLTrainer must have train_and_evaluate method."""
        from src.models.dl_trainer import DLTrainer

        trainer = DLTrainer(small_config)
        assert hasattr(trainer, "train_and_evaluate")
        assert callable(trainer.train_and_evaluate)

    def test_trainer_has_select_architecture_method(self, small_config):
        """DLTrainer must have select_best_architecture method."""
        from src.models.dl_trainer import DLTrainer

        trainer = DLTrainer(small_config)
        assert hasattr(trainer, "select_best_architecture")
        assert callable(trainer.select_best_architecture)


class TestDLTrainerTraining:
    """Test DLTrainer training functionality."""

    def test_train_single_architecture_lstm(self, small_config, sample_data):
        """Must train LSTM architecture without error."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.train_single_architecture(
            X_train, y_train, architecture="lstm"
        )

        assert "model" in result
        assert result["architecture"] == "lstm"
        assert len(result["history"]) > 0

    def test_train_single_architecture_transformer(self, small_config, sample_data):
        """Must train Transformer architecture without error."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.train_single_architecture(
            X_train, y_train, architecture="transformer"
        )

        assert "model" in result
        assert result["architecture"] == "transformer"
        assert len(result["history"]) > 0

    def test_prepare_sequence_inputs_accepts_real_sequence_payload(self, small_config):
        """Trainer should accept actual sequence payloads for DL training."""
        from src.models.dl_trainer import DLTrainer
        from src.models.sequence_utils import prepare_sequence_training_data

        np.random.seed(7)
        rows = []
        for cid in ["C1", "C2", "C3", "C4"]:
            for month in [1, 2, 3]:
                rows.append([cid, month, np.random.randn(), np.random.randn()])
        seq_df = pd.DataFrame(rows, columns=["customer_id", "month", "f1", "f2"])
        labels = pd.DataFrame(
            {"customer_id": ["C1", "C2", "C3", "C4"], "churn_label": [0, 1, 0, 1]}
        )

        trainer = DLTrainer(small_config)
        payload = prepare_sequence_training_data(
            seq_df, labels=labels, window_size=trainer.sequence_window
        )
        prepared = trainer.prepare_sequence_inputs(sequence_data=payload)
        assert prepared["sequence_source"] == "event_sequence"
        assert prepared["sequences"].ndim == 3

    def test_train_single_architecture_with_sequence_payload(self, small_config):
        """Training path should support prebuilt sequence tensors."""
        from src.models.dl_trainer import DLTrainer

        small_config["dl_model"]["epochs"] = 2
        trainer = DLTrainer(small_config)
        np.random.seed(11)
        n = 24
        seq = np.random.randn(n, trainer.sequence_window, 3).astype(np.float32)
        labels = np.random.randint(0, 2, size=n).astype(np.float32)
        payload = {"sequences": seq, "labels": labels}
        dummy_X = pd.DataFrame(np.random.randn(n, 3), columns=["a", "b", "c"])

        result = trainer.train_single_architecture(
            dummy_X,
            labels,
            architecture="lstm",
            sequence_train_data=payload,
        )
        assert result["model"] is not None
        assert len(result["history"]) > 0

    def test_train_returns_history(self, small_config, sample_data):
        """Training must return epoch-by-epoch history."""
        from src.models.dl_trainer import DLTrainer

        X_train, _, y_train, _ = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.train_single_architecture(X_train, y_train)

        history = result["history"]
        assert len(history) > 0
        assert "train_loss" in history[0]
        assert "val_loss" in history[0]
        assert "val_auc" in history[0]

    def test_early_stopping_during_training(self, small_config, sample_data):
        """Training must stop early when no improvement."""
        from src.models.dl_trainer import DLTrainer

        X_train, _, y_train, _ = sample_data
        small_config["dl_model"]["epochs"] = 100  # High max epochs
        small_config["dl_model"]["early_stopping"]["patience"] = 2

        trainer = DLTrainer(small_config)
        result = trainer.train_single_architecture(X_train, y_train)

        # Should stop before 100 epochs
        assert len(result["history"]) < 100


class TestArchitectureSelection:
    """Test model selection between LSTM and Transformer."""

    def test_select_best_architecture(self, small_config, sample_data):
        """Must compare and select best architecture."""
        from src.models.dl_trainer import DLTrainer

        X_train, _, y_train, _ = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.select_best_architecture(X_train, y_train)

        assert result["architecture"] in ("lstm", "transformer")
        assert "comparison" in result
        assert "lstm" in result["comparison"]
        assert "transformer" in result["comparison"]
        assert trainer.best_architecture is not None

    def test_architecture_comparison_has_metrics(self, small_config, sample_data):
        """Architecture comparison must include val_loss for each."""
        from src.models.dl_trainer import DLTrainer

        X_train, _, y_train, _ = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.select_best_architecture(X_train, y_train)

        for arch in ("lstm", "transformer"):
            assert "val_loss" in result["comparison"][arch]
            assert "best_epoch" in result["comparison"][arch]


class TestDLTrainerEvaluation:
    """Test full training and evaluation pipeline."""

    def test_train_and_evaluate_returns_metrics(self, small_config, sample_data):
        """train_and_evaluate must return evaluation metrics."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 5

        trainer = DLTrainer(small_config)
        result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

        eval_metrics = result["evaluation"]
        assert "auc" in eval_metrics
        assert "precision" in eval_metrics
        assert "recall" in eval_metrics
        assert "f1" in eval_metrics
        assert "accuracy" in eval_metrics
        assert "log_loss" in eval_metrics

    def test_train_and_evaluate_returns_dl_model(self, small_config, sample_data):
        """train_and_evaluate must return a usable DLChurnModel."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

        dl_model = result["dl_model"]
        probs = dl_model.predict_proba(X_test)
        assert len(probs) == len(X_test)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_train_and_evaluate_with_architecture_selection(
        self, small_config, sample_data
    ):
        """train_and_evaluate with select_architecture must work."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.train_and_evaluate(
            X_train, y_train, X_test, y_test,
            select_architecture=True,
        )

        assert result["architecture"] in ("lstm", "transformer")
        assert result["evaluation"]["auc"] > 0.0

    def test_evaluation_auc_reasonable(self, small_config, sample_data):
        """DL model should achieve reasonable AUC on learnable data."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 10

        trainer = DLTrainer(small_config)
        result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

        assert result["evaluation"]["auc"] >= 0.60, (
            f"AUC {result['evaluation']['auc']:.4f} too low"
        )


class TestDLTrainerMLflowIntegration:
    """Test MLflow integration in the trainer."""

    def test_train_with_mlflow_tracker(
        self, small_config, sample_data, tmp_path, monkeypatch
    ):
        """Training must integrate with MLflowTracker."""
        from src.models.dl_trainer import DLTrainer
        from src.models.mlflow_tracking import MLflowTracker

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 3
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        # Setup MLflow with temp dir
        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        small_config["mlflow"] = {
            "tracking_uri": tracking_uri,
            "artifact_location": str(tmp_path / "artifacts"),
            "experiment_name": "dl_trainer_test",
        }

        tracker = MLflowTracker(small_config)
        assert tracker.tracking_uri == tracking_uri
        tracker.create_experiment(name="dl_trainer_test")
        tracker.start_run(run_name="dl_test_run")

        trainer = DLTrainer(small_config)
        result = trainer.train_and_evaluate(
            X_train, y_train, X_test, y_test, tracker=tracker
        )

        tracker.end_run()

        # Verify metrics were logged
        assert result["evaluation"]["auc"] > 0.0

    def test_architecture_selection_with_mlflow(
        self, small_config, sample_data, tmp_path, monkeypatch
    ):
        """Architecture selection must log comparison to MLflow."""
        from src.models.dl_trainer import DLTrainer
        from src.models.mlflow_tracking import MLflowTracker

        X_train, _, y_train, _ = sample_data
        small_config["dl_model"]["epochs"] = 2
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
        small_config["mlflow"] = {
            "tracking_uri": tracking_uri,
            "artifact_location": str(tmp_path / "artifacts"),
            "experiment_name": "arch_selection_test",
        }

        tracker = MLflowTracker(small_config)
        assert tracker.tracking_uri == tracking_uri
        tracker.create_experiment(name="arch_selection_test")
        tracker.start_run(run_name="arch_select_run")

        trainer = DLTrainer(small_config)
        result = trainer.select_best_architecture(
            X_train, y_train, tracker=tracker
        )

        tracker.end_run()

        assert result["architecture"] in ("lstm", "transformer")


class TestDLTrainerSaveLoad:
    """Test model save/load through the trainer."""

    def test_save_and_load_model(self, small_config, sample_data, tmp_path):
        """Trainer must save and reload model with identical predictions."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 3

        trainer = DLTrainer(small_config)
        result = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

        # Save
        model_path = str(tmp_path / "dl_model.pt")
        trainer.save_model(model_path)
        assert os.path.exists(model_path)

        # Load
        loaded_model = trainer.load_model(model_path)
        probs_original = result["dl_model"].predict_proba(X_test)
        probs_loaded = loaded_model.predict_proba(X_test)

        np.testing.assert_array_almost_equal(
            probs_original, probs_loaded, decimal=5
        )


class TestDLTrainerReproducibility:
    """Test reproducibility of DL training."""

    def test_same_seed_same_results(self, small_config, sample_data):
        """Same seed must produce identical training results."""
        from src.models.dl_trainer import DLTrainer

        X_train, X_test, y_train, y_test = sample_data
        small_config["dl_model"]["epochs"] = 3
        # Disable early stopping for deterministic epoch count
        small_config["dl_model"]["early_stopping"]["enabled"] = False

        trainer1 = DLTrainer(small_config)
        result1 = trainer1.train_and_evaluate(X_train, y_train, X_test, y_test)

        trainer2 = DLTrainer(small_config)
        result2 = trainer2.train_and_evaluate(X_train, y_train, X_test, y_test)

        probs1 = result1["dl_model"].predict_proba(X_test)
        probs2 = result2["dl_model"].predict_proba(X_test)

        np.testing.assert_array_almost_equal(probs1, probs2, decimal=4)
