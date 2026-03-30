"""
ML/DL Churn Prediction Models.

Provides:
- time_based_split: Time-based train/test split (10 months train, 2 months test)
- MLChurnModel: Gradient boosting churn classifier with XGBoost and LightGBM,
  5-Fold cross-validation, hyperparameter tuning, and automatic model selection
- DLChurnModel: Transformer/LSTM-based PyTorch churn classifier with configurable
  sequence window, hidden dimensions, and number of layers
- EnsembleChurnModel: Weighted average ensemble (ML 0.6 + DL 0.4)

All configurable parameters are read from the YAML config dictionary.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: time-based train/test split
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    train_months: int = 10,
    test_months: int = 2,
    date_column: str = "reference_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split data by time into train and test sets.

    Uses the first ``train_months`` fraction of the time range for training
    and the remaining ``test_months`` fraction for testing, ensuring no
    temporal data leakage.

    Args:
        df: DataFrame containing features, labels, and a date column.
        train_months: Number of months for training.
        test_months: Number of months for testing.
        date_column: Name of the datetime column used for splitting.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    total_months = train_months + test_months
    train_ratio = train_months / total_months

    # Split by position based on time ordering
    split_idx = int(len(df) * train_ratio)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Separate features and labels
    label_col = "churn_label"
    drop_cols = ["customer_id", label_col, date_column]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_train = train_df[feature_cols + [date_column]].copy()
    X_test = test_df[feature_cols + [date_column]].copy()
    y_train = train_df[label_col].values.astype(int)
    y_test = test_df[label_col].values.astype(int)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# ML Model: XGBoost + LightGBM with CV, tuning, and model selection
# ---------------------------------------------------------------------------

class MLChurnModel:
    """Gradient boosting churn prediction model with XGBoost and LightGBM.

    Trains both XGBoost and LightGBM models with 5-Fold stratified
    cross-validation, performs hyperparameter tuning, and selects the
    best model based on mean CV AUC.

    Args:
        config: Configuration dictionary with keys from simulator_config.yaml.
    """

    # Default hyperparameter search spaces
    LGBM_PARAM_GRID = [
        {"num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9,
         "bagging_fraction": 0.8, "min_child_samples": 20, "num_boost_round": 200},
        {"num_leaves": 63, "learning_rate": 0.03, "feature_fraction": 0.8,
         "bagging_fraction": 0.7, "min_child_samples": 30, "num_boost_round": 300},
        {"num_leaves": 15, "learning_rate": 0.1, "feature_fraction": 0.95,
         "bagging_fraction": 0.9, "min_child_samples": 10, "num_boost_round": 150},
    ]

    XGB_PARAM_GRID = [
        {"max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
         "colsample_bytree": 0.9, "min_child_weight": 5, "n_estimators": 200},
        {"max_depth": 8, "learning_rate": 0.03, "subsample": 0.7,
         "colsample_bytree": 0.8, "min_child_weight": 10, "n_estimators": 300},
        {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.9,
         "colsample_bytree": 0.95, "min_child_weight": 3, "n_estimators": 150},
    ]

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ML model with config parameters."""
        self.config = config
        self.seed = config.get("simulation", {}).get("random_seed", 42)
        self.n_folds = config.get("ml_model", {}).get("n_folds", 5)
        self.model = None
        self.model_type: Optional[str] = None  # "xgboost" or "lightgbm"
        self.best_params: Optional[Dict[str, Any]] = None
        self.cv_scores: Optional[Dict[str, Any]] = None
        self.feature_names_: Optional[List[str]] = None
        self._xgb_model = None  # stored for feature importance
        self._lgb_model = None

    def _cv_score_lightgbm(
        self, X: np.ndarray, y: np.ndarray,
        feature_names: List[str], params_entry: Dict[str, Any],
    ) -> float:
        """Compute 5-Fold CV AUC for a LightGBM parameter set.

        Args:
            X: Feature matrix.
            y: Labels.
            feature_names: Feature column names.
            params_entry: Hyperparameter dict (includes num_boost_round).

        Returns:
            Mean AUC across folds.
        """
        import lightgbm as lgb

        num_boost_round = params_entry.pop("num_boost_round", 200)
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
            "bagging_freq": 5,
            **params_entry,
        }

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        fold_aucs = []
        for train_idx, val_idx in skf.split(X, y):
            dtrain = lgb.Dataset(
                X[train_idx], label=y[train_idx],
                feature_name=feature_names,
            )
            dval = lgb.Dataset(
                X[val_idx], label=y[val_idx],
                feature_name=feature_names, reference=dtrain,
            )

            model = lgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval],
                callbacks=[
                    lgb.log_evaluation(period=0),
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                ],
            )
            preds = model.predict(X[val_idx])

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y[val_idx], preds)
            fold_aucs.append(auc)

        # Restore num_boost_round for later use
        params_entry["num_boost_round"] = num_boost_round
        return float(np.mean(fold_aucs))

    def _cv_score_xgboost(
        self, X: np.ndarray, y: np.ndarray,
        params_entry: Dict[str, Any],
    ) -> float:
        """Compute 5-Fold CV AUC for an XGBoost parameter set.

        Args:
            X: Feature matrix.
            y: Labels.
            params_entry: Hyperparameter dict (includes n_estimators).

        Returns:
            Mean AUC across folds.
        """
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score

        n_estimators = params_entry.pop("n_estimators", 200)

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )

        fold_aucs = []
        for train_idx, val_idx in skf.split(X, y):
            model = XGBClassifier(
                n_estimators=n_estimators,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=self.seed,
                verbosity=0,
                early_stopping_rounds=20,
                **params_entry,
            )
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
            preds = model.predict_proba(X[val_idx])[:, 1]
            auc = roc_auc_score(y[val_idx], preds)
            fold_aucs.append(auc)

        # Restore n_estimators
        params_entry["n_estimators"] = n_estimators
        return float(np.mean(fold_aucs))

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        tracker: Optional[Any] = None,
    ) -> "MLChurnModel":
        """Train using hyperparameter tuning + model selection.

        Evaluates multiple hyperparameter configurations for both XGBoost
        and LightGBM using 5-Fold stratified cross-validation, then
        retrains the best configuration on the full training data.

        When a ``tracker`` (MLflowTracker) is provided, automatically logs
        hyperparameter tuning progress, CV scores, selected model type,
        and best parameters to the active MLflow run.

        Args:
            X: Feature DataFrame.
            y: Binary labels array.
            tracker: Optional MLflowTracker with an active run for logging.

        Returns:
            self
        """
        self.feature_names_ = list(X.columns)
        X_arr = X.values.astype(np.float64)

        # --- Hyperparameter tuning for LightGBM ---
        best_lgb_auc = -1.0
        best_lgb_params = None
        for idx, params in enumerate(self.LGBM_PARAM_GRID):
            p = dict(params)  # copy to avoid mutation
            auc = self._cv_score_lightgbm(X_arr, y, self.feature_names_, p)
            logger.info(f"LightGBM params={p} -> CV AUC={auc:.4f}")
            if tracker is not None:
                tracker.log_metrics(
                    {f"lgb_cv_auc_config_{idx}": auc}, step=idx
                )
            if auc > best_lgb_auc:
                best_lgb_auc = auc
                best_lgb_params = dict(p)

        # --- Hyperparameter tuning for XGBoost ---
        best_xgb_auc = -1.0
        best_xgb_params = None
        for idx, params in enumerate(self.XGB_PARAM_GRID):
            p = dict(params)
            auc = self._cv_score_xgboost(X_arr, y, p)
            logger.info(f"XGBoost params={p} -> CV AUC={auc:.4f}")
            if tracker is not None:
                tracker.log_metrics(
                    {f"xgb_cv_auc_config_{idx}": auc}, step=idx
                )
            if auc > best_xgb_auc:
                best_xgb_auc = auc
                best_xgb_params = dict(p)

        # Store CV results
        self.cv_scores = {
            "lightgbm_best_cv_auc": best_lgb_auc,
            "lightgbm_best_params": best_lgb_params,
            "xgboost_best_cv_auc": best_xgb_auc,
            "xgboost_best_params": best_xgb_params,
        }

        # --- Model selection: pick the best ---
        if best_lgb_auc >= best_xgb_auc:
            self.model_type = "lightgbm"
            self.best_params = best_lgb_params
            self._train_lightgbm_final(X_arr, y, best_lgb_params)
        else:
            self.model_type = "xgboost"
            self.best_params = best_xgb_params
            self._train_xgboost_final(X_arr, y, best_xgb_params)

        # Auto-log to tracker if provided
        if tracker is not None:
            tracker.log_metrics({
                "lightgbm_best_cv_auc": best_lgb_auc,
                "xgboost_best_cv_auc": best_xgb_auc,
            })
            tracker.log_params({
                "selected_model_type": self.model_type,
                "n_folds": self.n_folds,
            })
            if self.best_params:
                for k, v in self.best_params.items():
                    tracker.log_params({f"best_{k}": v})

        logger.info(
            f"Selected model: {self.model_type} "
            f"(LGB CV={best_lgb_auc:.4f}, XGB CV={best_xgb_auc:.4f})"
        )
        return self

    @staticmethod
    def _compute_scale_pos_weight(y: np.ndarray) -> float:
        """Compute scale_pos_weight for imbalanced binary classification.

        Calculates the ratio of negative to positive samples, which is used
        by XGBoost's ``scale_pos_weight`` parameter to handle class imbalance.

        Args:
            y: Binary label array (0 = negative, 1 = positive).

        Returns:
            Ratio of negatives to positives. Returns 1.0 if either class
            has zero samples to avoid division by zero.
        """
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        if n_pos == 0 or n_neg == 0:
            return 1.0
        ratio = n_neg / n_pos
        logger.info(
            "Class imbalance: n_neg=%d, n_pos=%d, scale_pos_weight=%.4f",
            n_neg, n_pos, ratio,
        )
        return float(ratio)

    def _train_lightgbm_final(
        self, X: np.ndarray, y: np.ndarray,
        params_entry: Dict[str, Any],
    ) -> None:
        """Retrain the best LightGBM config on full training data.

        Applies ``is_unbalance=True`` automatically when the positive class
        rate is below 30% to improve recall on the minority (churn) class.

        Args:
            X: Full training feature matrix.
            y: Full training labels.
            params_entry: Best hyperparameters from CV.
        """
        import lightgbm as lgb

        num_boost_round = params_entry.get("num_boost_round", 200)
        p = {k: v for k, v in params_entry.items() if k != "num_boost_round"}

        # Handle class imbalance: use is_unbalance when positive rate < 30%
        pos_rate = float(np.mean(y == 1))
        is_unbalance = pos_rate < 0.30
        logger.info(
            "LightGBM final train: pos_rate=%.4f, is_unbalance=%s",
            pos_rate, is_unbalance,
        )

        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": self.seed,
            "deterministic": True,
            "bagging_freq": 5,
            "is_unbalance": is_unbalance,
            **p,
        }

        dtrain = lgb.Dataset(
            X, label=y, feature_name=self.feature_names_
        )
        self._lgb_model = lgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        self.model = self._lgb_model

    def _train_xgboost_final(
        self, X: np.ndarray, y: np.ndarray,
        params_entry: Dict[str, Any],
    ) -> None:
        """Retrain the best XGBoost config on full training data.

        Automatically sets ``scale_pos_weight`` based on the negative-to-positive
        ratio in ``y`` to handle class imbalance.

        Args:
            X: Full training feature matrix.
            y: Full training labels.
            params_entry: Best hyperparameters from CV.
        """
        from xgboost import XGBClassifier

        n_estimators = params_entry.get("n_estimators", 200)
        p = {k: v for k, v in params_entry.items() if k != "n_estimators"}

        scale_pos_weight = self._compute_scale_pos_weight(y)

        self._xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=self.seed,
            verbosity=0,
            scale_pos_weight=scale_pos_weight,
            **p,
        )
        self._xgb_model.fit(X, y, verbose=False)
        self.model = self._xgb_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return churn probability predictions.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of churn probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_arr = X.values
        if self.model_type == "lightgbm":
            return self.model.predict(X_arr)
        else:
            # XGBoost returns 2-column proba
            return self.model.predict_proba(X_arr)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data and return metrics dict.

        Args:
            X: Feature DataFrame.
            y: True binary labels.

        Returns:
            Dictionary with auc_roc, accuracy, precision, recall, f1.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score,
        )
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        y_arr = np.asarray(y)
        return {
            "auc_roc": float(roc_auc_score(y_arr, proba)),
            "accuracy": float(accuracy_score(y_arr, preds)),
            "precision": float(precision_score(y_arr, preds, zero_division=0)),
            "recall": float(recall_score(y_arr, preds, zero_division=0)),
            "f1": float(f1_score(y_arr, preds, zero_division=0)),
        }

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance scores.

        Returns:
            Array of importance scores, one per feature.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type == "lightgbm":
            return np.array(
                self.model.feature_importance(importance_type="gain")
            )
        else:
            return self._xgb_model.feature_importances_

    def get_cv_results(self) -> Optional[Dict[str, Any]]:
        """Return cross-validation results from model selection.

        Returns:
            Dictionary with CV AUC scores and best params for both models,
            or None if fit() hasn't been called.
        """
        return self.cv_scores

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path (without extension). Saves as .joblib.
        """
        save_path = path if path.endswith(".joblib") else f"{path}.joblib"
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "best_params": self.best_params,
            "cv_scores": self.cv_scores,
            "feature_names": self.feature_names_,
            "config": self.config,
            "seed": self.seed,
        }, save_path)

    @classmethod
    def load(cls, path: str) -> "MLChurnModel":
        """Load model from disk.

        Args:
            path: File path. Tries path as-is, then with .joblib extension.

        Returns:
            Loaded MLChurnModel instance.
        """
        load_path = path
        if not os.path.exists(load_path):
            load_path = f"{path}.joblib"
        data = joblib.load(load_path)
        instance = cls(data["config"])
        instance.model = data["model"]
        instance.model_type = data.get("model_type", "lightgbm")
        instance.best_params = data.get("best_params")
        instance.cv_scores = data.get("cv_scores")
        instance.feature_names_ = data["feature_names"]
        instance.seed = data["seed"]
        # Restore typed model references
        if instance.model_type == "xgboost":
            instance._xgb_model = instance.model
        else:
            instance._lgb_model = instance.model
        return instance


# ---------------------------------------------------------------------------
# Utility: threshold analysis
# ---------------------------------------------------------------------------

def analyze_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Analyze Precision, Recall, and F1 across classification thresholds.

    Computes the Precision-Recall curve and evaluates Precision, Recall, and
    F1 score at a range of thresholds. Selects the optimal threshold for a
    churn-prediction business objective: maximising F1 score while keeping
    Recall >= 0.60 (i.e. capturing the majority of actual churners).

    The function also logs a summary table of key threshold values to the
    module logger at INFO level.

    Args:
        y_true: Binary ground-truth labels (0 = no churn, 1 = churn).
        y_prob: Predicted churn probabilities in [0, 1].

    Returns:
        Dictionary with the following keys:

        - ``"thresholds"`` (List[float]): Evaluated threshold values.
        - ``"precision"`` (List[float]): Precision at each threshold.
        - ``"recall"`` (List[float]): Recall at each threshold.
        - ``"f1"`` (List[float]): F1 score at each threshold.
        - ``"optimal_threshold"`` (float): Selected threshold.
        - ``"optimal_precision"`` (float): Precision at optimal threshold.
        - ``"optimal_recall"`` (float): Recall at optimal threshold.
        - ``"optimal_f1"`` (float): F1 at optimal threshold.

    Example:
        >>> result = analyze_threshold(y_test, model.predict_proba(X_test))
        >>> print(result["optimal_threshold"])
    """
    from sklearn.metrics import precision_recall_curve

    y_true_arr = np.asarray(y_true).ravel()
    y_prob_arr = np.asarray(y_prob).ravel()

    # precision_recall_curve returns arrays indexed by threshold;
    # the last entry has no corresponding threshold (appended sentinel).
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(
        y_true_arr, y_prob_arr
    )

    # Drop the sentinel entry so lengths match
    precision_arr = precision_arr[:-1]
    recall_arr = recall_arr[:-1]

    # Compute F1 at every threshold, guarding against zero denominators.
    # Use a safe denominator (minimum 1e-9) to avoid RuntimeWarning from
    # division; the np.where mask ensures zero F1 is returned when denom == 0.
    denom = precision_arr + recall_arr
    safe_denom = np.where(denom > 0, denom, 1e-9)
    f1_arr = np.where(
        denom > 0,
        2.0 * precision_arr * recall_arr / safe_denom,
        0.0,
    )

    thresholds_list = thresholds_arr.tolist()
    precision_list = precision_arr.tolist()
    recall_list = recall_arr.tolist()
    f1_list = f1_arr.tolist()

    # Business rule: maximise F1 among thresholds where Recall >= 0.60.
    # If no threshold satisfies the recall floor, fall back to global F1 max.
    MIN_RECALL = 0.60
    eligible_mask = recall_arr >= MIN_RECALL
    if eligible_mask.any():
        candidate_indices = np.where(eligible_mask)[0]
        best_idx = candidate_indices[np.argmax(f1_arr[eligible_mask])]
    else:
        best_idx = int(np.argmax(f1_arr))

    optimal_threshold = float(thresholds_arr[best_idx])
    optimal_precision = float(precision_arr[best_idx])
    optimal_recall = float(recall_arr[best_idx])
    optimal_f1 = float(f1_arr[best_idx])

    logger.info(
        "Threshold analysis: optimal=%.4f  P=%.4f  R=%.4f  F1=%.4f",
        optimal_threshold, optimal_precision, optimal_recall, optimal_f1,
    )

    # Log a summary table at key percentiles
    for pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
        idx = int(pct * (len(thresholds_list) - 1))
        logger.info(
            "  thr=%.3f  P=%.3f  R=%.3f  F1=%.3f",
            thresholds_list[idx],
            precision_list[idx],
            recall_list[idx],
            f1_list[idx],
        )

    return {
        "thresholds": thresholds_list,
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "optimal_threshold": optimal_threshold,
        "optimal_precision": optimal_precision,
        "optimal_recall": optimal_recall,
        "optimal_f1": optimal_f1,
    }


# ---------------------------------------------------------------------------
# LSTM Network Architecture
# ---------------------------------------------------------------------------

class LSTMChurnNetwork(nn.Module):
    """LSTM-based neural network for churn prediction.

    Processes sequential feature input through stacked LSTM layers,
    then uses a fully-connected head for binary classification.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate between LSTM layers and in the FC head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the LSTM network."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch,).
        """
        lstm_out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.fc(last_hidden).squeeze(-1)  # (batch,)
        return logits


# ---------------------------------------------------------------------------
# Transformer Network Architecture
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer input.

    Args:
        d_model: Embedding / model dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initialize positional encoding."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Positionally-encoded tensor of the same shape.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerChurnNetwork(nn.Module):
    """Transformer-based neural network for churn prediction.

    Projects raw features into d_model space, applies positional encoding,
    then processes through a stack of Transformer encoder layers.

    Args:
        input_size: Number of raw input features per timestep.
        d_model: Internal model dimension (must be divisible by nhead).
        nhead: Number of attention heads.
        num_encoder_layers: Number of stacked TransformerEncoder layers.
        dim_feedforward: Hidden dimension of the feed-forward sub-layers.
        dropout: Dropout rate throughout the network.
        sequence_window: Maximum sequence length (for positional encoding).
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
        sequence_window: int = 6,
    ) -> None:
        """Initialize the Transformer churn network."""
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max(sequence_window, 512),
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Logits tensor of shape (batch,).
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        logits = self.fc(x).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# DL Model: Transformer/LSTM-based churn classifier
# ---------------------------------------------------------------------------

class DLChurnModel:
    """Deep learning churn prediction model (Transformer or LSTM).

    Supports configurable architecture selection, sequence window, attention
    heads, encoder layers, and hidden dimensions via the config dictionary.
    Runs on CPU only.

    Args:
        config: Configuration dictionary with dl_model section.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize DL model from config."""
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

        self.device = torch.device("cpu")
        self.model: Optional[nn.Module] = None
        self.network: Optional[nn.Module] = None  # alias for tests
        self.input_size_: Optional[int] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_network(self, input_size: int) -> nn.Module:
        """Build the neural network based on configured architecture.

        Args:
            input_size: Number of raw input features.

        Returns:
            A PyTorch Module (TransformerChurnNetwork or LSTMChurnNetwork).
        """
        if self.architecture == "transformer":
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
        self, X: pd.DataFrame, fit: bool = False
    ) -> torch.Tensor:
        """Convert DataFrame features to a 3D tensor for the network.

        Since the tabular features don't have an inherent sequence dimension,
        we tile each sample's features across sequence_window timesteps with
        temporal scaling to provide a sequential signal.

        Args:
            X: Feature DataFrame of shape (n_samples, n_features).
            fit: If True, compute and store normalization statistics.

        Returns:
            Tensor of shape (n_samples, sequence_window, n_features).
        """
        values = X.values.astype(np.float32)

        # Normalize features
        if fit:
            self._feature_mean = values.mean(axis=0)
            self._feature_std = values.std(axis=0) + 1e-8
        if self._feature_mean is not None and self._feature_std is not None:
            values = (values - self._feature_mean) / self._feature_std

        n_samples, n_features = values.shape

        # Create pseudo-sequence: tile features across sequence_window
        sequences = np.tile(
            values[:, np.newaxis, :], (1, self.sequence_window, 1)
        )

        # Add slight temporal variation to give the network temporal signal
        for t in range(self.sequence_window):
            scale = (t + 1) / self.sequence_window
            sequences[:, t, :] *= scale

        return torch.tensor(sequences, dtype=torch.float32)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        tracker: Optional[Any] = None,
    ) -> "DLChurnModel":
        """Train the deep learning model.

        When a ``tracker`` (MLflowTracker) is provided, automatically logs
        per-epoch training loss and architecture parameters to the active
        MLflow run.

        Args:
            X: Feature DataFrame.
            y: Binary labels array.
            tracker: Optional MLflowTracker with an active run for logging.

        Returns:
            self
        """
        self._set_seed()

        self.input_size_ = X.shape[1]
        X_tensor = self._prepare_tensor(X, fit=True)
        y_tensor = torch.tensor(y.astype(np.float32), dtype=torch.float32)

        # Initialize model (Transformer or LSTM based on config)
        self.model = self._build_network(self.input_size_).to(self.device)
        self.network = self.model  # alias

        # Log DL architecture params
        if tracker is not None:
            tracker.log_params({
                "dl_architecture": self.architecture,
                "dl_sequence_window": self.sequence_window,
                "dl_hidden_size": self.hidden_size,
                "dl_num_layers": self.num_layers,
                "dl_dropout": self.dropout,
                "dl_learning_rate": self.learning_rate,
                "dl_batch_size": self.batch_size,
                "dl_epochs": self.epochs,
            })

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Training setup
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        self.training_history: List[Dict[str, float]] = []
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.training_history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
            })

            # Log per-epoch loss to tracker
            if tracker is not None:
                tracker.log_metrics(
                    {"dl_train_loss": avg_loss}, step=epoch
                )

        self.model.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return churn probability predictions.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of churn probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        self.model.eval()
        X_tensor = self._prepare_tensor(X, fit=False)

        with torch.no_grad():
            logits = self.model(X_tensor.to(self.device))
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model on test data and return metrics dict.

        Args:
            X: Feature DataFrame.
            y: True binary labels.

        Returns:
            Dictionary with auc_roc, accuracy, precision, recall, f1.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score,
        )
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        y_arr = np.asarray(y)
        return {
            "auc_roc": float(roc_auc_score(y_arr, proba)),
            "accuracy": float(accuracy_score(y_arr, preds)),
            "precision": float(precision_score(y_arr, preds, zero_division=0)),
            "recall": float(recall_score(y_arr, preds, zero_division=0)),
            "f1": float(f1_score(y_arr, preds, zero_division=0)),
        }

    def save(self, path: str) -> None:
        """Save model state to disk.

        Args:
            path: File path for the saved model (.pt).
        """
        save_path = path if path.endswith(".pt") else f"{path}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config,
            "input_size": self.input_size_,
            "architecture": self.architecture,
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            "seed": self.seed,
        }, save_path)

    @classmethod
    def load(cls, path: str) -> "DLChurnModel":
        """Load model from disk.

        Args:
            path: File path to the saved model.

        Returns:
            Loaded DLChurnModel instance.
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(data["config"])
        instance.input_size_ = data["input_size"]
        instance.architecture = data.get("architecture", instance.architecture)
        instance._feature_mean = data["feature_mean"]
        instance._feature_std = data["feature_std"]
        instance.seed = data["seed"]

        instance.model = instance._build_network(
            instance.input_size_
        ).to(instance.device)
        instance.model.load_state_dict(data["model_state_dict"])
        instance.model.eval()
        instance.network = instance.model
        return instance


# ---------------------------------------------------------------------------
# Ensemble Model: Weighted average of ML + DL
# ---------------------------------------------------------------------------

class EnsembleChurnModel:
    """Ensemble churn model combining ML and DL via weighted average.

    Default weights: ML 0.6, DL 0.4 (configurable via pipeline config).

    Args:
        config: Configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ensemble with config-driven weights."""
        self.config = config
        pipeline_cfg = config.get("pipeline", {})
        self.weight_ml = pipeline_cfg.get("ensemble_weight_ml", 0.6)
        self.weight_dl = pipeline_cfg.get("ensemble_weight_dl", 0.4)
        self.ml_model: Optional[MLChurnModel] = None
        self.dl_model: Optional[DLChurnModel] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        tracker: Optional[Any] = None,
    ) -> "EnsembleChurnModel":
        """Train both ML and DL sub-models.

        When a ``tracker`` (MLflowTracker) is provided, it is forwarded
        to both sub-models for automatic logging of their training
        parameters and per-step metrics.

        Args:
            X: Feature DataFrame.
            y: Binary labels array.
            tracker: Optional MLflowTracker with an active run.

        Returns:
            self
        """
        self.ml_model = MLChurnModel(self.config)
        self.dl_model = DLChurnModel(self.config)

        self.ml_model.fit(X, y, tracker=tracker)
        self.dl_model.fit(X, y, tracker=tracker)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted average churn probabilities.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of ensemble churn probabilities in [0, 1].
        """
        if self.ml_model is None or self.dl_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        ml_probs = self.ml_model.predict_proba(X)
        dl_probs = self.dl_model.predict_proba(X)

        return self.weight_ml * ml_probs + self.weight_dl * dl_probs

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble model on test data and return metrics dict.

        Args:
            X: Feature DataFrame.
            y: True binary labels.

        Returns:
            Dictionary with auc_roc, accuracy, precision, recall, f1.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_auc_score,
        )
        proba = self.predict_proba(X)
        preds = (proba >= 0.5).astype(int)
        y_arr = np.asarray(y)
        return {
            "auc_roc": float(roc_auc_score(y_arr, proba)),
            "accuracy": float(accuracy_score(y_arr, preds)),
            "precision": float(precision_score(y_arr, preds, zero_division=0)),
            "recall": float(recall_score(y_arr, preds, zero_division=0)),
            "f1": float(f1_score(y_arr, preds, zero_division=0)),
        }
