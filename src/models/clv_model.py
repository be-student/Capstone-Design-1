"""
Customer Lifetime Value (CLV) Prediction Module.

Implements a hybrid BG/NBD + Gamma-Gamma inspired CLV model that combines:
- Probabilistic RFM-based features (BG/NBD concepts for purchase frequency,
  Gamma-Gamma for monetary value estimation)
- Gradient Boosting regression for flexible CLV prediction from arbitrary features

Provides:
- fit / predict interface for CLV estimation
- Customer ranking by predicted CLV
- Churn-adjusted CLV computation
- CLV-proportional budget allocation
- Model persistence (save / load)
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Union, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CLVModel:
    """Customer Lifetime Value prediction model.

    Uses a Gradient Boosting regressor with BG/NBD + Gamma-Gamma inspired
    feature engineering to predict per-customer CLV in KRW.

    Parameters
    ----------
    config : dict
        Project configuration dictionary (loaded from YAML).
    """

    def __init__(self, config: dict):
        self.config = config
        self.seed = config.get("simulation", {}).get("random_seed", 42)

        # Core regression model
        self._model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=self.seed,
        )
        self._scaler = StandardScaler()
        self._is_fitted = False
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # BG/NBD + Gamma-Gamma inspired feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def _engineer_features(X: pd.DataFrame) -> pd.DataFrame:
        """Add probabilistic features inspired by BG/NBD and Gamma-Gamma.

        If recency, frequency, monetary columns are present, derives:
        - frequency_monetary_interaction (Gamma-Gamma concept)
        - recency_frequency_ratio (BG/NBD alive probability proxy)
        - log_monetary (monetary value log-transform)
        """
        X_out = X.copy()

        has_rfm = all(c in X_out.columns for c in ["recency", "frequency", "monetary"])

        if has_rfm:
            freq_safe = X_out["frequency"].clip(lower=1)
            # Gamma-Gamma: expected monetary value scales with avg transaction value
            X_out["freq_monetary_interaction"] = freq_safe * (
                X_out["monetary"] / freq_safe
            )
            # BG/NBD alive probability proxy: low recency + high frequency → alive
            X_out["recency_frequency_ratio"] = X_out["recency"] / freq_safe
            # Log monetary (common in Gamma-Gamma)
            X_out["log_monetary"] = np.log1p(X_out["monetary"])

        return X_out

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CLVModel":
        """Train the CLV model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (RFM + behavioural features).
        y : pd.Series or array-like
            Target CLV values (KRW, non-negative).
        """
        np.random.seed(self.seed)

        X_eng = self._engineer_features(X)
        self._feature_names = list(X_eng.columns)

        X_scaled = self._scaler.fit_transform(X_eng)
        y_arr = np.asarray(y, dtype=np.float64)

        self._model.fit(X_scaled, y_arr)
        self._is_fitted = True
        logger.info("CLV model fitted on %d samples with %d features",
                     len(y_arr), X_scaled.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CLV for each customer.

        Returns
        -------
        np.ndarray
            Non-negative CLV predictions (KRW).
        """
        if not self._is_fitted:
            raise RuntimeError("CLVModel has not been fitted yet.")

        X_eng = self._engineer_features(X)
        # Ensure same columns in same order
        for col in self._feature_names:
            if col not in X_eng.columns:
                X_eng[col] = 0.0
        X_eng = X_eng[self._feature_names]

        X_scaled = self._scaler.transform(X_eng)
        raw_preds = self._model.predict(X_scaled)

        # CLV must be non-negative
        predictions = np.clip(raw_preds, 0, None).astype(np.float64)
        return predictions

    # ------------------------------------------------------------------
    # Customer ranking
    # ------------------------------------------------------------------
    def rank_customers(
        self,
        customer_ids: Sequence[str],
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Rank customers by predicted CLV in descending order.

        Returns
        -------
        pd.DataFrame
            Columns: customer_id, predicted_clv  (sorted descending).
        """
        preds = self.predict(X)
        df = pd.DataFrame({
            "customer_id": np.asarray(customer_ids),
            "predicted_clv": preds,
        })
        df = df.sort_values("predicted_clv", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Churn integration
    # ------------------------------------------------------------------
    @staticmethod
    def adjust_for_churn(
        predicted_clv: np.ndarray,
        churn_prob: np.ndarray,
    ) -> np.ndarray:
        """Return churn-adjusted CLV = CLV × (1 − churn_prob).

        Parameters
        ----------
        predicted_clv : array-like
            Raw CLV predictions.
        churn_prob : array-like
            Per-customer churn probability in [0, 1].

        Returns
        -------
        np.ndarray
            Adjusted CLV values (non-negative).
        """
        clv = np.asarray(predicted_clv, dtype=np.float64)
        churn = np.asarray(churn_prob, dtype=np.float64)
        adjusted = clv * (1.0 - churn)
        return np.clip(adjusted, 0, None)

    # ------------------------------------------------------------------
    # Budget allocation
    # ------------------------------------------------------------------
    def allocate_budget(
        self,
        customer_ids: Sequence[str],
        X: pd.DataFrame,
        total_budget: float,
    ) -> pd.DataFrame:
        """Allocate retention budget proportional to predicted CLV.

        Higher-CLV customers receive a larger share of the budget.

        Parameters
        ----------
        customer_ids : sequence of str
            Customer identifiers.
        X : pd.DataFrame
            Feature matrix.
        total_budget : float
            Total budget in KRW.

        Returns
        -------
        pd.DataFrame
            Columns: customer_id, allocated_budget
        """
        preds = self.predict(X)
        total_clv = preds.sum()

        if total_clv > 0:
            shares = preds / total_clv
        else:
            # Uniform allocation fallback
            shares = np.ones(len(preds)) / len(preds)

        allocated = shares * total_budget

        return pd.DataFrame({
            "customer_id": np.asarray(customer_ids),
            "allocated_budget": allocated,
        })

    def evaluate_holdout(
        self,
        X_holdout: pd.DataFrame,
        y_holdout: Union[pd.Series, np.ndarray],
        customer_ids: Optional[Sequence[str]] = None,
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """Compare actual vs predicted CLV on a holdout set."""
        actual = np.asarray(y_holdout, dtype=np.float64)
        predicted = self.predict(X_holdout)

        if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0:
            correlation = float(np.corrcoef(actual, predicted)[0, 1])
        else:
            correlation = 0.0

        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

        report = pd.DataFrame({
            "customer_id": np.asarray(customer_ids) if customer_ids is not None
            else np.arange(len(predicted)),
            "actual_clv": actual,
            "predicted_clv": predicted,
            "absolute_error": np.abs(actual - predicted),
        }).sort_values("predicted_clv", ascending=False).reset_index(drop=True)

        return {
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "correlation": correlation,
            },
            "predictions": report,
            "top_n": report.head(top_n).copy(),
        }

    def build_value_report(
        self,
        customer_ids: Sequence[str],
        X: pd.DataFrame,
        top_n: int = 20,
        high_value_quantile: float = 0.8,
    ) -> Dict[str, Any]:
        """Return ranking and distribution helpers for downstream reporting."""
        ranked = self.rank_customers(customer_ids, X)
        threshold = float(ranked["predicted_clv"].quantile(high_value_quantile))
        ranked["high_value"] = ranked["predicted_clv"] >= threshold

        distribution = {
            "count": int(len(ranked)),
            "mean": float(ranked["predicted_clv"].mean()),
            "median": float(ranked["predicted_clv"].median()),
            "p80": float(ranked["predicted_clv"].quantile(0.8)),
            "p90": float(ranked["predicted_clv"].quantile(0.9)),
            "p95": float(ranked["predicted_clv"].quantile(0.95)),
            "high_value_threshold": threshold,
            "high_value_count": int(ranked["high_value"].sum()),
        }

        return {
            "ranking": ranked,
            "top_n": ranked.head(top_n).copy(),
            "distribution": distribution,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save the fitted model to disk.

        Creates a file at ``<path>.pkl`` containing the model state.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        save_path = f"{path}.pkl" if not str(path).endswith(".pkl") else path
        state = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
            "seed": self.seed,
            "config": self.config,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        logger.info("CLV model saved to %s", save_path)

    @classmethod
    def load(cls, path: str) -> "CLVModel":
        """Load a previously saved CLV model.

        Parameters
        ----------
        path : str
            Path prefix used in ``save()`` (without or with .pkl extension).

        Returns
        -------
        CLVModel
            Restored model ready for prediction.
        """
        load_path = f"{path}.pkl" if not str(path).endswith(".pkl") else path
        with open(load_path, "rb") as f:
            state = pickle.load(f)

        instance = cls(state["config"])
        instance._model = state["model"]
        instance._scaler = state["scaler"]
        instance._feature_names = state["feature_names"]
        instance._is_fitted = state["is_fitted"]
        instance.seed = state["seed"]
        return instance
