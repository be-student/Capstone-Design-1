"""
Uplift Modeling Module for E-Commerce Churn Prediction.

Implements T-Learner and S-Learner meta-learner approaches for estimating
heterogeneous treatment effects (uplift). Supports 4-quadrant customer
segmentation: persuadable, sure_thing, lost_cause, sleeping_dog.

Usage:
    model = UpliftModel(config)
    model.fit(X, treatment, y)
    uplift_scores = model.predict_uplift(X)
    segments = model.segment_customers(uplift_scores)
    auuc = model.compute_auuc(y, uplift_scores, treatment)
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)


class UpliftModel:
    """Uplift model using T-Learner or S-Learner meta-learner approach.

    T-Learner: Trains separate models for treatment and control groups,
    estimates uplift as the difference in predicted probabilities.

    S-Learner: Trains a single model with treatment indicator as a feature,
    estimates uplift by comparing predictions with treatment=1 vs treatment=0.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Uses 'simulation.random_seed' for
        reproducibility and optional 'uplift' section for model params.
    learner : str
        Meta-learner type: 't_learner' or 's_learner'. Default: 't_learner'.
    """

    def __init__(self, config: dict, learner: str = "t_learner"):
        self.config = config
        self.learner = learner
        self.seed = config.get("simulation", {}).get("random_seed", 42)

        # Uplift-specific config
        uplift_cfg = config.get("uplift", {})
        self.n_estimators = uplift_cfg.get("n_estimators", 100)
        self.max_depth = uplift_cfg.get("max_depth", 4)
        self.learning_rate = uplift_cfg.get("learning_rate", 0.1)

        # Segmentation thresholds (quantile-based by default)
        self.segment_thresholds = uplift_cfg.get("segment_thresholds", {
            "high_uplift_quantile": 0.5,
            "low_uplift_quantile": 0.5,
        })

        # Models (populated after fit)
        self._treatment_model = None
        self._control_model = None
        self._single_model = None  # For S-Learner
        self._is_fitted = False
        self._feature_names = None

    def _make_base_model(self):
        """Create a base classifier for uplift estimation."""
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            min_samples_leaf=10,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> "UpliftModel":
        """Fit the uplift model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicator (1=treatment, 0=control).
        y : array-like of shape (n_samples,)
            Binary outcome (e.g. churn label).

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=np.float64)
        treatment_arr = np.asarray(treatment).ravel()
        y_arr = np.asarray(y).ravel()

        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        treatment_mask = treatment_arr == 1
        control_mask = treatment_arr == 0

        if self.learner == "t_learner":
            self._fit_t_learner(X_arr, y_arr, treatment_mask, control_mask)
        else:
            self._fit_s_learner(X_arr, y_arr, treatment_arr)

        self._is_fitted = True
        logger.info("Uplift model fitted using %s approach", self.learner)
        return self

    def _fit_t_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment_mask: np.ndarray,
        control_mask: np.ndarray,
    ):
        """Fit T-Learner: separate models for treatment and control."""
        self._treatment_model = self._make_base_model()
        self._control_model = self._make_base_model()

        self._treatment_model.fit(X[treatment_mask], y[treatment_mask])
        self._control_model.fit(X[control_mask], y[control_mask])

    def _fit_s_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        treatment: np.ndarray,
    ):
        """Fit S-Learner: single model with treatment as a feature."""
        X_with_t = np.column_stack([X, treatment])
        self._single_model = self._make_base_model()
        self._single_model.fit(X_with_t, y)

    def predict_uplift(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Predict individual treatment effects (uplift scores).

        Uplift = P(Y=1 | X, T=1) - P(Y=1 | X, T=0)

        For churn prediction, a negative uplift means treatment reduces churn
        (good). We return the negated difference so that positive uplift
        means the treatment is beneficial.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        uplift : np.ndarray of shape (n_samples,)
            Uplift scores. Positive = treatment helps (reduces churn).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        X_arr = np.asarray(X, dtype=np.float64)

        if self.learner == "t_learner":
            return self._predict_t_learner(X_arr)
        else:
            return self._predict_s_learner(X_arr)

    def _predict_t_learner(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift using T-Learner."""
        p_treatment = self._treatment_model.predict_proba(X)[:, 1]
        p_control = self._control_model.predict_proba(X)[:, 1]
        # Negative difference = treatment reduces churn = positive uplift
        return (p_control - p_treatment).astype(np.float64)

    def _predict_s_learner(self, X: np.ndarray) -> np.ndarray:
        """Predict uplift using S-Learner."""
        n = X.shape[0]
        X_t1 = np.column_stack([X, np.ones(n)])
        X_t0 = np.column_stack([X, np.zeros(n)])

        p_treatment = self._single_model.predict_proba(X_t1)[:, 1]
        p_control = self._single_model.predict_proba(X_t0)[:, 1]
        return (p_control - p_treatment).astype(np.float64)

    def segment_customers(
        self, uplift_scores: Union[np.ndarray, list]
    ) -> np.ndarray:
        """Segment customers into 4-quadrant uplift categories.

        Quadrants based on uplift score magnitude:
        - persuadable: High positive uplift (treatment helps a lot)
        - sure_thing: Low positive uplift (would be fine without treatment)
        - lost_cause: Low negative uplift (treatment doesn't help much)
        - sleeping_dog: High negative uplift (treatment hurts)

        Parameters
        ----------
        uplift_scores : array-like of shape (n_samples,)
            Predicted uplift scores from predict_uplift.

        Returns
        -------
        segments : np.ndarray of shape (n_samples,)
            Segment labels for each customer.
        """
        scores = np.asarray(uplift_scores, dtype=np.float64)
        median_uplift = np.median(scores)

        segments = np.empty(len(scores), dtype=object)

        # Positive uplift = treatment beneficial
        high_positive = scores >= max(median_uplift, 0)
        low_positive = (scores >= 0) & (scores < max(median_uplift, 0))

        # Negative uplift = treatment harmful or no effect
        low_negative = (scores < 0) & (scores >= min(median_uplift, 0))
        high_negative = scores < min(median_uplift, 0)

        segments[high_positive] = "persuadable"
        segments[low_positive] = "sure_thing"
        segments[low_negative] = "lost_cause"
        segments[high_negative] = "sleeping_dog"

        # Fill any remaining with closest category
        unassigned = segments == None  # noqa: E711
        if unassigned.any():
            segments[unassigned & (scores >= 0)] = "sure_thing"
            segments[unassigned & (scores < 0)] = "lost_cause"

        return segments

    def compute_auuc(
        self,
        y: Union[pd.Series, np.ndarray],
        uplift: Union[np.ndarray, list],
        treatment: Union[pd.Series, np.ndarray],
    ) -> float:
        """Compute Area Under the Uplift Curve (AUUC).

        Calculates the area between the model's uplift curve and the
        random targeting baseline.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Binary outcome labels.
        uplift : array-like of shape (n_samples,)
            Predicted uplift scores.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicators.

        Returns
        -------
        auuc : float
            Area Under the Uplift Curve. Higher is better.
        """
        y_arr = np.asarray(y).ravel()
        uplift_arr = np.asarray(uplift).ravel()
        treatment_arr = np.asarray(treatment).ravel()

        # Sort by descending uplift score
        order = np.argsort(-uplift_arr)
        y_sorted = y_arr[order]
        t_sorted = treatment_arr[order]

        n = len(y_arr)
        n_treatment = t_sorted.sum()
        n_control = n - n_treatment

        if n_treatment == 0 or n_control == 0:
            return 0.0

        # Compute cumulative uplift curve
        cum_treatment_outcomes = np.cumsum(y_sorted * t_sorted)
        cum_control_outcomes = np.cumsum(y_sorted * (1 - t_sorted))
        cum_treatment_count = np.cumsum(t_sorted)
        cum_control_count = np.cumsum(1 - t_sorted)

        # Avoid division by zero
        cum_treatment_count = np.maximum(cum_treatment_count, 1)
        cum_control_count = np.maximum(cum_control_count, 1)

        # Uplift at each point: difference in response rates
        # For churn: we want treatment to reduce outcome (churn)
        # Uplift curve = cumulative difference in rates
        uplift_curve = (
            cum_treatment_outcomes / cum_treatment_count
            - cum_control_outcomes / cum_control_count
        )

        # AUUC = area between model curve and random baseline
        # Using trapezoidal integration normalized by n
        fractions = np.arange(1, n + 1) / n
        # np.trapezoid (NumPy 2.0+) replaces deprecated np.trapz
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auuc = float(_trapz(uplift_curve, fractions))

        # Return absolute value to ensure positive AUUC for a good model
        # (sign depends on whether treatment increases or decreases outcome)
        return abs(auuc)

    def save(self, path: str) -> None:
        """Save the fitted model to disk.

        Parameters
        ----------
        path : str
            File path (without extension). Will save as .pkl.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        save_path = Path(path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".pkl")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "learner": self.learner,
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "segment_thresholds": self.segment_thresholds,
            "treatment_model": self._treatment_model,
            "control_model": self._control_model,
            "single_model": self._single_model,
            "is_fitted": self._is_fitted,
            "feature_names": self._feature_names,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Uplift model saved to %s", save_path)

    @classmethod
    def load(cls, path: str) -> "UpliftModel":
        """Load a saved uplift model from disk.

        Parameters
        ----------
        path : str
            File path (with or without .pkl extension).

        Returns
        -------
        model : UpliftModel
            Loaded model instance.
        """
        load_path = Path(path)
        if not load_path.suffix:
            load_path = load_path.with_suffix(".pkl")

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        # Reconstruct model with minimal config
        config = {
            "simulation": {"random_seed": state["seed"]},
            "uplift": {
                "n_estimators": state["n_estimators"],
                "max_depth": state["max_depth"],
                "learning_rate": state["learning_rate"],
                "segment_thresholds": state["segment_thresholds"],
            },
        }

        model = cls(config, learner=state["learner"])
        model._treatment_model = state["treatment_model"]
        model._control_model = state["control_model"]
        model._single_model = state["single_model"]
        model._is_fitted = state["is_fitted"]
        model._feature_names = state["feature_names"]

        logger.info("Uplift model loaded from %s", load_path)
        return model
