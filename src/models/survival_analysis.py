"""
Survival Analysis Module for E-Commerce Churn Prediction.

Implements Cox Proportional Hazards survival modeling using lifelines,
along with Kaplan-Meier curve estimation. Provides:
- Cox PH model fitting with covariates
- Survival probability prediction at arbitrary time points
- Hazard function (partial hazard) prediction
- Median survival time estimation per customer
- Kaplan-Meier survival curve data for visualization
- Model save/load persistence

All configurable parameters are read from the project YAML config.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

logger = logging.getLogger(__name__)


class SurvivalModel:
    """Cox Proportional Hazards survival model for customer churn.

    Uses lifelines CoxPHFitter to model time-to-churn with covariates,
    and KaplanMeierFitter for non-parametric survival curve estimation.

    Args:
        config: Project configuration dictionary. Uses 'simulation.random_seed'
            for reproducibility and optional 'survival' section for model params.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.seed = self.config.get("simulation", {}).get("random_seed", 42)

        # Cox PH model parameters (from config or defaults)
        surv_cfg = self.config.get("survival", {})
        self.penalizer = surv_cfg.get("penalizer", 0.01)
        self.l1_ratio = surv_cfg.get("l1_ratio", 0.0)
        self.alpha = surv_cfg.get("alpha", 0.05)

        self.cox_model: Optional[CoxPHFitter] = None
        self.km_fitter: Optional[KaplanMeierFitter] = None
        self.feature_cols: Optional[List[str]] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        duration: pd.Series,
        event: pd.Series,
    ) -> "SurvivalModel":
        """Fit the Cox PH model on training data.

        Args:
            X: Covariate DataFrame (n_samples, n_features).
            duration: Observed duration (time-to-event or censoring time).
            event: Event indicator (1 = event/churn, 0 = censored).

        Returns:
            self for method chaining.
        """
        np.random.seed(self.seed)

        self.feature_cols = list(X.columns)

        # Build a single DataFrame for lifelines
        df = X.copy()
        df["duration"] = duration.values
        df["event"] = event.values

        # Fit Cox PH
        self.cox_model = CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
        )
        self.cox_model.fit(
            df,
            duration_col="duration",
            event_col="event",
        )

        # Fit Kaplan-Meier (population-level)
        self.km_fitter = KaplanMeierFitter()
        self.km_fitter.fit(
            durations=duration,
            event_observed=event,
        )

        self._is_fitted = True
        logger.info(
            "SurvivalModel fitted – concordance index: %.4f",
            self.cox_model.concordance_index_,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted or self.cox_model is None:
            raise RuntimeError("SurvivalModel has not been fitted yet.")

    def predict_survival(
        self,
        X: pd.DataFrame,
        t: float,
    ) -> np.ndarray:
        """Predict survival probability at time *t* for each row in X.

        Args:
            X: Covariate DataFrame (same columns as training data).
            t: Time point at which to evaluate survival probability.

        Returns:
            1-D numpy array of survival probabilities in [0, 1].
        """
        self._check_fitted()

        # predict_survival_function returns a DataFrame indexed by time
        surv_func = self.cox_model.predict_survival_function(X[self.feature_cols])

        # surv_func columns = one per sample, index = time grid
        # Find the closest time point <= t
        times = surv_func.index
        if t <= times.min():
            probs = surv_func.iloc[0].values
        elif t >= times.max():
            probs = surv_func.iloc[-1].values
        else:
            # Interpolate: use the last time point <= t
            idx = times.searchsorted(t, side="right") - 1
            probs = surv_func.iloc[idx].values

        return np.clip(probs.astype(float), 0.0, 1.0)

    def predict_hazard(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Predict partial hazard (exp(X @ beta)) for each customer.

        Higher values indicate higher risk of churning.

        Args:
            X: Covariate DataFrame.

        Returns:
            1-D numpy array of non-negative partial hazard values.
        """
        self._check_fitted()
        hazard = self.cox_model.predict_partial_hazard(X[self.feature_cols])
        return np.asarray(hazard).flatten().astype(float)

    def median_survival_time(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Predict median survival time for each customer.

        The median survival time is the time at which the predicted
        survival function crosses 0.5. If the curve never crosses 0.5,
        returns np.inf for that customer.

        Args:
            X: Covariate DataFrame.

        Returns:
            1-D numpy array of median survival times (days).
        """
        self._check_fitted()
        median_times = self.cox_model.predict_median(X[self.feature_cols])
        result = np.asarray(median_times).flatten().astype(float)
        # Replace NaN with inf (customer predicted to never churn)
        result = np.where(np.isnan(result), np.inf, result)
        return result

    # ------------------------------------------------------------------
    # Kaplan-Meier survival curve
    # ------------------------------------------------------------------

    def get_survival_curve(
        self,
        duration: pd.Series,
        event: pd.Series,
    ) -> pd.DataFrame:
        """Return Kaplan-Meier survival curve data for visualization.

        Args:
            duration: Observed durations.
            event: Event indicators.

        Returns:
            DataFrame with columns ['timeline', 'survival_probability'].
        """
        km = KaplanMeierFitter()
        km.fit(durations=duration, event_observed=event)

        surv = km.survival_function_
        curve_df = pd.DataFrame({
            "timeline": surv.index.values,
            "survival_probability": surv.iloc[:, 0].values,
        })
        return curve_df

    # ------------------------------------------------------------------
    # Summary / metrics
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Return the Cox PH model summary (coefficients, p-values, etc.)."""
        self._check_fitted()
        return self.cox_model.summary

    @property
    def concordance_index(self) -> float:
        """Return the concordance index of the fitted Cox PH model."""
        self._check_fitted()
        return float(self.cox_model.concordance_index_)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted survival model to disk.

        Args:
            path: Base path (without extension). Saves as '{path}.pkl'.
        """
        self._check_fitted()
        save_path = Path(f"{path}.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "cox_model": self.cox_model,
            "km_fitter": self.km_fitter,
            "feature_cols": self.feature_cols,
            "config": self.config,
            "seed": self.seed,
            "penalizer": self.penalizer,
            "l1_ratio": self.l1_ratio,
            "alpha": self.alpha,
        }
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        logger.info("Survival model saved to %s", save_path)

    @classmethod
    def load(cls, path: str) -> "SurvivalModel":
        """Load a saved survival model from disk.

        Args:
            path: Base path (without extension). Loads from '{path}.pkl'.

        Returns:
            A fitted SurvivalModel instance.
        """
        load_path = Path(f"{path}.pkl")
        with open(load_path, "rb") as f:
            state = pickle.load(f)

        model = cls(config=state["config"])
        model.cox_model = state["cox_model"]
        model.km_fitter = state["km_fitter"]
        model.feature_cols = state["feature_cols"]
        model.seed = state["seed"]
        model.penalizer = state["penalizer"]
        model.l1_ratio = state["l1_ratio"]
        model.alpha = state["alpha"]
        model._is_fitted = True

        logger.info("Survival model loaded from %s", load_path)
        return model
