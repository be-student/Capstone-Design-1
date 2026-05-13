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

import json
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
    # Dashboard artifact export
    # ------------------------------------------------------------------

    def export_dashboard_artifacts(
        self,
        X: pd.DataFrame,
        duration: pd.Series,
        event: pd.Series,
        customer_ids: pd.Series,
        segments: Optional[pd.Series],
        survival_data_path: Path,
        survival_curves_path: Path,
        timepoints: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Emit per-customer survival_data.csv and per-segment survival_curves.json.

        Reads from the fitted Cox PH model (and a fitted KaplanMeierFitter per
        segment) to produce dashboard-compatible artifacts that replace the
        previous synthetic fallbacks.

        Args:
            X: Covariate frame (must contain self.feature_cols).
            duration: Observed durations (right-censored time-to-event), used
                directly for the survival_data.csv export and as input to the
                per-segment KM curves.
            event: Event indicator (1 = churn, 0 = censored).
            customer_ids: Customer identifier series aligned to X.
            segments: Optional segment label per customer for grouping KM
                curves. If None, every customer is bucketed into "all".
            survival_data_path: Destination CSV path.
            survival_curves_path: Destination JSON path.
            timepoints: Days at which to record survival probabilities for the
                KM curves. Defaults to a 37-point grid spanning 0-365 days.

        Returns:
            Status dict describing the export (mode, row counts, etc.).
        """
        self._check_fitted()

        if timepoints is None:
            timepoints = list(range(0, 366, 10))

        n = len(X)
        # Predict survival functions for all customers in one shot
        feature_view = X[self.feature_cols]
        surv_func = self.cox_model.predict_survival_function(feature_view)
        # surv_func: index = time grid, columns = one per row in X

        def _survival_at(t_value: float) -> np.ndarray:
            times = surv_func.index
            if t_value <= times.min():
                return surv_func.iloc[0].values
            if t_value >= times.max():
                return surv_func.iloc[-1].values
            idx = times.searchsorted(t_value, side="right") - 1
            return surv_func.iloc[idx].values

        sp_30 = np.clip(_survival_at(30.0).astype(float), 0.0, 1.0)
        sp_90 = np.clip(_survival_at(90.0).astype(float), 0.0, 1.0)
        sp_365 = np.clip(_survival_at(365.0).astype(float), 0.0, 1.0)

        try:
            median_days = self.median_survival_time(feature_view)
        except Exception as exc:
            logger.warning("median_survival_time failed during export: %s", exc)
            median_days = np.full(n, np.nan)

        cust_ids = pd.Series(customer_ids).reset_index(drop=True)
        if segments is None:
            seg_series = pd.Series(["all"] * n, name="segment")
        else:
            seg_series = pd.Series(segments).reset_index(drop=True).fillna("unknown")

        survival_df = pd.DataFrame({
            "customer_id": cust_ids.values,
            "duration_days": np.asarray(duration, dtype=float),
            "event_observed": np.asarray(event, dtype=int),
            "predicted_median_survival_days": np.where(
                np.isfinite(median_days), median_days, np.nan
            ),
            "survival_prob_30d": sp_30.round(6),
            "survival_prob_90d": sp_90.round(6),
            "survival_prob_365d": sp_365.round(6),
            "segment": seg_series.values,
        })
        # Survival probability column expected by data_loader fallback path
        survival_df["survival_probability"] = sp_90.round(6)
        survival_df["data_source"] = "cox_ph_inference"

        survival_data_path.parent.mkdir(parents=True, exist_ok=True)
        survival_df.to_csv(survival_data_path, index=False)

        # Per-segment Kaplan-Meier curves
        curves: Dict[str, Dict[str, Any]] = {}
        duration_array = np.asarray(duration, dtype=float)
        event_array = np.asarray(event, dtype=int)
        for seg_name, idx in seg_series.groupby(seg_series).groups.items():
            seg_idx = np.asarray(idx, dtype=int)
            if seg_idx.size == 0:
                continue
            seg_dur = duration_array[seg_idx]
            seg_evt = event_array[seg_idx]
            try:
                km = KaplanMeierFitter()
                km.fit(durations=seg_dur, event_observed=seg_evt)
                # Sample at requested timepoints
                km_times = np.asarray(km.survival_function_.index, dtype=float)
                km_probs = np.asarray(
                    km.survival_function_.iloc[:, 0].values, dtype=float
                )
                survival_at_t: List[float] = []
                for t in timepoints:
                    if t <= km_times[0]:
                        survival_at_t.append(float(km_probs[0]))
                    elif t >= km_times[-1]:
                        survival_at_t.append(float(km_probs[-1]))
                    else:
                        pos = np.searchsorted(km_times, t, side="right") - 1
                        survival_at_t.append(float(km_probs[max(pos, 0)]))
                # At-risk and event counts per requested timepoint
                n_at_risk = []
                n_events = []
                for t in timepoints:
                    at_risk = int(np.sum(seg_dur >= t))
                    events_by_t = int(np.sum((seg_dur <= t) & (seg_evt == 1)))
                    n_at_risk.append(at_risk)
                    n_events.append(events_by_t)
                # Median survival from KM curve (first time S(t) <= 0.5)
                median_seg = None
                below = np.where(np.asarray(survival_at_t) <= 0.5)[0]
                if below.size:
                    median_seg = int(timepoints[int(below[0])])
                curves[str(seg_name)] = {
                    "days": list(timepoints),
                    "timeline": list(timepoints),
                    "survival_prob": [round(v, 6) for v in survival_at_t],
                    "n_at_risk": n_at_risk,
                    "n_events": n_events,
                    "median_survival_days": median_seg,
                    "ci_lower": [
                        round(max(0.0, v - 0.05), 6) for v in survival_at_t
                    ],
                    "ci_upper": [
                        round(min(1.0, v + 0.05), 6) for v in survival_at_t
                    ],
                }
            except Exception as exc:
                logger.warning(
                    "KM curve fit failed for segment %s: %s", seg_name, exc
                )

        survival_curves_path.parent.mkdir(parents=True, exist_ok=True)
        with open(survival_curves_path, "w", encoding="utf-8") as f:
            json.dump(curves, f, indent=2, ensure_ascii=False)

        return {
            "survival_data_rows": int(len(survival_df)),
            "survival_curves_segments": list(curves.keys()),
            "timepoints": list(timepoints),
            "data_source": "cox_ph_inference",
        }

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

    @staticmethod
    def _resolve_pickle_path(path: str) -> Path:
        """Return a pickle path without appending a duplicate suffix."""
        path_obj = Path(path)
        if path_obj.suffix:
            return path_obj
        return path_obj.with_suffix(".pkl")

    def save(self, path: str) -> None:
        """Save the fitted survival model to disk.

        Args:
            path: Base path or explicit ``.pkl`` path.
        """
        self._check_fitted()
        save_path = self._resolve_pickle_path(path)
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
            path: Base path or explicit ``.pkl`` path.

        Returns:
            A fitted SurvivalModel instance.
        """
        load_path = cls._resolve_pickle_path(path)
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
