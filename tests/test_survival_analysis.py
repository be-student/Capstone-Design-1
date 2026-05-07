"""
TDD Tests for Survival Analysis Module.

Tests cover:
- Survival model instantiation and interface
- Kaplan-Meier survival curve estimation
- Cox Proportional Hazards model fitting
- Hazard function computation
- Survival probability at time t
- Median survival time estimation
- Censored data handling (right-censoring)
- Customer risk scoring
- Time-to-churn prediction
- Survival curve visualization data
- Model save/load functionality
- Reproducibility with same random seed
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
def sample_survival_data():
    """Create synthetic survival/time-to-event data for testing.

    Each customer has:
    - duration: days until churn (or censoring)
    - event: 1 if churned, 0 if censored (still active at observation end)
    - covariates: features that may predict survival
    """
    np.random.seed(42)
    n = 2000

    # Covariates
    recency = np.random.exponential(30, n)
    frequency = np.random.poisson(5, n).astype(float)
    monetary = np.random.lognormal(10, 1, n)
    tenure_days = np.random.uniform(30, 365, n)
    visit_frequency = np.random.poisson(10, n).astype(float)
    coupon_usage_rate = np.random.beta(2, 5, n)
    session_duration = np.random.exponential(15, n)
    cs_contact_count = np.random.poisson(1, n).astype(float)

    # Generate survival times (Weibull-like distribution)
    # Higher frequency and lower recency → longer survival
    scale = np.exp(
        3.5
        + 0.1 * frequency
        - 0.02 * recency
        + 0.3 * coupon_usage_rate
        + np.random.randn(n) * 0.3
    )
    shape = 1.5
    duration = np.random.weibull(shape, n) * scale
    duration = np.clip(duration, 1, 365).astype(float)

    # Censoring: ~30% of observations are right-censored
    censoring_time = np.random.uniform(60, 365, n)
    event = (duration <= censoring_time).astype(int)
    observed_duration = np.minimum(duration, censoring_time)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "duration": observed_duration,
        "event": event,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "tenure_days": tenure_days,
        "visit_frequency": visit_frequency,
        "coupon_usage_rate": coupon_usage_rate,
        "session_duration": session_duration,
        "cs_contact_count": cs_contact_count,
    })

    return df


@pytest.fixture
def survival_model(config):
    """Create a survival analysis model instance."""
    from src.models.survival_analysis import SurvivalModel
    return SurvivalModel(config)


@pytest.fixture
def feature_cols():
    """Return the feature column names for survival models."""
    return [
        "recency", "frequency", "monetary", "tenure_days",
        "visit_frequency", "coupon_usage_rate", "session_duration",
        "cs_contact_count",
    ]


# ---------------------------------------------------------------------------
# Model interface tests
# ---------------------------------------------------------------------------

class TestSurvivalModelInterface:
    """Test survival model instantiation and interface."""

    def test_model_instantiation(self, survival_model):
        """Survival model must be instantiable from config."""
        assert survival_model is not None

    def test_has_fit_method(self, survival_model):
        """Survival model must implement a fit method."""
        assert hasattr(survival_model, "fit")
        assert callable(survival_model.fit)

    def test_has_predict_survival_method(self, survival_model):
        """Must implement survival probability prediction."""
        assert hasattr(survival_model, "predict_survival")
        assert callable(survival_model.predict_survival)

    def test_has_predict_hazard_method(self, survival_model):
        """Must implement hazard function prediction."""
        assert hasattr(survival_model, "predict_hazard")
        assert callable(survival_model.predict_hazard)

    def test_has_median_survival_time_method(self, survival_model):
        """Must implement median survival time computation."""
        assert hasattr(survival_model, "median_survival_time")
        assert callable(survival_model.median_survival_time)


# ---------------------------------------------------------------------------
# Model training tests
# ---------------------------------------------------------------------------

class TestSurvivalModelTraining:
    """Test survival model training functionality."""

    def test_model_trains(self, survival_model, sample_survival_data,
                          feature_cols):
        """Survival model must train without error."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

    def test_handles_censored_data(self, survival_model,
                                    sample_survival_data, feature_cols):
        """Model must handle right-censored observations."""
        # Verify data includes censored observations
        n_censored = (sample_survival_data["event"] == 0).sum()
        n_events = (sample_survival_data["event"] == 1).sum()
        assert n_censored > 0, "Test data has no censored observations"
        assert n_events > 0, "Test data has no event observations"

        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

    def test_trains_with_subset_features(self, survival_model,
                                          sample_survival_data):
        """Model must handle different feature subsets."""
        subset_cols = ["recency", "frequency", "monetary"]
        survival_model.fit(
            X=sample_survival_data[subset_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )


# ---------------------------------------------------------------------------
# Survival probability prediction tests
# ---------------------------------------------------------------------------

class TestSurvivalPrediction:
    """Test survival probability predictions."""

    def test_predict_survival_at_time(self, survival_model,
                                       sample_survival_data, feature_cols):
        """Must predict survival probability at a given time point."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        # Predict survival at 30 days
        surv_prob = survival_model.predict_survival(
            X=sample_survival_data[feature_cols],
            t=30,
        )

        assert len(surv_prob) == len(sample_survival_data)

    def test_survival_probabilities_between_0_and_1(
        self, survival_model, sample_survival_data, feature_cols
    ):
        """Survival probabilities must be in [0, 1]."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        surv_prob = survival_model.predict_survival(
            X=sample_survival_data[feature_cols],
            t=30,
        )
        surv_arr = np.array(surv_prob)

        assert np.all(surv_arr >= 0), "Survival probabilities below 0"
        assert np.all(surv_arr <= 1), "Survival probabilities above 1"

    def test_survival_decreases_over_time(self, survival_model,
                                            sample_survival_data,
                                            feature_cols):
        """Average survival probability should decrease over time."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        surv_30 = np.mean(survival_model.predict_survival(
            X=sample_survival_data[feature_cols], t=30,
        ))
        surv_180 = np.mean(survival_model.predict_survival(
            X=sample_survival_data[feature_cols], t=180,
        ))
        surv_365 = np.mean(survival_model.predict_survival(
            X=sample_survival_data[feature_cols], t=365,
        ))

        assert surv_30 >= surv_180, (
            f"Survival at 30d ({surv_30:.3f}) should be >= at 180d ({surv_180:.3f})"
        )
        assert surv_180 >= surv_365, (
            f"Survival at 180d ({surv_180:.3f}) should be >= at 365d ({surv_365:.3f})"
        )

    def test_no_nan_in_survival_predictions(self, survival_model,
                                              sample_survival_data,
                                              feature_cols):
        """Survival predictions must not contain NaN."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        surv_prob = survival_model.predict_survival(
            X=sample_survival_data[feature_cols], t=30,
        )
        assert not np.any(np.isnan(surv_prob)), (
            "NaN values in survival predictions"
        )

    def test_survival_has_variance(self, survival_model,
                                    sample_survival_data, feature_cols):
        """Survival predictions should vary across customers."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        surv_prob = survival_model.predict_survival(
            X=sample_survival_data[feature_cols], t=90,
        )
        assert np.std(surv_prob) > 0.01, (
            "Survival predictions have no variance"
        )


# ---------------------------------------------------------------------------
# Hazard function tests
# ---------------------------------------------------------------------------

class TestHazardPrediction:
    """Test hazard function predictions."""

    def test_predict_hazard(self, survival_model, sample_survival_data,
                            feature_cols):
        """Must predict hazard scores for each customer."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        hazard = survival_model.predict_hazard(
            X=sample_survival_data[feature_cols],
        )
        assert len(hazard) == len(sample_survival_data)

    def test_hazard_non_negative(self, survival_model, sample_survival_data,
                                  feature_cols):
        """Hazard values must be non-negative."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        hazard = survival_model.predict_hazard(
            X=sample_survival_data[feature_cols],
        )
        assert np.all(np.array(hazard) >= 0), "Negative hazard values found"

    def test_hazard_no_nan(self, survival_model, sample_survival_data,
                           feature_cols):
        """Hazard predictions must not contain NaN."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        hazard = survival_model.predict_hazard(
            X=sample_survival_data[feature_cols],
        )
        assert not np.any(np.isnan(hazard)), "NaN values in hazard predictions"

    def test_hazard_has_variance(self, survival_model, sample_survival_data,
                                  feature_cols):
        """Hazard scores should vary across customers."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        hazard = survival_model.predict_hazard(
            X=sample_survival_data[feature_cols],
        )
        assert np.std(hazard) > 0, "Hazard predictions have zero variance"


# ---------------------------------------------------------------------------
# Median survival time tests
# ---------------------------------------------------------------------------

class TestMedianSurvivalTime:
    """Test median survival time estimation."""

    def test_median_survival_returns_values(self, survival_model,
                                             sample_survival_data,
                                             feature_cols):
        """Must return median survival time for each customer."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        median_times = survival_model.median_survival_time(
            X=sample_survival_data[feature_cols],
        )
        assert len(median_times) == len(sample_survival_data)

    def test_median_survival_positive(self, survival_model,
                                       sample_survival_data, feature_cols):
        """Median survival times must be positive."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        median_times = survival_model.median_survival_time(
            X=sample_survival_data[feature_cols],
        )
        valid = np.array(median_times)
        # Filter out inf (customers predicted to never churn)
        finite_mask = np.isfinite(valid)
        if finite_mask.any():
            assert np.all(valid[finite_mask] > 0), (
                "Finite median survival times must be positive"
            )

    def test_median_survival_has_variance(self, survival_model,
                                           sample_survival_data,
                                           feature_cols):
        """Median survival times should vary across customers."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        median_times = survival_model.median_survival_time(
            X=sample_survival_data[feature_cols],
        )
        valid = np.array(median_times)
        finite_vals = valid[np.isfinite(valid)]
        if len(finite_vals) > 1:
            assert np.std(finite_vals) > 0, (
                "Median survival times have zero variance"
            )


# ---------------------------------------------------------------------------
# Survival curve data tests
# ---------------------------------------------------------------------------

class TestSurvivalCurve:
    """Test survival curve estimation (Kaplan-Meier style)."""

    def test_get_survival_curve(self, survival_model, sample_survival_data,
                                 feature_cols):
        """Must return survival curve data for visualization."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        curve = survival_model.get_survival_curve(
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        assert isinstance(curve, (pd.DataFrame, dict))

    def test_survival_curve_starts_at_one(self, survival_model,
                                           sample_survival_data,
                                           feature_cols):
        """Survival curve should start at or near 1.0 (all alive)."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        curve = survival_model.get_survival_curve(
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        if isinstance(curve, pd.DataFrame):
            first_prob = curve.iloc[0]["survival_probability"]
        else:
            first_prob = curve["survival_probability"][0]

        assert first_prob >= 0.95, (
            f"Survival curve should start near 1.0, got {first_prob:.3f}"
        )

    def test_survival_curve_monotone_decreasing(self, survival_model,
                                                  sample_survival_data,
                                                  feature_cols):
        """Survival curve must be monotonically non-increasing."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        curve = survival_model.get_survival_curve(
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        if isinstance(curve, pd.DataFrame):
            probs = curve["survival_probability"].values
        else:
            probs = np.array(curve["survival_probability"])

        # Allow tiny floating-point tolerance
        assert np.all(np.diff(probs) <= 1e-10), (
            "Survival curve is not monotonically non-increasing"
        )


# ---------------------------------------------------------------------------
# Model persistence tests
# ---------------------------------------------------------------------------

class TestSurvivalModelPersistence:
    """Test survival model save/load functionality."""

    def test_save_model(self, survival_model, sample_survival_data,
                        feature_cols, tmp_path):
        """Survival model must be saveable."""
        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        model_path = tmp_path / "survival_model"
        survival_model.save(str(model_path))

        saved_files = list(tmp_path.glob("survival_model*"))
        assert len(saved_files) > 0, "No survival model file saved"

    def test_load_model(self, survival_model, sample_survival_data,
                        feature_cols, tmp_path):
        """Saved survival model must be loadable and produce same results."""
        from src.models.survival_analysis import SurvivalModel

        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        model_path = tmp_path / "survival_model"
        survival_model.save(str(model_path))

        loaded_model = SurvivalModel.load(str(model_path))
        original_surv = survival_model.predict_survival(
            X=sample_survival_data[feature_cols], t=90,
        )
        loaded_surv = loaded_model.predict_survival(
            X=sample_survival_data[feature_cols], t=90,
        )

        np.testing.assert_array_almost_equal(
            original_surv, loaded_surv, decimal=5,
        )

    def test_save_with_explicit_pkl_suffix_does_not_duplicate(
        self, survival_model, sample_survival_data, feature_cols, tmp_path
    ):
        """Explicit .pkl paths should not become .pkl.pkl."""
        from src.models.survival_analysis import SurvivalModel

        survival_model.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )

        model_path = tmp_path / "survival_model.pkl"
        survival_model.save(str(model_path))

        assert model_path.exists()
        assert not (tmp_path / "survival_model.pkl.pkl").exists()
        loaded = SurvivalModel.load(str(model_path))
        assert loaded.concordance_index == survival_model.concordance_index


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestSurvivalReproducibility:
    """Test survival model reproducibility with same seed."""

    def test_same_seed_same_predictions(self, config, sample_survival_data,
                                         feature_cols):
        """Same seed must produce identical survival predictions."""
        from src.models.survival_analysis import SurvivalModel

        model1 = SurvivalModel(config)
        model1.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )
        surv1 = model1.predict_survival(
            X=sample_survival_data[feature_cols], t=90,
        )

        model2 = SurvivalModel(config)
        model2.fit(
            X=sample_survival_data[feature_cols],
            duration=sample_survival_data["duration"],
            event=sample_survival_data["event"],
        )
        surv2 = model2.predict_survival(
            X=sample_survival_data[feature_cols], t=90,
        )

        np.testing.assert_array_almost_equal(surv1, surv2, decimal=5)
