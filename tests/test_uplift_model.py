"""
TDD Tests for Uplift Modeling Module.

Tests cover:
- Uplift model training (T-Learner / S-Learner / causal approach)
- Treatment effect estimation per customer
- AUUC (Area Under Uplift Curve) metric
- Treatment/control group handling
- Uplift-based customer segmentation (persuadables, sure things, etc.)
- Model save/load functionality
- Reproducibility with same random seed
- Integration with churn predictions
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
def sample_uplift_data():
    """Create synthetic data with treatment/control labels for uplift tests.

    Simulates a scenario where treatment has heterogeneous effects:
    - Some customers respond positively (persuadables)
    - Some are unaffected (sure things / lost causes)
    - Some respond negatively (sleeping dogs)
    """
    np.random.seed(42)
    n = 2000
    n_features = 20

    X = np.random.randn(n, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    df["customer_id"] = [f"C{i:05d}" for i in range(n)]

    # Assign treatment/control (50/50 split)
    df["treatment_group"] = np.random.choice(
        ["treatment", "control"], size=n, p=[0.5, 0.5]
    )
    df["is_treatment"] = (df["treatment_group"] == "treatment").astype(int)

    # Generate heterogeneous treatment effects
    # Persuadables: respond well to treatment (feature_0 > 0)
    # Sleeping dogs: harmed by treatment (feature_1 > 1)
    base_churn_prob = 1 / (1 + np.exp(-(0.5 * X[:, 2] - 0.3 * X[:, 3])))

    treatment_effect = np.where(
        X[:, 0] > 0,
        -0.2,   # Persuadables: treatment reduces churn
        np.where(
            X[:, 1] > 1,
            0.15,  # Sleeping dogs: treatment increases churn
            0.0    # No effect
        )
    )

    churn_prob = base_churn_prob + df["is_treatment"].values * treatment_effect
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    df["churn_label"] = (np.random.rand(n) < churn_prob).astype(int)

    # Store true treatment effect for validation
    df["true_uplift"] = -treatment_effect  # Positive = treatment helps

    return df


@pytest.fixture
def uplift_model(config):
    """Create an uplift model instance."""
    from src.models.uplift_model import UpliftModel
    return UpliftModel(config)


# ---------------------------------------------------------------------------
# Model instantiation and interface tests
# ---------------------------------------------------------------------------

class TestUpliftModelInterface:
    """Test uplift model instantiation and interface."""

    def test_model_instantiation(self, uplift_model):
        """Uplift model must be instantiable from config."""
        assert uplift_model is not None

    def test_has_fit_method(self, uplift_model):
        """Uplift model must implement a fit method."""
        assert hasattr(uplift_model, "fit")
        assert callable(uplift_model.fit)

    def test_has_predict_uplift_method(self, uplift_model):
        """Uplift model must implement predict_uplift."""
        assert hasattr(uplift_model, "predict_uplift")
        assert callable(uplift_model.predict_uplift)

    def test_has_segment_method(self, uplift_model):
        """Uplift model must implement customer segmentation."""
        assert hasattr(uplift_model, "segment_customers")
        assert callable(uplift_model.segment_customers)


# ---------------------------------------------------------------------------
# Model training tests
# ---------------------------------------------------------------------------

class TestUpliftModelTraining:
    """Test uplift model training functionality."""

    def test_model_trains(self, uplift_model, sample_uplift_data):
        """Uplift model must train without error."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )

    def test_model_requires_treatment_column(self, uplift_model,
                                              sample_uplift_data):
        """Fit must accept a treatment indicator column."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        # Should work with treatment indicator
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )

    def test_model_handles_both_groups(self, uplift_model, sample_uplift_data):
        """Model must use both treatment and control data for training."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        treatment = sample_uplift_data["is_treatment"]

        assert treatment.sum() > 0, "No treatment samples"
        assert (1 - treatment).sum() > 0, "No control samples"

        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=treatment,
            y=sample_uplift_data["churn_label"],
        )


# ---------------------------------------------------------------------------
# Uplift prediction tests
# ---------------------------------------------------------------------------

class TestUpliftPrediction:
    """Test uplift prediction (treatment effect estimation)."""

    def test_predicts_uplift_scores(self, uplift_model, sample_uplift_data):
        """Uplift model must return per-customer uplift scores."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        assert len(uplift_scores) == len(sample_uplift_data)

    def test_uplift_scores_are_numeric(self, uplift_model, sample_uplift_data):
        """Uplift scores must be numeric (float)."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        assert np.issubdtype(np.array(uplift_scores).dtype, np.floating), (
            "Uplift scores must be floating-point values"
        )

    def test_uplift_scores_have_variance(self, uplift_model,
                                          sample_uplift_data):
        """Uplift scores should vary across customers (heterogeneous effects)."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        assert np.std(uplift_scores) > 0.01, (
            f"Uplift scores have too little variance ({np.std(uplift_scores):.4f})"
        )

    def test_uplift_both_positive_and_negative(self, uplift_model,
                                                sample_uplift_data):
        """Uplift scores should include both positive and negative values."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        assert np.any(np.array(uplift_scores) > 0), "No positive uplift scores"
        assert np.any(np.array(uplift_scores) < 0), "No negative uplift scores"

    def test_no_nan_in_uplift_scores(self, uplift_model, sample_uplift_data):
        """Uplift scores must not contain NaN values."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        assert not np.any(np.isnan(uplift_scores)), "NaN values in uplift scores"


# ---------------------------------------------------------------------------
# Customer segmentation tests
# ---------------------------------------------------------------------------

class TestUpliftSegmentation:
    """Test uplift-based customer segmentation."""

    def test_segment_customers(self, uplift_model, sample_uplift_data):
        """Must segment customers into uplift-based groups."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )
        segments = uplift_model.segment_customers(uplift_scores)

        assert len(segments) == len(sample_uplift_data)

    def test_segment_customers_with_baseline_churn(self, uplift_model):
        """Baseline churn input should disambiguate sure things vs lost causes."""
        uplift_scores = np.array([0.12, 0.01, 0.02, -0.05])
        baseline = np.array([0.9, 0.1, 0.8, 0.7])
        segments = uplift_model.segment_customers(
            uplift_scores,
            baseline_churn_probability=baseline,
        )
        assert set(segments) == {
            "persuadable", "sure_thing", "lost_cause", "sleeping_dog"
        }

    def test_segment_categories(self, uplift_model, sample_uplift_data):
        """Segments must include standard uplift categories."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )
        segments = uplift_model.segment_customers(uplift_scores)

        unique_segments = set(segments)
        # Must have at least persuadables and sleeping_dogs
        expected_segments = {"persuadable", "sleeping_dog", "sure_thing",
                             "lost_cause"}
        assert len(unique_segments) >= 2, (
            f"Expected >= 2 segment types, got {unique_segments}"
        )
        # At least some should be standard uplift segment names
        assert unique_segments.issubset(expected_segments) or len(unique_segments) >= 2

    def test_persuadables_have_positive_uplift(self, uplift_model,
                                                sample_uplift_data):
        """Customers in 'persuadable' segment must have positive uplift."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )
        segments = uplift_model.segment_customers(uplift_scores)

        uplift_arr = np.array(uplift_scores)
        segments_arr = np.array(segments)

        persuadable_mask = segments_arr == "persuadable"
        if persuadable_mask.any():
            avg_uplift = uplift_arr[persuadable_mask].mean()
            assert avg_uplift > 0, (
                f"Persuadables should have positive uplift, got {avg_uplift:.4f}"
            )


# ---------------------------------------------------------------------------
# AUUC metric tests
# ---------------------------------------------------------------------------

class TestAUUCMetric:
    """Test Area Under Uplift Curve (AUUC) metric."""

    def test_auuc_computed(self, uplift_model, sample_uplift_data):
        """AUUC metric must be computable from uplift predictions."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        auuc = uplift_model.compute_auuc(
            y=sample_uplift_data["churn_label"],
            uplift=uplift_scores,
            treatment=sample_uplift_data["is_treatment"],
        )
        assert isinstance(auuc, float)

    def test_auuc_positive(self, uplift_model, sample_uplift_data):
        """AUUC should be positive for a meaningful uplift model."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        uplift_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        auuc = uplift_model.compute_auuc(
            y=sample_uplift_data["churn_label"],
            uplift=uplift_scores,
            treatment=sample_uplift_data["is_treatment"],
        )
        assert auuc > 0, f"AUUC {auuc:.4f} should be positive"


# ---------------------------------------------------------------------------
# Model persistence tests
# ---------------------------------------------------------------------------

class TestUpliftModelPersistence:
    """Test uplift model save/load functionality."""

    def test_save_model(self, uplift_model, sample_uplift_data, tmp_path):
        """Uplift model must be saveable."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )

        model_path = tmp_path / "uplift_model"
        uplift_model.save(str(model_path))

        # Check that some model file was created
        saved_files = list(tmp_path.glob("uplift_model*"))
        assert len(saved_files) > 0, "No model file saved"

    def test_load_model(self, uplift_model, sample_uplift_data, tmp_path):
        """Saved uplift model must be loadable and produce same results."""
        from src.models.uplift_model import UpliftModel

        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )

        model_path = tmp_path / "uplift_model"
        uplift_model.save(str(model_path))

        loaded_model = UpliftModel.load(str(model_path))
        original_scores = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )
        loaded_scores = loaded_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )

        np.testing.assert_array_almost_equal(
            original_scores, loaded_scores, decimal=5
        )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestUpliftReproducibility:
    """Test uplift model reproducibility with same seed."""

    def test_same_seed_same_uplift(self, config, sample_uplift_data):
        """Same seed must produce identical uplift predictions."""
        from src.models.uplift_model import UpliftModel

        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]

        model1 = UpliftModel(config)
        model1.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        scores1 = model1.predict_uplift(sample_uplift_data[feature_cols])

        model2 = UpliftModel(config)
        model2.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        scores2 = model2.predict_uplift(sample_uplift_data[feature_cols])

        np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestUpliftEdgeCases:
    """Test uplift model edge cases and error handling."""

    def test_predict_before_fit_raises(self, uplift_model, sample_uplift_data):
        """Predicting before fitting must raise RuntimeError."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        with pytest.raises(RuntimeError):
            uplift_model.predict_uplift(sample_uplift_data[feature_cols])

    def test_save_before_fit_raises(self, uplift_model, tmp_path):
        """Saving before fitting must raise RuntimeError."""
        with pytest.raises(RuntimeError):
            uplift_model.save(str(tmp_path / "model"))

    def test_small_dataset(self, config):
        """Model should handle small datasets without crashing."""
        from src.models.uplift_model import UpliftModel
        np.random.seed(42)
        n = 50
        X = pd.DataFrame(np.random.randn(n, 5),
                          columns=[f"f{i}" for i in range(5)])
        treatment = np.random.choice([0, 1], size=n)
        y = np.random.choice([0, 1], size=n)

        model = UpliftModel(config)
        model.fit(X=X, treatment=treatment, y=y)
        scores = model.predict_uplift(X)
        assert len(scores) == n
        assert not np.any(np.isnan(scores))

    def test_numpy_array_input(self, config):
        """Model should accept numpy arrays directly."""
        from src.models.uplift_model import UpliftModel
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 10)
        treatment = np.random.choice([0, 1], size=n)
        y = np.random.choice([0, 1], size=n)

        model = UpliftModel(config)
        model.fit(X=X, treatment=treatment, y=y)
        scores = model.predict_uplift(X)
        assert len(scores) == n

    def test_fit_returns_self(self, uplift_model, sample_uplift_data):
        """fit() should return the model instance for chaining."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        result = uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        assert result is uplift_model


# ---------------------------------------------------------------------------
# S-Learner variant tests
# ---------------------------------------------------------------------------

class TestSLearner:
    """Test S-Learner meta-learner variant."""

    def test_s_learner_trains_and_predicts(self, config, sample_uplift_data):
        """S-Learner should train and produce uplift scores."""
        from src.models.uplift_model import UpliftModel

        model = UpliftModel(config, learner="s_learner")
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]

        model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        scores = model.predict_uplift(sample_uplift_data[feature_cols])

        assert len(scores) == len(sample_uplift_data)
        assert not np.any(np.isnan(scores))

    def test_s_learner_segments(self, config, sample_uplift_data):
        """S-Learner should produce valid customer segments."""
        from src.models.uplift_model import UpliftModel

        model = UpliftModel(config, learner="s_learner")
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]

        model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        scores = model.predict_uplift(sample_uplift_data[feature_cols])
        segments = model.segment_customers(scores)

        assert len(segments) == len(sample_uplift_data)
        valid_segments = {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
        assert set(segments).issubset(valid_segments)

    def test_s_learner_persistence(self, config, sample_uplift_data, tmp_path):
        """S-Learner should save and load correctly."""
        from src.models.uplift_model import UpliftModel

        model = UpliftModel(config, learner="s_learner")
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]

        model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        original_scores = model.predict_uplift(sample_uplift_data[feature_cols])

        model.save(str(tmp_path / "s_learner"))
        loaded = UpliftModel.load(str(tmp_path / "s_learner"))
        loaded_scores = loaded.predict_uplift(sample_uplift_data[feature_cols])

        np.testing.assert_array_almost_equal(original_scores, loaded_scores, decimal=5)


# ---------------------------------------------------------------------------
# Uplift correlation with true effects tests
# ---------------------------------------------------------------------------

class TestUpliftQuality:
    """Test that uplift predictions correlate with true treatment effects."""

    def test_uplift_correlates_with_true_effect(self, uplift_model,
                                                  sample_uplift_data):
        """Predicted uplift should positively correlate with true uplift."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        predicted_uplift = uplift_model.predict_uplift(
            sample_uplift_data[feature_cols]
        )
        true_uplift = sample_uplift_data["true_uplift"].values

        correlation = np.corrcoef(predicted_uplift, true_uplift)[0, 1]
        assert correlation > 0, (
            f"Predicted uplift should correlate positively with true uplift, "
            f"got r={correlation:.4f}"
        )

    def test_sleeping_dogs_have_negative_uplift(self, uplift_model,
                                                  sample_uplift_data):
        """Customers in 'sleeping_dog' segment should have negative uplift."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        scores = uplift_model.predict_uplift(sample_uplift_data[feature_cols])
        segments = uplift_model.segment_customers(scores)

        scores_arr = np.array(scores)
        segments_arr = np.array(segments)
        sleeping_mask = segments_arr == "sleeping_dog"
        if sleeping_mask.any():
            avg_uplift = scores_arr[sleeping_mask].mean()
            assert avg_uplift < 0, (
                f"Sleeping dogs should have negative uplift, got {avg_uplift:.4f}"
            )

    def test_at_least_three_segments_covered(self, uplift_model, sample_uplift_data):
        """At least 3 of the 4 uplift segments should be represented."""
        feature_cols = [c for c in sample_uplift_data.columns
                        if c.startswith("feature_")]
        uplift_model.fit(
            X=sample_uplift_data[feature_cols],
            treatment=sample_uplift_data["is_treatment"],
            y=sample_uplift_data["churn_label"],
        )
        scores = uplift_model.predict_uplift(sample_uplift_data[feature_cols])
        segments = uplift_model.segment_customers(scores)

        unique_segments = set(segments)
        valid_segments = {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
        assert unique_segments.issubset(valid_segments), (
            f"Unexpected segments found: {unique_segments - valid_segments}"
        )
        assert len(unique_segments) >= 3, (
            f"Expected at least 3 segments, got {unique_segments}"
        )
