"""
Unit Tests for Uplift Modeling Module – supplementary test_uplift.py.

Covers additional key functions and edge cases not in test_uplift_model.py:
- S-Learner variant training and prediction
- Unfitted model error handling
- Segment customers with edge-case score distributions
- AUUC with degenerate inputs (all treatment / all control)
- DataFrame input handling and feature name preservation
- Config-driven hyperparameter usage
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.uplift_model import UpliftModel

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def simple_data():
    """Small synthetic dataset for quick unit tests."""
    np.random.seed(123)
    n = 400
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
    treatment = np.random.randint(0, 2, n)
    # treatment effect: feature_0 > 0 → treatment reduces churn
    base_prob = 0.5
    effect = np.where(X["f0"].values > 0, -0.2, 0.1)
    prob = np.clip(base_prob + treatment * effect, 0.05, 0.95)
    y = (np.random.rand(n) < prob).astype(int)
    return X, treatment, y


# ---------------------------------------------------------------------------
# S-Learner tests
# ---------------------------------------------------------------------------

class TestSLearner:
    """Test S-Learner meta-learner variant."""

    def test_s_learner_trains(self, config, simple_data):
        X, treatment, y = simple_data
        model = UpliftModel(config, learner="s_learner")
        model.fit(X, treatment, y)
        assert model._is_fitted

    def test_s_learner_predicts(self, config, simple_data):
        X, treatment, y = simple_data
        model = UpliftModel(config, learner="s_learner")
        model.fit(X, treatment, y)
        scores = model.predict_uplift(X)
        assert scores.shape == (len(X),)
        assert not np.any(np.isnan(scores))

    def test_s_learner_segments(self, config, simple_data):
        X, treatment, y = simple_data
        model = UpliftModel(config, learner="s_learner")
        model.fit(X, treatment, y)
        scores = model.predict_uplift(X)
        segments = model.segment_customers(scores)
        assert len(segments) == len(X)
        assert all(s in {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}
                   for s in segments)

    def test_s_learner_auuc(self, config, simple_data):
        X, treatment, y = simple_data
        model = UpliftModel(config, learner="s_learner")
        model.fit(X, treatment, y)
        scores = model.predict_uplift(X)
        auuc = model.compute_auuc(y, scores, treatment)
        assert isinstance(auuc, float)
        assert auuc >= 0


# ---------------------------------------------------------------------------
# Unfitted model error handling
# ---------------------------------------------------------------------------

class TestUnfittedErrors:
    """Test error handling for unfitted model."""

    def test_predict_before_fit_raises(self, config):
        model = UpliftModel(config)
        X = np.random.randn(10, 5)
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict_uplift(X)

    def test_save_before_fit_raises(self, config, tmp_path):
        model = UpliftModel(config)
        with pytest.raises(RuntimeError, match="unfitted"):
            model.save(str(tmp_path / "model"))


# ---------------------------------------------------------------------------
# Segmentation edge cases
# ---------------------------------------------------------------------------

class TestSegmentationEdgeCases:
    """Test segment_customers with edge-case distributions."""

    def test_all_positive_scores(self, config):
        model = UpliftModel(config)
        scores = np.array([0.1, 0.2, 0.3, 0.5, 0.8])
        segments = model.segment_customers(scores)
        assert len(segments) == 5
        assert "persuadable" in segments

    def test_all_negative_scores(self, config):
        model = UpliftModel(config)
        scores = np.array([-0.1, -0.2, -0.3, -0.5, -0.8])
        segments = model.segment_customers(scores)
        assert len(segments) == 5
        assert "sleeping_dog" in segments

    def test_all_zero_scores(self, config):
        model = UpliftModel(config)
        scores = np.zeros(10)
        segments = model.segment_customers(scores)
        assert len(segments) == 10
        # All should be assigned (no None)
        assert all(s is not None for s in segments)

    def test_single_customer(self, config):
        model = UpliftModel(config)
        scores = np.array([0.5])
        segments = model.segment_customers(scores)
        assert len(segments) == 1

    def test_segment_with_baseline_churn_probability(self, config):
        """Optional churn axis should produce 4-quadrant semantics."""
        model = UpliftModel(config)
        scores = np.array([0.2, 0.01, 0.0, -0.1])
        churn = np.array([0.8, 0.2, 0.9, 0.7])
        segments = model.segment_customers(scores, baseline_churn_probability=churn)
        assert list(segments) == [
            "persuadable", "sure_thing", "lost_cause", "sleeping_dog"
        ]

    def test_compare_learners_returns_both_rows(self, config, simple_data):
        """Helper should make T/S-learner comparison easy."""
        X, treatment, y = simple_data
        model = UpliftModel(config)
        comparison = model.compare_learners(X, treatment, y)
        assert set(comparison["learner"]) >= {"t_learner", "s_learner"}
        assert "auuc" in comparison.columns

    def test_default_auto_selects_best_auuc_learner(self, config, simple_data):
        """Default model output should use the best available learner."""
        X, treatment, y = simple_data
        auto_model = UpliftModel(config)
        auto_model.fit(X, treatment, y)

        assert auto_model.requested_learner == "auto"
        assert auto_model.learner in {"t_learner", "s_learner"}
        assert auto_model.selection_metrics_ is not None
        best = auto_model.selection_metrics_.sort_values(
            ["auuc", "mean_uplift"], ascending=[False, False]
        ).iloc[0]
        assert auto_model.learner == best["learner"]

    def test_persuadables_analysis_returns_targeting_data(self, config):
        """Persuadables analysis should produce data-backed targeting inputs."""
        model = UpliftModel(config)
        X = pd.DataFrame({
            "recency": [90.0, 80.0, 10.0, 5.0],
            "frequency": [1.0, 2.0, 8.0, 9.0],
            "monetary": [100.0, 200.0, 900.0, 1000.0],
        })
        scores = np.array([0.30, 0.20, 0.01, -0.10])
        segments = np.array([
            "persuadable", "persuadable", "sure_thing", "sleeping_dog"
        ])
        churn = np.array([0.8, 0.7, 0.2, 0.6])

        analysis = model.analyze_persuadables(
            X,
            uplift_scores=scores,
            segments=segments,
            baseline_churn_probability=churn,
            customer_ids=["C1", "C2", "C3", "C4"],
            top_features=2,
        )

        assert analysis["summary"]["persuadable_count"] == 2.0
        assert len(analysis["feature_lift"]) == 2
        assert list(analysis["top_customers"]["customer_id"]) == ["C1", "C2"]


# ---------------------------------------------------------------------------
# AUUC edge cases
# ---------------------------------------------------------------------------

class TestAUUCEdgeCases:
    """Test AUUC computation edge cases."""

    def test_auuc_all_treatment_returns_zero(self, config):
        model = UpliftModel(config)
        y = np.array([1, 0, 1, 0, 1])
        uplift = np.array([0.5, 0.3, 0.1, -0.1, -0.3])
        treatment = np.ones(5, dtype=int)  # all treatment
        auuc = model.compute_auuc(y, uplift, treatment)
        assert auuc == 0.0

    def test_auuc_all_control_returns_zero(self, config):
        model = UpliftModel(config)
        y = np.array([1, 0, 1, 0, 1])
        uplift = np.array([0.5, 0.3, 0.1, -0.1, -0.3])
        treatment = np.zeros(5, dtype=int)  # all control
        auuc = model.compute_auuc(y, uplift, treatment)
        assert auuc == 0.0

    def test_auuc_accepts_list_inputs(self, config):
        model = UpliftModel(config)
        y = [1, 0, 1, 0]
        uplift = [0.5, 0.3, -0.1, -0.3]
        treatment = [1, 0, 1, 0]
        auuc = model.compute_auuc(y, uplift, treatment)
        assert isinstance(auuc, float)


# ---------------------------------------------------------------------------
# DataFrame feature name handling
# ---------------------------------------------------------------------------

class TestDataFrameHandling:
    """Test that DataFrame inputs preserve feature names."""

    def test_feature_names_stored(self, config, simple_data):
        X, treatment, y = simple_data
        model = UpliftModel(config)
        model.fit(X, treatment, y)
        assert model._feature_names == list(X.columns)

    def test_numpy_input_no_feature_names(self, config, simple_data):
        X, treatment, y = simple_data
        model = UpliftModel(config)
        model.fit(X.values, treatment, y)
        assert model._feature_names is None


# ---------------------------------------------------------------------------
# Config-driven hyperparameters
# ---------------------------------------------------------------------------

class TestConfigHyperparams:
    """Test that config controls model hyperparameters."""

    def test_custom_n_estimators(self):
        config = {
            "simulation": {"random_seed": 99},
            "uplift": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05},
        }
        model = UpliftModel(config)
        assert model.n_estimators == 50
        assert model.max_depth == 3
        assert model.learning_rate == 0.05
        assert model.seed == 99

    def test_default_hyperparams(self):
        config = {"simulation": {"random_seed": 42}}
        model = UpliftModel(config)
        assert model.n_estimators == 100
        assert model.max_depth == 4
        assert model.learning_rate == 0.1


# ---------------------------------------------------------------------------
# Persistence round-trip with S-Learner
# ---------------------------------------------------------------------------

class TestSLearnerPersistence:
    """Test save/load with S-Learner variant."""

    def test_s_learner_save_load(self, config, simple_data, tmp_path):
        X, treatment, y = simple_data
        model = UpliftModel(config, learner="s_learner")
        model.fit(X, treatment, y)

        path = str(tmp_path / "s_learner_model")
        model.save(path)

        loaded = UpliftModel.load(path)
        assert loaded.learner == "s_learner"

        orig_scores = model.predict_uplift(X)
        load_scores = loaded.predict_uplift(X)
        np.testing.assert_array_almost_equal(orig_scores, load_scores, decimal=5)
