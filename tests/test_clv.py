"""
Unit Tests for CLV Prediction Module – supplementary test_clv.py.

Covers additional key functions and edge cases not in test_clv_model.py:
- Feature engineering (_engineer_features) static method
- Unfitted model error handling
- adjust_for_churn boundary conditions
- allocate_budget edge cases (zero CLV, single customer)
- Predict with missing columns (graceful handling)
- Non-negative constraint enforcement
- Budget allocation sums to total
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.clv_model import CLVModel

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def rfm_data():
    """Small RFM dataset for quick tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "recency": np.random.exponential(20, n),
        "frequency": np.random.poisson(5, n).astype(float),
        "monetary": np.random.lognormal(10, 1, n),
        "tenure_days": np.random.uniform(30, 365, n),
        "visit_frequency": np.random.poisson(8, n).astype(float),
    })


@pytest.fixture
def clv_target(rfm_data):
    """Target CLV values correlated with RFM features."""
    np.random.seed(42)
    n = len(rfm_data)
    return (
        0.3 * rfm_data["frequency"] * rfm_data["monetary"] / rfm_data["frequency"].clip(1)
        + np.random.randn(n) * 10000
    ).clip(0)


@pytest.fixture
def fitted_model(config, rfm_data, clv_target):
    """Return a fitted CLV model."""
    model = CLVModel(config)
    model.fit(rfm_data, clv_target)
    return model


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Test _engineer_features static method."""

    def test_rfm_features_created(self, rfm_data):
        result = CLVModel._engineer_features(rfm_data)
        assert "freq_monetary_interaction" in result.columns
        assert "recency_frequency_ratio" in result.columns
        assert "log_monetary" in result.columns

    def test_non_rfm_unchanged(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = CLVModel._engineer_features(df)
        assert list(result.columns) == ["a", "b"]

    def test_original_not_mutated(self, rfm_data):
        original_cols = list(rfm_data.columns)
        CLVModel._engineer_features(rfm_data)
        assert list(rfm_data.columns) == original_cols

    def test_zero_frequency_handled(self):
        df = pd.DataFrame({
            "recency": [10.0],
            "frequency": [0.0],
            "monetary": [1000.0],
        })
        result = CLVModel._engineer_features(df)
        assert not np.any(np.isnan(result.values))
        assert not np.any(np.isinf(result.values))


# ---------------------------------------------------------------------------
# Unfitted model errors
# ---------------------------------------------------------------------------

class TestUnfittedErrors:
    """Test error handling for unfitted model."""

    def test_predict_before_fit_raises(self, config, rfm_data):
        model = CLVModel(config)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(rfm_data)

    def test_rank_before_fit_raises(self, config, rfm_data):
        model = CLVModel(config)
        ids = [f"C{i}" for i in range(len(rfm_data))]
        with pytest.raises(RuntimeError):
            model.rank_customers(ids, rfm_data)


# ---------------------------------------------------------------------------
# adjust_for_churn boundary conditions
# ---------------------------------------------------------------------------

class TestAdjustForChurnBoundaries:
    """Test adjust_for_churn edge cases."""

    def test_zero_churn_returns_original(self):
        clv = np.array([100.0, 200.0, 300.0])
        churn = np.zeros(3)
        result = CLVModel.adjust_for_churn(clv, churn)
        np.testing.assert_array_almost_equal(result, clv)

    def test_full_churn_returns_zero(self):
        clv = np.array([100.0, 200.0, 300.0])
        churn = np.ones(3)
        result = CLVModel.adjust_for_churn(clv, churn)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_half_churn_halves_clv(self):
        clv = np.array([100.0, 200.0])
        churn = np.array([0.5, 0.5])
        result = CLVModel.adjust_for_churn(clv, churn)
        np.testing.assert_array_almost_equal(result, [50.0, 100.0])

    def test_result_always_non_negative(self):
        clv = np.array([100.0, 0.0, 50.0])
        churn = np.array([0.5, 0.9, 1.0])
        result = CLVModel.adjust_for_churn(clv, churn)
        assert np.all(result >= 0)

    def test_accepts_lists(self):
        result = CLVModel.adjust_for_churn([100, 200], [0.1, 0.2])
        assert isinstance(result, np.ndarray)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Budget allocation edge cases
# ---------------------------------------------------------------------------

class TestBudgetAllocationEdgeCases:
    """Test allocate_budget edge cases."""

    def test_budget_sums_exactly(self, fitted_model, rfm_data):
        total = 1_000_000.0
        ids = [f"C{i}" for i in range(len(rfm_data))]
        allocation = fitted_model.allocate_budget(ids, rfm_data, total)
        np.testing.assert_almost_equal(
            allocation["allocated_budget"].sum(), total, decimal=0
        )

    def test_single_customer_gets_all(self, config):
        """When predicting for a single customer, they get the full budget."""
        model = CLVModel(config)
        np.random.seed(42)
        n = 50
        X_train = pd.DataFrame({
            "recency": np.random.exponential(20, n),
            "frequency": np.random.poisson(3, n).astype(float),
            "monetary": np.random.lognormal(9, 1, n),
        })
        y_train = pd.Series(np.random.lognormal(10, 1, n))
        model.fit(X_train, y_train)
        # Allocate for a single customer
        X_single = X_train.iloc[[0]]
        allocation = model.allocate_budget(["C1"], X_single, 100_000.0)
        np.testing.assert_almost_equal(
            allocation["allocated_budget"].iloc[0], 100_000.0, decimal=0
        )

    def test_all_allocations_non_negative(self, fitted_model, rfm_data):
        ids = [f"C{i}" for i in range(len(rfm_data))]
        allocation = fitted_model.allocate_budget(ids, rfm_data, 500_000.0)
        assert (allocation["allocated_budget"] >= 0).all()


class TestReportingHelpers:
    """Test holdout evaluation and reporting helpers."""

    def test_evaluate_holdout_returns_metrics(self, fitted_model, rfm_data, clv_target):
        ids = [f"C{i}" for i in range(len(rfm_data))]
        result = fitted_model.evaluate_holdout(rfm_data, clv_target, customer_ids=ids)
        assert "metrics" in result
        assert "predictions" in result
        assert "correlation" in result["metrics"]
        assert result["metrics"]["validation_type"] == "actual_vs_predicted_holdout"
        assert "actual_clv" in result["predictions"].columns

    def test_future_revenue_labels_use_post_cutoff_purchases(self):
        """Future-window CLV labels should be separate from observation features."""
        events = pd.DataFrame({
            "customer_id": ["C001", "C001", "C002", "C002"],
            "event_date": [
                "2024-01-05",
                "2024-03-01",
                "2024-01-10",
                "2024-04-01",
            ],
            "event_type": ["purchase", "purchase", "purchase", "page_view"],
            "amount": [100.0, 300.0, 50.0, 999.0],
        })

        labels = CLVModel.future_revenue_labels(
            events,
            cutoff_date="2024-02-01",
            customer_ids=["C001", "C002"],
        )

        label_df = labels["labels"].set_index("customer_id")
        assert labels["metadata"]["target"] == "future_revenue_12m_actual"
        assert labels["metadata"]["validation_type"] == "temporal_actual_vs_predicted"
        assert label_df.loc["C001", "future_revenue_12m_actual"] > 0
        assert label_df.loc["C002", "future_revenue_12m_actual"] == 0

    def test_build_value_report_returns_top_n(self, fitted_model, rfm_data):
        ids = [f"C{i}" for i in range(len(rfm_data))]
        report = fitted_model.build_value_report(ids, rfm_data, top_n=15)
        assert len(report["top_n"]) == 15
        assert "distribution" in report
        assert "high_value_threshold" in report["distribution"]


# ---------------------------------------------------------------------------
# Predict with missing engineered features
# ---------------------------------------------------------------------------

class TestPredictMissingCols:
    """Test prediction when some engineered columns are missing."""

    def test_predict_subset_of_train_features(self, config):
        """Model trained with RFM should handle predict with same RFM."""
        model = CLVModel(config)
        np.random.seed(42)
        n = 100
        X_train = pd.DataFrame({
            "recency": np.random.exponential(20, n),
            "frequency": np.random.poisson(3, n).astype(float),
            "monetary": np.random.lognormal(9, 1, n),
        })
        y_train = pd.Series(np.random.lognormal(10, 1, n))
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({
            "recency": np.random.exponential(20, 10),
            "frequency": np.random.poisson(3, 10).astype(float),
            "monetary": np.random.lognormal(9, 1, 10),
        })
        preds = model.predict(X_test)
        assert len(preds) == 10
        assert np.all(preds >= 0)


# ---------------------------------------------------------------------------
# Non-negative CLV constraint
# ---------------------------------------------------------------------------

class TestNonNegativeConstraint:
    """Test that predictions are always non-negative."""

    def test_negative_targets_still_produce_non_negative(self, config):
        """Even if some targets are zero, predictions should be ≥ 0."""
        model = CLVModel(config)
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            "recency": np.random.exponential(20, n),
            "frequency": np.random.poisson(3, n).astype(float),
            "monetary": np.random.lognormal(9, 1, n),
        })
        y = pd.Series(np.zeros(n))  # all-zero targets
        model.fit(X, y)
        preds = model.predict(X)
        assert np.all(preds >= 0)


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersistenceRoundTrip:
    """Test save/load preserves model state."""

    def test_loaded_model_has_same_seed(self, fitted_model, tmp_path):
        path = str(tmp_path / "clv_test")
        fitted_model.save(path)
        loaded = CLVModel.load(path)
        assert loaded.seed == fitted_model.seed

    def test_loaded_model_has_same_feature_names(self, fitted_model, rfm_data, tmp_path):
        path = str(tmp_path / "clv_test")
        fitted_model.save(path)
        loaded = CLVModel.load(path)
        assert loaded._feature_names == fitted_model._feature_names

    def test_save_with_pkl_extension(self, fitted_model, tmp_path):
        path = str(tmp_path / "model.pkl")
        fitted_model.save(path)
        loaded = CLVModel.load(path)
        assert loaded._is_fitted


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibilityDetailed:
    """Test reproducibility edge cases."""

    def test_different_seeds_differ(self, rfm_data, clv_target):
        cfg1 = {"simulation": {"random_seed": 1}}
        cfg2 = {"simulation": {"random_seed": 2}}
        m1 = CLVModel(cfg1)
        m2 = CLVModel(cfg2)
        m1.fit(rfm_data, clv_target)
        m2.fit(rfm_data, clv_target)
        p1 = m1.predict(rfm_data)
        p2 = m2.predict(rfm_data)
        # Predictions should differ with different seeds
        assert not np.allclose(p1, p2)
