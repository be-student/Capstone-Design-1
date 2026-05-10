"""
TDD Tests for Customer Lifetime Value (CLV) Prediction Module.

Tests cover:
- CLV model training and prediction
- CLV computation approaches (probabilistic / regression-based)
- Per-customer CLV estimation in KRW
- CLV-based customer ranking and segmentation
- Integration with churn probability
- Model save/load functionality
- Reproducibility with same random seed
- Non-negative CLV constraint
- Budget allocation based on CLV
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
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_clv_data():
    """Create synthetic customer data for CLV prediction tests.

    Includes purchase history features that correlate with lifetime value.
    """
    np.random.seed(42)
    n = 1500

    # RFM-like features
    recency = np.random.exponential(30, n)
    frequency = np.random.poisson(5, n).astype(float)
    monetary = np.random.lognormal(10, 1, n)  # KRW-scale amounts

    # Additional features
    tenure_days = np.random.uniform(30, 365, n)
    avg_order_value = monetary / np.maximum(frequency, 1)
    purchase_cycle_days = np.random.exponential(14, n)
    coupon_usage_rate = np.random.beta(2, 5, n)
    review_rate = np.random.beta(1, 10, n)

    # Behavioral features
    visit_frequency = np.random.poisson(10, n).astype(float)
    cart_conversion = np.random.beta(3, 7, n)
    session_duration = np.random.exponential(15, n)

    # Extra numeric features
    search_count = np.random.poisson(8, n).astype(float)
    cs_contact_count = np.random.poisson(1, n).astype(float)
    weekend_ratio = np.random.beta(2, 3, n)

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "tenure_days": tenure_days,
        "avg_order_value": avg_order_value,
        "purchase_cycle_days": purchase_cycle_days,
        "coupon_usage_rate": coupon_usage_rate,
        "review_rate": review_rate,
        "visit_frequency": visit_frequency,
        "cart_conversion": cart_conversion,
        "session_duration": session_duration,
        "search_count": search_count,
        "cs_contact_count": cs_contact_count,
        "weekend_ratio": weekend_ratio,
    })

    # Generate CLV target: correlated with frequency and monetary
    # CLV = expected future value based on historical behavior
    df["clv_target"] = (
        0.4 * frequency * avg_order_value
        + 0.3 * (365 / np.maximum(purchase_cycle_days, 1)) * avg_order_value
        - 0.1 * recency * 1000
        + np.random.randn(n) * 50000
    ).clip(0)

    # Churn probability for integration testing
    df["churn_prob"] = 1 / (1 + np.exp(
        -(0.02 * recency - 0.1 * frequency + np.random.randn(n) * 0.5)
    ))

    return df


@pytest.fixture
def clv_model(config):
    """Create a CLV model instance."""
    from src.models.clv_model import CLVModel
    return CLVModel(config)


# ---------------------------------------------------------------------------
# Model interface tests
# ---------------------------------------------------------------------------

class TestCLVModelInterface:
    """Test CLV model instantiation and interface."""

    def test_model_instantiation(self, clv_model):
        """CLV model must be instantiable from config."""
        assert clv_model is not None

    def test_has_fit_method(self, clv_model):
        """CLV model must implement a fit method."""
        assert hasattr(clv_model, "fit")
        assert callable(clv_model.fit)

    def test_has_predict_method(self, clv_model):
        """CLV model must implement a predict method."""
        assert hasattr(clv_model, "predict")
        assert callable(clv_model.predict)

    def test_has_rank_customers_method(self, clv_model):
        """CLV model must implement customer ranking by predicted CLV."""
        assert hasattr(clv_model, "rank_customers")
        assert callable(clv_model.rank_customers)


# ---------------------------------------------------------------------------
# Model training tests
# ---------------------------------------------------------------------------

class TestCLVModelTraining:
    """Test CLV model training functionality."""

    def test_model_trains(self, clv_model, sample_clv_data):
        """CLV model must train without error."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

    def test_model_trains_with_various_features(self, clv_model,
                                                  sample_clv_data):
        """Model must handle different feature subsets."""
        # Train with only RFM features
        rfm_cols = ["recency", "frequency", "monetary"]
        clv_model.fit(
            X=sample_clv_data[rfm_cols],
            y=sample_clv_data["clv_target"],
        )


# ---------------------------------------------------------------------------
# CLV prediction tests
# ---------------------------------------------------------------------------

class TestCLVPrediction:
    """Test CLV prediction functionality."""

    def test_predicts_clv_values(self, clv_model, sample_clv_data):
        """CLV model must return per-customer CLV predictions."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        assert len(predictions) == len(sample_clv_data)

    def test_predictions_non_negative(self, clv_model, sample_clv_data):
        """CLV predictions must be non-negative (KRW values)."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        assert np.all(np.array(predictions) >= 0), (
            "CLV predictions must be non-negative"
        )

    def test_predictions_are_numeric(self, clv_model, sample_clv_data):
        """CLV predictions must be numeric float values."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        pred_arr = np.array(predictions)
        assert np.issubdtype(pred_arr.dtype, np.floating) or \
               np.issubdtype(pred_arr.dtype, np.integer), (
            "CLV predictions must be numeric"
        )

    def test_no_nan_in_predictions(self, clv_model, sample_clv_data):
        """CLV predictions must not contain NaN values."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        assert not np.any(np.isnan(predictions)), (
            "NaN values found in CLV predictions"
        )

    def test_predictions_have_variance(self, clv_model, sample_clv_data):
        """CLV predictions should have meaningful variance."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        assert np.std(predictions) > 0, (
            "CLV predictions have zero variance — model may not be learning"
        )

    def test_predictions_in_reasonable_range(self, clv_model, sample_clv_data):
        """CLV predictions should be in a reasonable KRW range."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        pred_arr = np.array(predictions)
        # CLV should not be astronomically large (sanity check)
        assert pred_arr.max() < 1e10, (
            f"Max CLV {pred_arr.max():.0f} KRW seems unreasonably large"
        )


# ---------------------------------------------------------------------------
# Prediction accuracy tests
# ---------------------------------------------------------------------------

class TestCLVAccuracy:
    """Test CLV prediction accuracy metrics."""

    def test_prediction_correlation(self, clv_model, sample_clv_data):
        """Predicted CLV should positively correlate with actual CLV."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]

        # Train/test split
        n = len(sample_clv_data)
        train_idx = int(n * 0.8)
        train_data = sample_clv_data.iloc[:train_idx]
        test_data = sample_clv_data.iloc[train_idx:]

        clv_model.fit(
            X=train_data[feature_cols],
            y=train_data["clv_target"],
        )
        predictions = clv_model.predict(test_data[feature_cols])

        correlation = np.corrcoef(
            test_data["clv_target"].values, np.array(predictions)
        )[0, 1]
        assert correlation > 0.3, (
            f"CLV prediction correlation {correlation:.4f} too low (< 0.3)"
        )

    def test_high_value_customers_ranked_higher(self, clv_model,
                                                  sample_clv_data):
        """Top predicted CLV customers should overlap with actual top CLV."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]

        n = len(sample_clv_data)
        train_idx = int(n * 0.8)
        train_data = sample_clv_data.iloc[:train_idx]
        test_data = sample_clv_data.iloc[train_idx:]

        clv_model.fit(
            X=train_data[feature_cols],
            y=train_data["clv_target"],
        )
        predictions = clv_model.predict(test_data[feature_cols])

        # Top 20% by predicted vs actual
        top_k = max(int(len(test_data) * 0.2), 1)
        pred_top = set(np.argsort(-np.array(predictions))[:top_k])
        actual_top = set(np.argsort(-test_data["clv_target"].values)[:top_k])

        overlap = len(pred_top & actual_top) / top_k
        assert overlap > 0.2, (
            f"Top 20% overlap {overlap:.2%} too low (expected > 20%)"
        )


# ---------------------------------------------------------------------------
# Customer ranking tests
# ---------------------------------------------------------------------------

class TestCustomerRanking:
    """Test CLV-based customer ranking."""

    def test_rank_customers_returns_sorted(self, clv_model, sample_clv_data):
        """rank_customers must return customers sorted by predicted CLV."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        ranked = clv_model.rank_customers(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
        )
        assert isinstance(ranked, pd.DataFrame)
        assert "customer_id" in ranked.columns
        assert "predicted_clv" in ranked.columns

    def test_ranked_in_descending_order(self, clv_model, sample_clv_data):
        """Ranked customers should be in descending CLV order."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        ranked = clv_model.rank_customers(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
        )
        clv_values = ranked["predicted_clv"].values
        assert np.all(clv_values[:-1] >= clv_values[1:]), (
            "Customers not ranked in descending CLV order"
        )

    def test_ranking_covers_all_customers(self, clv_model, sample_clv_data):
        """Ranking must include all customers."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        ranked = clv_model.rank_customers(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
        )
        assert len(ranked) == len(sample_clv_data), (
            f"Ranking has {len(ranked)} customers, expected {len(sample_clv_data)}"
        )

    def test_holdout_helper_supports_actual_vs_predicted(
        self, clv_model, sample_clv_data
    ):
        """Holdout helper should surface actual-vs-predicted validation."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        n = len(sample_clv_data)
        train_idx = int(n * 0.8)
        train = sample_clv_data.iloc[:train_idx]
        test = sample_clv_data.iloc[train_idx:]
        clv_model.fit(train[feature_cols], train["clv_target"])
        report = clv_model.evaluate_holdout(
            test[feature_cols],
            test["clv_target"],
            customer_ids=test["customer_id"],
        )
        assert "mae" in report["metrics"]
        assert report["metrics"]["validation_type"] == "actual_vs_predicted_holdout"
        assert len(report["top_n"]) > 0

    def test_future_revenue_labels_schema_and_metadata(self, sample_clv_data):
        """Future labels should encode actual post-observation revenue evidence."""
        from src.models.clv_model import CLVModel

        events = pd.DataFrame({
            "customer_id": ["C00000", "C00000", "C00001", "C00002"],
            "event_timestamp": [
                "2024-01-01",
                "2024-04-01",
                "2024-04-15",
                "2024-01-20",
            ],
            "event_type": ["purchase", "purchase", "purchase", "purchase"],
            "revenue": [1000.0, 5000.0, 3000.0, 7000.0],
        })

        bundle = CLVModel.future_revenue_labels(
            events,
            cutoff_date="2024-03-01",
            customer_ids=["C00000", "C00001", "C00002"],
        )

        labels = bundle["labels"]
        assert list(labels.columns) == ["customer_id", "future_revenue_12m_actual"]
        assert bundle["metadata"]["target"] == "future_revenue_12m_actual"
        assert bundle["metadata"]["future_purchase_rows"] == 2
        assert labels.loc[
            labels["customer_id"] == "C00002", "future_revenue_12m_actual"
        ].iloc[0] == 0

    def test_value_report_contains_high_value_flag(self, clv_model, sample_clv_data):
        """Top-N/distribution helper should create a high-value cut."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(sample_clv_data[feature_cols], sample_clv_data["clv_target"])
        report = clv_model.build_value_report(
            sample_clv_data["customer_id"], sample_clv_data[feature_cols]
        )
        assert "high_value" in report["ranking"].columns
        assert report["distribution"]["high_value_count"] > 0


# ---------------------------------------------------------------------------
# Churn integration tests
# ---------------------------------------------------------------------------

class TestCLVChurnIntegration:
    """Test CLV integration with churn probability."""

    def test_adjusted_clv_with_churn_prob(self, clv_model, sample_clv_data):
        """CLV should be adjustable by churn probability."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        # Adjusted CLV = predicted CLV * (1 - churn_prob)
        adjusted = clv_model.adjust_for_churn(
            predicted_clv=predictions,
            churn_prob=sample_clv_data["churn_prob"].values,
        )
        assert len(adjusted) == len(predictions)
        assert np.all(np.array(adjusted) >= 0)

    def test_higher_churn_lowers_adjusted_clv(self, clv_model,
                                                sample_clv_data):
        """Higher churn probability should result in lower adjusted CLV."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = clv_model.predict(sample_clv_data[feature_cols])

        low_churn = np.full(len(predictions), 0.1)
        high_churn = np.full(len(predictions), 0.9)

        adj_low = clv_model.adjust_for_churn(predictions, low_churn)
        adj_high = clv_model.adjust_for_churn(predictions, high_churn)

        assert np.mean(adj_low) > np.mean(adj_high), (
            "Higher churn probability should reduce adjusted CLV"
        )


# ---------------------------------------------------------------------------
# Budget allocation tests
# ---------------------------------------------------------------------------

class TestCLVBudgetAllocation:
    """Test CLV-based budget allocation."""

    def test_allocate_budget(self, clv_model, sample_clv_data, config):
        """Must allocate retention budget proportional to CLV."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        total_budget = config["budget"]["total_krw"]
        allocation = clv_model.allocate_budget(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
            total_budget=total_budget,
        )
        assert isinstance(allocation, pd.DataFrame)
        assert "customer_id" in allocation.columns
        assert "allocated_budget" in allocation.columns

    def test_budget_sum_within_total(self, clv_model, sample_clv_data, config):
        """Total allocated budget must not exceed total budget."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        total_budget = config["budget"]["total_krw"]
        allocation = clv_model.allocate_budget(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
            total_budget=total_budget,
        )
        actual_total = allocation["allocated_budget"].sum()
        assert actual_total <= total_budget * 1.01, (
            f"Allocated {actual_total:,.0f} KRW exceeds budget "
            f"{total_budget:,.0f} KRW"
        )

    def test_budget_non_negative(self, clv_model, sample_clv_data, config):
        """Individual budget allocations must be non-negative."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        total_budget = config["budget"]["total_krw"]
        allocation = clv_model.allocate_budget(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
            total_budget=total_budget,
        )
        assert (allocation["allocated_budget"] >= 0).all(), (
            "Negative budget allocations found"
        )

    def test_higher_clv_gets_more_budget(self, clv_model, sample_clv_data,
                                          config):
        """Higher-CLV customers should receive more budget on average."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        total_budget = config["budget"]["total_krw"]
        allocation = clv_model.allocate_budget(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
            total_budget=total_budget,
        )

        predictions = clv_model.predict(sample_clv_data[feature_cols])
        pred_arr = np.array(predictions)
        median_clv = np.median(pred_arr)

        merged = allocation.copy()
        merged["predicted_clv"] = pred_arr
        high_clv_budget = merged[merged["predicted_clv"] >= median_clv][
            "allocated_budget"
        ].mean()
        low_clv_budget = merged[merged["predicted_clv"] < median_clv][
            "allocated_budget"
        ].mean()

        assert high_clv_budget >= low_clv_budget, (
            f"High-CLV budget ({high_clv_budget:,.0f}) should be >= "
            f"low-CLV budget ({low_clv_budget:,.0f})"
        )


# ---------------------------------------------------------------------------
# Model persistence tests
# ---------------------------------------------------------------------------

class TestCLVModelPersistence:
    """Test CLV model save/load functionality."""

    def test_save_model(self, clv_model, sample_clv_data, tmp_path):
        """CLV model must be saveable."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        model_path = tmp_path / "clv_model"
        clv_model.save(str(model_path))

        saved_files = list(tmp_path.glob("clv_model*"))
        assert len(saved_files) > 0, "No CLV model file saved"

    def test_load_model(self, clv_model, sample_clv_data, tmp_path):
        """Saved CLV model must be loadable and produce same results."""
        from src.models.clv_model import CLVModel

        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )

        model_path = tmp_path / "clv_model"
        clv_model.save(str(model_path))

        loaded_model = CLVModel.load(str(model_path))
        original_preds = clv_model.predict(sample_clv_data[feature_cols])
        loaded_preds = loaded_model.predict(sample_clv_data[feature_cols])

        np.testing.assert_array_almost_equal(
            original_preds, loaded_preds, decimal=5
        )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestCLVReproducibility:
    """Test CLV model reproducibility with same seed."""

    def test_same_seed_same_predictions(self, config, sample_clv_data):
        """Same seed must produce identical CLV predictions."""
        from src.models.clv_model import CLVModel

        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]

        model1 = CLVModel(config)
        model1.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        preds1 = model1.predict(sample_clv_data[feature_cols])

        model2 = CLVModel(config)
        model2.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        preds2 = model2.predict(sample_clv_data[feature_cols])

        np.testing.assert_array_almost_equal(preds1, preds2, decimal=5)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestCLVEdgeCases:
    """Test CLV model edge cases and error handling."""

    def test_predict_before_fit_raises(self, clv_model, sample_clv_data):
        """Predicting before fitting must raise RuntimeError."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        with pytest.raises(RuntimeError):
            clv_model.predict(sample_clv_data[feature_cols])

    def test_fit_returns_self(self, clv_model, sample_clv_data):
        """fit() should return the model instance for chaining."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        result = clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        assert result is clv_model

    def test_single_customer_prediction(self, clv_model, sample_clv_data):
        """Model should handle single-row predictions."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        single_row = sample_clv_data[feature_cols].iloc[[0]]
        pred = clv_model.predict(single_row)
        assert len(pred) == 1
        assert pred[0] >= 0

    def test_zero_budget_allocation(self, clv_model, sample_clv_data):
        """Zero total budget should allocate zero to all customers."""
        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]
        clv_model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        allocation = clv_model.allocate_budget(
            customer_ids=sample_clv_data["customer_id"],
            X=sample_clv_data[feature_cols],
            total_budget=0,
        )
        assert (allocation["allocated_budget"] == 0).all()


# ---------------------------------------------------------------------------
# Feature engineering tests (BG/NBD + Gamma-Gamma)
# ---------------------------------------------------------------------------

class TestCLVFeatureEngineering:
    """Test BG/NBD and Gamma-Gamma inspired feature engineering."""

    def test_engineer_features_with_rfm(self, sample_clv_data):
        """Feature engineering should add RFM-derived features."""
        from src.models.clv_model import CLVModel
        result = CLVModel._engineer_features(sample_clv_data)

        assert "freq_monetary_interaction" in result.columns
        assert "recency_frequency_ratio" in result.columns
        assert "log_monetary" in result.columns

    def test_engineer_features_without_rfm(self):
        """Feature engineering should work without RFM columns."""
        from src.models.clv_model import CLVModel
        df = pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
        })
        result = CLVModel._engineer_features(df)

        # Should not add RFM features if RFM columns are absent
        assert "freq_monetary_interaction" not in result.columns
        assert len(result.columns) == 2

    def test_log_monetary_positive(self, sample_clv_data):
        """log_monetary should produce non-negative values."""
        from src.models.clv_model import CLVModel
        result = CLVModel._engineer_features(sample_clv_data)
        assert (result["log_monetary"] >= 0).all()


# ---------------------------------------------------------------------------
# Churn-adjusted CLV edge case tests
# ---------------------------------------------------------------------------

class TestCLVChurnEdgeCases:
    """Test churn-adjusted CLV edge cases."""

    def test_zero_churn_preserves_clv(self):
        """Zero churn probability should preserve full CLV."""
        from src.models.clv_model import CLVModel
        clv = np.array([100.0, 200.0, 300.0])
        churn = np.array([0.0, 0.0, 0.0])
        adjusted = CLVModel.adjust_for_churn(clv, churn)
        np.testing.assert_array_almost_equal(adjusted, clv)

    def test_full_churn_zeroes_clv(self):
        """Churn probability of 1.0 should zero out CLV."""
        from src.models.clv_model import CLVModel
        clv = np.array([100.0, 200.0, 300.0])
        churn = np.array([1.0, 1.0, 1.0])
        adjusted = CLVModel.adjust_for_churn(clv, churn)
        np.testing.assert_array_almost_equal(adjusted, [0.0, 0.0, 0.0])

    def test_adjusted_clv_always_non_negative(self):
        """Adjusted CLV should always be non-negative."""
        from src.models.clv_model import CLVModel
        clv = np.array([100.0, 0.0, 50.0])
        churn = np.array([0.5, 0.9, 1.1])  # churn > 1 edge case
        adjusted = CLVModel.adjust_for_churn(clv, churn)
        assert np.all(adjusted >= 0)

    def test_half_churn_halves_clv(self):
        """Churn probability of 0.5 should halve CLV."""
        from src.models.clv_model import CLVModel
        clv = np.array([100.0, 200.0])
        churn = np.array([0.5, 0.5])
        adjusted = CLVModel.adjust_for_churn(clv, churn)
        np.testing.assert_array_almost_equal(adjusted, [50.0, 100.0])


# ---------------------------------------------------------------------------
# Integration tests (Uplift + CLV)
# ---------------------------------------------------------------------------

class TestCLVUpliftIntegration:
    """Test CLV and Uplift model integration patterns."""

    def test_clv_weighted_uplift_ranking(self, config, sample_clv_data):
        """Should be able to combine CLV predictions with uplift for priority."""
        from src.models.clv_model import CLVModel

        feature_cols = [c for c in sample_clv_data.columns
                        if c not in ("customer_id", "clv_target", "churn_prob")]

        model = CLVModel(config)
        model.fit(
            X=sample_clv_data[feature_cols],
            y=sample_clv_data["clv_target"],
        )
        predictions = model.predict(sample_clv_data[feature_cols])

        # Simulate uplift scores
        np.random.seed(42)
        uplift_scores = np.random.randn(len(predictions))

        # Combined priority: CLV * uplift (higher = more valuable to treat)
        combined = predictions * np.maximum(uplift_scores, 0)
        assert len(combined) == len(predictions)
        assert np.all(combined >= 0)
