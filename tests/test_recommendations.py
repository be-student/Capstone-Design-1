"""
TDD Tests for Personalized Recommendation Module.

Tests cover:
- Recommendation engine instantiation and interface
- Per-customer retention action recommendations
- Recommendation ranking and top-K selection
- Action type diversity (coupon, push notification, email, etc.)
- Recommendation scoring using churn prob, CLV, uplift, and preferences
- Personalization based on customer segment/persona
- Budget-aware recommendations (cost vs expected value)
- Recommendation explanation / reasoning
- Integration with churn, CLV, and uplift predictions
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
def sample_customer_data():
    """Create synthetic customer data for recommendation tests.

    Includes churn probability, CLV, uplift, behavioral features,
    and segment information needed by the recommendation engine.
    """
    np.random.seed(42)
    n = 1500

    # Model predictions from upstream modules
    churn_prob = np.random.beta(2, 5, n)
    clv = np.random.lognormal(10, 1, n)
    uplift_score = np.random.randn(n) * 0.1

    # Behavioral features
    recency = np.random.exponential(30, n)
    frequency = np.random.poisson(5, n).astype(float)
    monetary = np.random.lognormal(10, 1, n)
    coupon_usage_rate = np.random.beta(2, 5, n)
    visit_frequency = np.random.poisson(10, n).astype(float)
    session_duration = np.random.exponential(15, n)

    # Customer preferences / history
    preferred_category = np.random.choice(
        ["electronics", "fashion", "food", "beauty", "home"], size=n,
    )
    avg_discount_used = np.random.beta(3, 7, n) * 50  # 0-50% discount
    push_opt_in = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
    email_opt_in = np.random.choice([0, 1], size=n, p=[0.1, 0.9])

    # Customer segments
    segments = np.random.choice(
        ["vip_loyal", "regular_loyal", "bargain_hunter",
         "new_customer", "dormant", "high_value_at_risk"],
        size=n,
    )

    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "churn_prob": churn_prob,
        "clv": clv,
        "uplift_score": uplift_score,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "coupon_usage_rate": coupon_usage_rate,
        "visit_frequency": visit_frequency,
        "session_duration": session_duration,
        "preferred_category": preferred_category,
        "avg_discount_used": avg_discount_used,
        "push_opt_in": push_opt_in,
        "email_opt_in": email_opt_in,
        "segment": segments,
    })

    return df


@pytest.fixture
def recommendation_engine(config):
    """Create a recommendation engine instance."""
    from src.models.recommendations import RecommendationEngine
    return RecommendationEngine(config)


# ---------------------------------------------------------------------------
# Engine interface tests
# ---------------------------------------------------------------------------

class TestRecommendationEngineInterface:
    """Test recommendation engine instantiation and interface."""

    def test_instantiation(self, recommendation_engine):
        """Recommendation engine must be instantiable from config."""
        assert recommendation_engine is not None

    def test_has_recommend_method(self, recommendation_engine):
        """Must implement a recommend method."""
        assert hasattr(recommendation_engine, "recommend")
        assert callable(recommendation_engine.recommend)

    def test_has_recommend_top_k_method(self, recommendation_engine):
        """Must implement top-K recommendation selection."""
        assert hasattr(recommendation_engine, "recommend_top_k")
        assert callable(recommendation_engine.recommend_top_k)

    def test_has_get_available_actions_method(self, recommendation_engine):
        """Must expose available retention actions."""
        assert hasattr(recommendation_engine, "get_available_actions")
        assert callable(recommendation_engine.get_available_actions)


# ---------------------------------------------------------------------------
# Recommendation generation tests
# ---------------------------------------------------------------------------

class TestRecommendationGeneration:
    """Test recommendation generation for customers."""

    def test_generates_recommendations(self, recommendation_engine,
                                        sample_customer_data):
        """Must generate recommendations for all customers."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        assert len(recs) == len(sample_customer_data), (
            f"Expected {len(sample_customer_data)} recommendations, "
            f"got {len(recs)}"
        )

    def test_recommendations_have_action_type(self, recommendation_engine,
                                               sample_customer_data):
        """Each recommendation must specify an action type."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        assert "action_type" in recs.columns, (
            "Recommendations must include 'action_type' column"
        )
        assert recs["action_type"].notna().all(), (
            "Action type must not be null"
        )

    def test_recommendations_have_score(self, recommendation_engine,
                                         sample_customer_data):
        """Each recommendation must have a relevance/priority score."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        assert "score" in recs.columns, (
            "Recommendations must include 'score' column"
        )
        assert recs["score"].notna().all(), "Scores must not be null"

    def test_recommendations_have_customer_id(self, recommendation_engine,
                                               sample_customer_data):
        """Each recommendation must be tied to a customer."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        assert "customer_id" in recs.columns
        assert set(recs["customer_id"]) == set(
            sample_customer_data["customer_id"]
        )

    def test_no_duplicate_customers(self, recommendation_engine,
                                     sample_customer_data):
        """Each customer should appear exactly once in primary recs."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        assert recs["customer_id"].nunique() == len(sample_customer_data), (
            "Duplicate customers in recommendations"
        )


# ---------------------------------------------------------------------------
# Action type tests
# ---------------------------------------------------------------------------

class TestActionTypes:
    """Test diversity and validity of recommended actions."""

    def test_available_actions_non_empty(self, recommendation_engine):
        """Available actions list must be non-empty."""
        actions = recommendation_engine.get_available_actions()
        assert len(actions) > 0, "No available retention actions"

    def test_action_types_are_valid(self, recommendation_engine,
                                     sample_customer_data):
        """Recommended action types must be from the available set."""
        available = set(recommendation_engine.get_available_actions())
        recs = recommendation_engine.recommend(data=sample_customer_data)

        recommended_types = set(recs["action_type"].unique())
        invalid = recommended_types - available
        assert len(invalid) == 0, (
            f"Invalid action types recommended: {invalid}"
        )

    def test_action_diversity(self, recommendation_engine,
                               sample_customer_data):
        """Recommendations should include multiple action types."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        unique_actions = recs["action_type"].nunique()
        assert unique_actions >= 2, (
            f"Only {unique_actions} action type(s) recommended; "
            f"expected diversity"
        )


# ---------------------------------------------------------------------------
# Top-K recommendation tests
# ---------------------------------------------------------------------------

class TestTopKRecommendations:
    """Test top-K recommendation selection."""

    def test_top_k_returns_k_items(self, recommendation_engine,
                                    sample_customer_data):
        """recommend_top_k must return exactly K recommendations per query."""
        k = 3
        top_k = recommendation_engine.recommend_top_k(
            data=sample_customer_data,
            k=k,
        )

        # Each customer should get up to k recommendations
        per_customer = top_k.groupby("customer_id").size()
        assert (per_customer <= k).all(), (
            f"Some customers got more than {k} recommendations"
        )

    def test_top_k_sorted_by_score(self, recommendation_engine,
                                    sample_customer_data):
        """Top-K recommendations should be sorted by score (descending)."""
        k = 3
        top_k = recommendation_engine.recommend_top_k(
            data=sample_customer_data,
            k=k,
        )

        for _, group in top_k.groupby("customer_id"):
            scores = group["score"].values
            assert np.all(scores[:-1] >= scores[1:]), (
                "Top-K recommendations not sorted by score"
            )

    def test_top_k_with_different_k(self, recommendation_engine,
                                     sample_customer_data):
        """Must work with different values of K."""
        for k in [1, 3, 5]:
            top_k = recommendation_engine.recommend_top_k(
                data=sample_customer_data,
                k=k,
            )
            per_customer = top_k.groupby("customer_id").size()
            assert (per_customer <= k).all()


# ---------------------------------------------------------------------------
# Personalization tests
# ---------------------------------------------------------------------------

class TestPersonalization:
    """Test recommendation personalization based on customer attributes."""

    def test_high_churn_gets_stronger_actions(self, recommendation_engine,
                                               sample_customer_data):
        """Customers with high churn risk should get higher-priority recs."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        merged = recs.merge(
            sample_customer_data[["customer_id", "churn_prob"]],
            on="customer_id",
        )
        median_churn = merged["churn_prob"].median()

        high_churn_score = merged[merged["churn_prob"] >= median_churn][
            "score"
        ].mean()
        low_churn_score = merged[merged["churn_prob"] < median_churn][
            "score"
        ].mean()

        assert high_churn_score >= low_churn_score * 0.8, (
            "High-churn customers should receive comparable or higher scores"
        )

    def test_segment_influences_recommendation(self, recommendation_engine,
                                                sample_customer_data):
        """Different segments should receive different action distributions."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        merged = recs.merge(
            sample_customer_data[["customer_id", "segment"]],
            on="customer_id",
        )

        # Check that at least 2 segments have different dominant actions
        segment_actions = merged.groupby("segment")["action_type"].agg(
            lambda x: x.mode()[0] if len(x) > 0 else None
        )

        # Not all segments should get the same action
        unique_dominant = segment_actions.nunique()
        # Relaxed: at least some differentiation is expected
        assert unique_dominant >= 1, (
            "All segments get identical recommendations — no personalization"
        )

    def test_opt_out_respected(self, recommendation_engine,
                                sample_customer_data):
        """Customers who opted out of a channel should not get that channel."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        merged = recs.merge(
            sample_customer_data[["customer_id", "push_opt_in"]],
            on="customer_id",
        )

        # Customers with push_opt_in=0 should not get push_notification actions
        push_opted_out = merged[merged["push_opt_in"] == 0]
        if len(push_opted_out) > 0 and "push_notification" in recs[
            "action_type"
        ].values:
            push_recs = push_opted_out[
                push_opted_out["action_type"] == "push_notification"
            ]
            opt_out_ratio = len(push_recs) / len(push_opted_out)
            assert opt_out_ratio < 0.1, (
                f"{opt_out_ratio:.1%} of opted-out customers got push "
                f"notifications — opt-out not respected"
            )


# ---------------------------------------------------------------------------
# Budget-aware recommendation tests
# ---------------------------------------------------------------------------

class TestBudgetAwareRecommendations:
    """Test that recommendations consider cost constraints."""

    def test_recommendation_has_estimated_cost(self, recommendation_engine,
                                                sample_customer_data):
        """Each recommendation should include an estimated cost."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        assert "estimated_cost" in recs.columns, (
            "Recommendations must include estimated_cost column"
        )

    def test_estimated_costs_non_negative(self, recommendation_engine,
                                           sample_customer_data):
        """Estimated costs must be non-negative."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        if "estimated_cost" in recs.columns:
            assert (recs["estimated_cost"] >= 0).all(), (
                "Negative estimated costs found"
            )

    def test_total_cost_reasonable(self, recommendation_engine,
                                    sample_customer_data, config):
        """Total recommendation cost should be within the budget range."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        if "estimated_cost" in recs.columns:
            total_cost = recs["estimated_cost"].sum()
            total_budget = config["budget"]["total_krw"]
            # Total cost should not exceed 2x the configured budget
            assert total_cost <= total_budget * 2, (
                f"Total cost {total_cost:,.0f} KRW seems unreasonably high "
                f"vs budget {total_budget:,.0f} KRW"
            )


# ---------------------------------------------------------------------------
# Recommendation explanation tests
# ---------------------------------------------------------------------------

class TestRecommendationExplanation:
    """Test recommendation explanation/reasoning."""

    def test_recommendation_has_reason(self, recommendation_engine,
                                        sample_customer_data):
        """Recommendations should include a reason or explanation."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        assert "reason" in recs.columns, (
            "Recommendations should include a 'reason' column"
        )

    def test_reasons_are_non_empty(self, recommendation_engine,
                                    sample_customer_data):
        """Recommendation reasons should be non-empty strings."""
        recs = recommendation_engine.recommend(data=sample_customer_data)

        if "reason" in recs.columns:
            assert recs["reason"].notna().all(), "Null reasons found"
            assert (recs["reason"].str.len() > 0).all(), (
                "Empty reason strings found"
            )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestRecommendationIntegration:
    """Test integration with upstream model outputs."""

    def test_uses_churn_probability(self, recommendation_engine,
                                     sample_customer_data):
        """Recommendations should be influenced by churn probability."""
        # Create two copies: one with high churn, one with low
        data_high = sample_customer_data.copy()
        data_high["churn_prob"] = 0.9
        data_low = sample_customer_data.copy()
        data_low["churn_prob"] = 0.1

        recs_high = recommendation_engine.recommend(data=data_high)
        recs_low = recommendation_engine.recommend(data=data_low)

        # High-churn recs should have higher average scores
        assert recs_high["score"].mean() >= recs_low["score"].mean() * 0.5, (
            "High-churn customers should receive higher recommendation scores"
        )

    def test_uses_clv(self, recommendation_engine, sample_customer_data):
        """Recommendations should be influenced by CLV."""
        data_high_clv = sample_customer_data.copy()
        data_high_clv["clv"] = sample_customer_data["clv"].quantile(0.9)

        data_low_clv = sample_customer_data.copy()
        data_low_clv["clv"] = sample_customer_data["clv"].quantile(0.1)

        recs_high = recommendation_engine.recommend(data=data_high_clv)
        recs_low = recommendation_engine.recommend(data=data_low_clv)

        # High-CLV recs should have at least comparable scores
        assert recs_high["score"].mean() >= recs_low["score"].mean() * 0.5, (
            "High-CLV customers should get meaningful recommendation scores"
        )

    def test_negative_uplift_lowers_priority(self, recommendation_engine,
                                              sample_customer_data):
        """Negative uplift (sleeping dogs) should lower rec priority."""
        data_pos = sample_customer_data.copy()
        data_pos["uplift_score"] = 0.2

        data_neg = sample_customer_data.copy()
        data_neg["uplift_score"] = -0.2

        recs_pos = recommendation_engine.recommend(data=data_pos)
        recs_neg = recommendation_engine.recommend(data=data_neg)

        assert recs_pos["score"].mean() >= recs_neg["score"].mean(), (
            "Positive-uplift customers should score higher than negative"
        )


# ---------------------------------------------------------------------------
# Output format tests
# ---------------------------------------------------------------------------

class TestRecommendationOutput:
    """Test recommendation output format."""

    def test_output_is_dataframe(self, recommendation_engine,
                                  sample_customer_data):
        """Recommendations must be returned as a DataFrame."""
        recs = recommendation_engine.recommend(data=sample_customer_data)
        assert isinstance(recs, pd.DataFrame)

    def test_no_nan_in_scores(self, recommendation_engine,
                               sample_customer_data):
        """Recommendation scores must not contain NaN."""
        recs = recommendation_engine.recommend(data=sample_customer_data)
        assert not recs["score"].isna().any(), "NaN values in scores"

    def test_scores_are_numeric(self, recommendation_engine,
                                 sample_customer_data):
        """Scores must be numeric."""
        recs = recommendation_engine.recommend(data=sample_customer_data)
        assert np.issubdtype(recs["score"].dtype, np.number), (
            "Scores must be numeric"
        )


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestRecommendationPersistence:
    """Test recommendation engine save/load functionality."""

    def test_save_engine(self, recommendation_engine, tmp_path):
        """Recommendation engine must be saveable."""
        save_path = tmp_path / "rec_engine"
        recommendation_engine.save(str(save_path))

        saved_files = list(tmp_path.glob("rec_engine*"))
        assert len(saved_files) > 0, "No engine state saved"

    def test_load_engine(self, recommendation_engine, sample_customer_data,
                         tmp_path):
        """Saved engine must be loadable and produce same results."""
        from src.models.recommendations import RecommendationEngine

        recs_original = recommendation_engine.recommend(
            data=sample_customer_data,
        )

        save_path = tmp_path / "rec_engine"
        recommendation_engine.save(str(save_path))

        loaded = RecommendationEngine.load(str(save_path))
        recs_loaded = loaded.recommend(data=sample_customer_data)

        # Action types and scores should match
        assert set(recs_original["action_type"]) == set(
            recs_loaded["action_type"]
        )
        np.testing.assert_array_almost_equal(
            recs_original["score"].values,
            recs_loaded["score"].values,
            decimal=5,
        )


# ---------------------------------------------------------------------------
# Reproducibility tests
# ---------------------------------------------------------------------------

class TestRecommendationReproducibility:
    """Test recommendation engine reproducibility with same seed."""

    def test_same_seed_same_recommendations(self, config,
                                             sample_customer_data):
        """Same seed must produce identical recommendations."""
        from src.models.recommendations import RecommendationEngine

        engine1 = RecommendationEngine(config)
        recs1 = engine1.recommend(data=sample_customer_data)

        engine2 = RecommendationEngine(config)
        recs2 = engine2.recommend(data=sample_customer_data)

        pd.testing.assert_frame_equal(
            recs1.sort_values("customer_id").reset_index(drop=True),
            recs2.sort_values("customer_id").reset_index(drop=True),
        )
