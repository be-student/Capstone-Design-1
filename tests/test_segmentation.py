"""
Tests for Customer Segmentation Module (TDD).

Tests cover:
- 6+ customer segments with configurable thresholds
- RFM-based segmentation with quantile scoring
- K-Means based segmentation
- Segment assignment correctness
- Configuration-driven thresholds
- Segment statistics and summary
- Edge cases (empty data, single customer)
"""

import numpy as np
import pandas as pd
import pytest
import yaml
import os

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def segmentation_config():
    """Load segmentation config from YAML."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "simulator_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("segmentation", {})


@pytest.fixture
def sample_rfm_data():
    """Create sample RFM feature data for segmentation testing."""
    np.random.seed(42)
    n = 200

    data = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "recency": np.random.exponential(30, n).clip(1, 365),
        "frequency": np.random.poisson(5, n).clip(0, 50),
        "monetary": np.random.lognormal(10, 1.5, n).clip(0, 5000000),
        "avg_order_value": np.random.lognormal(9, 1.0, n).clip(0, 500000),
        "churn_probability": np.random.beta(2, 5, n),
    })
    return data


@pytest.fixture
def sample_features_with_behavioral(sample_rfm_data):
    """RFM data augmented with behavioral features."""
    n = len(sample_rfm_data)
    np.random.seed(42)
    df = sample_rfm_data.copy()
    df["visit_frequency_change"] = np.random.normal(0, 0.3, n)
    df["purchase_cycle_change"] = np.random.normal(0, 0.3, n)
    df["session_duration_change"] = np.random.normal(0, 0.2, n)
    df["cart_conversion_change"] = np.random.normal(0, 0.2, n)
    df["tenure_days"] = np.random.randint(10, 365, n)
    df["journey_stage"] = np.random.choice([0, 1, 2, 3, 4], n)
    return df


@pytest.fixture
def segmenter(segmentation_config):
    """Create a CustomerSegmenter with config."""
    from src.features.segmentation import CustomerSegmenter
    return CustomerSegmenter(config=segmentation_config)


@pytest.fixture
def segmenter_default():
    """Create a CustomerSegmenter with default config."""
    from src.features.segmentation import CustomerSegmenter
    return CustomerSegmenter()


# ---------------------------------------------------------------------------
# Test: Module Imports
# ---------------------------------------------------------------------------

class TestSegmentationImports:
    """Verify segmentation module is importable."""

    def test_import_customer_segmenter(self):
        """CustomerSegmenter class must be importable."""
        from src.features.segmentation import CustomerSegmenter
        assert CustomerSegmenter is not None

    def test_import_from_features_init(self):
        """CustomerSegmenter must be exported from features package."""
        from src.features import CustomerSegmenter
        assert CustomerSegmenter is not None


# ---------------------------------------------------------------------------
# Test: Configuration
# ---------------------------------------------------------------------------

class TestSegmentationConfig:
    """Test configuration-driven segmentation."""

    def test_config_has_segments(self, segmentation_config):
        """Config must define segment list."""
        assert "segments" in segmentation_config
        assert len(segmentation_config["segments"]) >= 6

    def test_config_segment_structure(self, segmentation_config):
        """Each segment must have name, criteria, and retention_action."""
        for seg in segmentation_config["segments"]:
            assert "name" in seg, f"Segment missing 'name': {seg}"
            assert "criteria" in seg, f"Segment {seg['name']} missing 'criteria'"
            assert "retention_action" in seg, (
                f"Segment {seg['name']} missing 'retention_action'"
            )

    def test_config_segment_names_unique(self, segmentation_config):
        """Segment names must be unique."""
        names = [s["name"] for s in segmentation_config["segments"]]
        assert len(names) == len(set(names)), "Duplicate segment names found"

    def test_custom_config_overrides(self):
        """Custom config should override defaults."""
        from src.features.segmentation import CustomerSegmenter

        custom_config = {
            "method": "kmeans",
            "kmeans_clusters": 8,
            "n_rfm_bins": 4,
            "segments": [
                {
                    "name": "custom_seg",
                    "criteria": {"recency_score_min": 3},
                    "retention_action": "custom_action",
                }
            ],
        }
        seg = CustomerSegmenter(config=custom_config)
        assert seg.config["method"] == "kmeans"
        assert seg.config["kmeans_clusters"] == 8


# ---------------------------------------------------------------------------
# Test: RFM Scoring
# ---------------------------------------------------------------------------

class TestRFMScoring:
    """Test RFM quantile scoring."""

    def test_compute_rfm_scores(self, segmenter, sample_rfm_data):
        """Must compute R, F, M scores in 1-5 range."""
        scored = segmenter.compute_rfm_scores(sample_rfm_data)

        assert "recency_score" in scored.columns
        assert "frequency_score" in scored.columns
        assert "monetary_score" in scored.columns

        for col in ["recency_score", "frequency_score", "monetary_score"]:
            assert scored[col].min() >= 1, f"{col} min below 1"
            assert scored[col].max() <= 5, f"{col} max above 5"

    def test_recency_inverse_scoring(self, segmenter, sample_rfm_data):
        """Lower recency (more recent) should get higher score."""
        scored = segmenter.compute_rfm_scores(sample_rfm_data)

        # Customers with lowest recency should have highest recency_score
        low_recency = scored.nsmallest(20, "recency")
        high_recency = scored.nlargest(20, "recency")

        assert low_recency["recency_score"].mean() > high_recency[
            "recency_score"
        ].mean(), "Recency scoring should be inverse"

    def test_frequency_direct_scoring(self, segmenter, sample_rfm_data):
        """Higher frequency should get higher score."""
        scored = segmenter.compute_rfm_scores(sample_rfm_data)

        high_freq = scored.nlargest(20, "frequency")
        low_freq = scored.nsmallest(20, "frequency")

        assert high_freq["frequency_score"].mean() > low_freq[
            "frequency_score"
        ].mean(), "Frequency scoring should be direct"

    def test_rfm_scores_configurable_bins(self, sample_rfm_data):
        """Number of RFM bins should be configurable."""
        from src.features.segmentation import CustomerSegmenter

        seg3 = CustomerSegmenter(config={"n_rfm_bins": 3})
        scored3 = seg3.compute_rfm_scores(sample_rfm_data)
        assert scored3["recency_score"].max() <= 3

        seg10 = CustomerSegmenter(config={"n_rfm_bins": 10})
        scored10 = seg10.compute_rfm_scores(sample_rfm_data)
        assert scored10["recency_score"].max() <= 10


# ---------------------------------------------------------------------------
# Test: Segment Assignment
# ---------------------------------------------------------------------------

class TestSegmentAssignment:
    """Test segment assignment logic."""

    def test_segment_customers_returns_all(
        self, segmenter, sample_rfm_data
    ):
        """Every customer must be assigned exactly one segment."""
        result = segmenter.segment_customers(sample_rfm_data)

        assert "segment" in result.columns
        assert len(result) == len(sample_rfm_data)
        assert result["segment"].isna().sum() == 0, (
            "Some customers have no segment"
        )

    def test_at_least_six_segments(self, segmenter, sample_rfm_data):
        """Must produce at least 6 distinct segments."""
        result = segmenter.segment_customers(sample_rfm_data)
        unique_segments = result["segment"].nunique()
        assert unique_segments >= 6, (
            f"Expected >= 6 segments, got {unique_segments}: "
            f"{result['segment'].unique()}"
        )

    def test_segment_names_match_config(
        self, segmenter, segmentation_config, sample_rfm_data
    ):
        """Assigned segments must be from the configured segment list."""
        result = segmenter.segment_customers(sample_rfm_data)
        config_names = {s["name"] for s in segmentation_config["segments"]}
        # Allow an "other" fallback segment
        config_names.add("other")
        assigned = set(result["segment"].unique())
        assert assigned.issubset(config_names), (
            f"Unknown segments: {assigned - config_names}"
        )

    def test_vip_loyal_has_high_rfm(self, segmenter, sample_rfm_data):
        """VIP loyal segment should have high RFM scores."""
        result = segmenter.segment_customers(sample_rfm_data)
        scored = segmenter.compute_rfm_scores(sample_rfm_data)
        merged = result.merge(scored, on="customer_id")

        vip = merged[merged["segment"] == "vip_loyal"]
        if len(vip) > 0:
            assert vip["recency_score"].mean() >= 3.5
            assert vip["frequency_score"].mean() >= 3.5
            assert vip["monetary_score"].mean() >= 3.5

    def test_at_risk_has_low_recency_score(
        self, segmenter, sample_rfm_data
    ):
        """At-risk segment should have low recency scores."""
        result = segmenter.segment_customers(sample_rfm_data)
        scored = segmenter.compute_rfm_scores(sample_rfm_data)
        merged = result.merge(scored, on="customer_id")

        at_risk = merged[merged["segment"] == "at_risk"]
        if len(at_risk) > 0:
            assert at_risk["recency_score"].mean() <= 2.5


# ---------------------------------------------------------------------------
# Test: K-Means Segmentation
# ---------------------------------------------------------------------------

class TestKMeansSegmentation:
    """Test K-Means based segmentation method."""

    def test_kmeans_method(self, sample_rfm_data):
        """K-Means method must produce correct number of clusters."""
        from src.features.segmentation import CustomerSegmenter

        seg = CustomerSegmenter(
            config={"method": "kmeans", "kmeans_clusters": 6}
        )
        result = seg.segment_customers(sample_rfm_data)

        assert "segment" in result.columns
        assert len(result) == len(sample_rfm_data)
        # K-Means should produce up to kmeans_clusters clusters
        assert result["segment"].nunique() <= 6

    def test_kmeans_assigns_all(self, sample_rfm_data):
        """Every customer must have a segment in kmeans mode."""
        from src.features.segmentation import CustomerSegmenter

        seg = CustomerSegmenter(
            config={"method": "kmeans", "kmeans_clusters": 6}
        )
        result = seg.segment_customers(sample_rfm_data)
        assert result["segment"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Test: Segment Summary Statistics
# ---------------------------------------------------------------------------

class TestSegmentSummary:
    """Test segment summary/statistics generation."""

    def test_get_segment_summary(self, segmenter, sample_rfm_data):
        """Must produce summary stats per segment."""
        result = segmenter.segment_customers(sample_rfm_data)
        summary = segmenter.get_segment_summary(result)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) >= 6
        assert "segment" in summary.columns
        assert "count" in summary.columns
        assert "percentage" in summary.columns

    def test_summary_counts_sum_to_total(
        self, segmenter, sample_rfm_data
    ):
        """Summary counts must sum to total customers."""
        result = segmenter.segment_customers(sample_rfm_data)
        summary = segmenter.get_segment_summary(result)

        total = summary["count"].sum()
        assert total == len(sample_rfm_data), (
            f"Summary count {total} != data size {len(sample_rfm_data)}"
        )

    def test_summary_percentages_sum_to_100(
        self, segmenter, sample_rfm_data
    ):
        """Summary percentages must sum to ~100%."""
        result = segmenter.segment_customers(sample_rfm_data)
        summary = segmenter.get_segment_summary(result)

        pct_sum = summary["percentage"].sum()
        assert abs(pct_sum - 100.0) < 0.1, (
            f"Percentage sum {pct_sum} not close to 100"
        )

    def test_summary_includes_rfm_means(
        self, segmenter, sample_rfm_data
    ):
        """Summary should include mean RFM values per segment."""
        result = segmenter.segment_customers(sample_rfm_data)
        summary = segmenter.get_segment_summary(result)

        for col in ["avg_recency", "avg_frequency", "avg_monetary"]:
            assert col in summary.columns, f"Missing {col} in summary"


class TestValueUpliftSegmentation:
    """Test 6+ operational segmentation using churn/uplift/CLV."""

    def test_segment_value_uplift_customers(self, segmenter, sample_rfm_data):
        df = sample_rfm_data.copy()
        df["uplift_score"] = np.linspace(-0.2, 0.3, len(df))
        df["predicted_clv"] = np.linspace(10_000, 500_000, len(df))
        df["tenure_days"] = np.where(np.arange(len(df)) < 10, 10, 90)

        result = segmenter.segment_value_uplift_customers(df)
        assert "priority_score" in result.columns
        assert "segment" in result.columns
        assert result["segment"].nunique() >= 6

    def test_value_uplift_summary_includes_new_metrics(self, segmenter, sample_rfm_data):
        df = sample_rfm_data.copy()
        df["uplift_score"] = np.random.normal(0, 0.1, len(df))
        df["predicted_clv"] = np.random.lognormal(11, 0.7, len(df))
        df["tenure_days"] = 120

        segmented = segmenter.segment_value_uplift_customers(df)
        summary = segmenter.get_segment_summary(segmented)
        assert "avg_uplift_score" in summary.columns
        assert "avg_predicted_clv" in summary.columns
        assert "avg_priority_score" in summary.columns


# ---------------------------------------------------------------------------
# Test: Retention Actions
# ---------------------------------------------------------------------------

class TestRetentionActions:
    """Test retention action recommendations per segment."""

    def test_get_retention_actions(self, segmenter, sample_rfm_data):
        """Must assign retention actions to all customers."""
        result = segmenter.segment_customers(sample_rfm_data)
        actions = segmenter.get_retention_actions(result)

        assert "retention_action" in actions.columns
        assert len(actions) == len(sample_rfm_data)
        assert actions["retention_action"].isna().sum() == 0

    def test_actions_match_segments(
        self, segmenter, segmentation_config, sample_rfm_data
    ):
        """Retention actions must match configured segment actions."""
        result = segmenter.segment_customers(sample_rfm_data)
        actions = segmenter.get_retention_actions(result)

        # Build expected mapping
        action_map = {
            s["name"]: s.get("retention_action", "general")
            for s in segmentation_config["segments"]
        }
        action_map["other"] = "general"

        for _, row in actions.iterrows():
            seg = row["segment"]
            expected = action_map.get(seg, "general")
            assert row["retention_action"] == expected, (
                f"Segment {seg}: expected action {expected}, "
                f"got {row['retention_action']}"
            )


# ---------------------------------------------------------------------------
# Test: Edge Cases
# ---------------------------------------------------------------------------

class TestSegmentationEdgeCases:
    """Test edge cases for segmentation."""

    def test_single_customer(self, segmenter):
        """Must handle single customer."""
        df = pd.DataFrame({
            "customer_id": ["C0001"],
            "recency": [10],
            "frequency": [5],
            "monetary": [100000],
        })
        result = segmenter.segment_customers(df)
        assert len(result) == 1
        assert result["segment"].isna().sum() == 0

    def test_zero_values(self, segmenter):
        """Must handle customers with zero RFM values."""
        df = pd.DataFrame({
            "customer_id": ["C0001", "C0002"],
            "recency": [0, 365],
            "frequency": [0, 0],
            "monetary": [0, 0],
        })
        result = segmenter.segment_customers(df)
        assert len(result) == 2
        assert result["segment"].isna().sum() == 0

    def test_all_identical_customers(self, segmenter):
        """Must handle all-identical customer data gracefully."""
        n = 50
        df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(n)],
            "recency": [30] * n,
            "frequency": [5] * n,
            "monetary": [50000] * n,
        })
        result = segmenter.segment_customers(df)
        assert len(result) == n
        assert result["segment"].isna().sum() == 0

    def test_reproducibility_with_seed(self, sample_rfm_data):
        """Same data + config should produce same segments."""
        from src.features.segmentation import CustomerSegmenter

        seg1 = CustomerSegmenter(
            config={"method": "kmeans", "kmeans_clusters": 6}
        )
        seg2 = CustomerSegmenter(
            config={"method": "kmeans", "kmeans_clusters": 6}
        )

        r1 = seg1.segment_customers(sample_rfm_data)
        r2 = seg2.segment_customers(sample_rfm_data)

        pd.testing.assert_frame_equal(r1, r2)
