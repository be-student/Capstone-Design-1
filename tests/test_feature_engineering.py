"""
TDD Tests for Feature Engineering Module.

Tests cover:
- RFM feature computation (Recency, Frequency, Monetary)
- Behavioral change rate features (5+ features)
- Purchase cycle anomaly feature
- Session quality features (3+ features)
- Sequence features (2+ features)
- Time-based behavior features (weekend/weekday, time-of-day)
- Customer journey stage features
- Missing/outlier handling in features
- Feature store save/load (file-based)
- Feature count >= 30
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_customers():
    """Create sample customer data for feature engineering tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "persona": np.random.choice(
            ["vip_loyal", "regular_loyal", "price_sensitive",
             "explorer", "churning", "new_user"],
            n
        ),
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "treatment_group": np.random.choice(["treatment", "control"], n),
        "churn_label": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })


@pytest.fixture
def sample_events(sample_customers):
    """Create sample events for feature engineering tests."""
    np.random.seed(42)
    event_types = [
        "page_view", "search", "add_to_cart", "remove_from_cart",
        "purchase", "coupon_use", "review", "cs_contact"
    ]
    events = []
    for _, cust in sample_customers.iterrows():
        n_events = np.random.randint(20, 200)
        base_date = pd.Timestamp(cust["signup_date"])
        for _ in range(n_events):
            day_offset = np.random.randint(0, 180)
            hour = int(np.random.choice(range(24)))
            event_date = base_date + timedelta(days=int(day_offset), hours=hour)
            event_type = np.random.choice(
                event_types, p=[0.3, 0.15, 0.12, 0.05, 0.15, 0.08, 0.05, 0.10]
            )
            amount = (
                float(np.random.lognormal(10, 0.5))
                if event_type == "purchase" else 0.0
            )
            session_duration = np.random.exponential(300)  # seconds
            events.append({
                "customer_id": cust["customer_id"],
                "event_type": event_type,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "timestamp": event_date.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": amount,
                "session_duration": session_duration,
            })
    return pd.DataFrame(events)


@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance."""
    from src.features.feature_engineering import FeatureEngineer
    return FeatureEngineer()


@pytest.fixture
def computed_features(feature_engineer, sample_customers, sample_events):
    """Compute all features for testing."""
    return feature_engineer.compute_all_features(
        sample_customers, sample_events,
        reference_date="2024-07-01"
    )


class TestRFMFeatures:
    """Test Recency, Frequency, Monetary feature computation."""

    def test_rfm_features_computed(self, feature_engineer, sample_customers,
                                   sample_events):
        """RFM features must be computed for each customer."""
        rfm = feature_engineer.compute_rfm(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert isinstance(rfm, pd.DataFrame)
        assert "recency" in rfm.columns
        assert "frequency" in rfm.columns
        assert "monetary" in rfm.columns

    def test_recency_non_negative(self, feature_engineer, sample_customers,
                                   sample_events):
        """Recency (days since last purchase) must be non-negative."""
        rfm = feature_engineer.compute_rfm(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert (rfm["recency"] >= 0).all(), "Negative recency values found"

    def test_frequency_non_negative(self, feature_engineer, sample_customers,
                                     sample_events):
        """Frequency (purchase count) must be non-negative integer."""
        rfm = feature_engineer.compute_rfm(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert (rfm["frequency"] >= 0).all(), "Negative frequency values found"

    def test_monetary_non_negative(self, feature_engineer, sample_customers,
                                    sample_events):
        """Monetary (total spend) must be non-negative."""
        rfm = feature_engineer.compute_rfm(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert (rfm["monetary"] >= 0).all(), "Negative monetary values found"

    def test_rfm_covers_all_customers(self, feature_engineer, sample_customers,
                                       sample_events):
        """RFM features must exist for every customer."""
        rfm = feature_engineer.compute_rfm(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert len(rfm) == len(sample_customers), (
            f"RFM has {len(rfm)} rows, expected {len(sample_customers)}"
        )


class TestBehavioralChangeFeatures:
    """Test behavioral change rate features (min 5 required)."""

    def test_change_features_computed(self, feature_engineer, sample_customers,
                                      sample_events):
        """Must compute at least 5 behavioral change features."""
        change_features = feature_engineer.compute_behavioral_changes(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert isinstance(change_features, pd.DataFrame)
        assert len(change_features.columns) >= 5, (
            f"Expected >= 5 change features, got {len(change_features.columns)}"
        )

    def test_visit_frequency_change(self, feature_engineer, sample_customers,
                                     sample_events):
        """Visit frequency change ratio must be computed."""
        change_features = feature_engineer.compute_behavioral_changes(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert "visit_frequency_change" in change_features.columns

    def test_purchase_cycle_change(self, feature_engineer, sample_customers,
                                    sample_events):
        """Purchase cycle change rate must be computed."""
        change_features = feature_engineer.compute_behavioral_changes(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert "purchase_cycle_change" in change_features.columns

    def test_session_duration_change(self, feature_engineer, sample_customers,
                                      sample_events):
        """Session duration change must be computed."""
        change_features = feature_engineer.compute_behavioral_changes(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert "session_duration_change" in change_features.columns

    def test_cart_conversion_change(self, feature_engineer, sample_customers,
                                     sample_events):
        """Cart conversion rate change must be computed."""
        change_features = feature_engineer.compute_behavioral_changes(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert "cart_conversion_change" in change_features.columns

    def test_coupon_response_change(self, feature_engineer, sample_customers,
                                     sample_events):
        """Coupon response rate change must be computed."""
        change_features = feature_engineer.compute_behavioral_changes(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert "coupon_response_change" in change_features.columns


class TestPurchaseCycleAnomaly:
    """Test purchase cycle anomaly feature."""

    def test_purchase_cycle_anomaly_computed(self, feature_engineer,
                                             sample_customers, sample_events):
        """Purchase cycle anomaly = current_no_purchase_days / avg_purchase_cycle."""
        anomaly = feature_engineer.compute_purchase_cycle_anomaly(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert isinstance(anomaly, pd.DataFrame)
        assert "purchase_cycle_anomaly" in anomaly.columns

    def test_anomaly_non_negative(self, feature_engineer, sample_customers,
                                   sample_events):
        """Anomaly score must be non-negative."""
        anomaly = feature_engineer.compute_purchase_cycle_anomaly(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        assert (anomaly["purchase_cycle_anomaly"] >= 0).all()

    def test_high_anomaly_for_overdue_customers(self, feature_engineer,
                                                 sample_customers,
                                                 sample_events):
        """Customers overdue for purchase should have anomaly > 1."""
        anomaly = feature_engineer.compute_purchase_cycle_anomaly(
            sample_customers, sample_events,
            reference_date="2024-07-01"
        )
        # At least some customers should have high anomaly scores
        assert (anomaly["purchase_cycle_anomaly"] > 1.0).any(), (
            "No customers with high purchase cycle anomaly"
        )


class TestSessionQualityFeatures:
    """Test session quality features (min 3 required)."""

    def test_session_features_computed(self, feature_engineer, sample_customers,
                                       sample_events):
        """Must compute at least 3 session quality features."""
        session_features = feature_engineer.compute_session_quality(
            sample_customers, sample_events
        )
        assert isinstance(session_features, pd.DataFrame)
        assert len(session_features.columns) >= 3, (
            f"Expected >= 3 session features, got "
            f"{len(session_features.columns)}"
        )

    def test_avg_session_duration(self, feature_engineer, sample_customers,
                                   sample_events):
        """Average session duration must be computed."""
        session_features = feature_engineer.compute_session_quality(
            sample_customers, sample_events
        )
        assert "avg_session_duration" in session_features.columns

    def test_pageviews_per_session(self, feature_engineer, sample_customers,
                                    sample_events):
        """Page views per session must be computed."""
        session_features = feature_engineer.compute_session_quality(
            sample_customers, sample_events
        )
        assert "pageviews_per_session" in session_features.columns

    def test_search_to_purchase_rate(self, feature_engineer, sample_customers,
                                      sample_events):
        """Search-to-purchase conversion rate must be computed."""
        session_features = feature_engineer.compute_session_quality(
            sample_customers, sample_events
        )
        assert "search_to_purchase_rate" in session_features.columns


class TestSequenceFeatures:
    """Test sequence-based features (min 2 required)."""

    def test_sequence_features_computed(self, feature_engineer, sample_customers,
                                        sample_events):
        """Must compute at least 2 sequence features."""
        seq_features = feature_engineer.compute_sequence_features(
            sample_customers, sample_events
        )
        assert isinstance(seq_features, pd.DataFrame)
        assert len(seq_features.columns) >= 2, (
            f"Expected >= 2 sequence features, got "
            f"{len(seq_features.columns)}"
        )

    def test_event_sequence_embedding(self, feature_engineer, sample_customers,
                                       sample_events):
        """Event sequence embedding feature must be computed."""
        seq_features = feature_engineer.compute_sequence_features(
            sample_customers, sample_events
        )
        embedding_cols = [
            c for c in seq_features.columns if "embedding" in c.lower()
            or "sequence" in c.lower()
        ]
        assert len(embedding_cols) >= 1, (
            "No sequence embedding feature found"
        )

    def test_behavior_pattern_cluster(self, feature_engineer, sample_customers,
                                       sample_events):
        """Behavior pattern cluster ID must be computed."""
        seq_features = feature_engineer.compute_sequence_features(
            sample_customers, sample_events
        )
        cluster_cols = [
            c for c in seq_features.columns if "cluster" in c.lower()
            or "pattern" in c.lower()
        ]
        assert len(cluster_cols) >= 1, (
            "No behavior pattern cluster feature found"
        )


class TestTimeBasedFeatures:
    """Test time-based behavior features."""

    def test_time_features_computed(self, feature_engineer, sample_customers,
                                     sample_events):
        """Must compute time-based features."""
        time_features = feature_engineer.compute_time_features(
            sample_customers, sample_events
        )
        assert isinstance(time_features, pd.DataFrame)

    def test_weekend_weekday_ratio(self, feature_engineer, sample_customers,
                                    sample_events):
        """Weekend/weekday purchase ratio must be computed."""
        time_features = feature_engineer.compute_time_features(
            sample_customers, sample_events
        )
        assert "weekend_purchase_ratio" in time_features.columns

    def test_time_of_day_features(self, feature_engineer, sample_customers,
                                   sample_events):
        """Time-of-day activity ratio must be computed."""
        time_features = feature_engineer.compute_time_features(
            sample_customers, sample_events
        )
        time_cols = [
            c for c in time_features.columns if "hour" in c.lower()
            or "time_of_day" in c.lower() or "morning" in c.lower()
            or "evening" in c.lower() or "night" in c.lower()
        ]
        assert len(time_cols) >= 1, "No time-of-day features found"


class TestJourneyStageFeatures:
    """Test customer journey stage features."""

    def test_journey_stage_computed(self, feature_engineer, sample_customers,
                                    sample_events):
        """Customer journey stage must be computed."""
        journey_features = feature_engineer.compute_journey_features(
            sample_customers, sample_events
        )
        assert isinstance(journey_features, pd.DataFrame)
        assert "journey_stage" in journey_features.columns

    def test_journey_stages_valid(self, feature_engineer, sample_customers,
                                   sample_events):
        """Journey stages must be valid stage names or IDs."""
        journey_features = feature_engineer.compute_journey_features(
            sample_customers, sample_events
        )
        stages = journey_features["journey_stage"].unique()
        assert len(stages) >= 2, (
            f"Expected multiple journey stages, got {len(stages)}"
        )

    def test_stage_tenure_computed(self, feature_engineer, sample_customers,
                                    sample_events):
        """Stage tenure (days in current stage) must be computed."""
        journey_features = feature_engineer.compute_journey_features(
            sample_customers, sample_events
        )
        assert "stage_tenure_days" in journey_features.columns
        assert (journey_features["stage_tenure_days"] >= 0).all()


class TestMissingAndOutlierHandling:
    """Test that computed features handle missing values and outliers."""

    def test_no_nan_in_final_features(self, computed_features):
        """Final feature matrix must have no NaN values."""
        assert computed_features.isna().sum().sum() == 0, (
            f"Found NaN values in features:\n"
            f"{computed_features.isna().sum()[computed_features.isna().sum() > 0]}"
        )

    def test_no_inf_in_final_features(self, computed_features):
        """Final feature matrix must have no infinite values."""
        numeric_cols = computed_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(computed_features[col]).any(), (
                f"Infinite values found in feature: {col}"
            )


class TestFeatureStore:
    """Test file-based feature store save/load."""

    def test_save_features(self, feature_engineer, computed_features, tmp_path):
        """Features must be saveable to file-based store."""
        store_path = tmp_path / "feature_store"
        feature_engineer.save_to_feature_store(
            computed_features, str(store_path)
        )
        assert (store_path / "features.parquet").exists() or \
               (store_path / "features.csv").exists()

    def test_load_features(self, feature_engineer, computed_features, tmp_path):
        """Features must be loadable from file-based store."""
        store_path = tmp_path / "feature_store"
        feature_engineer.save_to_feature_store(
            computed_features, str(store_path)
        )

        loaded = feature_engineer.load_from_feature_store(str(store_path))
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(computed_features)
        assert set(loaded.columns) == set(computed_features.columns)


class TestFeatureCount:
    """Test total feature count meets requirements."""

    def test_minimum_30_features(self, computed_features):
        """Final feature set must include at least 30 features."""
        # Exclude customer_id and target columns
        feature_cols = [
            c for c in computed_features.columns
            if c not in ("customer_id", "churn_label", "persona",
                         "treatment_group", "signup_date")
        ]
        assert len(feature_cols) >= 30, (
            f"Expected >= 30 features, got {len(feature_cols)}: "
            f"{feature_cols}"
        )

    def test_all_feature_groups_present(self, computed_features):
        """All required feature groups must be represented."""
        cols = set(computed_features.columns)

        # RFM
        assert "recency" in cols, "Missing RFM: recency"
        assert "frequency" in cols, "Missing RFM: frequency"
        assert "monetary" in cols, "Missing RFM: monetary"

        # Behavioral change (at least one)
        change_cols = [c for c in cols if "change" in c.lower()]
        assert len(change_cols) >= 1, "Missing behavioral change features"

        # Session quality (at least one)
        session_cols = [c for c in cols if "session" in c.lower()
                        or "pageview" in c.lower()]
        assert len(session_cols) >= 1, "Missing session quality features"

        # Journey
        assert "journey_stage" in cols or any(
            "journey" in c.lower() for c in cols
        ), "Missing journey features"
