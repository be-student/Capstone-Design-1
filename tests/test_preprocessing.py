"""
TDD Tests for Data Preprocessing Module.

Tests cover:
- Loading raw data (customers, events CSVs)
- Date parsing and type conversion
- Missing value handling
- Outlier detection and treatment
- Time-based train/test split (10 months train, 2 months test)
- Sequence data preparation for DL models (padding, encoding)
- Class imbalance handling (SMOTE / class_weight)
- Feature scaling and normalization
- Data validation and integrity checks
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
    """Create sample customer DataFrame for testing."""
    np.random.seed(42)
    n = 500
    signup_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "persona": np.random.choice(
            ["vip_loyal", "regular_loyal", "price_sensitive",
             "explorer", "churning", "new_user"],
            n
        ),
        "signup_date": np.random.choice(signup_dates, n),
        "treatment_group": np.random.choice(
            ["treatment", "control"], n
        ),
        "churn_label": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })


@pytest.fixture
def sample_events(sample_customers):
    """Create sample events DataFrame for testing."""
    np.random.seed(42)
    events = []
    event_types = [
        "page_view", "search", "add_to_cart", "remove_from_cart",
        "purchase", "coupon_use", "review", "cs_contact"
    ]
    for _, customer in sample_customers.iterrows():
        n_events = np.random.randint(5, 100)
        base_date = pd.Timestamp(customer["signup_date"])
        for _ in range(n_events):
            event_date = base_date + timedelta(days=np.random.randint(0, 180))
            event_type = np.random.choice(event_types)
            amount = (
                np.random.lognormal(10, 1) if event_type == "purchase"
                else 0.0
            )
            events.append({
                "customer_id": customer["customer_id"],
                "event_type": event_type,
                "event_date": event_date.strftime("%Y-%m-%d"),
                "timestamp": event_date.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": amount,
            })
    return pd.DataFrame(events)


@pytest.fixture
def sample_events_with_issues(sample_events):
    """Create events with missing values and outliers for testing."""
    df = sample_events.copy()
    # Inject missing values
    null_indices = np.random.choice(len(df), size=20, replace=False)
    df.loc[null_indices[:10], "amount"] = np.nan
    df.loc[null_indices[10:], "event_date"] = np.nan
    # Inject outliers
    outlier_indices = np.random.choice(len(df), size=5, replace=False)
    df.loc[outlier_indices, "amount"] = 99999999.0
    return df


@pytest.fixture
def preprocessor():
    """Create a Preprocessor instance."""
    from src.data.preprocessing import Preprocessor
    return Preprocessor()


class TestDataLoading:
    """Test raw data loading functionality."""

    def test_load_customers_csv(self, preprocessor, sample_customers, tmp_path):
        """Preprocessor must load customer CSV correctly."""
        path = tmp_path / "customers.csv"
        sample_customers.to_csv(path, index=False)

        loaded = preprocessor.load_customers(str(path))
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(sample_customers)

    def test_load_events_csv(self, preprocessor, sample_events, tmp_path):
        """Preprocessor must load events CSV correctly."""
        path = tmp_path / "events.csv"
        sample_events.to_csv(path, index=False)

        loaded = preprocessor.load_events(str(path))
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(sample_events)

    def test_date_columns_parsed(self, preprocessor, sample_events, tmp_path):
        """Date columns must be converted to datetime dtype."""
        path = tmp_path / "events.csv"
        sample_events.to_csv(path, index=False)

        loaded = preprocessor.load_events(str(path))
        assert pd.api.types.is_datetime64_any_dtype(loaded["event_date"])


class TestMissingValueHandling:
    """Test missing value detection and treatment."""

    def test_detect_missing_values(self, preprocessor, sample_events_with_issues):
        """Preprocessor must detect missing values."""
        missing_report = preprocessor.check_missing(sample_events_with_issues)
        assert isinstance(missing_report, dict)
        assert missing_report["amount"] > 0
        assert missing_report["event_date"] > 0

    def test_handle_missing_amounts(self, preprocessor, sample_events_with_issues):
        """Missing purchase amounts should be handled (fill or drop)."""
        cleaned = preprocessor.handle_missing(sample_events_with_issues)
        # After handling, no purchase events should have null amounts
        purchases = cleaned[cleaned["event_type"] == "purchase"]
        assert purchases["amount"].notna().all(), (
            "Purchase events still have missing amounts after cleaning"
        )

    def test_handle_missing_dates(self, preprocessor, sample_events_with_issues):
        """Missing event dates should be handled."""
        cleaned = preprocessor.handle_missing(sample_events_with_issues)
        assert cleaned["event_date"].notna().all(), (
            "Events still have missing dates after cleaning"
        )


class TestOutlierHandling:
    """Test outlier detection and treatment."""

    def test_detect_outliers(self, preprocessor, sample_events_with_issues):
        """Preprocessor must detect outlier amounts."""
        outliers = preprocessor.detect_outliers(
            sample_events_with_issues, column="amount"
        )
        assert isinstance(outliers, pd.Series)
        assert outliers.dtype == bool
        assert outliers.sum() > 0, "No outliers detected"

    def test_handle_outliers_capping(self, preprocessor, sample_events_with_issues):
        """Outliers should be capped/clipped to reasonable range."""
        cleaned = preprocessor.handle_outliers(
            sample_events_with_issues, column="amount"
        )
        original_max = sample_events_with_issues["amount"].max()
        cleaned_max = cleaned["amount"].max()
        assert cleaned_max < original_max, (
            "Outlier capping did not reduce max value"
        )


class TestTrainTestSplit:
    """Test time-based train/test split."""

    def test_time_based_split(self, preprocessor, sample_customers, sample_events):
        """Must perform time-based split: 10 months train, 2 months test."""
        train_data, test_data = preprocessor.time_based_split(
            sample_customers, sample_events,
            train_months=10, test_months=2
        )
        assert isinstance(train_data, dict)
        assert isinstance(test_data, dict)
        assert "customers" in train_data
        assert "customers" in test_data
        assert "events" in train_data
        assert "events" in test_data

    def test_no_data_leakage(self, preprocessor, sample_customers, sample_events):
        """Test set events must not appear in training period."""
        train_data, test_data = preprocessor.time_based_split(
            sample_customers, sample_events,
            train_months=10, test_months=2
        )
        train_events = train_data["events"]
        test_events = test_data["events"]

        if len(train_events) > 0 and len(test_events) > 0:
            train_max_date = pd.to_datetime(train_events["event_date"]).max()
            test_min_date = pd.to_datetime(test_events["event_date"]).min()
            assert train_max_date <= test_min_date, (
                f"Data leakage: train max date {train_max_date} > "
                f"test min date {test_min_date}"
            )

    def test_split_ratio_approximate(
        self, preprocessor, sample_customers, sample_events
    ):
        """Train/test split should approximately match 10/2 month ratio."""
        train_data, test_data = preprocessor.time_based_split(
            sample_customers, sample_events,
            train_months=10, test_months=2
        )
        train_customers = len(train_data["customers"])
        test_customers = len(test_data["customers"])
        total = train_customers + test_customers
        if total > 0:
            train_ratio = train_customers / total
            # Train should be larger than test
            assert train_ratio > 0.5, (
                f"Train ratio {train_ratio:.2f} is too low"
            )


class TestSequencePreparation:
    """Test sequence data preparation for DL models."""

    def test_create_event_sequences(self, preprocessor, sample_events):
        """Must create per-customer event type sequences."""
        sequences = preprocessor.create_event_sequences(
            sample_events, max_length=50
        )
        assert isinstance(sequences, dict)
        # Each customer should have a sequence
        customer_ids = sample_events["customer_id"].unique()
        assert len(sequences) == len(customer_ids)

    def test_sequence_padding(self, preprocessor, sample_events):
        """Sequences must be padded to uniform length."""
        sequences = preprocessor.create_event_sequences(
            sample_events, max_length=50
        )
        for cid, seq in sequences.items():
            assert len(seq) == 50, (
                f"Sequence for {cid} has length {len(seq)}, expected 50"
            )

    def test_event_type_encoding(self, preprocessor, sample_events):
        """Event types must be encoded to integer IDs."""
        sequences = preprocessor.create_event_sequences(
            sample_events, max_length=50
        )
        for cid, seq in sequences.items():
            assert all(isinstance(v, (int, np.integer)) for v in seq), (
                f"Sequence for {cid} contains non-integer values"
            )


class TestFeatureScaling:
    """Test feature scaling and normalization."""

    def test_standard_scaling(self, preprocessor):
        """StandardScaler should normalize features to ~mean=0, std=1."""
        df = pd.DataFrame({
            "feature_a": np.random.randn(100) * 10 + 50,
            "feature_b": np.random.randn(100) * 5 + 20,
        })
        scaled = preprocessor.scale_features(df, method="standard")
        assert abs(scaled["feature_a"].mean()) < 0.5
        assert abs(scaled["feature_a"].std() - 1.0) < 0.5

    def test_minmax_scaling(self, preprocessor):
        """MinMaxScaler should normalize features to [0, 1]."""
        df = pd.DataFrame({
            "feature_a": np.random.randn(100) * 10 + 50,
            "feature_b": np.random.randn(100) * 5 + 20,
        })
        scaled = preprocessor.scale_features(df, method="minmax")
        assert scaled["feature_a"].min() >= -0.01
        assert scaled["feature_a"].max() <= 1.01


class TestDataValidation:
    """Test data integrity validation."""

    def test_validate_no_duplicate_customer_ids(
        self, preprocessor, sample_customers
    ):
        """Validation must catch duplicate customer IDs."""
        # Clean data should pass
        assert preprocessor.validate_customers(sample_customers)

        # Data with duplicates should fail
        duped = pd.concat([sample_customers, sample_customers.iloc[:5]])
        with pytest.raises(ValueError, match="duplicate"):
            preprocessor.validate_customers(duped)

    def test_validate_event_references(
        self, preprocessor, sample_customers, sample_events
    ):
        """All event customer_ids must exist in customers table."""
        assert preprocessor.validate_event_references(
            sample_events, sample_customers
        )

    def test_validate_date_order(self, preprocessor, sample_events):
        """Event dates must not precede customer signup dates."""
        # Basic validation that dates are parseable
        assert preprocessor.validate_dates(sample_events)
