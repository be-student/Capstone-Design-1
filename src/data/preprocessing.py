"""
Data Preprocessing Module for E-Commerce Churn Prediction.

Handles loading raw data, cleaning, missing value treatment, outlier detection,
time-based train/test splitting, sequence preparation for DL models,
feature scaling, and data validation.

Usage:
    preprocessor = Preprocessor()
    customers = preprocessor.load_customers("data/raw/customers.csv")
    events = preprocessor.load_events("data/raw/events.csv")
    cleaned = preprocessor.handle_missing(events)
    train, test = preprocessor.time_based_split(customers, events, 10, 2)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Preprocessor:
    """Preprocess raw customer and event data for churn prediction.

    Provides methods for loading, cleaning, splitting, scaling,
    sequence preparation, and validation of e-commerce data.
    """

    # Default event type encoding (0 reserved for padding)
    EVENT_TYPE_MAP: Dict[str, int] = {
        "page_view": 1,
        "search": 2,
        "add_to_cart": 3,
        "remove_from_cart": 4,
        "purchase": 5,
        "coupon_use": 6,
        "review": 7,
        "cs_contact": 8,
    }

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def load_customers(self, path: str) -> pd.DataFrame:
        """Load customer profiles from CSV.

        Args:
            path: Path to customers CSV file.

        Returns:
            DataFrame with parsed date columns.
        """
        df = pd.read_csv(path)
        if "signup_date" in df.columns:
            df["signup_date"] = pd.to_datetime(df["signup_date"])
        return df

    def load_events(self, path: str) -> pd.DataFrame:
        """Load event logs from CSV with date parsing.

        Args:
            path: Path to events CSV file.

        Returns:
            DataFrame with datetime-typed date columns.
        """
        df = pd.read_csv(path)
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"])
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ------------------------------------------------------------------
    # Missing Value Handling
    # ------------------------------------------------------------------

    def check_missing(self, df: pd.DataFrame) -> Dict[str, int]:
        """Report the number of missing values per column.

        Args:
            df: Input DataFrame.

        Returns:
            Dictionary mapping column names to missing-value counts.
        """
        return df.isnull().sum().to_dict()

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in event data.

        Strategy:
        - Missing amounts for purchase events: fill with median of purchases.
        - Missing amounts for non-purchase events: fill with 0.
        - Missing event_date / timestamp: drop rows.

        Args:
            df: Events DataFrame potentially containing NaN values.

        Returns:
            Cleaned DataFrame with no missing values in critical columns.
        """
        df = df.copy()

        # Handle missing dates by dropping rows
        if "event_date" in df.columns:
            df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
            df = df.dropna(subset=["event_date"])

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # For timestamp, fill from event_date if possible
            if "event_date" in df.columns:
                mask = df["timestamp"].isna()
                df.loc[mask, "timestamp"] = df.loc[mask, "event_date"]
            df = df.dropna(subset=["timestamp"])

        # Handle missing amounts
        if "amount" in df.columns:
            # For purchase events, fill with median purchase amount
            purchase_mask = df["event_type"] == "purchase"
            purchase_amounts = df.loc[
                purchase_mask & df["amount"].notna(), "amount"
            ]
            if len(purchase_amounts) > 0:
                median_amount = purchase_amounts.median()
            else:
                median_amount = 0.0

            df.loc[purchase_mask & df["amount"].isna(), "amount"] = median_amount
            # For non-purchase events, fill with 0
            df["amount"] = df["amount"].fillna(0.0)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Outlier Detection & Treatment
    # ------------------------------------------------------------------

    def detect_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.Series:
        """Detect outliers in a numeric column.

        Uses IQR method by default: values beyond Q1-1.5*IQR or Q3+1.5*IQR.

        Args:
            df: Input DataFrame.
            column: Column name to check for outliers.
            method: Detection method ('iqr' or 'zscore').
            threshold: Z-score threshold (only used with 'zscore' method).

        Returns:
            Boolean Series where True indicates an outlier.
        """
        series = df[column].dropna()

        if method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outlier_mask = z_scores > threshold
        else:  # iqr
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)

        # Reindex to match original DataFrame
        result = pd.Series(False, index=df.index)
        result.loc[outlier_mask.index] = outlier_mask
        return result

    def handle_outliers(
        self,
        df: pd.DataFrame,
        column: str,
        method: str = "cap",
    ) -> pd.DataFrame:
        """Handle outliers by capping to IQR bounds.

        Args:
            df: Input DataFrame.
            column: Column name to cap.
            method: Treatment method ('cap' for capping).

        Returns:
            DataFrame with outliers capped.
        """
        df = df.copy()
        series = df[column].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[column] = df[column].clip(lower=lower, upper=upper)
        return df

    # ------------------------------------------------------------------
    # Time-Based Train/Test Split
    # ------------------------------------------------------------------

    def time_based_split(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
        train_months: int = 10,
        test_months: int = 2,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split data by time into train and test sets.

        The total period is divided into train_months for training and
        test_months for testing. Events and customers are assigned
        accordingly.

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.
            train_months: Number of months for training period.
            test_months: Number of months for test period.

        Returns:
            Tuple of (train_data, test_data), each a dict with
            'customers' and 'events' keys.
        """
        events = events.copy()
        customers = customers.copy()

        # Ensure datetime types
        if not pd.api.types.is_datetime64_any_dtype(events["event_date"]):
            events["event_date"] = pd.to_datetime(events["event_date"])
        if "signup_date" in customers.columns:
            if not pd.api.types.is_datetime64_any_dtype(customers["signup_date"]):
                customers["signup_date"] = pd.to_datetime(
                    customers["signup_date"]
                )

        # Determine date range
        min_date = events["event_date"].min()
        max_date = events["event_date"].max()

        total_months = train_months + test_months
        total_days = (max_date - min_date).days
        train_days = int(total_days * (train_months / total_months))

        split_date = min_date + pd.Timedelta(days=train_days)

        # Split events
        train_events = events[events["event_date"] <= split_date].copy()
        test_events = events[events["event_date"] > split_date].copy()

        # Split customers: train includes those who signed up before split,
        # test includes those active in test period
        train_customer_ids = set(train_events["customer_id"].unique())
        test_customer_ids = set(test_events["customer_id"].unique())

        # All customers with activity in train period
        train_customers = customers[
            customers["customer_id"].isin(train_customer_ids)
        ].copy()
        # Customers with activity in test period
        test_customers = customers[
            customers["customer_id"].isin(test_customer_ids)
        ].copy()

        train_data = {"customers": train_customers, "events": train_events}
        test_data = {"customers": test_customers, "events": test_events}

        return train_data, test_data

    # ------------------------------------------------------------------
    # Sequence Preparation for DL Models
    # ------------------------------------------------------------------

    def create_event_sequences(
        self,
        events: pd.DataFrame,
        max_length: int = 50,
    ) -> Dict[str, List[int]]:
        """Create padded event-type sequences per customer for DL models.

        Each customer's events are sorted chronologically, encoded to
        integer IDs, truncated or padded to max_length.

        Args:
            events: Events DataFrame.
            max_length: Fixed sequence length (pad/truncate).

        Returns:
            Dictionary mapping customer_id to list of integer event codes.
        """
        events = events.copy()
        if "timestamp" in events.columns:
            events["timestamp"] = pd.to_datetime(
                events["timestamp"], errors="coerce"
            )
            events = events.sort_values("timestamp")
        elif "event_date" in events.columns:
            events["event_date"] = pd.to_datetime(
                events["event_date"], errors="coerce"
            )
            events = events.sort_values("event_date")

        sequences: Dict[str, List[int]] = {}

        for cid, group in events.groupby("customer_id"):
            encoded = [
                self.EVENT_TYPE_MAP.get(et, 0)
                for et in group["event_type"].values
            ]
            # Truncate if too long, pad with 0 if too short
            if len(encoded) > max_length:
                encoded = encoded[-max_length:]  # Keep most recent
            else:
                encoded = [0] * (max_length - len(encoded)) + encoded

            sequences[cid] = encoded

        return sequences

    # ------------------------------------------------------------------
    # Feature Scaling
    # ------------------------------------------------------------------

    def scale_features(
        self,
        df: pd.DataFrame,
        method: str = "standard",
    ) -> pd.DataFrame:
        """Scale numeric features.

        Args:
            df: DataFrame with numeric features.
            method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler.

        Returns:
            DataFrame with scaled features (same column names).
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return df

        if method == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    # ------------------------------------------------------------------
    # Data Validation
    # ------------------------------------------------------------------

    def validate_customers(self, df: pd.DataFrame) -> bool:
        """Validate customer DataFrame has no duplicate IDs.

        Args:
            df: Customer profiles DataFrame.

        Returns:
            True if validation passes.

        Raises:
            ValueError: If duplicate customer_ids are found.
        """
        if df["customer_id"].duplicated().any():
            n_dupes = df["customer_id"].duplicated().sum()
            raise ValueError(
                f"Found {n_dupes} duplicate customer_id entries"
            )
        return True

    def validate_event_references(
        self,
        events: pd.DataFrame,
        customers: pd.DataFrame,
    ) -> bool:
        """Validate all event customer_ids exist in customers table.

        Args:
            events: Events DataFrame.
            customers: Customers DataFrame.

        Returns:
            True if all references are valid.

        Raises:
            ValueError: If orphan event customer_ids are found.
        """
        event_cids = set(events["customer_id"].unique())
        customer_cids = set(customers["customer_id"].unique())
        orphans = event_cids - customer_cids

        if orphans:
            raise ValueError(
                f"Found {len(orphans)} event customer_ids not in "
                f"customers table: {list(orphans)[:5]}"
            )
        return True

    def validate_dates(self, events: pd.DataFrame) -> bool:
        """Validate that event dates are parseable.

        Args:
            events: Events DataFrame.

        Returns:
            True if all dates are valid.

        Raises:
            ValueError: If unparseable dates exist.
        """
        if "event_date" in events.columns:
            parsed = pd.to_datetime(events["event_date"], errors="coerce")
            n_invalid = parsed.isna().sum()
            if n_invalid > 0:
                raise ValueError(
                    f"Found {n_invalid} unparseable event_date values"
                )
        return True
