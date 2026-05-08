"""
Feature Engineering Module for E-Commerce Churn Prediction.

Computes 30+ features across multiple feature groups:
- RFM (Recency, Frequency, Monetary)
- Behavioral change rates (5+ features)
- Purchase cycle anomaly
- Session quality (3+ features)
- Sequence features (2+ features)
- Time-based behavior (weekend/weekday, time-of-day)
- Customer journey stage

Supports file-based feature store for persistence.

Usage:
    fe = FeatureEngineer()
    features = fe.compute_all_features(customers, events, "2024-07-01")
    fe.save_to_feature_store(features, "data/feature_store")
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class FeatureEngineer:
    """Compute and manage features for churn prediction.

    Computes RFM, behavioral change, session quality, sequence,
    time-based, and journey stage features from customer and event data.
    """

    EVENT_TYPES = [
        "page_view", "search", "add_to_cart", "remove_from_cart",
        "purchase", "coupon_use", "review", "cs_contact",
    ]
    FEATURE_EVENT_COLUMNS = [
        "customer_id", "event_type", "event_date", "timestamp",
        "event_timestamp", "amount", "session_duration",
    ]
    CHANGE_FEATURE_COLUMNS = [
        "visit_frequency_change",
        "purchase_cycle_change",
        "session_duration_change",
        "cart_conversion_change",
        "coupon_response_change",
        "search_intensity_change",
        "cs_contact_change",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the feature engineer.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}

    # ------------------------------------------------------------------
    # Master Feature Computation
    # ------------------------------------------------------------------

    def compute_all_features(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
        reference_date: Union[str, pd.Timestamp] = "2024-07-01",
    ) -> pd.DataFrame:
        """Compute all feature groups and merge into a single DataFrame.

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.
            reference_date: Reference date for recency calculations.

        Returns:
            DataFrame with customer_id, all features, and metadata columns.
        """
        events = self._prepare_events_for_features(events)
        if "signup_date" in customers.columns:
            signup_dates = pd.to_datetime(customers["signup_date"])
        else:
            signup_dates = None

        ref_date = pd.Timestamp(reference_date)

        # Compute each feature group
        rfm = self.compute_rfm(customers, events, reference_date)
        behavioral = self.compute_behavioral_changes(
            customers, events, reference_date
        )
        anomaly = self.compute_purchase_cycle_anomaly(
            customers, events, reference_date
        )
        session = self.compute_session_quality(customers, events)
        sequence = self.compute_sequence_features(customers, events)
        time_feats = self.compute_time_features(customers, events)
        journey = self.compute_journey_features(customers, events)

        # Start with customer base and merge compact per-customer feature frames.
        customer_cols = ["customer_id"]
        for col in ["persona", "churn_label", "treatment_group", "signup_date"]:
            if col in customers.columns:
                customer_cols.append(col)
        result = customers[customer_cols].copy()
        if signup_dates is not None:
            result["signup_date"] = signup_dates

        for feat_df in [
            rfm, behavioral, anomaly, session, sequence, time_feats, journey
        ]:
            if "customer_id" in feat_df.columns:
                result = result.merge(feat_df, on="customer_id", how="left")

        # Compute tenure
        if "signup_date" in result.columns:
            result["signup_date"] = pd.to_datetime(result["signup_date"])
            result["tenure_days"] = (
                ref_date - result["signup_date"]
            ).dt.days.clip(lower=0)

        return self._sanitize_feature_matrix(result)

    # ------------------------------------------------------------------
    # RFM Features
    # ------------------------------------------------------------------

    def compute_rfm(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
        reference_date: Union[str, pd.Timestamp] = "2024-07-01",
    ) -> pd.DataFrame:
        """Compute Recency, Frequency, Monetary features.

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.
            reference_date: Reference date for recency calculation.

        Returns:
            DataFrame with customer_id, recency, frequency, monetary columns.
        """
        ref_date = pd.Timestamp(reference_date)
        events = self._prepare_events_for_features(events)

        result = customers[["customer_id"]].copy()
        if events.empty or "event_type" not in events.columns:
            result["recency"] = 365.0
            result["frequency"] = 0
            result["monetary"] = 0.0
            result["avg_order_value"] = 0.0
            result["monetary_per_day"] = 0.0
            return result

        purchases = events.loc[events["event_type"] == "purchase"]

        if len(purchases) > 0:
            purchase_agg = purchases.groupby("customer_id", observed=True).agg(
                last_purchase=("event_date", "max"),
                frequency=("event_date", "count"),
                monetary=("amount", "sum"),
            ).reset_index()
        else:
            purchase_agg = pd.DataFrame(
                columns=["customer_id", "last_purchase", "frequency", "monetary"]
            )

        rfm = result.merge(purchase_agg, on="customer_id", how="left")

        # Recency: days since last purchase
        rfm["last_purchase"] = pd.to_datetime(rfm["last_purchase"])
        rfm["recency"] = (ref_date - rfm["last_purchase"]).dt.days
        # For customers with no purchases, set high recency
        max_recency = 365
        rfm["recency"] = rfm["recency"].fillna(max_recency).clip(lower=0)

        rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
        rfm["monetary"] = rfm["monetary"].fillna(0.0)

        # Additional RFM-derived features
        rfm["avg_order_value"] = np.where(
            rfm["frequency"] > 0,
            rfm["monetary"] / rfm["frequency"],
            0.0,
        )
        rfm["monetary_per_day"] = np.where(
            rfm["recency"] > 0,
            rfm["monetary"] / (max_recency - rfm["recency"] + 1),
            0.0,
        )

        return rfm[
            ["customer_id", "recency", "frequency", "monetary",
             "avg_order_value", "monetary_per_day"]
        ]

    # ------------------------------------------------------------------
    # Behavioral Change Features
    # ------------------------------------------------------------------

    def compute_behavioral_changes(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
        reference_date: Union[str, pd.Timestamp] = "2024-07-01",
    ) -> pd.DataFrame:
        """Compute behavioral change rate features.

        Compares first half vs second half of each customer's activity
        to detect changes in behavior patterns.

        Features:
        - visit_frequency_change: Change in daily visit rate
        - purchase_cycle_change: Change in avg days between purchases
        - session_duration_change: Change in avg session duration
        - cart_conversion_change: Change in cart-to-purchase rate
        - coupon_response_change: Change in coupon usage rate
        - search_intensity_change: Change in searches per visit
        - cs_contact_change: Change in CS contact rate

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.
            reference_date: Reference date for splitting periods.

        Returns:
            DataFrame with customer_id and change features.
        """
        _ = pd.Timestamp(reference_date)
        events = self._prepare_events_for_features(events)
        result = customers[["customer_id"]].copy()
        for col in self.CHANGE_FEATURE_COLUMNS:
            result[col] = 0.0
        if events.empty:
            return result

        date_stats = events.groupby("customer_id", observed=True)["event_date"].agg(
            ["min", "max"]
        )
        median_date = date_stats["min"] + (date_stats["max"] - date_stats["min"]) / 2
        first_days = (median_date - date_stats["min"]).dt.days.clip(lower=1)
        second_days = (date_stats["max"] - median_date).dt.days.clip(lower=1)

        event_median = events["customer_id"].map(median_date)
        period = pd.Series(
            np.where(events["event_date"] <= event_median, 0, 1),
            index=events.index,
            name="_period",
        )
        period_counts = (
            events.groupby(["customer_id", period], observed=True)
            .size()
            .unstack("_period", fill_value=0)
        )
        event_counts = (
            events.groupby(["customer_id", period, "event_type"], observed=True)
            .size()
            .unstack(["_period", "event_type"], fill_value=0)
        )

        def count(period_id: int, event_type: str) -> pd.Series:
            key = (period_id, event_type)
            if key in event_counts.columns:
                return event_counts[key].reindex(date_stats.index, fill_value=0)
            return pd.Series(0, index=date_stats.index, dtype="float64")

        first_page = count(0, "page_view")
        second_page = count(1, "page_view")
        first_purchase = count(0, "purchase")
        second_purchase = count(1, "purchase")
        first_cart = count(0, "add_to_cart")
        second_cart = count(1, "add_to_cart")

        changes = pd.DataFrame(index=date_stats.index)
        changes["visit_frequency_change"] = self._safe_ratio_series(
            second_page / second_days,
            first_page / first_days,
        )
        changes["purchase_cycle_change"] = self._safe_ratio_series(
            second_purchase / second_days,
            first_purchase / first_days,
        )

        if "session_duration" in events.columns:
            durations = (
                events.groupby(["customer_id", period], observed=True)[
                    "session_duration"
                ]
                .mean()
                .unstack("_period")
            )
            first_dur = durations.get(0, pd.Series(np.nan, index=date_stats.index))
            second_dur = durations.get(1, pd.Series(np.nan, index=date_stats.index))
        else:
            first_dur = period_counts.get(0, pd.Series(0, index=date_stats.index)) / first_days
            second_dur = period_counts.get(1, pd.Series(0, index=date_stats.index)) / second_days
        changes["session_duration_change"] = self._safe_ratio_series(
            second_dur.reindex(date_stats.index),
            first_dur.reindex(date_stats.index),
        )

        first_conv = self._safe_ratio_series(first_purchase, first_cart)
        second_conv = self._safe_ratio_series(second_purchase, second_cart)
        changes["cart_conversion_change"] = self._safe_ratio_series(
            second_conv,
            first_conv,
        )
        changes["coupon_response_change"] = self._safe_ratio_series(
            count(1, "coupon_use") / second_days,
            count(0, "coupon_use") / first_days,
        )
        changes["search_intensity_change"] = self._safe_ratio_series(
            count(1, "search") / second_days,
            count(0, "search") / first_days,
        )
        changes["cs_contact_change"] = self._safe_ratio_series(
            count(1, "cs_contact") / second_days,
            count(0, "cs_contact") / first_days,
        )

        for col in self.CHANGE_FEATURE_COLUMNS:
            result[col] = result["customer_id"].map(changes[col]).fillna(0.0)
        return result

    # ------------------------------------------------------------------
    # Purchase Cycle Anomaly
    # ------------------------------------------------------------------

    def compute_purchase_cycle_anomaly(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
        reference_date: Union[str, pd.Timestamp] = "2024-07-01",
    ) -> pd.DataFrame:
        """Compute purchase cycle anomaly score.

        Anomaly = days_since_last_purchase / avg_purchase_cycle.
        A value > 1 indicates the customer is overdue for a purchase.

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.
            reference_date: Reference date.

        Returns:
            DataFrame with customer_id and purchase_cycle_anomaly.
        """
        ref_date = pd.Timestamp(reference_date)
        events = self._prepare_events_for_features(events)
        result = customers[["customer_id"]].copy()
        result["purchase_cycle_anomaly"] = 2.0
        result["avg_purchase_cycle_days"] = 0.0
        result["days_since_last_purchase"] = 0.0
        if events.empty or "event_type" not in events.columns:
            return result

        purchases = events.loc[events["event_type"] == "purchase", [
            "customer_id", "event_date",
        ]]
        if purchases.empty:
            return result

        purchases = purchases.sort_values(["customer_id", "event_date"], kind="mergesort")
        counts = purchases.groupby("customer_id", observed=True).size()
        last_purchase = purchases.groupby("customer_id", observed=True)["event_date"].max()
        intervals = (
            purchases.groupby("customer_id", observed=True)["event_date"]
            .diff()
            .dt.days
        )
        avg_cycle = intervals.groupby(purchases["customer_id"], observed=True).mean()

        anomaly = pd.Series(2.0, index=counts.index, dtype="float64")
        anomaly.loc[counts == 1] = 1.5
        enough = counts >= 2
        days_since = (ref_date - last_purchase).dt.days.clip(lower=0).astype(float)
        valid_avg = avg_cycle.reindex(counts.index).fillna(0.0)
        anomaly.loc[enough] = self._safe_ratio_series(
            days_since.loc[enough],
            valid_avg.loc[enough],
            default=2.0,
        ).clip(lower=0.0)

        avg_for_output = valid_avg.where(enough, 0.0)
        days_for_output = days_since.where(enough, 0.0)
        result["purchase_cycle_anomaly"] = (
            result["customer_id"].map(anomaly).fillna(2.0)
        )
        result["avg_purchase_cycle_days"] = (
            result["customer_id"].map(avg_for_output).fillna(0.0)
        )
        result["days_since_last_purchase"] = (
            result["customer_id"].map(days_for_output).fillna(0.0)
        )
        return result

    # ------------------------------------------------------------------
    # Session Quality Features
    # ------------------------------------------------------------------

    def compute_session_quality(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute session quality features.

        Features:
        - avg_session_duration: Average session duration (or event density)
        - pageviews_per_session: Average page views per visit day
        - search_to_purchase_rate: Ratio of purchases to searches
        - cart_abandonment_rate: Ratio of cart removals to cart adds
        - avg_events_per_day: Overall event density

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.

        Returns:
            DataFrame with customer_id and session quality features.
        """
        events = self._prepare_events_for_features(events)
        result = customers[["customer_id"]].copy()
        defaults = {
            "avg_session_duration": 0.0,
            "pageviews_per_session": 0.0,
            "search_to_purchase_rate": 0.0,
            "cart_abandonment_rate": 0.0,
            "avg_events_per_day": 0.0,
        }
        for col, value in defaults.items():
            result[col] = value
        if events.empty:
            return result

        grouped = events.groupby("customer_id", observed=True)
        event_count = grouped.size()
        visit_days = grouped["event_date"].nunique().clip(lower=1)
        date_range = (grouped["event_date"].max() - grouped["event_date"].min())
        date_range = date_range.dt.days.add(1).clip(lower=1)
        event_counts = self._event_count_matrix(events)

        if "session_duration" in events.columns:
            avg_session = grouped["session_duration"].mean()
            density_fallback = event_count / visit_days
            avg_session = avg_session.fillna(density_fallback)
        else:
            avg_session = event_count / visit_days

        features = pd.DataFrame(index=event_count.index)
        features["avg_session_duration"] = avg_session
        features["pageviews_per_session"] = (
            self._matrix_col(event_counts, "page_view") / visit_days
        )
        features["search_to_purchase_rate"] = self._safe_ratio_series(
            self._matrix_col(event_counts, "purchase"),
            self._matrix_col(event_counts, "search"),
        )
        features["cart_abandonment_rate"] = self._safe_ratio_series(
            self._matrix_col(event_counts, "remove_from_cart"),
            self._matrix_col(event_counts, "add_to_cart"),
        )
        features["avg_events_per_day"] = event_count / date_range

        for col in defaults:
            result[col] = result["customer_id"].map(features[col]).fillna(0.0)
        return result

    # ------------------------------------------------------------------
    # Sequence Features
    # ------------------------------------------------------------------

    def compute_sequence_features(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute sequence-based features.

        Features:
        - sequence_diversity: Number of unique event types used
        - sequence_length: Total number of events
        - behavior_pattern_cluster: Cluster ID from event type distribution
        - purchase_sequence_trend: Trend in purchase amounts over time

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.

        Returns:
            DataFrame with customer_id and sequence features.
        """
        events = self._prepare_events_for_features(events)
        result = customers[["customer_id"]].copy()
        result["sequence_diversity"] = 0
        result["sequence_length"] = 0
        result["purchase_sequence_trend"] = 0.0
        result["behavior_pattern_cluster"] = 0
        if events.empty:
            return result

        event_counts = self._event_count_matrix(events, self.EVENT_TYPES)
        sequence_diversity = event_counts.gt(0).sum(axis=1)
        sequence_length = event_counts.sum(axis=1)
        customer_ids = result["customer_id"]
        distribution = (
            event_counts.reindex(customer_ids, fill_value=0)[self.EVENT_TYPES]
            .div(
                sequence_length.reindex(customer_ids, fill_value=0)
                .replace(0, np.nan),
                axis=0,
            )
            .fillna(0.0)
        )

        result["sequence_diversity"] = (
            customer_ids.map(sequence_diversity).fillna(0).astype(int)
        )
        result["sequence_length"] = (
            customer_ids.map(sequence_length).fillna(0).astype(int)
        )

        if "amount" in events.columns:
            purchases = events.loc[
                events["event_type"] == "purchase",
                ["customer_id", "event_date", "amount"],
            ]
            if not purchases.empty:
                purchases = purchases.sort_values(
                    ["customer_id", "event_date"], kind="mergesort"
                )
                trend = purchases.groupby("customer_id", observed=True)["amount"].apply(
                    self._purchase_amount_trend
                )
                result["purchase_sequence_trend"] = (
                    customer_ids.map(trend).fillna(0.0)
                )

        dist_matrix = distribution.to_numpy(dtype=np.float32, copy=False)
        n_clusters = min(5, len(dist_matrix))
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            result["behavior_pattern_cluster"] = kmeans.fit_predict(dist_matrix)

        return result

    # ------------------------------------------------------------------
    # Time-Based Features
    # ------------------------------------------------------------------

    def compute_time_features(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute time-based behavior features.

        Features:
        - weekend_purchase_ratio: Fraction of purchases on weekends
        - evening_activity_ratio: Fraction of events in evening hours
        - morning_activity_ratio: Fraction of events in morning hours
        - night_activity_ratio: Fraction of events at night
        - peak_hour: Most common activity hour
        - day_variance: Variance in daily event counts

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.

        Returns:
            DataFrame with customer_id and time-based features.
        """
        events = self._prepare_events_for_features(events)
        result = customers[["customer_id"]].copy()
        defaults = {
            "weekend_purchase_ratio": 0.0,
            "evening_activity_ratio": 0.0,
            "morning_activity_ratio": 0.0,
            "night_activity_ratio": 0.0,
            "peak_hour": 12,
            "day_variance": 0.0,
        }
        for col, value in defaults.items():
            result[col] = value
        if events.empty:
            return result

        features = pd.DataFrame(index=events.groupby("customer_id", observed=True).size().index)

        purchases = events.loc[events["event_type"] == "purchase", [
            "customer_id", "event_date",
        ]]
        if not purchases.empty:
            weekend_ratio = (
                (purchases["event_date"].dt.dayofweek >= 5)
                .groupby(purchases["customer_id"], observed=True)
                .mean()
            )
            features["weekend_purchase_ratio"] = weekend_ratio

        if "timestamp" in events.columns:
            hours = events["timestamp"].dt.hour
            valid_hours = hours.notna()
            if valid_hours.any():
                hour_values = hours[valid_hours].astype("int16")
                hour_ids = events.loc[valid_hours, "customer_id"]
                totals = hour_ids.groupby(hour_ids, observed=True).size()
                features["morning_activity_ratio"] = (
                    ((hour_values >= 6) & (hour_values < 12))
                    .groupby(hour_ids, observed=True)
                    .sum()
                    .div(totals)
                )
                features["evening_activity_ratio"] = (
                    ((hour_values >= 18) & (hour_values < 22))
                    .groupby(hour_ids, observed=True)
                    .sum()
                    .div(totals)
                )
                features["night_activity_ratio"] = (
                    ((hour_values >= 22) | (hour_values < 6))
                    .groupby(hour_ids, observed=True)
                    .sum()
                    .div(totals)
                )
                hour_counts = (
                    pd.DataFrame({
                        "customer_id": hour_ids,
                        "hour": hour_values.to_numpy(),
                    })
                    .groupby(["customer_id", "hour"], observed=True)
                    .size()
                    .reset_index(name="count")
                )
                peak_hour = (
                    hour_counts.sort_values(
                        ["customer_id", "count", "hour"],
                        ascending=[True, False, True],
                    )
                    .drop_duplicates("customer_id")
                    .set_index("customer_id")["hour"]
                )
                features["peak_hour"] = peak_hour

        event_day = events["event_date"].dt.floor("D")
        daily_counts = events.groupby(["customer_id", event_day], observed=True).size()
        features["day_variance"] = (
            daily_counts.groupby(level=0, observed=True).var().fillna(0.0)
        )

        for col, default in defaults.items():
            values = (
                features[col]
                if col in features.columns
                else pd.Series(dtype="float64")
            )
            result[col] = result["customer_id"].map(values).fillna(default)
        result["peak_hour"] = result["peak_hour"].astype(int)
        return result

    # ------------------------------------------------------------------
    # Customer Journey Stage Features
    # ------------------------------------------------------------------

    def compute_journey_features(
        self,
        customers: pd.DataFrame,
        events: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute customer journey stage features.

        Assigns each customer to a journey stage based on behavior:
        - 'onboarding': < 30 days of activity, few purchases
        - 'growing': Increasing purchase frequency
        - 'mature': Stable, regular purchasing pattern
        - 'declining': Decreasing activity
        - 'dormant': Very low recent activity

        Features:
        - journey_stage: Stage label (encoded as int)
        - stage_tenure_days: Days in current behavioral pattern

        Args:
            customers: Customer profiles DataFrame.
            events: Events DataFrame.

        Returns:
            DataFrame with customer_id, journey_stage, stage_tenure_days.
        """
        events = self._prepare_events_for_features(events)
        result = customers[["customer_id"]].copy()
        result["journey_stage"] = 4
        result["stage_tenure_days"] = 0
        if events.empty:
            return result

        grouped = events.groupby("customer_id", observed=True)
        stats = grouped["event_date"].agg(["min", "max", "count"])
        activity_span = (stats["max"] - stats["min"]).dt.days.clip(lower=0)
        last_date = events["customer_id"].map(stats["max"]).astype("datetime64[ns]")
        recent_mask = events["event_date"] > (last_date - pd.Timedelta(days=30))
        recent_count = (
            recent_mask.groupby(events["customer_id"], observed=True)
            .sum()
            .reindex(stats.index, fill_value=0)
            .astype(float)
        )
        older_count = stats["count"].astype(float) - recent_count
        recent_rate = recent_count / 30.0
        older_denominator = (activity_span - 30).clip(lower=1).astype(float)
        older_rate = pd.Series(
            np.where(
                activity_span > 30,
                older_count / older_denominator,
                recent_rate,
            ),
            index=stats.index,
        )
        purchase_counts = (
            events.loc[events["event_type"] == "purchase"]
            .groupby("customer_id", observed=True)
            .size()
            .reindex(stats.index, fill_value=0)
        )

        stage = pd.Series(2, index=stats.index, dtype="int64")
        tenure = activity_span.astype("int64").copy()

        onboarding = activity_span < 30
        growing = (~onboarding) & (recent_rate > older_rate * 1.2)
        mature = (
            (~onboarding)
            & (~growing)
            & (recent_rate > older_rate * 0.8)
            & (purchase_counts >= 3)
        )
        declining = (
            (~onboarding)
            & (~growing)
            & (~mature)
            & (recent_rate < older_rate * 0.5)
        )
        remaining = ~(onboarding | growing | mature | declining)
        dormant = remaining & (recent_rate < 0.1)
        still_onboarding = remaining & (~dormant) & (purchase_counts < 2)

        stage.loc[onboarding | still_onboarding] = 0
        stage.loc[growing] = 1
        stage.loc[mature] = 2
        stage.loc[declining] = 3
        stage.loc[dormant] = 4

        tenure.loc[growing] = np.minimum(30, activity_span.loc[growing]).astype(int)
        tenure.loc[declining | dormant] = 30
        tenure = tenure.clip(lower=0)

        result["journey_stage"] = result["customer_id"].map(stage).fillna(4).astype(int)
        result["stage_tenure_days"] = (
            result["customer_id"].map(tenure).fillna(0).astype(int)
        )
        return result

    # ------------------------------------------------------------------
    # Feature Store (File-Based)
    # ------------------------------------------------------------------

    def save_to_feature_store(
        self,
        features: pd.DataFrame,
        store_path: str,
    ) -> None:
        """Save features to file-based feature store.

        Saves as both Parquet (primary) and CSV (backup).

        Args:
            features: Feature DataFrame.
            store_path: Directory path for the feature store.
        """
        os.makedirs(store_path, exist_ok=True)
        parquet_path = os.path.join(store_path, "features.parquet")
        try:
            features.to_parquet(parquet_path, index=False)
        except Exception as exc:
            warnings.warn(
                f"Parquet feature store save failed; CSV fallback only: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            if os.path.exists(parquet_path):
                os.remove(parquet_path)
        features.to_csv(
            os.path.join(store_path, "features.csv"), index=False
        )

    def load_from_feature_store(
        self,
        store_path: str,
    ) -> pd.DataFrame:
        """Load features from file-based feature store.

        Prefers Parquet format, falls back to CSV.

        Args:
            store_path: Directory path for the feature store.

        Returns:
            Feature DataFrame.
        """
        parquet_path = os.path.join(store_path, "features.parquet")
        csv_path = os.path.join(store_path, "features.csv")

        if os.path.exists(parquet_path):
            try:
                return pd.read_parquet(parquet_path)
            except Exception as exc:
                if not os.path.exists(csv_path):
                    raise
                warnings.warn(
                    f"Parquet feature store load failed; using CSV fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        raise FileNotFoundError(
            f"No feature files found in {store_path}"
        )

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _ensure_datetime(self, events: pd.DataFrame) -> pd.DataFrame:
        """Ensure date columns are datetime type.

        Args:
            events: Events DataFrame.

        Returns:
            DataFrame with datetime columns.
        """
        needs_copy = False
        for col in ("event_date", "timestamp", "event_timestamp"):
            if (
                col in events.columns
                and not pd.api.types.is_datetime64_any_dtype(events[col])
            ):
                needs_copy = True
                break
        if needs_copy:
            events = events.copy()
        if "event_date" in events.columns:
            if not pd.api.types.is_datetime64_any_dtype(events["event_date"]):
                events["event_date"] = pd.to_datetime(
                    events["event_date"], errors="coerce"
                )
        if "timestamp" in events.columns:
            if not pd.api.types.is_datetime64_any_dtype(events["timestamp"]):
                events["timestamp"] = pd.to_datetime(
                    events["timestamp"], errors="coerce"
                )
        if "event_timestamp" in events.columns:
            if not pd.api.types.is_datetime64_any_dtype(events["event_timestamp"]):
                events["event_timestamp"] = pd.to_datetime(
                    events["event_timestamp"], errors="coerce"
                )
        return events

    def _prepare_events_for_features(self, events: pd.DataFrame) -> pd.DataFrame:
        """Return a compact event view with feature-required columns and dtypes."""
        if events.empty:
            return events
        keep = [col for col in self.FEATURE_EVENT_COLUMNS if col in events.columns]
        if keep and len(keep) < len(events.columns):
            events = events.loc[:, keep].copy()
        events = self._ensure_datetime(events)
        if "timestamp" not in events.columns and "event_timestamp" in events.columns:
            events = events.copy()
            events["timestamp"] = events["event_timestamp"]
        categorical_cols = [
            col for col in ("customer_id", "event_type")
            if col in events.columns
            and not isinstance(events[col].dtype, pd.CategoricalDtype)
        ]
        numeric_object_cols = [
            col for col in ("amount", "session_duration")
            if col in events.columns and events[col].dtype == object
        ]
        if categorical_cols or numeric_object_cols:
            events = events.copy()
            for col in categorical_cols:
                events[col] = events[col].astype("category")
            for col in numeric_object_cols:
                events[col] = pd.to_numeric(events[col], errors="coerce")
        return events

    @staticmethod
    def _event_count_matrix(
        events: pd.DataFrame,
        event_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Build a compact customer x event_type count matrix."""
        if events.empty or "event_type" not in events.columns:
            return pd.DataFrame()
        counts = (
            events.groupby(["customer_id", "event_type"], observed=True)
            .size()
            .unstack("event_type", fill_value=0)
        )
        if event_types:
            for event_type in event_types:
                if event_type not in counts.columns:
                    counts[event_type] = 0
            counts = counts[event_types]
        return counts

    @staticmethod
    def _matrix_col(matrix: pd.DataFrame, column: str) -> pd.Series:
        """Return a count column, or zeros aligned to the matrix index."""
        if column in matrix.columns:
            return matrix[column]
        return pd.Series(0, index=matrix.index, dtype="float64")

    @staticmethod
    def _safe_ratio_series(
        numerator: pd.Series,
        denominator: pd.Series,
        default: float = 0.0,
    ) -> pd.Series:
        """Vectorized safe ratio with zero, inf, and NaN protection."""
        aligned_num, aligned_den = numerator.align(denominator, join="outer")
        result = aligned_num.astype(float) / aligned_den.replace(0, np.nan).astype(float)
        return result.replace([np.inf, -np.inf], np.nan).fillna(default)

    @staticmethod
    def _purchase_amount_trend(amounts: pd.Series) -> float:
        """Correlation between purchase order and amount for one customer."""
        values = amounts.fillna(0.0).to_numpy(dtype=np.float64, copy=False)
        if len(values) < 2 or np.std(values) == 0:
            return 0.0
        x = np.arange(len(values), dtype=np.float64)
        trend = np.corrcoef(x, values)[0, 1]
        return 0.0 if np.isnan(trend) else float(trend)

    @staticmethod
    def _safe_ratio(
        numerator: float,
        denominator: float,
        default: float = 0.0,
    ) -> float:
        """Compute ratio safely, returning default on zero division.

        Args:
            numerator: Numerator value.
            denominator: Denominator value.
            default: Default value when denominator is zero.

        Returns:
            Ratio or default value.
        """
        if denominator == 0 or np.isnan(denominator):
            return default
        result = numerator / denominator
        if np.isnan(result) or np.isinf(result):
            return default
        return float(result)

    def _sanitize_feature_matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply final missing-value and outlier handling to numeric features."""
        result = features
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        exclude = {
            "customer_id",
            "churn_label",
            "journey_stage",
            "behavior_pattern_cluster",
            "peak_hour",
        }

        for col in numeric_cols:
            cleaned = result[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            if col in exclude or cleaned.nunique(dropna=True) <= 2:
                result[col] = cleaned
                continue

            lower = cleaned.quantile(0.01)
            upper = cleaned.quantile(0.99)
            if pd.notna(lower) and pd.notna(upper) and lower < upper:
                cleaned = cleaned.astype(float).clip(lower=lower, upper=upper)
            result[col] = cleaned

        return result

    @staticmethod
    def _empty_change_row(customer_id: str) -> Dict[str, Any]:
        """Create an empty behavioral change row for a customer.

        Args:
            customer_id: Customer identifier.

        Returns:
            Dictionary with all change features set to 0.
        """
        return {
            "customer_id": customer_id,
            "visit_frequency_change": 0.0,
            "purchase_cycle_change": 0.0,
            "session_duration_change": 0.0,
            "cart_conversion_change": 0.0,
            "coupon_response_change": 0.0,
            "search_intensity_change": 0.0,
            "cs_contact_change": 0.0,
        }
