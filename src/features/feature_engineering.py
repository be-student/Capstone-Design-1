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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class FeatureEngineer:
    """Compute and manage features for churn prediction.

    Computes RFM, behavioral change, session quality, sequence,
    time-based, and journey stage features from customer and event data.
    """

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
        customers = customers.copy()
        events = events.copy()

        # Ensure datetime types
        events = self._ensure_datetime(events)
        if "signup_date" in customers.columns:
            customers["signup_date"] = pd.to_datetime(
                customers["signup_date"]
            )

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

        # Start with customer base
        result = customers[["customer_id"]].copy()

        # Merge all feature groups
        for feat_df in [
            rfm, behavioral, anomaly, session, sequence, time_feats, journey
        ]:
            if "customer_id" in feat_df.columns:
                result = result.merge(feat_df, on="customer_id", how="left")

        # Add metadata columns back
        for col in ["persona", "churn_label", "treatment_group", "signup_date"]:
            if col in customers.columns:
                mapping = customers.set_index("customer_id")[col]
                result[col] = result["customer_id"].map(mapping)

        # Compute tenure
        if "signup_date" in result.columns:
            result["signup_date"] = pd.to_datetime(result["signup_date"])
            result["tenure_days"] = (
                ref_date - result["signup_date"]
            ).dt.days.clip(lower=0)

        # Fill remaining NaN with 0 and replace inf
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            result[col] = result[col].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0)

        return result

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
        events = self._ensure_datetime(events)

        purchases = events[events["event_type"] == "purchase"].copy()

        if len(purchases) > 0:
            purchase_agg = purchases.groupby("customer_id").agg(
                last_purchase=("event_date", "max"),
                frequency=("event_date", "count"),
                monetary=("amount", "sum"),
            ).reset_index()
        else:
            purchase_agg = pd.DataFrame(
                columns=["customer_id", "last_purchase", "frequency", "monetary"]
            )

        # Start with all customers
        rfm = customers[["customer_id"]].copy()
        rfm = rfm.merge(purchase_agg, on="customer_id", how="left")

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
        ref_date = pd.Timestamp(reference_date)
        events = self._ensure_datetime(events)

        # Split each customer's events into first half and second half
        result_rows = []

        for cid in customers["customer_id"].unique():
            cust_events = events[events["customer_id"] == cid].copy()

            if len(cust_events) == 0:
                result_rows.append(self._empty_change_row(cid))
                continue

            # Split by median date
            dates = cust_events["event_date"]
            median_date = dates.min() + (dates.max() - dates.min()) / 2

            first_half = cust_events[cust_events["event_date"] <= median_date]
            second_half = cust_events[cust_events["event_date"] > median_date]

            first_days = max(1, (median_date - dates.min()).days)
            second_days = max(1, (dates.max() - median_date).days)

            row = {"customer_id": cid}

            # Visit frequency change
            first_visits = len(
                first_half[first_half["event_type"] == "page_view"]
            )
            second_visits = len(
                second_half[second_half["event_type"] == "page_view"]
            )
            first_rate = first_visits / first_days
            second_rate = second_visits / second_days
            row["visit_frequency_change"] = self._safe_ratio(
                second_rate, first_rate
            )

            # Purchase cycle change
            first_purchases = first_half[
                first_half["event_type"] == "purchase"
            ]
            second_purchases = second_half[
                second_half["event_type"] == "purchase"
            ]
            first_pfreq = len(first_purchases) / first_days
            second_pfreq = len(second_purchases) / second_days
            row["purchase_cycle_change"] = self._safe_ratio(
                second_pfreq, first_pfreq
            )

            # Session duration change
            if "session_duration" in events.columns:
                first_dur = first_half["session_duration"].mean()
                second_dur = second_half["session_duration"].mean()
            else:
                # Estimate from event counts per day
                first_dur = len(first_half) / first_days
                second_dur = len(second_half) / second_days
            row["session_duration_change"] = self._safe_ratio(
                second_dur, first_dur
            )

            # Cart conversion change
            first_carts = len(
                first_half[first_half["event_type"] == "add_to_cart"]
            )
            second_carts = len(
                second_half[second_half["event_type"] == "add_to_cart"]
            )
            first_conv = (
                len(first_purchases) / first_carts if first_carts > 0 else 0
            )
            second_conv = (
                len(second_purchases) / second_carts
                if second_carts > 0
                else 0
            )
            row["cart_conversion_change"] = self._safe_ratio(
                second_conv, first_conv
            )

            # Coupon response change
            first_coupons = len(
                first_half[first_half["event_type"] == "coupon_use"]
            )
            second_coupons = len(
                second_half[second_half["event_type"] == "coupon_use"]
            )
            first_coupon_rate = first_coupons / first_days
            second_coupon_rate = second_coupons / second_days
            row["coupon_response_change"] = self._safe_ratio(
                second_coupon_rate, first_coupon_rate
            )

            # Search intensity change
            first_searches = len(
                first_half[first_half["event_type"] == "search"]
            )
            second_searches = len(
                second_half[second_half["event_type"] == "search"]
            )
            first_search_rate = first_searches / first_days
            second_search_rate = second_searches / second_days
            row["search_intensity_change"] = self._safe_ratio(
                second_search_rate, first_search_rate
            )

            # CS contact change
            first_cs = len(
                first_half[first_half["event_type"] == "cs_contact"]
            )
            second_cs = len(
                second_half[second_half["event_type"] == "cs_contact"]
            )
            first_cs_rate = first_cs / first_days
            second_cs_rate = second_cs / second_days
            row["cs_contact_change"] = self._safe_ratio(
                second_cs_rate, first_cs_rate
            )

            result_rows.append(row)

        return pd.DataFrame(result_rows)

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
        events = self._ensure_datetime(events)

        purchases = events[events["event_type"] == "purchase"].copy()
        result_rows = []

        for cid in customers["customer_id"].unique():
            cust_purchases = purchases[
                purchases["customer_id"] == cid
            ].sort_values("event_date")

            if len(cust_purchases) < 2:
                # Not enough purchases to compute cycle
                # Use high anomaly for 0 purchases, moderate for 1
                anomaly = 2.0 if len(cust_purchases) == 0 else 1.5
                result_rows.append({
                    "customer_id": cid,
                    "purchase_cycle_anomaly": anomaly,
                    "avg_purchase_cycle_days": 0.0,
                    "days_since_last_purchase": 0.0,
                })
                continue

            # Compute average inter-purchase interval
            dates = cust_purchases["event_date"].values
            intervals = np.diff(dates).astype("timedelta64[D]").astype(float)
            avg_cycle = float(np.mean(intervals))

            # Days since last purchase
            last_purchase = pd.Timestamp(dates[-1])
            days_since = (ref_date - last_purchase).days

            # Anomaly score
            anomaly = days_since / avg_cycle if avg_cycle > 0 else 2.0

            result_rows.append({
                "customer_id": cid,
                "purchase_cycle_anomaly": max(0.0, anomaly),
                "avg_purchase_cycle_days": avg_cycle,
                "days_since_last_purchase": float(max(0, days_since)),
            })

        return pd.DataFrame(result_rows)

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
        events = self._ensure_datetime(events)
        result_rows = []

        for cid in customers["customer_id"].unique():
            cust_events = events[events["customer_id"] == cid]
            row = {"customer_id": cid}

            if len(cust_events) == 0:
                row.update({
                    "avg_session_duration": 0.0,
                    "pageviews_per_session": 0.0,
                    "search_to_purchase_rate": 0.0,
                    "cart_abandonment_rate": 0.0,
                    "avg_events_per_day": 0.0,
                })
                result_rows.append(row)
                continue

            # Session duration
            if "session_duration" in events.columns:
                row["avg_session_duration"] = float(
                    cust_events["session_duration"].mean()
                )
            else:
                # Estimate from events per unique day
                n_days = cust_events["event_date"].nunique()
                row["avg_session_duration"] = (
                    len(cust_events) / max(1, n_days)
                )

            # Pageviews per session (per visit day)
            pageviews = len(
                cust_events[cust_events["event_type"] == "page_view"]
            )
            visit_days = cust_events["event_date"].nunique()
            row["pageviews_per_session"] = pageviews / max(1, visit_days)

            # Search to purchase rate
            searches = len(
                cust_events[cust_events["event_type"] == "search"]
            )
            purchases = len(
                cust_events[cust_events["event_type"] == "purchase"]
            )
            row["search_to_purchase_rate"] = (
                purchases / searches if searches > 0 else 0.0
            )

            # Cart abandonment rate
            cart_adds = len(
                cust_events[cust_events["event_type"] == "add_to_cart"]
            )
            cart_removes = len(
                cust_events[cust_events["event_type"] == "remove_from_cart"]
            )
            row["cart_abandonment_rate"] = (
                cart_removes / cart_adds if cart_adds > 0 else 0.0
            )

            # Avg events per day
            date_range = (
                cust_events["event_date"].max()
                - cust_events["event_date"].min()
            ).days + 1
            row["avg_events_per_day"] = len(cust_events) / max(1, date_range)

            result_rows.append(row)

        return pd.DataFrame(result_rows)

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
        events = self._ensure_datetime(events)

        event_types = [
            "page_view", "search", "add_to_cart", "remove_from_cart",
            "purchase", "coupon_use", "review", "cs_contact",
        ]

        result_rows = []
        distribution_vectors = []
        customer_ids = []

        for cid in customers["customer_id"].unique():
            cust_events = events[events["customer_id"] == cid]
            row = {"customer_id": cid}

            # Sequence diversity
            row["sequence_diversity"] = cust_events["event_type"].nunique()
            row["sequence_length"] = len(cust_events)

            # Event type distribution vector for clustering
            total = max(1, len(cust_events))
            dist = []
            for et in event_types:
                count = len(cust_events[cust_events["event_type"] == et])
                dist.append(count / total)
            distribution_vectors.append(dist)
            customer_ids.append(cid)

            # Purchase amount trend
            cust_purchases = cust_events[
                cust_events["event_type"] == "purchase"
            ].sort_values("event_date")

            if len(cust_purchases) >= 2 and "amount" in cust_purchases.columns:
                amounts = cust_purchases["amount"].fillna(0).values
                # Simple linear trend: positive = increasing spend
                x = np.arange(len(amounts))
                if np.std(x) > 0:
                    trend = np.corrcoef(x, amounts)[0, 1]
                    row["purchase_sequence_trend"] = (
                        0.0 if np.isnan(trend) else float(trend)
                    )
                else:
                    row["purchase_sequence_trend"] = 0.0
            else:
                row["purchase_sequence_trend"] = 0.0

            result_rows.append(row)

        result = pd.DataFrame(result_rows)

        # K-means clustering on event type distribution
        if len(distribution_vectors) > 0:
            dist_matrix = np.array(distribution_vectors)
            n_clusters = min(5, len(dist_matrix))
            if n_clusters >= 2:
                kmeans = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10
                )
                clusters = kmeans.fit_predict(dist_matrix)
                cluster_df = pd.DataFrame({
                    "customer_id": customer_ids,
                    "behavior_pattern_cluster": clusters,
                })
                result = result.merge(cluster_df, on="customer_id", how="left")
            else:
                result["behavior_pattern_cluster"] = 0

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
        events = self._ensure_datetime(events)
        result_rows = []

        for cid in customers["customer_id"].unique():
            cust_events = events[events["customer_id"] == cid]
            row = {"customer_id": cid}

            if len(cust_events) == 0:
                row.update({
                    "weekend_purchase_ratio": 0.0,
                    "evening_activity_ratio": 0.0,
                    "morning_activity_ratio": 0.0,
                    "night_activity_ratio": 0.0,
                    "peak_hour": 12,
                    "day_variance": 0.0,
                })
                result_rows.append(row)
                continue

            # Weekend purchase ratio
            purchases = cust_events[cust_events["event_type"] == "purchase"]
            if len(purchases) > 0:
                weekend_mask = purchases["event_date"].dt.dayofweek >= 5
                row["weekend_purchase_ratio"] = float(weekend_mask.mean())
            else:
                row["weekend_purchase_ratio"] = 0.0

            # Time-of-day features (from timestamp)
            if "timestamp" in events.columns:
                hours = cust_events["timestamp"].dt.hour
                total = len(hours)
                # Morning: 6-12, Afternoon: 12-18, Evening: 18-22, Night: 22-6
                row["morning_activity_ratio"] = float(
                    ((hours >= 6) & (hours < 12)).sum() / total
                )
                row["evening_activity_ratio"] = float(
                    ((hours >= 18) & (hours < 22)).sum() / total
                )
                row["night_activity_ratio"] = float(
                    ((hours >= 22) | (hours < 6)).sum() / total
                )
                row["peak_hour"] = int(hours.mode().iloc[0]) if len(hours) > 0 else 12
            else:
                row["morning_activity_ratio"] = 0.0
                row["evening_activity_ratio"] = 0.0
                row["night_activity_ratio"] = 0.0
                row["peak_hour"] = 12

            # Day variance: variance of daily event counts
            daily_counts = cust_events.groupby(
                cust_events["event_date"].dt.date
            ).size()
            row["day_variance"] = float(daily_counts.var()) if len(daily_counts) > 1 else 0.0

            result_rows.append(row)

        return pd.DataFrame(result_rows)

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
        events = self._ensure_datetime(events)
        result_rows = []

        for cid in customers["customer_id"].unique():
            cust_events = events[events["customer_id"] == cid]
            row = {"customer_id": cid}

            if len(cust_events) == 0:
                row["journey_stage"] = 4  # dormant
                row["stage_tenure_days"] = 0
                result_rows.append(row)
                continue

            dates = cust_events["event_date"]
            activity_span = (dates.max() - dates.min()).days

            # Recent activity (last 30 days of their activity)
            last_date = dates.max()
            recent_cutoff = last_date - pd.Timedelta(days=30)
            recent_events = cust_events[cust_events["event_date"] > recent_cutoff]
            older_events = cust_events[cust_events["event_date"] <= recent_cutoff]

            recent_rate = len(recent_events) / 30.0
            older_rate = (
                len(older_events) / max(1, activity_span - 30)
                if activity_span > 30
                else recent_rate
            )

            # Number of purchases
            n_purchases = len(
                cust_events[cust_events["event_type"] == "purchase"]
            )

            # Determine journey stage
            if activity_span < 30:
                stage = 0  # onboarding
                tenure = activity_span
            elif recent_rate > older_rate * 1.2:
                stage = 1  # growing
                tenure = min(30, activity_span)
            elif recent_rate > older_rate * 0.8 and n_purchases >= 3:
                stage = 2  # mature
                tenure = activity_span
            elif recent_rate < older_rate * 0.5:
                stage = 3  # declining
                # Estimate when decline started
                tenure = 30
            else:
                if recent_rate < 0.1:
                    stage = 4  # dormant
                    tenure = 30
                elif n_purchases < 2:
                    stage = 0  # still onboarding
                    tenure = activity_span
                else:
                    stage = 2  # mature
                    tenure = activity_span

            row["journey_stage"] = stage
            row["stage_tenure_days"] = max(0, tenure)

            result_rows.append(row)

        return pd.DataFrame(result_rows)

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
        features.to_parquet(
            os.path.join(store_path, "features.parquet"), index=False
        )
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
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
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
        return events

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
