"""
Customer Segmentation Module for E-Commerce Churn Prediction.

Provides RFM-based and K-Means-based customer segmentation with
configurable thresholds. Supports 6+ customer segments defined
via YAML configuration.

Segments:
    1. VIP Loyal (VIP충성고객) - Top-tier high RFM
    2. Loyal Customer (충성고객) - High frequency, moderate+ spend
    3. Potential Loyalist (잠재충성고객) - Recent, growing engagement
    4. At Risk (이탈위험고객) - Declining recency, historically active
    5. Hibernating (휴면고객) - Low recency, low frequency
    6. Explorer (탐색형) - Recent but limited history
    7. New Customer (신규가입자) - High spend, declining recency
    8. Bargain Hunter (가격민감형) - Frequent low-value buyers

Usage:
    segmenter = CustomerSegmenter(config=segmentation_config)
    result = segmenter.segment_customers(rfm_features)
    summary = segmenter.get_segment_summary(result)
    actions = segmenter.get_retention_actions(result)
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# Default segment definitions (used when no config provided)
DEFAULT_SEGMENTS = [
    {
        "name": "vip_loyal",
        "name_kr": "VIP충성고객",
        "description": "Top-tier customers with highest RFM scores",
        "color": "#2ecc71",
        "priority": 1,
        "criteria": {
            "recency_score_min": 4,
            "frequency_score_min": 4,
            "monetary_score_min": 4,
        },
        "retention_action": "exclusive_rewards",
    },
    {
        "name": "loyal_customer",
        "name_kr": "충성고객",
        "description": "Consistent buyers with strong frequency",
        "color": "#27ae60",
        "priority": 2,
        "criteria": {
            "recency_score_min": 3,
            "frequency_score_min": 4,
            "monetary_score_min": 2,
        },
        "retention_action": "loyalty_program",
    },
    {
        "name": "potential_loyalist",
        "name_kr": "잠재충성고객",
        "description": "Recent customers with growing engagement",
        "color": "#3498db",
        "priority": 3,
        "criteria": {
            "recency_score_min": 4,
            "frequency_score_min": 2,
            "monetary_score_min": 2,
        },
        "retention_action": "engagement_campaign",
    },
    {
        "name": "at_risk",
        "name_kr": "이탈위험고객",
        "description": "Previously active customers showing decline",
        "color": "#e67e22",
        "priority": 4,
        "criteria": {
            "recency_score_max": 2,
            "frequency_score_min": 3,
            "monetary_score_min": 2,
        },
        "retention_action": "win_back_campaign",
    },
    {
        "name": "hibernating",
        "name_kr": "휴면고객",
        "description": "Low recency, low frequency customers",
        "color": "#e74c3c",
        "priority": 5,
        "criteria": {
            "recency_score_max": 2,
            "frequency_score_max": 2,
            "monetary_score_min": 1,
        },
        "retention_action": "reactivation_offer",
    },
    {
        "name": "explorer",
        "name_kr": "탐색형",
        "description": "Recently acquired with limited history",
        "color": "#9b59b6",
        "priority": 6,
        "criteria": {
            "recency_score_min": 4,
            "frequency_score_max": 1,
            "monetary_score_max": 2,
        },
        "retention_action": "onboarding_sequence",
    },
    {
        "name": "new_customer",
        "name_kr": "신규가입자",
        "description": "Newly registered customers",
        "color": "#c0392b",
        "priority": 7,
        "criteria": {
            "recency_score_max": 3,
            "frequency_score_min": 2,
            "monetary_score_min": 4,
        },
        "retention_action": "premium_win_back",
    },
    {
        "name": "bargain_hunter",
        "name_kr": "가격민감형",
        "description": "Frequent low-value buyers driven by promotions",
        "color": "#f39c12",
        "priority": 8,
        "criteria": {
            "recency_score_min": 2,
            "frequency_score_min": 3,
            "monetary_score_max": 2,
        },
        "retention_action": "targeted_promotion",
    },
]

DEFAULT_CONFIG = {
    "method": "rfm_behavioral",
    "n_rfm_bins": 5,
    "kmeans_clusters": 6,
    "segments": DEFAULT_SEGMENTS,
}


class CustomerSegmenter:
    """Customer segmentation using RFM scoring or K-Means clustering.

    Supports 6+ configurable segments with threshold-based assignment
    from YAML config. Falls back to K-Means nearest-centroid for
    customers not matching any rule-based segment.

    Attributes:
        config: Segmentation configuration dictionary.
        segment_definitions: List of segment definitions with criteria.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the customer segmenter.

        Args:
            config: Segmentation configuration. If None, uses defaults.
                Expected keys: method, n_rfm_bins, kmeans_clusters, segments.
        """
        self.config = {**DEFAULT_CONFIG}
        if config:
            self.config.update(config)

        self.segment_definitions = self.config.get(
            "segments", DEFAULT_SEGMENTS
        )
        self.n_rfm_bins = self.config.get("n_rfm_bins", 5)
        self._random_state = 42

    # ------------------------------------------------------------------
    # RFM Scoring
    # ------------------------------------------------------------------

    def compute_rfm_scores(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute RFM quantile scores for each customer.

        Recency is inverse-scored (lower recency = higher score).
        Frequency and Monetary are direct-scored (higher = better).

        Args:
            data: DataFrame with customer_id, recency, frequency, monetary.

        Returns:
            DataFrame with original columns plus recency_score,
            frequency_score, monetary_score (1 to n_rfm_bins).
        """
        result = data.copy()
        n_bins = self.n_rfm_bins

        # Recency: lower is better, so invert the quantile labels
        result["recency_score"] = self._quantile_score(
            result["recency"], n_bins, inverse=True
        )

        # Frequency: higher is better
        result["frequency_score"] = self._quantile_score(
            result["frequency"], n_bins, inverse=False
        )

        # Monetary: higher is better
        result["monetary_score"] = self._quantile_score(
            result["monetary"], n_bins, inverse=False
        )

        return result

    # ------------------------------------------------------------------
    # Segment Assignment
    # ------------------------------------------------------------------

    def segment_customers(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Assign each customer to a segment.

        Uses the configured method (rfm_behavioral or kmeans).

        Args:
            data: DataFrame with customer_id, recency, frequency, monetary.

        Returns:
            DataFrame with customer_id and segment columns.
        """
        method = self.config.get("method", "rfm_behavioral")

        if method == "kmeans":
            return self._segment_kmeans(data)
        else:
            return self._segment_rfm_rules(data)

    def _segment_rfm_rules(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Assign segments using RFM rule-based criteria.

        Evaluates each segment definition's criteria against RFM scores.
        Segments are evaluated in priority order; first match wins.
        Unmatched customers get assigned via nearest-centroid fallback.

        Args:
            data: DataFrame with RFM features.

        Returns:
            DataFrame with customer_id and segment.
        """
        scored = self.compute_rfm_scores(data)

        # Sort segment definitions by priority
        sorted_segments = sorted(
            self.segment_definitions,
            key=lambda s: s.get("priority", 99),
        )

        # Initialize segment column
        scored["segment"] = None

        for seg_def in sorted_segments:
            name = seg_def["name"]
            criteria = seg_def.get("criteria", {})

            # Build mask from criteria
            mask = pd.Series(True, index=scored.index)

            for key, value in criteria.items():
                if key == "recency_score_min":
                    mask &= scored["recency_score"] >= value
                elif key == "recency_score_max":
                    mask &= scored["recency_score"] <= value
                elif key == "frequency_score_min":
                    mask &= scored["frequency_score"] >= value
                elif key == "frequency_score_max":
                    mask &= scored["frequency_score"] <= value
                elif key == "monetary_score_min":
                    mask &= scored["monetary_score"] >= value
                elif key == "monetary_score_max":
                    mask &= scored["monetary_score"] <= value

            # Only assign to unassigned customers
            unassigned = scored["segment"].isna()
            scored.loc[mask & unassigned, "segment"] = name

        # Fallback: assign remaining customers using nearest centroid
        unassigned_mask = scored["segment"].isna()
        if unassigned_mask.any():
            scored = self._assign_fallback(scored, unassigned_mask)

        result = scored[["customer_id", "segment"]].copy()

        # Merge back original RFM columns
        for col in ["recency", "frequency", "monetary"]:
            if col in data.columns:
                result[col] = data[col].values

        # Also carry over churn_probability if present
        if "churn_probability" in data.columns:
            result["churn_probability"] = data["churn_probability"].values

        return result

    def _assign_fallback(
        self,
        scored: pd.DataFrame,
        unassigned_mask: pd.Series,
    ) -> pd.DataFrame:
        """Assign unmatched customers to nearest segment centroid.

        Computes the centroid (mean RFM scores) of each assigned segment,
        then assigns unassigned customers to the closest centroid.

        Args:
            scored: DataFrame with RFM scores and partial segment assignment.
            unassigned_mask: Boolean mask of unassigned customers.

        Returns:
            DataFrame with all customers assigned.
        """
        assigned = scored[~unassigned_mask]
        rfm_cols = ["recency_score", "frequency_score", "monetary_score"]

        if len(assigned) == 0:
            # No rules matched at all — use "other"
            scored.loc[unassigned_mask, "segment"] = "other"
            return scored

        # Compute centroids per segment
        centroids = assigned.groupby("segment")[rfm_cols].mean()

        # For each unassigned customer, find nearest centroid
        centroid_matrix = centroids.values.astype(float)
        segment_names = centroids.index.tolist()

        for idx in scored.index[unassigned_mask]:
            customer_rfm = np.array(
                [float(scored.loc[idx, c]) for c in rfm_cols]
            ).reshape(1, -1)
            diff = centroid_matrix - customer_rfm
            distances = np.sqrt(np.sum(diff ** 2, axis=1))
            nearest_idx = int(np.argmin(distances))
            scored.loc[idx, "segment"] = segment_names[nearest_idx]

        return scored

    def _segment_kmeans(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        """Assign segments using K-Means clustering on RFM features.

        Args:
            data: DataFrame with RFM features.

        Returns:
            DataFrame with customer_id and segment columns.
        """
        n_clusters = self.config.get("kmeans_clusters", 6)
        rfm_cols = ["recency", "frequency", "monetary"]

        # Prepare features
        features = data[rfm_cols].copy()

        # Handle edge cases
        if len(data) <= 1:
            result = data[["customer_id"]].copy()
            result["segment"] = "cluster_0"
            for col in rfm_cols:
                result[col] = data[col].values
            return result

        # Normalize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features.fillna(0))

        # Fit K-Means
        actual_k = min(n_clusters, len(data))
        kmeans = KMeans(
            n_clusters=actual_k,
            random_state=self._random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(features_scaled)

        # Create result with cluster labels as segment names
        result = data[["customer_id"]].copy()

        # Name clusters by their characteristics
        cluster_names = self._name_kmeans_clusters(
            data, labels, actual_k
        )
        result["segment"] = [cluster_names[label] for label in labels]

        for col in rfm_cols:
            result[col] = data[col].values

        if "churn_probability" in data.columns:
            result["churn_probability"] = data["churn_probability"].values

        return result

    def _name_kmeans_clusters(
        self,
        data: pd.DataFrame,
        labels: np.ndarray,
        n_clusters: int,
    ) -> Dict[int, str]:
        """Name K-Means clusters based on RFM characteristics.

        Args:
            data: Original data DataFrame.
            labels: Cluster labels array.
            n_clusters: Number of clusters.

        Returns:
            Dictionary mapping cluster index to segment name.
        """
        temp = data[["recency", "frequency", "monetary"]].copy()
        temp["cluster"] = labels

        centroids = temp.groupby("cluster").mean()

        # Rank clusters by composite score (lower recency + higher F + M)
        centroids["composite"] = (
            -centroids["recency"].rank()
            + centroids["frequency"].rank()
            + centroids["monetary"].rank()
        )

        ranked = centroids["composite"].rank(ascending=False).astype(int)

        name_pool = [
            "vip_loyal", "loyal_customer", "potential_loyalist",
            "at_risk", "hibernating", "explorer",
            "new_customer", "bargain_hunter",
        ]

        cluster_names = {}
        for cluster_id in range(n_clusters):
            rank = ranked.get(cluster_id, cluster_id + 1)
            idx = min(rank - 1, len(name_pool) - 1)
            cluster_names[cluster_id] = name_pool[idx]

        return cluster_names

    # ------------------------------------------------------------------
    # Segment Summary
    # ------------------------------------------------------------------

    def get_segment_summary(
        self, segmented_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute summary statistics per segment.

        Args:
            segmented_data: DataFrame with segment, recency, frequency,
                monetary columns.

        Returns:
            DataFrame with segment, count, percentage, avg_recency,
            avg_frequency, avg_monetary, and optional avg_churn_probability.
        """
        total = len(segmented_data)

        agg_dict = {"customer_id": "count"}

        # Aggregate available RFM columns
        for col, agg_name in [
            ("recency", "avg_recency"),
            ("frequency", "avg_frequency"),
            ("monetary", "avg_monetary"),
            ("churn_probability", "avg_churn_probability"),
            ("uplift_score", "avg_uplift_score"),
            ("predicted_clv", "avg_predicted_clv"),
            ("clv", "avg_clv"),
            ("priority_score", "avg_priority_score"),
        ]:
            if col in segmented_data.columns:
                agg_dict[col] = "mean"

        summary = segmented_data.groupby("segment").agg(agg_dict)
        summary = summary.rename(columns={
            "customer_id": "count",
            "recency": "avg_recency",
            "frequency": "avg_frequency",
            "monetary": "avg_monetary",
            "churn_probability": "avg_churn_probability",
            "uplift_score": "avg_uplift_score",
            "predicted_clv": "avg_predicted_clv",
            "clv": "avg_clv",
            "priority_score": "avg_priority_score",
        })

        summary["percentage"] = (summary["count"] / total) * 100.0
        summary = summary.reset_index()

        # Add segment metadata
        meta_map = {
            s["name"]: s
            for s in self.segment_definitions
        }

        summary["name_kr"] = summary["segment"].map(
            lambda s: meta_map.get(s, {}).get("name_kr", s)
        )
        summary["description"] = summary["segment"].map(
            lambda s: meta_map.get(s, {}).get("description", "")
        )
        summary["color"] = summary["segment"].map(
            lambda s: meta_map.get(s, {}).get("color", "#95a5a6")
        )

        # Sort by count descending
        summary = summary.sort_values("count", ascending=False).reset_index(
            drop=True
        )

        return summary

    def segment_value_uplift_customers(
        self,
        data: pd.DataFrame,
        churn_col: str = "churn_probability",
        uplift_col: str = "uplift_score",
        clv_col: str = "predicted_clv",
        customer_col: str = "customer_id",
        new_customer_col: str = "tenure_days",
    ) -> pd.DataFrame:
        """Create 6+ operational segments from churn, uplift, and CLV."""
        required = {customer_col, churn_col, uplift_col, clv_col}
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"Missing required columns for value segmentation: {missing}")

        result = data.copy()
        churn = result[churn_col].astype(float)
        uplift = result[uplift_col].astype(float)
        clv = result[clv_col].astype(float)

        high_churn_threshold = float(self.config.get("high_churn_threshold", 0.5))
        high_value_threshold = float(clv.quantile(self.config.get("high_value_quantile", 0.8)))
        neutral_band = float(self.config.get("neutral_uplift_threshold", 0.05))
        onboarding_days = float(self.config.get("new_customer_days", 30))

        result["value_tier"] = np.where(clv >= high_value_threshold, "high_value", "low_value")
        result["high_value"] = result["value_tier"].eq("high_value")
        result["high_churn"] = churn >= high_churn_threshold
        result["positive_uplift"] = uplift > 0
        result["uplift_class"] = np.where(
            uplift < 0,
            "sleeping_dog",
            np.where(np.abs(uplift) <= neutral_band, "neutral", "persuadable"),
        )
        result["priority_score"] = uplift * clv

        segments = []
        for _, row in result.iterrows():
            if new_customer_col in result.columns and float(row[new_customer_col]) <= onboarding_days:
                segments.append("new_customer_onboarding")
                continue

            if row["uplift_class"] == "sleeping_dog":
                segments.append("sleeping_dog")
            elif row["uplift_class"] == "persuadable" and float(row[churn_col]) >= high_churn_threshold:
                prefix = "high_value" if row["value_tier"] == "high_value" else "low_value"
                segments.append(f"{prefix}_persuadable")
            elif float(row[churn_col]) >= high_churn_threshold:
                prefix = "high_value" if row["value_tier"] == "high_value" else "low_value"
                segments.append(f"{prefix}_lost_cause")
            else:
                prefix = "high_value" if row["value_tier"] == "high_value" else "low_value"
                segments.append(f"{prefix}_sure_thing")

        result["segment"] = segments
        cols = [customer_col, "segment", churn_col, uplift_col, clv_col, "priority_score"]
        optional = [
            c
            for c in [
                "value_tier",
                "high_value",
                "high_churn",
                "positive_uplift",
                "uplift_class",
                new_customer_col,
            ]
            if c in result.columns
        ]
        passthrough = [c for c in result.columns if c not in cols + optional and c in ("recency", "frequency", "monetary")]
        output = result[cols + optional + passthrough].copy()
        output.attrs["high_value_actionable_evidence"] = self.build_value_uplift_evidence(
            output,
            churn_col=churn_col,
            uplift_col=uplift_col,
            clv_col=clv_col,
            customer_col=customer_col,
            high_value_threshold=high_value_threshold,
            high_churn_threshold=high_churn_threshold,
            neutral_uplift_threshold=neutral_band,
        )
        return output

    def build_value_uplift_evidence(
        self,
        segmented_data: pd.DataFrame,
        churn_col: str = "churn_probability",
        uplift_col: str = "uplift_score",
        clv_col: str = "predicted_clv",
        customer_col: str = "customer_id",
        high_value_threshold: Optional[float] = None,
        high_churn_threshold: Optional[float] = None,
        neutral_uplift_threshold: Optional[float] = None,
        sample_size: int = 20,
    ) -> Dict[str, Any]:
        """Summarize high-value actionable segment evidence.

        The report passes validation when high-value persuadable or lost-cause
        customers exist. If they do not, it carries an explicit absence report
        with the calibration thresholds and population counts used.
        """
        required = {customer_col, "segment", churn_col, uplift_col, clv_col}
        missing = sorted(required - set(segmented_data.columns))
        if missing:
            raise ValueError(f"Missing columns for segment evidence: {missing}")

        frame = segmented_data.copy()
        clv = frame[clv_col].astype(float)
        churn = frame[churn_col].astype(float)
        uplift = frame[uplift_col].astype(float)

        if high_value_threshold is None:
            high_value_threshold = float(
                clv.quantile(self.config.get("high_value_quantile", 0.8))
            )
        if high_churn_threshold is None:
            high_churn_threshold = float(self.config.get("high_churn_threshold", 0.5))
        if neutral_uplift_threshold is None:
            neutral_uplift_threshold = float(
                self.config.get("neutral_uplift_threshold", 0.05)
            )

        high_value = (
            frame["high_value"].astype(bool)
            if "high_value" in frame.columns
            else clv >= high_value_threshold
        )
        high_churn = (
            frame["high_churn"].astype(bool)
            if "high_churn" in frame.columns
            else churn >= high_churn_threshold
        )
        positive_uplift = (
            frame["positive_uplift"].astype(bool)
            if "positive_uplift" in frame.columns
            else uplift > 0
        )

        high_value_persuadable = frame["segment"].eq("high_value_persuadable")
        high_value_lost_cause = frame["segment"].eq("high_value_lost_cause")
        high_value_actionable = high_value_persuadable | high_value_lost_cause
        high_value_at_risk_positive = high_value & high_churn & positive_uplift
        high_value_at_risk_nonpositive = high_value & high_churn & ~positive_uplift

        sample_cols = [
            c
            for c in [
                customer_col,
                "segment",
                churn_col,
                uplift_col,
                clv_col,
                "priority_score",
            ]
            if c in frame.columns
        ]
        sample = (
            frame.loc[high_value_actionable, sample_cols]
            .sort_values(
                by="priority_score" if "priority_score" in sample_cols else clv_col,
                ascending=False,
            )
            .head(sample_size)
            .to_dict(orient="records")
        )

        report: Dict[str, Any] = {
            "high_value_threshold": float(high_value_threshold),
            "high_churn_threshold": float(high_churn_threshold),
            "neutral_uplift_threshold": float(neutral_uplift_threshold),
            "total_customers": int(len(frame)),
            "high_value_count": int(high_value.sum()),
            "high_risk_count": int(high_churn.sum()),
            "positive_uplift_count": int(positive_uplift.sum()),
            "high_value_persuadable_count": int(high_value_persuadable.sum()),
            "high_value_lost_cause_count": int(high_value_lost_cause.sum()),
            "high_value_actionable_count": int(high_value_actionable.sum()),
            "high_value_at_risk_positive_uplift_count": int(
                high_value_at_risk_positive.sum()
            ),
            "high_value_at_risk_nonpositive_uplift_count": int(
                high_value_at_risk_nonpositive.sum()
            ),
            "actionable_samples": sample,
            "absence_report": None,
            "absence_reason": None,
        }

        if report["high_value_actionable_count"] == 0:
            report["absence_report"] = {
                "reason": (
                    "No high-value customers met persuadable or lost-cause "
                    "criteria under the configured churn/uplift/CLV thresholds."
                ),
                "thresholds": {
                    "high_value_threshold": float(high_value_threshold),
                    "high_churn_threshold": float(high_churn_threshold),
                    "neutral_uplift_threshold": float(neutral_uplift_threshold),
                },
                "counts": {
                    "total_customers": report["total_customers"],
                    "high_value_count": report["high_value_count"],
                    "high_risk_count": report["high_risk_count"],
                    "positive_uplift_count": report["positive_uplift_count"],
                    "high_value_at_risk_positive_uplift_count": report[
                        "high_value_at_risk_positive_uplift_count"
                    ],
                    "high_value_at_risk_nonpositive_uplift_count": report[
                        "high_value_at_risk_nonpositive_uplift_count"
                    ],
                },
            }
            report["absence_reason"] = report["absence_report"]["reason"]

        return report

    def validate_value_uplift_evidence(
        self,
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate high-value actionable evidence or structured absence."""
        has_actionable_segment = (
            int(evidence.get("high_value_persuadable_count", 0)) > 0
            or int(evidence.get("high_value_lost_cause_count", 0)) > 0
        )
        absence = evidence.get("absence_report")

        if has_actionable_segment:
            return {"valid": True, "reason": "high_value_actionable_evidence_present"}

        if isinstance(absence, dict) and absence.get("reason") and absence.get("counts"):
            return {"valid": True, "reason": "structured_absence_report_present"}

        return {
            "valid": False,
            "reason": "missing_high_value_actionable_evidence_or_absence_report",
        }

    # ------------------------------------------------------------------
    # Retention Actions
    # ------------------------------------------------------------------

    def get_retention_actions(
        self, segmented_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Map each customer's segment to a retention action.

        Args:
            segmented_data: DataFrame with customer_id and segment columns.

        Returns:
            DataFrame with customer_id, segment, retention_action columns.
        """
        result = segmented_data[["customer_id", "segment"]].copy()

        # Build action mapping from config
        action_map = {
            s["name"]: s.get("retention_action", "general")
            for s in self.segment_definitions
        }
        action_map["other"] = "general"

        result["retention_action"] = result["segment"].map(action_map)
        # Fallback for any unmapped segments
        result["retention_action"] = result["retention_action"].fillna(
            "general"
        )

        return result

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_score(
        series: pd.Series,
        n_bins: int,
        inverse: bool = False,
    ) -> pd.Series:
        """Score a series using quantile-based binning.

        Args:
            series: Numeric series to score.
            n_bins: Number of quantile bins.
            inverse: If True, lower values get higher scores.

        Returns:
            Series of integer scores from 1 to n_bins.
        """
        # Handle constant or near-constant series
        if series.nunique() <= 1:
            return pd.Series(
                [n_bins if not inverse else 1] * len(series),
                index=series.index,
            )

        try:
            labels = list(range(1, n_bins + 1))
            if inverse:
                labels = labels[::-1]

            scores = pd.qcut(
                series.rank(method="first"),
                q=n_bins,
                labels=labels,
                duplicates="drop",
            ).astype(int)
        except (ValueError, IndexError):
            # Fallback: use rank-based scoring
            ranks = series.rank(pct=True)
            if inverse:
                ranks = 1 - ranks
            scores = (ranks * (n_bins - 1)).round().astype(int) + 1
            scores = scores.clip(1, n_bins)

        return scores
