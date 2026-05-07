"""
Personalized Recommendation Engine for Customer Retention Actions.

Implements a collaborative-filtering-inspired approach that scores retention
actions per customer based on churn probability, CLV, uplift score, behavioral
features, segment membership, and channel opt-in preferences.

Actions are ranked and filtered to respect opt-out constraints and budget
limits, producing explainable, personalized recommendations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Retention action catalogue ────────────────────────────────────────────────

_DEFAULT_ACTIONS: List[Dict[str, Any]] = [
    {
        "action_type": "coupon",
        "base_cost": 5000,
        "description": "Discount coupon for next purchase",
        "channel": None,
        "segment_affinity": {
            "bargain_hunter": 1.5,
            "explorer": 1.2,
            "dormant": 1.1,
            "new_customer": 1.0,
            "regular_loyal": 0.9,
            "vip_loyal": 0.7,
        },
    },
    {
        "action_type": "push_notification",
        "base_cost": 100,
        "description": "Personalized push notification",
        "channel": "push",
        "segment_affinity": {
            "explorer": 1.3,
            "regular_loyal": 1.2,
            "bargain_hunter": 1.1,
            "new_customer": 1.0,
            "vip_loyal": 0.9,
            "dormant": 0.8,
        },
    },
    {
        "action_type": "email_campaign",
        "base_cost": 200,
        "description": "Targeted email campaign",
        "channel": "email",
        "segment_affinity": {
            "regular_loyal": 1.3,
            "vip_loyal": 1.2,
            "new_customer": 1.2,
            "explorer": 1.0,
            "bargain_hunter": 0.9,
            "dormant": 0.8,
        },
    },
    {
        "action_type": "loyalty_points",
        "base_cost": 3000,
        "description": "Bonus loyalty points reward",
        "channel": None,
        "segment_affinity": {
            "vip_loyal": 1.4,
            "regular_loyal": 1.3,
            "new_customer": 1.2,
            "explorer": 1.0,
            "bargain_hunter": 0.8,
            "dormant": 0.7,
        },
    },
    {
        "action_type": "personal_outreach",
        "base_cost": 10000,
        "description": "Personal call or message from account manager",
        "channel": None,
        "segment_affinity": {
            "new_customer": 1.5,
            "vip_loyal": 1.3,
            "dormant": 1.1,
            "regular_loyal": 0.8,
            "explorer": 0.7,
            "bargain_hunter": 0.6,
        },
    },
    {
        "action_type": "exclusive_offer",
        "base_cost": 8000,
        "description": "Exclusive limited-time offer",
        "channel": None,
        "segment_affinity": {
            "new_customer": 1.4,
            "vip_loyal": 1.3,
            "dormant": 1.2,
            "regular_loyal": 1.0,
            "bargain_hunter": 1.0,
            "explorer": 0.9,
        },
    },
    {
        "action_type": "no_action",
        "base_cost": 0,
        "description": "No retention action",
        "channel": None,
        "segment_affinity": {},
    },
]

# ── Reason templates ──────────────────────────────────────────────────────────

_REASON_TEMPLATES: Dict[str, str] = {
    "coupon": "Customer in segment '{segment}' with churn risk {churn:.0%}; coupon aligns with purchase behavior.",
    "push_notification": "Push notification selected for segment '{segment}' with engagement uplift {uplift:.2f}.",
    "email_campaign": "Email campaign for segment '{segment}' customer (CLV={clv:,.0f} KRW).",
    "loyalty_points": "Loyalty points reward for '{segment}' customer to reinforce retention.",
    "personal_outreach": "High-touch outreach for '{segment}' customer with CLV={clv:,.0f} KRW and churn risk {churn:.0%}.",
    "exclusive_offer": "Exclusive offer targeting '{segment}' segment with uplift score {uplift:.2f}.",
    "no_action": "No action: customer has non-positive uplift or is classified as a sleeping dog.",
}


class RecommendationEngine:
    """Collaborative-filtering-inspired retention action recommender.

    Scores each (customer, action) pair using a weighted combination of:
    - Churn probability  (higher churn → higher urgency)
    - CLV                (higher value → higher priority)
    - Uplift score       (positive uplift → higher expected benefit)
    - Segment affinity   (action-segment alignment)
    - Channel opt-in     (hard filter)

    Parameters
    ----------
    config : dict
        Project configuration (loaded from ``config/simulator_config.yaml``).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.seed = config.get("simulation", {}).get("random_seed", 42)
        self.rng = np.random.RandomState(self.seed)

        self.total_budget = config.get("budget", {}).get("total_krw", 50_000_000)

        # Action catalogue (mutable copy)
        self.actions = [dict(a) for a in _DEFAULT_ACTIONS]

        # Scoring weights
        self.w_churn = 0.35
        self.w_clv = 0.25
        self.w_uplift = 0.20
        self.w_segment = 0.20

        logger.info(
            "RecommendationEngine initialised (seed=%d, %d actions)",
            self.seed,
            len(self.actions),
        )

    # ── Public API ────────────────────────────────────────────────────────

    def get_available_actions(self) -> List[str]:
        """Return list of available retention action type names."""
        return [a["action_type"] for a in self.actions]

    def recommend(
        self,
        *,
        data: pd.DataFrame,
        include_context: bool = False,
    ) -> pd.DataFrame:
        """Generate the single best recommendation per customer.

        Parameters
        ----------
        data : pd.DataFrame
            Customer feature DataFrame. Required columns:
            ``customer_id``, ``churn_prob``, ``clv``, ``uplift_score``,
            ``segment``. Optional: ``push_opt_in``, ``email_opt_in``.

        Returns
        -------
        pd.DataFrame
            One row per customer with columns:
            ``customer_id``, ``action_type``, ``score``, ``estimated_cost``,
            ``reason``.
        """
        scores_df = self._score_all_pairs(data)

        # For each customer pick the highest-scoring action
        idx = scores_df.groupby("customer_id")["score"].idxmax()
        best = scores_df.loc[idx].reset_index(drop=True)
        best = self._apply_no_action_policy(best, data)

        # Attach explanations
        best["reason"] = best.apply(
            lambda r: self._build_reason(r, data), axis=1,
        )
        if include_context:
            best = self._attach_decision_context(best, data)
            columns = [
                "customer_id", "action_type", "recommendation_type", "score",
                "estimated_cost", "reason", "segment", "uplift_segment",
                "uplift_score", "clv", "churn_probability", "priority_score",
                "expected_revenue_saved", "expected_roi", "recommended_offer",
                "expected_uplift",
            ]
            return best[columns]

        return best[[
            "customer_id", "action_type", "score", "estimated_cost", "reason"
        ]]

    def recommend_top_k(
        self,
        *,
        data: pd.DataFrame,
        k: int = 3,
        include_context: bool = False,
    ) -> pd.DataFrame:
        """Return top-*k* recommendations per customer, sorted by score.

        Parameters
        ----------
        data : pd.DataFrame
            Same schema as :meth:`recommend`.
        k : int
            Maximum number of recommendations per customer.

        Returns
        -------
        pd.DataFrame
            Up to *k* rows per customer.
        """
        scores_df = self._score_all_pairs(data)

        # Sort descending, keep top-k per customer
        scores_df = scores_df.sort_values(
            ["customer_id", "score"], ascending=[True, False],
        )
        top_k = scores_df.groupby("customer_id").head(k).reset_index(drop=True)
        no_action_ids = set(
            data.loc[self._no_action_mask(data), "customer_id"].astype(str)
        )
        if no_action_ids:
            top_k = top_k[~top_k["customer_id"].astype(str).isin(no_action_ids)]
            no_action_rows = pd.DataFrame({
                "customer_id": list(no_action_ids),
                "action_type": "no_action",
                "score": 0.0,
                "estimated_cost": 0.0,
            })
            top_k = pd.concat([top_k, no_action_rows], ignore_index=True)

        top_k["reason"] = top_k.apply(
            lambda r: self._build_reason(r, data), axis=1,
        )
        if include_context:
            top_k = self._attach_decision_context(top_k, data)
            return top_k[[
                "customer_id", "action_type", "recommendation_type", "score",
                "estimated_cost", "reason", "segment", "uplift_segment",
                "uplift_score", "clv", "churn_probability", "priority_score",
                "expected_revenue_saved", "expected_roi", "recommended_offer",
                "expected_uplift",
            ]]

        return top_k[[
            "customer_id", "action_type", "score", "estimated_cost", "reason"
        ]]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist engine state to disk.

        Saves configuration, action catalogue, and weights as JSON.
        """
        path = str(path)
        if not path.endswith(".json"):
            path = path + ".json"

        state = {
            "config": self.config,
            "seed": self.seed,
            "actions": self.actions,
            "weights": {
                "w_churn": self.w_churn,
                "w_clv": self.w_clv,
                "w_uplift": self.w_uplift,
                "w_segment": self.w_segment,
            },
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info("RecommendationEngine saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "RecommendationEngine":
        """Load a previously saved engine.

        Parameters
        ----------
        path : str
            File path (with or without ``.json`` suffix).

        Returns
        -------
        RecommendationEngine
        """
        path = str(path)
        if not path.endswith(".json"):
            path = path + ".json"

        with open(path, "r") as f:
            state = json.load(f)

        engine = cls(state["config"])
        engine.seed = state["seed"]
        engine.rng = np.random.RandomState(engine.seed)
        engine.actions = state["actions"]

        weights = state.get("weights", {})
        engine.w_churn = weights.get("w_churn", engine.w_churn)
        engine.w_clv = weights.get("w_clv", engine.w_clv)
        engine.w_uplift = weights.get("w_uplift", engine.w_uplift)
        engine.w_segment = weights.get("w_segment", engine.w_segment)

        logger.info("RecommendationEngine loaded from %s", path)
        return engine

    # ── Internal helpers ──────────────────────────────────────────────────

    def _score_all_pairs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Score every (customer, action) pair and apply hard filters.

        Returns a DataFrame with columns:
        ``customer_id``, ``action_type``, ``score``, ``estimated_cost``.
        """
        records: List[Dict[str, Any]] = []

        # Normalise CLV to [0, 1] for scoring
        clv_vals = data["clv"].values.astype(float)
        clv_min, clv_max = clv_vals.min(), clv_vals.max()
        clv_range = clv_max - clv_min if clv_max > clv_min else 1.0

        for _, row in data.iterrows():
            cid = row["customer_id"]
            churn = float(row["churn_prob"])
            clv_norm = (float(row["clv"]) - clv_min) / clv_range
            uplift = float(row["uplift_score"])
            segment = str(row.get("segment", "unknown"))

            push_ok = bool(row.get("push_opt_in", 1))
            email_ok = bool(row.get("email_opt_in", 1))

            for action in self.actions:
                atype = action["action_type"]
                if atype == "no_action":
                    continue

                # ── Hard channel filter ───────────────────────────────
                if action.get("channel") == "push" and not push_ok:
                    continue
                if action.get("channel") == "email" and not email_ok:
                    continue

                # ── Segment affinity ──────────────────────────────────
                seg_aff = action.get("segment_affinity", {}).get(segment, 1.0)

                # ── Composite score ───────────────────────────────────
                score = (
                    self.w_churn * churn
                    + self.w_clv * clv_norm
                    + self.w_uplift * max(uplift, 0)  # clamp negative
                    + self.w_segment * seg_aff
                )

                # Penalise negative uplift (sleeping dogs)
                if uplift < 0:
                    score *= max(0.3, 1.0 + uplift)

                # Scale cost by CLV-tier (high-value customers get bigger
                # investments)
                cost_multiplier = 0.5 + clv_norm  # 0.5 – 1.5×
                estimated_cost = action["base_cost"] * cost_multiplier

                records.append(
                    {
                        "customer_id": cid,
                        "action_type": atype,
                        "score": round(score, 6),
                        "estimated_cost": round(estimated_cost, 2),
                    }
                )

        return pd.DataFrame(records)

    def _no_action_mask(self, data: pd.DataFrame) -> pd.Series:
        """Identify customers where retention intervention should be skipped."""
        uplift = data.get("uplift_score", pd.Series(0.0, index=data.index)).astype(float)
        mask = uplift <= 0
        for col in ("segment", "uplift_segment"):
            if col in data.columns:
                mask = mask | data[col].astype(str).str.contains(
                    "sleeping_dog", case=False, na=False
                )
        return mask

    def _apply_no_action_policy(
        self,
        recommendations: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Force no-action outputs for sleeping dogs and non-positive uplift."""
        adjusted = recommendations.copy()
        data_indexed = data.copy()
        data_indexed["customer_id"] = data_indexed["customer_id"].astype(str)
        no_action_ids = set(
            data_indexed.loc[self._no_action_mask(data_indexed), "customer_id"]
        )
        mask = adjusted["customer_id"].astype(str).isin(no_action_ids)
        adjusted.loc[mask, "action_type"] = "no_action"
        adjusted.loc[mask, "score"] = 0.0
        adjusted.loc[mask, "estimated_cost"] = 0.0
        return adjusted

    def _attach_decision_context(
        self,
        recommendations: pd.DataFrame,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Attach model context required for final retention decisions."""
        context_cols = [
            c for c in [
                "customer_id", "segment", "uplift_segment", "uplift_score",
                "clv", "churn_prob", "churn_probability", "priority_score",
            ]
            if c in data.columns
        ]
        context = data[context_cols].copy()
        recs = recommendations.copy()
        recs["customer_id"] = recs["customer_id"].astype(str)
        context["customer_id"] = context["customer_id"].astype(str)
        recs = recs.merge(context, on="customer_id", how="left")
        if "churn_probability" not in recs.columns:
            recs["churn_probability"] = recs.get("churn_prob", 0.0)
        recs.drop(columns=["churn_prob"], inplace=True, errors="ignore")
        for col, default in [
            ("segment", "unknown"),
            ("uplift_segment", "unknown"),
            ("uplift_score", 0.0),
            ("clv", 0.0),
            ("churn_probability", 0.0),
            ("priority_score", recs["score"]),
        ]:
            if col not in recs.columns:
                recs[col] = default
        recs["expected_revenue_saved"] = (
            np.maximum(recs["uplift_score"].astype(float), 0.0)
            * recs["clv"].astype(float)
        )
        recs.loc[recs["action_type"].eq("no_action"), "expected_revenue_saved"] = 0.0
        recs["expected_roi"] = np.where(
            recs["estimated_cost"].astype(float) > 0,
            recs["expected_revenue_saved"] / recs["estimated_cost"].astype(float),
            0.0,
        )
        recs["recommendation_type"] = recs["action_type"]
        recs["recommended_offer"] = recs["action_type"]
        recs["expected_uplift"] = recs["uplift_score"]
        return recs

    def _build_reason(self, rec_row: pd.Series, data: pd.DataFrame) -> str:
        """Build a human-readable explanation for a recommendation."""
        cid = rec_row["customer_id"]
        atype = rec_row["action_type"]

        cust = data.loc[data["customer_id"].astype(str) == str(cid)].iloc[0]
        segment = str(cust.get("segment", "unknown"))
        churn = float(cust.get("churn_prob", 0))
        clv = float(cust.get("clv", 0))
        uplift = float(cust.get("uplift_score", 0))

        template = _REASON_TEMPLATES.get(
            atype,
            "Recommended '{action}' for segment '{segment}'.",
        )
        try:
            return template.format(
                segment=segment,
                churn=churn,
                clv=clv,
                uplift=uplift,
                action=atype,
            )
        except (KeyError, ValueError):
            return f"Action '{atype}' recommended for segment '{segment}'."
