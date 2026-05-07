"""
Dashboard Data Loader Module.

Loads predictions, model metrics, A/B test results, budget optimization
results, survival data, recommendations, uplift results, and CLV data
from file-based feature store artifacts for the Streamlit dashboard.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default artifacts directory
DEFAULT_ARTIFACTS_DIR = Path("data/artifacts")
DEFAULT_RESULTS_DIR = Path("results")


class DashboardDataLoader:
    """Loads all dashboard data from file-based feature store.

    Attributes:
        config: Parsed YAML configuration dictionary.
        artifacts_dir: Path to the artifacts directory.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader from config.

        Args:
            config: Parsed simulator_config.yaml dict.
        """
        self.config = config
        dashboard_cfg = config.get("dashboard", {}) or {}
        self.artifacts_dir = Path(
            dashboard_cfg.get("artifacts_dir", str(DEFAULT_ARTIFACTS_DIR))
        )
        self.results_dir = Path(
            dashboard_cfg.get("results_dir", str(DEFAULT_RESULTS_DIR))
        )
        if "results_dir" in dashboard_cfg:
            self.search_dirs = [self.results_dir, self.artifacts_dir]
        elif "artifacts_dir" in dashboard_cfg:
            self.search_dirs = [self.artifacts_dir, self.results_dir]
        else:
            self.search_dirs = [self.results_dir, self.artifacts_dir]
        self._artifact_issues: Dict[str, str] = {}

    def _record_artifact_issue(self, artifact_name: str, message: str) -> None:
        """Remember a dashboard-visible issue for the most recent load."""
        self._artifact_issues[artifact_name] = message
        logger.warning("%s: %s", artifact_name, message)

    def _clear_artifact_issue(self, artifact_name: str) -> None:
        self._artifact_issues.pop(artifact_name, None)

    def get_artifact_issue(self, artifact_name: str) -> Optional[str]:
        """Return the latest dashboard-visible issue for a loader."""
        return self._artifact_issues.get(artifact_name)

    def get_artifact_issues(self) -> Dict[str, str]:
        """Return current dashboard-visible loader issues."""
        return dict(self._artifact_issues)

    def _resolve_existing_path(self, *candidates: str) -> Optional[Path]:
        """Find the first existing file across dashboard artifact locations."""
        for directory in self.search_dirs:
            for candidate in candidates:
                path = directory / candidate
                if path.exists():
                    return path
        return None

    def _read_json(self, path: Path) -> Any:
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _empty_frame(
        columns: List[str],
        artifact_name: str,
        issue: str,
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=columns)
        df.attrs["artifact_name"] = artifact_name
        df.attrs["dashboard_issue"] = issue
        return df

    def _required_csv(
        self,
        artifact_name: str,
        candidates: List[str],
        required_columns: List[str],
    ) -> Optional[pd.DataFrame]:
        path = self._resolve_existing_path(*candidates)
        if path is None:
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact missing: one of {', '.join(candidates)}.",
            )
            return None

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} could not be read: {exc}.",
            )
            return None

        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is invalid; missing columns: {', '.join(missing)}.",
            )
            return None
        if df.empty:
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is empty.",
            )
            return None

        self._clear_artifact_issue(artifact_name)
        return df

    def _required_json(
        self,
        artifact_name: str,
        candidates: List[str],
        required_keys: List[str],
    ) -> Optional[Dict[str, Any]]:
        path = self._resolve_existing_path(*candidates)
        if path is None:
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact missing: one of {', '.join(candidates)}.",
            )
            return None

        try:
            payload = self._read_json(path)
        except Exception as exc:
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} could not be read: {exc}.",
            )
            return None
        if not isinstance(payload, dict):
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is invalid; expected a JSON object.",
            )
            return None

        missing = [k for k in required_keys if k not in payload]
        if missing:
            self._record_artifact_issue(
                artifact_name,
                f"Required artifact {path} is invalid; missing keys: {', '.join(missing)}.",
            )
            return None

        self._clear_artifact_issue(artifact_name)
        return payload

    def _customers_path(self) -> Optional[Path]:
        for path in [
            self.results_dir / "customers.csv",
            self.artifacts_dir / "customers.csv",
            Path("data/raw/customers.csv"),
        ]:
            if path.exists():
                return path
        return None

    def _load_metric_timeseries(self, metric_name: str) -> pd.DataFrame:
        """Load time-series metrics from dedicated files or MLflow exports."""
        path = self._resolve_existing_path(
            f"{metric_name}_history.csv",
            f"{metric_name}_timeseries.csv",
            "model_performance_timeseries.csv",
            "model_performance_history.csv",
        )
        if path is not None:
            df = pd.read_csv(path)
            if "model_type" not in df.columns and "model" in df.columns:
                df = df.rename(columns={"model": "model_type"})
            if metric_name in df.columns:
                cols = [c for c in ["timestamp", "model_type", metric_name] if c in df.columns]
                return df[cols]
            value_cols = [
                c for c in df.columns
                if c.lower() in {metric_name.lower(), f"{metric_name.lower()}_value"}
            ]
            if value_cols:
                rename_map = {value_cols[0]: metric_name}
                return df.rename(columns=rename_map)

        runs = self.load_mlflow_runs()
        if metric_name in runs.columns:
            cols = [c for c in ["timestamp", "model_type", metric_name] if c in runs.columns]
            return runs[cols].copy()
        return pd.DataFrame(columns=["timestamp", "model_type", metric_name])

    def _adapt_budget_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map pipeline budget outputs to dashboard schema."""
        adapted = df.copy()
        if "allocated_budget_krw" not in adapted.columns and "allocated_budget" in adapted.columns:
            adapted["allocated_budget_krw"] = adapted["allocated_budget"]
        if "expected_revenue_saved_krw" not in adapted.columns:
            for candidate in [
                "expected_revenue_saved",
                "retained_value",
                "expected_retained_value",
            ]:
                if candidate in adapted.columns:
                    adapted["expected_revenue_saved_krw"] = adapted[candidate]
                    break
        if "expected_retained" not in adapted.columns:
            if "customers_treated" in adapted.columns:
                adapted["expected_retained"] = adapted["customers_treated"]
            elif "customer_id" in adapted.columns:
                adapted["expected_retained"] = 1
        if "roi" not in adapted.columns:
            spent = adapted.get("allocated_budget_krw", pd.Series(dtype=float)).replace(0, np.nan)
            revenue = adapted.get("expected_revenue_saved_krw", pd.Series(dtype=float))
            adapted["roi"] = (revenue / spent).fillna(0.0)
        if "segment" not in adapted.columns:
            if "customer_id" in adapted.columns:
                adapted["segment"] = "all_customers"
            else:
                adapted["segment"] = [f"scenario_{i+1}" for i in range(len(adapted))]

        required = [
            "segment", "allocated_budget_krw", "expected_retained",
            "expected_revenue_saved_krw", "roi",
        ]
        for col in required:
            if col not in adapted.columns:
                adapted[col] = 0.0 if col != "segment" else "unknown"
        return adapted[required]

    def _adapt_clv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        adapted = df.copy()
        if "clv_predicted" not in adapted.columns and "predicted_clv" in adapted.columns:
            adapted["clv_predicted"] = adapted["predicted_clv"]
        if "clv_predicted" in adapted.columns:
            adapted["clv_predicted"] = pd.to_numeric(
                adapted["clv_predicted"], errors="coerce"
            )
            issues = []
            null_count = int(adapted["clv_predicted"].isna().sum())
            negative_count = int((adapted["clv_predicted"] < 0).sum())
            if null_count:
                issues.append(f"{null_count:,} null/non-numeric CLV rows")
            if negative_count:
                issues.append(f"{negative_count:,} negative CLV rows")
            if "customer_id" in adapted.columns:
                duplicate_count = int(
                    adapted["customer_id"].astype(str).duplicated().sum()
                )
                if duplicate_count:
                    issues.append(f"{duplicate_count:,} duplicate customer rows")
                customers_path = self._customers_path()
                if customers_path is not None:
                    try:
                        expected_ids = pd.read_csv(
                            customers_path, usecols=["customer_id"]
                        )["customer_id"].astype(str)
                        actual_ids = adapted["customer_id"].astype(str)
                        missing_count = int(len(set(expected_ids) - set(actual_ids)))
                        if missing_count:
                            issues.append(
                                f"{missing_count:,} customers missing from CLV artifact"
                            )
                    except Exception as exc:
                        issues.append(f"customer coverage could not be checked: {exc}")
            if issues:
                self._record_artifact_issue(
                    "clv_data",
                    "CLV artifact has invalid coverage: " + "; ".join(issues) + ".",
                )
            else:
                self._clear_artifact_issue("clv_data")
        if "segment" not in adapted.columns:
            adapted["segment"] = "all_customers"
        segment_path = self._resolve_existing_path("segments_6plus.csv")
        if segment_path is not None and "customer_id" in adapted.columns:
            segments = pd.read_csv(segment_path)
            if {"customer_id", "segment"}.issubset(segments.columns):
                adapted["customer_id"] = adapted["customer_id"].astype(str)
                segments["customer_id"] = segments["customer_id"].astype(str)
                adapted = adapted.drop(columns=["segment"], errors="ignore").merge(
                    segments[["customer_id", "segment"]],
                    on="customer_id",
                    how="left",
                )
                adapted["segment"] = adapted["segment"].fillna("all_customers")
        return adapted

    def _adapt_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add optional CLV fields expected by dashboard scatter views."""
        adapted = df.copy()
        adapted["churn_probability"] = pd.to_numeric(
            adapted["churn_probability"], errors="coerce"
        ).clip(0.0, 1.0)
        adapted = adapted.dropna(subset=["customer_id", "churn_probability"])
        if "risk_level" not in adapted.columns:
            adapted["risk_level"] = pd.cut(
                adapted["churn_probability"],
                bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
                labels=["low", "medium", "high", "critical"],
            ).astype(str)
        if "segment" not in adapted.columns:
            adapted["segment"] = adapted.get("persona", "unknown")
        adapted["customer_id"] = adapted["customer_id"].astype(str)
        if "clv_predicted" not in adapted.columns:
            clv = self.load_clv_data()
            if not clv.empty and "customer_id" in adapted.columns:
                adapted = adapted.merge(
                    clv[["customer_id", "clv_predicted"]],
                    on="customer_id",
                    how="left",
                )
            elif self.get_artifact_issue("clv_data") is None:
                self._record_artifact_issue(
                    "clv_data",
                    "Required CLV artifact unavailable for churn prediction enrichment.",
                )
        if "clv_predicted" not in adapted.columns:
            adapted["clv_predicted"] = pd.NA
        adapted["clv_predicted"] = pd.to_numeric(
            adapted["clv_predicted"], errors="coerce"
        )
        return adapted

    def get_prediction_coverage(self) -> Dict[str, Any]:
        """Compare churn prediction rows with the current customer universe."""
        issue = self.get_artifact_issue("churn_predictions")
        predictions_path = self._resolve_existing_path("churn_predictions.csv")
        customers_path = self._customers_path()
        if issue is not None:
            return {
                "status": "invalid",
                "message": issue,
                "prediction_count": 0,
                "customer_count": 0,
                "covered_count": 0,
                "missing_count": None,
                "coverage_ratio": 0.0,
                "is_full_coverage": False,
            }
        if predictions_path is None or customers_path is None:
            return {
                "status": "unknown",
                "message": "Customer universe or churn prediction artifact is unavailable.",
                "prediction_count": 0,
                "customer_count": 0,
                "covered_count": 0,
                "missing_count": None,
                "coverage_ratio": 0.0,
                "is_full_coverage": False,
            }

        try:
            predictions = pd.read_csv(predictions_path, usecols=["customer_id"])
            customers = pd.read_csv(customers_path, usecols=["customer_id"])
        except Exception as exc:
            return {
                "status": "invalid",
                "message": f"Could not evaluate churn prediction coverage: {exc}.",
                "prediction_count": 0,
                "customer_count": 0,
                "covered_count": 0,
                "missing_count": None,
                "coverage_ratio": 0.0,
                "is_full_coverage": False,
            }

        prediction_ids = set(predictions["customer_id"].astype(str))
        customer_ids = set(customers["customer_id"].astype(str))
        covered = len(prediction_ids & customer_ids)
        total = len(customer_ids)
        missing = max(total - covered, 0)
        ratio = covered / total if total else 0.0
        full = total > 0 and missing == 0
        message = (
            f"Churn predictions cover all {total:,} customers."
            if full
            else (
                f"Churn predictions cover {covered:,}/{total:,} customers; "
                f"{missing:,} customers are missing."
            )
        )
        return {
            "status": "complete" if full else "partial",
            "message": message,
            "prediction_count": len(prediction_ids),
            "customer_count": total,
            "covered_count": covered,
            "missing_count": missing,
            "coverage_ratio": ratio,
            "is_full_coverage": full,
        }

    def _adapt_uplift_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map uplift pipeline output to dashboard schema."""
        adapted = df.copy()
        if "treatment_effect" not in adapted.columns:
            adapted["treatment_effect"] = adapted.get("uplift_score", 0.0)
        if "segment" not in adapted.columns:
            adapted["segment"] = "unknown"
        return adapted

    def _load_survival_from_segments(self) -> Optional[pd.DataFrame]:
        """Build survival view rows from 6+ segmentation output."""
        segment_path = self._resolve_existing_path("segments_6plus.csv")
        if segment_path is None:
            return None
        segments = pd.read_csv(segment_path)
        required = {"customer_id", "segment"}
        if not required.issubset(segments.columns):
            return None
        churn = segments.get(
            "churn_probability",
            pd.Series(0.2, index=segments.index),
        ).astype(float).clip(0.0, 1.0)
        duration = (
            365 * (1.0 - churn)
        ).clip(lower=1).round().astype(int)
        return pd.DataFrame({
            "customer_id": segments["customer_id"],
            "duration_days": duration,
            "event_observed": (churn >= 0.5).astype(int),
            "segment": segments["segment"],
            "survival_probability": (1.0 - churn).clip(0.0, 1.0),
        })

    def _adapt_model_metrics(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Keep only dashboard-ready model metric mappings."""
        aliases = {
            "ml_metrics": "ml_model",
            "ml_model": "ml_model",
            "dl_metrics": "dl_model",
            "dl_model": "dl_model",
            "ensemble_metrics": "ensemble",
            "ensemble": "ensemble",
        }
        adapted: Dict[str, Dict[str, float]] = {}
        for raw_key, canonical_key in aliases.items():
            metrics = payload.get(raw_key)
            if not isinstance(metrics, dict):
                continue
            auc = metrics.get("auc", metrics.get("auc_roc"))
            if auc is None:
                continue
            adapted[canonical_key] = {
                "auc": float(auc),
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1_score": float(metrics.get("f1_score", metrics.get("f1", 0.0))),
                "accuracy": float(metrics.get("accuracy", 0.0)),
            }
        return adapted

    def _adapt_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map recommendation pipeline outputs to dashboard schema."""
        adapted = df.copy()
        if "recommendation_type" not in adapted.columns and "action" in adapted.columns:
            adapted["recommendation_type"] = adapted["action"]
        if "recommendation_type" not in adapted.columns and "action_type" in adapted.columns:
            adapted["recommendation_type"] = adapted["action_type"]
        if "priority_score" not in adapted.columns:
            if "score" in adapted.columns:
                adapted["priority_score"] = adapted["score"]
            else:
                adapted["priority_score"] = 0.0
        if "expected_uplift" not in adapted.columns:
            adapted["expected_uplift"] = adapted.get("uplift_score", adapted["priority_score"])
        if "recommended_offer" not in adapted.columns:
            adapted["recommended_offer"] = adapted["recommendation_type"].astype(str)
        if "estimated_cost" not in adapted.columns:
            default_costs = {
                "email": 1000,
                "push_notification": 500,
                "coupon": 5000,
                "loyalty_points": 3000,
                "personal_outreach": 20000,
                "exclusive_offer": 10000,
                "no_action": 0,
            }
            adapted["estimated_cost"] = (
                adapted["recommendation_type"].astype(str).map(default_costs).fillna(1000)
            )
        if "segment" not in adapted.columns:
            segment_path = self._resolve_existing_path("segments_6plus.csv")
            if segment_path is not None and "customer_id" in adapted.columns:
                segments = pd.read_csv(segment_path)
                if {"customer_id", "segment"}.issubset(segments.columns):
                    adapted["customer_id"] = adapted["customer_id"].astype(str)
                    segments["customer_id"] = segments["customer_id"].astype(str)
                    adapted = adapted.merge(
                        segments[["customer_id", "segment"]],
                        on="customer_id",
                        how="left",
                    )
            if "segment" not in adapted.columns:
                adapted["segment"] = "vip_loyal"
            else:
                adapted["segment"] = adapted["segment"].fillna("vip_loyal")
        adapted["priority_score"] = adapted["priority_score"].clip(0.0, 1.0)
        adapted["expected_uplift"] = adapted["expected_uplift"].clip(0.0, 1.0)
        return adapted

    def _adapt_ab_results(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        adapted = dict(payload)
        if "lift" not in adapted:
            treatment_rate = adapted.get(
                "treatment_churn_rate",
                adapted.get("treatment_mean", 0.0),
            )
            control_rate = adapted.get("control_churn_rate", adapted.get("control_mean", 0.0))
            adapted["lift"] = (
                (control_rate - treatment_rate) / abs(control_rate)
                if control_rate not in (0, None)
                else 0.0
            )
        return adapted

    def _adapt_ab_detailed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from src.models.ab_testing import ABTestFramework

        framework = ABTestFramework(self.config)
        return framework.to_dashboard_detailed_results(payload)

    def load_predictions(self) -> pd.DataFrame:
        """Load churn prediction results.

        Returns:
            DataFrame with customer_id, churn_probability, risk_level,
            segment, recommended_action, clv_predicted, etc.
        """
        artifact_name = "churn_predictions"
        df = self._required_csv(
            artifact_name,
            ["churn_predictions.csv"],
            ["customer_id", "churn_probability"],
        )
        if df is None:
            return self._empty_frame(
                [
                    "customer_id", "churn_probability", "risk_level",
                    "segment", "clv_predicted",
                ],
                artifact_name,
                self.get_artifact_issue(artifact_name) or "Churn predictions unavailable.",
            )
        adapted = self._adapt_predictions(df)
        if adapted.empty:
            issue = "Required churn prediction artifact has no valid prediction rows."
            self._record_artifact_issue(artifact_name, issue)
            return self._empty_frame(
                [
                    "customer_id", "churn_probability", "risk_level",
                    "segment", "clv_predicted",
                ],
                artifact_name,
                issue,
            )
        return adapted

    def load_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load model performance metrics.

        Returns:
            Dict with ml_model, dl_model, ensemble keys, each containing
            auc, precision, recall, f1_score, accuracy.
        """
        payload = self._required_json(
            "model_metrics",
            ["model_metrics.json"],
            [],
        )
        if payload is not None:
            adapted = self._adapt_model_metrics(payload)
            if adapted:
                return adapted
            self._record_artifact_issue(
                "model_metrics",
                "Required model_metrics.json is invalid; no dashboard-ready model metrics found.",
            )
        return {}

    def load_ab_test_results(self) -> Dict[str, Any]:
        """Load A/B test experiment results.

        Returns:
            Dict with experiment_name, treatment/control sizes,
            churn rates, lift, p_value, is_significant, CI.
        """
        payload = self._required_json(
            "ab_test_results",
            ["ab_test_results.json"],
            ["p_value"],
        )
        if payload is not None:
            return self._adapt_ab_results(payload)
        return {}

    def load_budget_results(self) -> pd.DataFrame:
        """Load budget optimization results by segment.

        Returns:
            DataFrame with segment, allocated_budget_krw,
            expected_retained, expected_revenue_saved_krw, roi.
        """
        artifact_name = "budget_results"
        df = self._required_csv(
            artifact_name,
            ["budget_results.csv", "budget_optimization.csv"],
            [],
        )
        if df is not None:
            adapted = self._adapt_budget_results(df)
            if not adapted.empty:
                return adapted
        return self._empty_frame(
            [
                "segment", "allocated_budget_krw", "expected_retained",
                "expected_revenue_saved_krw", "roi",
            ],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Budget optimization artifact unavailable.",
        )

    def load_survival_data(self) -> pd.DataFrame:
        """Load survival analysis data.

        Returns:
            DataFrame with customer_id, duration_days, event_observed,
            segment, survival_probability.
        """
        path = self._resolve_existing_path("survival_data.csv")
        if path is not None:
            return pd.read_csv(path)

        segmented = self._load_survival_from_segments()
        if segmented is not None:
            return segmented
        return self._generate_sample_survival_data()

    def load_recommendations(self) -> pd.DataFrame:
        """Load personalized recommendation results.

        Returns:
            DataFrame with customer_id, recommendation_type,
            expected_uplift, priority_score, recommended_offer.
        """
        artifact_name = "recommendations"
        df = self._required_csv(
            artifact_name,
            ["recommendations.csv"],
            ["customer_id"],
        )
        if df is not None:
            return self._adapt_recommendations(df)
        return self._empty_frame(
            [
                "customer_id", "recommendation_type", "expected_uplift",
                "priority_score", "recommended_offer",
            ],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Recommendations artifact unavailable.",
        )

    def load_uplift_results(self) -> pd.DataFrame:
        """Load uplift modeling results.

        Returns:
            DataFrame with customer_id, uplift_score, treatment_effect,
            segment.
        """
        artifact_name = "uplift_results"
        df = self._required_csv(
            artifact_name,
            ["uplift_results.csv"],
            ["customer_id", "uplift_score"],
        )
        if df is not None:
            return self._adapt_uplift_results(df)
        return self._empty_frame(
            ["customer_id", "uplift_score", "treatment_effect", "segment"],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Uplift results artifact unavailable.",
        )

    def load_clv_data(self) -> pd.DataFrame:
        """Load CLV prediction data.

        Returns:
            DataFrame with customer_id, clv_predicted, segment.
        """
        artifact_name = "clv_data"
        df = self._required_csv(
            artifact_name,
            ["clv_data.csv", "clv_predictions.csv"],
            ["customer_id"],
        )
        if df is not None:
            adapted = self._adapt_clv_data(df)
            if "clv_predicted" in adapted.columns and not adapted.empty:
                return adapted
            self._record_artifact_issue(
                artifact_name,
                "Required CLV artifact is invalid; missing non-negative "
                "clv_predicted or predicted_clv values.",
            )
        return self._empty_frame(
            ["customer_id", "clv_predicted", "segment"],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "CLV artifact unavailable.",
        )

    def load_feature_importance(self) -> pd.DataFrame:
        """Load feature importance scores from trained model.

        Returns:
            DataFrame with feature, importance columns sorted descending.
        """
        artifact_name = "feature_importance"
        df = self._required_csv(
            artifact_name,
            ["feature_importance.csv"],
            ["feature", "importance"],
        )
        if df is not None:
            return df.sort_values("importance", ascending=False).reset_index(
                drop=True
            )
        return self._empty_frame(
            ["feature", "importance"],
            artifact_name,
            self.get_artifact_issue(artifact_name) or "Feature importance artifact unavailable.",
        )

    # -----------------------------------------------------------------
    # Sample data generators (used when no artifacts exist)
    # -----------------------------------------------------------------

    def _generate_sample_predictions(self) -> pd.DataFrame:
        """Generate sample churn prediction data."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 500
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "churn_probability": np.random.beta(2, 5, n),
            "risk_level": np.random.choice(
                ["low", "medium", "high", "critical"], n,
                p=[0.4, 0.3, 0.2, 0.1],
            ),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"], n,
            ),
            "recommended_action": np.random.choice(
                ["coupon", "push_notification", "email", "no_action"], n,
            ),
            "clv_predicted": np.random.lognormal(11, 1, n),
            "days_since_last_purchase": np.random.exponential(15, n),
            "days_since_last_login": np.random.exponential(8, n),
        })

    def _generate_sample_metrics(self) -> Dict[str, Dict[str, float]]:
        """Generate sample model metrics."""
        return {
            "ml_model": {
                "auc": 0.82, "precision": 0.76,
                "recall": 0.70, "f1_score": 0.73, "accuracy": 0.81,
            },
            "dl_model": {
                "auc": 0.79, "precision": 0.72,
                "recall": 0.67, "f1_score": 0.69, "accuracy": 0.78,
            },
            "ensemble": {
                "auc": 0.84, "precision": 0.78,
                "recall": 0.72, "f1_score": 0.75, "accuracy": 0.83,
            },
        }

    def _generate_sample_ab_results(self) -> Dict[str, Any]:
        """Generate sample A/B test results."""
        return {
            "experiment_name": "retention_coupon_campaign",
            "treatment_size": 500,
            "control_size": 500,
            "treatment_churn_rate": 0.12,
            "control_churn_rate": 0.20,
            "lift": 0.40,
            "p_value": 0.003,
            "is_significant": True,
            "confidence_interval": [0.03, 0.13],
        }

    def _generate_sample_budget_results(self) -> pd.DataFrame:
        """Generate sample budget optimization results."""
        total = self.config.get("budget", {}).get("total_krw", 50_000_000)
        segments = [
            "vip_loyal", "regular_loyal", "bargain_hunter",
            "explorer", "dormant", "new_customer",
        ]
        allocs = [0.10, 0.24, 0.16, 0.20, 0.06, 0.24]
        allocated = [int(total * a) for a in allocs]

        return pd.DataFrame({
            "segment": segments,
            "allocated_budget_krw": allocated,
            "expected_retained": [450, 1800, 1200, 1500, 200, 850],
            "expected_revenue_saved_krw": [
                67500000, 144000000, 54000000,
                82500000, 12000000, 102000000,
            ],
            "roi": [13.5, 12.0, 6.75, 8.25, 4.0, 8.5],
        })

    def _generate_sample_survival_data(self) -> pd.DataFrame:
        """Generate sample survival analysis data."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 300
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "duration_days": np.random.exponential(90, n),
            "event_observed": np.random.binomial(1, 0.3, n),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"], n,
            ),
            "survival_probability": np.random.beta(5, 2, n),
        })

    def _generate_sample_recommendations(self) -> pd.DataFrame:
        """Generate sample recommendations with segment and cost data."""
        np.random.seed(
            self.config.get("simulation", {}).get("random_seed", 42)
        )
        n = 50
        action_types = [
            "coupon", "push_notification", "email",
            "loyalty_points", "personal_outreach", "exclusive_offer",
            "no_action",
        ]
        action_costs = {
            "coupon": 5000, "push_notification": 100, "email": 200,
            "loyalty_points": 3000, "personal_outreach": 10000,
            "exclusive_offer": 8000, "no_action": 0,
        }
        segments = [
            "vip_loyal", "regular_loyal", "bargain_hunter",
            "explorer", "dormant", "new_customer",
        ]
        rec_types = np.random.choice(action_types, n, p=[
            0.2, 0.15, 0.2, 0.15, 0.05, 0.1, 0.15,
        ])
        offers = {
            "coupon": "20% discount coupon",
            "push_notification": "Flash sale alert",
            "email": "Personalized product picks",
            "loyalty_points": "2x loyalty points bonus",
            "personal_outreach": "VIP concierge call",
            "exclusive_offer": "Early access exclusive",
            "no_action": "None",
        }
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "recommendation_type": rec_types,
            "expected_uplift": np.clip(
                np.random.beta(2, 8, n) * 0.5, 0.0, 1.0
            ),
            "priority_score": np.clip(
                np.random.beta(3, 2, n), 0.0, 1.0
            ),
            "recommended_offer": [offers[r] for r in rec_types],
            "segment": np.random.choice(segments, n),
            "estimated_cost": [
                action_costs[r] for r in rec_types
            ],
        })

    def _generate_sample_uplift_results(self) -> pd.DataFrame:
        """Generate sample uplift results."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 200
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "uplift_score": np.random.normal(0.05, 0.03, n),
            "treatment_effect": np.random.normal(0.08, 0.04, n),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"], n,
            ),
        })

    def _generate_sample_clv_data(self) -> pd.DataFrame:
        """Generate sample CLV data."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 500
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "clv_predicted": np.random.lognormal(11, 1, n),
            "segment": np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"], n,
            ),
        })

    def load_cohort_data(self) -> pd.DataFrame:
        """Load cohort analysis event data.

        Returns:
            DataFrame with customer_id, event_date, revenue, segment
            suitable for CohortAnalyzer.assign_cohorts().
        """
        path = self._resolve_existing_path("cohort_data.csv")
        if path is not None:
            df = pd.read_csv(path)
            if "event_date" in df.columns:
                df["event_date"] = pd.to_datetime(df["event_date"])
            return df

        return self._generate_sample_cohort_data()

    def load_cohort_retention_matrix(self) -> pd.DataFrame:
        """Load pre-computed cohort retention matrix.

        Returns:
            DataFrame with cohort labels as index, period indices as columns,
            and retention rates (0.0 to 1.0) as values.
        """
        path = self._resolve_existing_path("cohort_retention_matrix.csv")
        if path is not None:
            df = pd.read_csv(path, index_col=0)
            if df.shape[0] >= 2 and df.shape[1] >= 2:
                return df
            logger.warning("Ignoring incomplete cohort retention matrix at %s", path)

        # Compute from cohort data using CohortAnalyzer
        cohort_data = self.load_cohort_data()
        if cohort_data.empty:
            return pd.DataFrame()

        try:
            from src.analysis.cohort_analysis import CohortAnalyzer
            analyzer = CohortAnalyzer()
            assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
            retention = analyzer.compute_retention_matrix(assigned)
            if retention.shape[0] >= 2 and retention.shape[1] >= 2:
                return retention
            return self._generate_sample_retention_matrix()
        except Exception:
            return self._generate_sample_retention_matrix()

    def _generate_sample_cohort_data(self) -> pd.DataFrame:
        """Generate sample cohort event data."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n_customers = 200
        events_per_customer = 5

        customers = [f"C{i:05d}" for i in range(n_customers)]
        rows = []
        base_date = pd.Timestamp("2024-01-01")

        for cust in customers:
            # Random first event date in first 6 months
            first_offset = np.random.randint(0, 180)
            first_date = base_date + pd.Timedelta(days=int(first_offset))
            segment = np.random.choice(
                ["vip_loyal", "regular_loyal", "bargain_hunter",
                 "explorer", "dormant", "new_customer"]
            )
            for j in range(np.random.randint(1, events_per_customer + 1)):
                event_offset = np.random.randint(0, 365)
                event_date = first_date + pd.Timedelta(days=int(event_offset))
                revenue = np.random.lognormal(9, 1)
                rows.append({
                    "customer_id": cust,
                    "event_date": event_date,
                    "revenue": revenue,
                    "segment": segment,
                })

        return pd.DataFrame(rows)

    def _generate_sample_retention_matrix(self) -> pd.DataFrame:
        """Generate sample retention matrix for display."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        cohorts = [f"2024-{m:02d}" for m in range(1, 7)]
        periods = list(range(7))

        data = {}
        for cohort in cohorts:
            retention = [1.0]
            for p in range(1, len(periods)):
                decay = np.random.uniform(0.80, 0.95)
                retention.append(round(retention[-1] * decay, 4))
            data[cohort] = retention

        df = pd.DataFrame(data, index=periods).T
        df.index.name = "cohort"
        df.columns = periods
        return df

    # -----------------------------------------------------------------
    # Model Performance & A/B Testing enhanced loaders
    # -----------------------------------------------------------------

    def load_model_performance_history(self) -> pd.DataFrame:
        """Load real model performance history in dashboard run-history shape."""
        artifact_name = "model_performance_history"
        df = self._required_csv(
            artifact_name,
            ["model_performance_history.csv", "mlflow_runs.csv"],
            [],
        )
        if df is None:
            return self._empty_frame(
                [
                    "run_id", "model_type", "auc", "precision", "recall",
                    "f1_score", "accuracy", "training_time_s", "timestamp",
                    "params_lr", "params_epochs",
                ],
                artifact_name,
                self.get_artifact_issue(artifact_name)
                or "Model performance history unavailable.",
            )
        if "model_type" not in df.columns and "model" in df.columns:
            df = df.rename(columns={"model": "model_type"})
        if "auc" not in df.columns and "auc_roc" in df.columns:
            df["auc"] = df["auc_roc"]
        if "f1_score" not in df.columns and "f1" in df.columns:
            df["f1_score"] = df["f1"]
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.Timestamp.utcnow().isoformat()
        if "run_id" not in df.columns:
            run_col = (
                df["run"].astype(str)
                if "run" in df.columns
                else pd.Series("current", index=df.index)
            )
            df["run_id"] = [f"{run}_{idx}" for idx, run in enumerate(run_col)]
        for metric in ["auc", "precision", "recall", "f1_score", "accuracy"]:
            if metric not in df.columns:
                df[metric] = 0.0
            df[metric] = pd.to_numeric(df[metric], errors="coerce").fillna(0.0)
        if "model_type" not in df.columns:
            df["model_type"] = "unknown"
        if "training_time_s" not in df.columns:
            df["training_time_s"] = 1.0
        else:
            df["training_time_s"] = pd.to_numeric(
                df["training_time_s"], errors="coerce"
            ).fillna(1.0).clip(lower=1e-6)
        if "params_lr" not in df.columns:
            df["params_lr"] = 0.1
        if "params_epochs" not in df.columns:
            df["params_epochs"] = 1
        if df.empty:
            self._record_artifact_issue(
                artifact_name,
                "Required model performance history artifact has no rows.",
            )
        else:
            self._clear_artifact_issue(artifact_name)
        return df

    def load_mlflow_runs(self) -> pd.DataFrame:
        """Load MLflow experiment run metrics.

        Returns:
            DataFrame with run_id, model_type, auc, precision, recall,
            f1_score, accuracy, training_time_s, timestamp.
        """
        return self.load_model_performance_history()

    def load_roc_data(self) -> Dict[str, Dict[str, list]]:
        """Load ROC curve data for each model.

        Returns:
            Dict mapping model name to dict with fpr, tpr lists.
        """
        path = self.artifacts_dir / "roc_data.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return self._generate_sample_roc_data()

    def load_confusion_matrices(self) -> Dict[str, list]:
        """Load confusion matrix data for each model.

        Returns:
            Dict mapping model name to 2x2 matrix as nested list.
        """
        path = self.artifacts_dir / "confusion_matrices.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return self._generate_sample_confusion_matrices()

    def load_ab_test_detailed(self) -> Dict[str, Any]:
        """Load detailed A/B test results including multiple experiments.

        Returns:
            Dict with experiments list, each containing name, metrics,
            and statistical test details.
        """
        path = self._resolve_existing_path("ab_test_detailed.json")
        if path is not None:
            return self._adapt_ab_detailed(self._read_json(path))
        path = self._resolve_existing_path("ab_test_results.json")
        if path is not None:
            return self._adapt_ab_detailed(self._read_json(path))
        return self._generate_sample_ab_detailed()

    def load_survival_curves(self) -> Dict[str, Dict[str, list]]:
        """Load Kaplan-Meier survival curves per segment.

        Returns:
            Dict mapping segment name to dict with timeline, survival_prob.
        """
        path = self.artifacts_dir / "survival_curves.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return self._generate_sample_survival_curves()

    # -----------------------------------------------------------------
    # New sample data generators for enhanced views
    # -----------------------------------------------------------------

    def _generate_sample_mlflow_runs(self) -> pd.DataFrame:
        """Generate sample MLflow experiment run data."""
        np.random.seed(
            self.config.get("simulation", {}).get("random_seed", 42)
        )
        runs = []
        model_types = [
            "xgboost", "lightgbm", "lstm", "transformer", "ensemble",
        ]
        for i, mt in enumerate(model_types):
            base_auc = 0.78 + np.random.uniform(0, 0.08)
            runs.append({
                "run_id": f"run_{i:04d}",
                "model_type": mt,
                "auc": round(base_auc, 4),
                "precision": round(
                    base_auc - np.random.uniform(0.03, 0.08), 4,
                ),
                "recall": round(
                    base_auc - np.random.uniform(0.05, 0.12), 4,
                ),
                "f1_score": round(
                    base_auc - np.random.uniform(0.04, 0.10), 4,
                ),
                "accuracy": round(
                    base_auc - np.random.uniform(0.01, 0.05), 4,
                ),
                "training_time_s": round(np.random.uniform(5, 120), 1),
                "timestamp": f"2024-{10 + i % 3:02d}-{15 + i:02d}T10:00:00",
                "params_lr": round(
                    float(np.random.choice([0.001, 0.01, 0.1])), 4,
                ),
                "params_epochs": int(np.random.choice([5, 10, 20, 50])),
            })
        return pd.DataFrame(runs)

    def _generate_sample_roc_data(self) -> Dict[str, Dict[str, list]]:
        """Generate sample ROC curve data for ML, DL, and ensemble."""
        np.random.seed(
            self.config.get("simulation", {}).get("random_seed", 42)
        )
        result = {}
        model_aucs = {
            "ml_model": 0.82, "dl_model": 0.79, "ensemble": 0.84,
        }
        for name, target_auc in model_aucs.items():
            n_points = 100
            fpr = np.sort(np.concatenate(
                [[0], np.random.beta(1, 3, n_points - 2), [1]]
            ))
            tpr = np.sort(np.concatenate(
                [[0], np.random.beta(3, 1, n_points - 2), [1]]
            ))
            # np.trapezoid for numpy>=2.0, fallback to np.trapz
            try:
                area = np.trapezoid(tpr, fpr)
            except AttributeError:
                area = np.trapz(tpr, fpr)
            scale = target_auc / max(area, 0.01)
            tpr = np.clip(tpr * min(scale, 1.2), 0, 1)
            tpr = np.sort(tpr)
            tpr[-1] = 1.0
            result[name] = {
                "fpr": fpr.round(4).tolist(),
                "tpr": tpr.round(4).tolist(),
            }
        return result

    def _generate_sample_confusion_matrices(self) -> Dict[str, list]:
        """Generate sample confusion matrices for each model."""
        return {
            "ml_model": [[350, 50], [80, 120]],
            "dl_model": [[340, 60], [90, 110]],
            "ensemble": [[360, 40], [70, 130]],
        }

    def _generate_sample_ab_detailed(self) -> Dict[str, Any]:
        """Generate detailed A/B test results with multiple experiments."""
        return {
            "experiments": [
                {
                    "name": "retention_coupon_campaign",
                    "treatment_size": 500,
                    "control_size": 500,
                    "treatment_churn_rate": 0.12,
                    "control_churn_rate": 0.20,
                    "lift": 0.40,
                    "absolute_effect": 0.08,
                    "p_value": 0.003,
                    "is_significant": True,
                    "confidence_interval": [0.03, 0.13],
                    "effect_size_cohens_h": 0.22,
                    "power": 0.87,
                    "test_type": "two-proportion z-test",
                    "alpha": 0.05,
                    "duration_days": 30,
                },
                {
                    "name": "push_notification_retention",
                    "treatment_size": 400,
                    "control_size": 400,
                    "treatment_churn_rate": 0.16,
                    "control_churn_rate": 0.19,
                    "lift": 0.158,
                    "absolute_effect": 0.03,
                    "p_value": 0.28,
                    "is_significant": False,
                    "confidence_interval": [-0.02, 0.08],
                    "effect_size_cohens_h": 0.08,
                    "power": 0.32,
                    "test_type": "two-proportion z-test",
                    "alpha": 0.05,
                    "duration_days": 21,
                },
                {
                    "name": "email_personalization",
                    "treatment_size": 600,
                    "control_size": 600,
                    "treatment_churn_rate": 0.14,
                    "control_churn_rate": 0.21,
                    "lift": 0.333,
                    "absolute_effect": 0.07,
                    "p_value": 0.001,
                    "is_significant": True,
                    "confidence_interval": [0.03, 0.11],
                    "effect_size_cohens_h": 0.19,
                    "power": 0.91,
                    "test_type": "two-proportion z-test",
                    "alpha": 0.05,
                    "duration_days": 45,
                },
            ],
            "summary": {
                "total_experiments": 3,
                "significant_count": 2,
                "best_experiment": "retention_coupon_campaign",
                "avg_lift": 0.297,
            },
        }

    def _generate_sample_survival_curves(
        self,
    ) -> Dict[str, Dict[str, list]]:
        """Generate sample Kaplan-Meier survival curves per segment."""
        np.random.seed(
            self.config.get("simulation", {}).get("random_seed", 42)
        )
        result = {}
        segments = {
            "vip_loyal": 0.92,
            "regular_loyal": 0.80,
            "bargain_hunter": 0.65,
            "explorer": 0.70,
            "dormant": 0.40,
            "new_customer": 0.55,
        }
        for seg, base_survival in segments.items():
            timeline = list(range(0, 361, 10))
            decay_rate = -np.log(max(base_survival, 0.01)) / 360
            survival_prob = [
                round(float(np.exp(-decay_rate * t)), 4)
                for t in timeline
            ]
            ci_lower = [
                round(max(0, s - np.random.uniform(0.02, 0.08)), 4)
                for s in survival_prob
            ]
            ci_upper = [
                round(min(1, s + np.random.uniform(0.02, 0.08)), 4)
                for s in survival_prob
            ]
            median_surv = (
                int(np.log(2) / max(decay_rate, 1e-10))
                if base_survival < 1.0 and decay_rate > 0 else None
            )
            result[seg] = {
                "timeline": timeline,
                "survival_prob": survival_prob,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "median_survival_days": median_surv,
            }
        return result

    def load_scoring_history(self) -> pd.DataFrame:
        """Load real-time scoring history log.

        Returns:
            DataFrame with customer_id, churn_probability, risk_level,
            recommended_action, model_type, scored_at columns.
        """
        path = self.artifacts_dir / "scoring_history.csv"
        if path.exists():
            return pd.read_csv(path)
        return self._generate_sample_scoring_history()

    def load_drift_history(self) -> pd.DataFrame:
        """Load model drift detection history.

        Returns:
            DataFrame with timestamp, alert_level, num_drifted_features,
            psi_mean, ks_mean columns.
        """
        path = self._resolve_existing_path("drift_history.csv")
        if path is not None:
            self._clear_artifact_issue("monitoring_report")
            return pd.read_csv(path)

        report_path = self._resolve_existing_path("monitoring_report.json")
        empty_columns = [
            "timestamp", "alert_level", "num_drifted_features",
            "psi_mean", "ks_mean",
        ]
        if report_path is None:
            issue = (
                "Required artifact missing: monitoring_report.json. "
                "Drift history cannot use generated samples for required dashboard evidence."
            )
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        try:
            report = self._read_json(report_path)
        except Exception as exc:
            issue = f"Required artifact {report_path} could not be read: {exc}."
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        if not isinstance(report, dict):
            issue = f"Required artifact {report_path} is invalid; expected a JSON object."
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        psi_report = report.get("psi_report", {})
        ks_report = report.get("ks_report", {})
        if not psi_report and not ks_report:
            issue = (
                f"Required artifact {report_path} is invalid; "
                "missing PSI/KS drift reports."
            )
            self._record_artifact_issue("monitoring_report", issue)
            return self._empty_frame(empty_columns, "monitoring_report", issue)

        psi_alerts = (
            psi_report.get("feature_alerts")
            or psi_report.get("alerts", {})
            or {}
        )
        ks_alerts = (
            ks_report.get("feature_alerts")
            or ks_report.get("alerts", {})
            or {}
        )
        psi_values = [
            float(alert.get("psi_value", 0.0))
            for alert in psi_alerts.values()
        ]
        ks_values = [
            float(alert.get("statistic", 0.0))
            for alert in ks_alerts.values()
        ]
        self._clear_artifact_issue("monitoring_report")
        return pd.DataFrame([{
            "timestamp": report.get("timestamp"),
            "alert_level": report.get("overall_alert_level", "green"),
            "num_drifted_features": len(report.get("drifted_features", [])),
            "psi_mean": float(np.mean(psi_values)) if psi_values else 0.0,
            "ks_mean": float(np.mean(ks_values)) if ks_values else 0.0,
        }])

    def load_auc_history(self) -> pd.DataFrame:
        """Load AUC metric time series."""
        return self._load_metric_timeseries("auc")

    def load_precision_history(self) -> pd.DataFrame:
        """Load Precision metric time series."""
        return self._load_metric_timeseries("precision")

    def load_recall_history(self) -> pd.DataFrame:
        """Load Recall metric time series."""
        return self._load_metric_timeseries("recall")

    def load_scoring_throughput(self) -> pd.DataFrame:
        """Load scoring throughput metrics over time.

        Returns:
            DataFrame with timestamp, requests_per_minute,
            avg_latency_ms, error_rate columns.
        """
        path = self.artifacts_dir / "scoring_throughput.csv"
        if path.exists():
            return pd.read_csv(path)
        return self._generate_sample_scoring_throughput()

    def load_retention_offers(self) -> pd.DataFrame:
        """Load personalized retention offer details per customer.

        Returns:
            DataFrame with customer_id, segment, risk_level,
            churn_probability, offer_type, offer_detail,
            expected_uplift, estimated_cost_krw, estimated_revenue_save_krw,
            priority_rank.
        """
        path = self.artifacts_dir / "retention_offers.csv"
        if path.exists():
            return pd.read_csv(path)
        return self._generate_sample_retention_offers()

    # -----------------------------------------------------------------
    # Sample data generators for real-time scoring views
    # -----------------------------------------------------------------

    def _generate_sample_scoring_history(self) -> pd.DataFrame:
        """Generate sample scoring history data."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 200
        timestamps = pd.date_range(
            "2024-10-01", periods=n, freq="15min"
        ).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        probs = np.random.beta(2, 5, n)
        risk_levels = []
        actions = []
        for p in probs:
            if p >= 0.75:
                risk_levels.append("critical")
                actions.append("immediate_personal_outreach")
            elif p >= 0.50:
                risk_levels.append("high")
                actions.append("win_back_campaign_with_discount")
            elif p >= 0.25:
                risk_levels.append("medium")
                actions.append("engagement_email_campaign")
            else:
                risk_levels.append("low")
                actions.append("standard_loyalty_program")
        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in np.random.randint(0, 500, n)],
            "churn_probability": probs.round(4),
            "risk_level": risk_levels,
            "recommended_action": actions,
            "model_type": np.random.choice(
                ["ensemble", "xgboost", "lightgbm"], n, p=[0.7, 0.15, 0.15]
            ),
            "scored_at": timestamps,
        })

    def _generate_sample_drift_history(self) -> pd.DataFrame:
        """Generate sample drift detection history."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 30
        timestamps = pd.date_range(
            "2024-09-01", periods=n, freq="1D"
        ).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        alert_levels = np.random.choice(
            ["green", "yellow", "red"], n, p=[0.7, 0.2, 0.1]
        )
        return pd.DataFrame({
            "timestamp": timestamps,
            "alert_level": alert_levels,
            "num_drifted_features": np.where(
                alert_levels == "green", 0,
                np.where(alert_levels == "yellow",
                         np.random.randint(1, 3, n),
                         np.random.randint(3, 8, n))
            ),
            "psi_mean": np.random.uniform(0.01, 0.25, n).round(4),
            "ks_mean": np.random.uniform(0.02, 0.15, n).round(4),
        })

    def _generate_sample_scoring_throughput(self) -> pd.DataFrame:
        """Generate sample scoring throughput metrics."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 48  # 48 half-hour intervals (1 day)
        timestamps = pd.date_range(
            "2024-10-15", periods=n, freq="30min"
        ).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        # Simulate daily pattern with peak hours
        hours = np.arange(n) * 0.5
        base_rate = 50 + 30 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
        return pd.DataFrame({
            "timestamp": timestamps,
            "requests_per_minute": (base_rate + np.random.normal(0, 5, n)).clip(5).round(1),
            "avg_latency_ms": (15 + np.random.exponential(5, n)).round(1),
            "error_rate": np.random.uniform(0, 0.02, n).round(4),
        })

    def _generate_sample_retention_offers(self) -> pd.DataFrame:
        """Generate sample personalized retention offers."""
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        n = 50
        segments = np.random.choice(
            ["vip_loyal", "regular_loyal", "bargain_hunter",
             "explorer", "dormant", "new_customer"], n,
        )
        probs = np.random.beta(3, 3, n)
        risk_levels = []
        for p in probs:
            if p >= 0.75:
                risk_levels.append("critical")
            elif p >= 0.50:
                risk_levels.append("high")
            elif p >= 0.25:
                risk_levels.append("medium")
            else:
                risk_levels.append("low")

        offer_types = []
        offer_details = []
        costs = []
        for seg, risk in zip(segments, risk_levels):
            if risk in ("critical", "high"):
                if seg in ("vip_loyal", "new_customer"):
                    offer_types.append("premium_discount")
                    offer_details.append("30% off next 3 orders + free shipping")
                    costs.append(int(np.random.uniform(50000, 150000)))
                else:
                    offer_types.append("discount_coupon")
                    offer_details.append("20% off next order")
                    costs.append(int(np.random.uniform(10000, 50000)))
            elif risk == "medium":
                offer_types.append("engagement_email")
                offer_details.append("Personalized product picks + 10% coupon")
                costs.append(int(np.random.uniform(2000, 10000)))
            else:
                offer_types.append("loyalty_points")
                offer_details.append("2x loyalty points for 30 days")
                costs.append(int(np.random.uniform(1000, 5000)))

        expected_uplift = np.where(
            np.array(risk_levels) == "critical", np.random.uniform(0.10, 0.25, n),
            np.where(np.array(risk_levels) == "high", np.random.uniform(0.08, 0.18, n),
                     np.where(np.array(risk_levels) == "medium",
                              np.random.uniform(0.03, 0.12, n),
                              np.random.uniform(0.01, 0.05, n)))
        )
        revenue_save = (np.array(costs) * np.random.uniform(3, 15, n)).astype(int)

        return pd.DataFrame({
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "segment": segments,
            "risk_level": risk_levels,
            "churn_probability": probs.round(4),
            "offer_type": offer_types,
            "offer_detail": offer_details,
            "expected_uplift": expected_uplift.round(4),
            "estimated_cost_krw": costs,
            "estimated_revenue_save_krw": revenue_save,
            "priority_rank": list(range(1, n + 1)),
        })

    def _generate_sample_feature_importance(self) -> pd.DataFrame:
        """Generate sample feature importance data."""
        features = [
            "days_since_last_purchase",
            "days_since_last_login",
            "purchase_frequency",
            "avg_order_value",
            "total_revenue",
            "page_views_30d",
            "cart_abandonment_rate",
            "coupon_usage_rate",
            "customer_tenure_days",
            "review_count",
            "cs_contact_count",
            "purchase_recency_score",
            "login_frequency",
            "category_diversity",
            "avg_session_duration",
        ]
        np.random.seed(self.config.get("simulation", {}).get("random_seed", 42))
        importances = np.random.dirichlet(np.ones(len(features)))
        # Make top features more prominent
        importances = np.sort(importances)[::-1]
        return pd.DataFrame({
            "feature": features,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
