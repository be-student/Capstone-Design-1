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
        self.artifacts_dir = Path(
            config.get("dashboard", {}).get(
                "artifacts_dir", str(DEFAULT_ARTIFACTS_DIR)
            )
        )

    def load_predictions(self) -> pd.DataFrame:
        """Load churn prediction results.

        Returns:
            DataFrame with customer_id, churn_probability, risk_level,
            segment, recommended_action, clv_predicted, etc.
        """
        path = self.artifacts_dir / "churn_predictions.csv"
        if path.exists():
            return pd.read_csv(path)

        # Return sample data if no artifacts exist yet
        return self._generate_sample_predictions()

    def load_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Load model performance metrics.

        Returns:
            Dict with ml_model, dl_model, ensemble keys, each containing
            auc, precision, recall, f1_score, accuracy.
        """
        path = self.artifacts_dir / "model_metrics.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)

        return self._generate_sample_metrics()

    def load_ab_test_results(self) -> Dict[str, Any]:
        """Load A/B test experiment results.

        Returns:
            Dict with experiment_name, treatment/control sizes,
            churn rates, lift, p_value, is_significant, CI.
        """
        path = self.artifacts_dir / "ab_test_results.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)

        return self._generate_sample_ab_results()

    def load_budget_results(self) -> pd.DataFrame:
        """Load budget optimization results by segment.

        Returns:
            DataFrame with segment, allocated_budget_krw,
            expected_retained, expected_revenue_saved_krw, roi.
        """
        path = self.artifacts_dir / "budget_results.csv"
        if path.exists():
            return pd.read_csv(path)

        return self._generate_sample_budget_results()

    def load_survival_data(self) -> pd.DataFrame:
        """Load survival analysis data.

        Returns:
            DataFrame with customer_id, duration_days, event_observed,
            segment, survival_probability.
        """
        path = self.artifacts_dir / "survival_data.csv"
        if path.exists():
            return pd.read_csv(path)

        return self._generate_sample_survival_data()

    def load_recommendations(self) -> pd.DataFrame:
        """Load personalized recommendation results.

        Returns:
            DataFrame with customer_id, recommendation_type,
            expected_uplift, priority_score, recommended_offer.
        """
        path = self.artifacts_dir / "recommendations.csv"
        if path.exists():
            return pd.read_csv(path)

        return self._generate_sample_recommendations()

    def load_uplift_results(self) -> pd.DataFrame:
        """Load uplift modeling results.

        Returns:
            DataFrame with customer_id, uplift_score, treatment_effect,
            segment.
        """
        path = self.artifacts_dir / "uplift_results.csv"
        if path.exists():
            return pd.read_csv(path)

        return self._generate_sample_uplift_results()

    def load_clv_data(self) -> pd.DataFrame:
        """Load CLV prediction data.

        Returns:
            DataFrame with customer_id, clv_predicted, segment.
        """
        path = self.artifacts_dir / "clv_data.csv"
        if path.exists():
            return pd.read_csv(path)

        return self._generate_sample_clv_data()

    def load_feature_importance(self) -> pd.DataFrame:
        """Load feature importance scores from trained model.

        Returns:
            DataFrame with feature, importance columns sorted descending.
        """
        path = self.artifacts_dir / "feature_importance.csv"
        if path.exists():
            df = pd.read_csv(path)
            return df.sort_values("importance", ascending=False).reset_index(
                drop=True
            )

        return self._generate_sample_feature_importance()

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
        path = self.artifacts_dir / "cohort_data.csv"
        if path.exists():
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
        path = self.artifacts_dir / "cohort_retention_matrix.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            return df

        # Compute from cohort data using CohortAnalyzer
        cohort_data = self.load_cohort_data()
        if cohort_data.empty:
            return pd.DataFrame()

        try:
            from src.analysis.cohort_analysis import CohortAnalyzer
            analyzer = CohortAnalyzer()
            assigned = analyzer.assign_cohorts(cohort_data, cohort_type="monthly")
            retention = analyzer.compute_retention_matrix(assigned)
            return retention
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

    def load_mlflow_runs(self) -> pd.DataFrame:
        """Load MLflow experiment run metrics.

        Returns:
            DataFrame with run_id, model_type, auc, precision, recall,
            f1_score, accuracy, training_time_s, timestamp.
        """
        path = self.artifacts_dir / "mlflow_runs.csv"
        if path.exists():
            return pd.read_csv(path)
        return self._generate_sample_mlflow_runs()

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
        path = self.artifacts_dir / "ab_test_detailed.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
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
        path = self.artifacts_dir / "drift_history.csv"
        if path.exists():
            return pd.read_csv(path)
        return self._generate_sample_drift_history()

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
