"""
A/B Testing Framework Module.

Provides:
- ABTestFramework: Full A/B and A/B/n experiment framework
  - Power analysis for sample size calculation
  - Treatment/control group assignment with configurable ratios
  - Statistical significance testing (Chi-square, Z-test, T-test)
  - P-value computation with 95% confidence intervals
  - Effect size estimation (Cohen's d / h)
  - Multi-variant (A/B/n) experiment support
  - Experiment persistence (save/load)

All configurable parameters are read from the YAML config dictionary.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _safe_rate_from_mean(value: float) -> float:
    """Clamp a mean-like value into the [0, 1] rate range."""
    if np.isnan(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Power Analysis
# ---------------------------------------------------------------------------

class PowerAnalysis:
    """Sample size and power calculations for A/B tests."""

    @staticmethod
    def required_sample_size(
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        power: float = 0.80,
        two_sided: bool = True,
    ) -> int:
        """Calculate required sample size per group for a proportion test.

        Parameters
        ----------
        baseline_rate : float
            Expected baseline conversion/churn rate (0-1).
        mde : float
            Minimum detectable effect (absolute difference).
        alpha : float
            Significance level (default 0.05).
        power : float
            Statistical power (default 0.80).
        two_sided : bool
            Whether to use a two-sided test.

        Returns
        -------
        int
            Required sample size per group.
        """
        if mde == 0:
            raise ValueError("MDE cannot be zero")

        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_avg = (p1 + p2) / 2.0

        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        z_beta = stats.norm.ppf(power)

        numerator = (
            z_alpha * np.sqrt(2 * p_avg * (1 - p_avg))
            + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        denominator = (p2 - p1) ** 2

        n = int(np.ceil(numerator / denominator))
        return n

    @staticmethod
    def minimum_detectable_effect(
        n: int,
        baseline_rate: float,
        alpha: float = 0.05,
        power: float = 0.80,
        two_sided: bool = True,
    ) -> float:
        """Compute the minimum detectable effect given sample size.

        Uses binary search to find the smallest absolute MDE that
        achieves the desired power for the given sample size per group.

        Parameters
        ----------
        n : int
            Sample size per group.
        baseline_rate : float
            Baseline conversion/churn rate (0-1).
        alpha : float
            Significance level (default 0.05).
        power : float
            Desired statistical power (default 0.80).
        two_sided : bool
            Whether to use a two-sided test.

        Returns
        -------
        float
            Minimum detectable effect (absolute difference).
        """
        if n <= 0:
            raise ValueError("Sample size must be positive")
        if not (0 < baseline_rate < 1):
            raise ValueError("baseline_rate must be between 0 and 1 (exclusive)")

        # Binary search for the MDE that yields the target power
        lo, hi = 1e-6, min(baseline_rate, 1 - baseline_rate)
        for _ in range(100):  # sufficient iterations for convergence
            mid = (lo + hi) / 2.0
            computed_power = PowerAnalysis.compute_power(
                n=n,
                baseline_rate=baseline_rate,
                mde=mid,
                alpha=alpha,
                two_sided=two_sided,
            )
            if computed_power < power:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0

    @staticmethod
    def compute_power(
        n: int,
        baseline_rate: float,
        mde: float,
        alpha: float = 0.05,
        two_sided: bool = True,
    ) -> float:
        """Compute statistical power given sample size.

        Parameters
        ----------
        n : int
            Sample size per group.
        baseline_rate : float
            Baseline rate.
        mde : float
            Minimum detectable effect.
        alpha : float
            Significance level.
        two_sided : bool
            Two-sided test flag.

        Returns
        -------
        float
            Statistical power (0-1).
        """
        p1 = baseline_rate
        p2 = baseline_rate + mde
        p_avg = (p1 + p2) / 2.0

        if two_sided:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)

        se_null = np.sqrt(2 * p_avg * (1 - p_avg) / n)
        se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)

        if se_alt == 0:
            return 1.0

        z_beta = (abs(mde) / se_alt) - (z_alpha * se_null / se_alt)
        power = stats.norm.cdf(z_beta)
        return float(power)


# ---------------------------------------------------------------------------
# A/B Test Framework
# ---------------------------------------------------------------------------

class ABTestFramework:
    """A/B testing framework with statistical significance testing.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

        # Treatment settings from config
        treatment_cfg = config.get("treatment", {})
        self.treatment_ratio = treatment_cfg.get("treatment_ratio", 0.50)
        self.min_group_size = treatment_cfg.get("min_group_size", 10000)

        # Random seed from config
        sim_cfg = config.get("simulation", {})
        self.random_seed = sim_cfg.get("random_seed", 42)

        # Experiment registry
        self.experiments: Dict[str, Dict[str, Any]] = {}

        # Power analysis utility
        self.power_analysis = PowerAnalysis()

        logger.info(
            "ABTestFramework initialized (treatment_ratio=%.2f, seed=%d)",
            self.treatment_ratio,
            self.random_seed,
        )

    # ------------------------------------------------------------------
    # Experiment creation
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        metrics: Optional[List[str]] = None,
    ) -> str:
        """Create a new A/B test experiment.

        Parameters
        ----------
        name : str
            Experiment name.
        description : str
            Experiment description.
        metrics : list of str, optional
            Metrics to track.

        Returns
        -------
        str
            Unique experiment identifier.
        """
        exp_id = f"{name}_{uuid.uuid4().hex[:8]}"
        self.experiments[exp_id] = {
            "name": name,
            "description": description,
            "metrics": metrics or [],
            "status": "created",
        }
        logger.info("Created experiment '%s' (id=%s)", name, exp_id)
        return exp_id

    # ------------------------------------------------------------------
    # Group assignment
    # ------------------------------------------------------------------

    def assign_groups(
        self,
        customer_ids: List[str],
        n_variants: int = 2,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Assign customers to treatment/control (or multi-variant) groups.

        Uses deterministic hashing so the same customer always lands in the
        same group across calls with the same seed.

        Parameters
        ----------
        customer_ids : list of str
            Customer identifiers.
        n_variants : int
            Number of variant groups (2 = standard A/B).
        seed : int, optional
            Random seed override (defaults to config seed).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``customer_id`` and ``group``.
        """
        seed = seed if seed is not None else self.random_seed
        rng = np.random.RandomState(seed)

        n = len(customer_ids)

        if n_variants == 2:
            # Standard A/B split using configured treatment_ratio
            labels = ["control"] * n
            indices = rng.permutation(n)
            n_treatment = int(round(n * self.treatment_ratio))
            for idx in indices[:n_treatment]:
                labels[idx] = "treatment"
        else:
            # Multi-variant: balanced random assignment
            group_labels = [f"variant_{i}" for i in range(n_variants)]
            indices = rng.permutation(n)
            labels = np.array([group_labels[i % n_variants] for i in range(n)])
            # Shuffle to match random permutation
            final_labels = np.empty(n, dtype=object)
            for pos, idx in enumerate(indices):
                final_labels[idx] = labels[pos]
            labels = final_labels

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "group": labels,
        })

        return df

    # ------------------------------------------------------------------
    # Covariate/persona balance checks
    # ------------------------------------------------------------------

    def compute_balance_check(
        self,
        data: pd.DataFrame,
        covariates: Optional[List[str]] = None,
        categorical_covariates: Optional[List[str]] = None,
        group_col: str = "group",
        treatment_label: str = "treatment",
        control_label: str = "control",
        imbalance_threshold: float = 0.10,
        covariate_roles: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Check treatment/control covariate balance using SMD.

        Random assignment controls confounding in expectation, but the
        realized sample still needs a balance diagnostic. Numeric covariates
        use standardized mean difference (SMD). Categorical covariates are
        expanded to level indicators and checked with the same SMD formula.
        ``abs(SMD) <= 0.10`` is treated as balanced by default.
        """
        if (
            group_col not in data.columns
            and group_col == "group"
            and "treatment_group" in data.columns
        ):
            group_col = "treatment_group"
        if group_col not in data.columns:
            raise ValueError(f"Balance check requires '{group_col}' column")
        data = data.copy()
        if "signup_month" not in data.columns and "signup_date" in data.columns:
            signup_month = pd.to_datetime(
                data["signup_date"], errors="coerce"
            ).dt.strftime("%Y-%m")
            fallback = data["signup_date"].astype(str).str.slice(0, 7)
            data["signup_month"] = signup_month.fillna(fallback)

        observed_groups = set(data[group_col].dropna().astype(str).unique())
        required_groups = {treatment_label, control_label}
        if not required_groups.issubset(observed_groups):
            raise ValueError(
                "Balance check requires treatment/control groups: "
                f"missing {required_groups - observed_groups}"
            )

        excluded = {
            "customer_id",
            "group",
            "treatment_group",
            "metric",
            "churn_label",
            "churned",
            "converted",
            "revenue",
        }
        if categorical_covariates is None:
            categorical_covariates = [
                col for col in ["persona", "signup_month"] if col in data.columns
            ]
        categorical_covariates = [
            col for col in categorical_covariates if col in data.columns
        ]

        if covariates is None:
            covariates = [
                col
                for col in data.columns
                if col not in excluded
                and col not in categorical_covariates
                and pd.api.types.is_numeric_dtype(data[col])
            ]
        covariates = [col for col in covariates if col in data.columns]

        rows: List[Dict[str, Any]] = []
        treatment_mask = data[group_col].astype(str) == treatment_label
        control_mask = data[group_col].astype(str) == control_label
        n_treatment = int(treatment_mask.sum())
        n_control = int(control_mask.sum())
        default_pre_treatment = {"persona", "signup_month", "tenure_days"}
        covariate_roles = {
            col: "pre_treatment_confounder_balance"
            for col in default_pre_treatment
            if col in data.columns
        } | (covariate_roles or {})

        def _pass_fail(abs_smd: float) -> str:
            if np.isfinite(abs_smd) and abs_smd <= imbalance_threshold:
                return "pass"
            return "fail"

        for covariate in covariates:
            treatment = pd.to_numeric(
                data.loc[treatment_mask, covariate], errors="coerce"
            ).dropna()
            control = pd.to_numeric(
                data.loc[control_mask, covariate], errors="coerce"
            ).dropna()
            treatment_mean = float(treatment.mean()) if len(treatment) else np.nan
            control_mean = float(control.mean()) if len(control) else np.nan
            difference = treatment_mean - control_mean
            pooled_sd = float(
                np.sqrt((treatment.var(ddof=1) + control.var(ddof=1)) / 2)
            ) if len(treatment) > 1 and len(control) > 1 else np.nan
            if np.isfinite(pooled_sd) and pooled_sd > 0:
                smd = float(difference / pooled_sd)
            elif np.isfinite(difference) and abs(difference) <= 1e-12:
                smd = 0.0
            else:
                smd = float("inf")
            abs_smd = abs(smd)
            pass_fail = _pass_fail(abs_smd)
            rows.append({
                "covariate": covariate,
                "level": "",
                "covariate_type": "numeric",
                "analysis_role": covariate_roles.get(covariate, "covariate_balance"),
                "treatment_mean_or_share": treatment_mean,
                "control_mean_or_share": control_mean,
                "group_difference": float(difference),
                "standardized_mean_difference": smd,
                "abs_standardized_mean_difference": abs_smd,
                "threshold": float(imbalance_threshold),
                "pass_fail": pass_fail,
                "balance_pass": pass_fail == "pass",
                "n_treatment": n_treatment,
                "n_control": n_control,
                "missing_treatment": int(
                    data.loc[treatment_mask, covariate].isna().sum()
                ),
                "missing_control": int(
                    data.loc[control_mask, covariate].isna().sum()
                ),
            })

        for covariate in categorical_covariates:
            values = data.loc[treatment_mask | control_mask, covariate].dropna()
            levels = sorted(values.astype(str).unique())
            for level in levels:
                t_indicator = data.loc[treatment_mask, covariate].astype(str) == level
                c_indicator = data.loc[control_mask, covariate].astype(str) == level
                treatment_share = float(t_indicator.mean()) if n_treatment else np.nan
                control_share = float(c_indicator.mean()) if n_control else np.nan
                difference = treatment_share - control_share
                pooled_sd = np.sqrt(
                    (
                        treatment_share * (1 - treatment_share)
                        + control_share * (1 - control_share)
                    ) / 2
                )
                if np.isfinite(pooled_sd) and pooled_sd > 0:
                    smd = float(difference / pooled_sd)
                elif abs(difference) <= 1e-12:
                    smd = 0.0
                else:
                    smd = float("inf")
                abs_smd = abs(smd)
                pass_fail = _pass_fail(abs_smd)
                rows.append({
                    "covariate": covariate,
                    "level": level,
                    "covariate_type": "categorical_level",
                    "analysis_role": covariate_roles.get(
                        covariate, "covariate_balance"
                    ),
                    "treatment_mean_or_share": treatment_share,
                    "control_mean_or_share": control_share,
                    "group_difference": float(difference),
                    "standardized_mean_difference": smd,
                    "abs_standardized_mean_difference": abs_smd,
                    "threshold": float(imbalance_threshold),
                    "pass_fail": pass_fail,
                    "balance_pass": pass_fail == "pass",
                    "n_treatment": n_treatment,
                    "n_control": n_control,
                    "missing_treatment": int(
                        data.loc[treatment_mask, covariate].isna().sum()
                    ),
                    "missing_control": int(
                        data.loc[control_mask, covariate].isna().sum()
                    ),
                })

        return pd.DataFrame(rows)

    def save_balance_check(
        self,
        data: pd.DataFrame,
        output_dir: Union[str, Path] = "results",
        experiment_name: str = "simulated_retention_campaign",
        covariates: Optional[List[str]] = None,
        categorical_covariates: Optional[List[str]] = None,
        group_col: str = "group",
        imbalance_threshold: float = 0.10,
        covariate_roles: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Persist A/B test balance-check evidence as CSV and JSON."""
        balance = self.compute_balance_check(
            data=data,
            covariates=covariates,
            categorical_covariates=categorical_covariates,
            group_col=group_col,
            imbalance_threshold=imbalance_threshold,
            covariate_roles=covariate_roles,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_path / "ab_test_balance_check.csv"
        json_path = output_path / "ab_test_balance_check.json"
        balance.to_csv(csv_path, index=False)

        checked_numeric = sorted(
            balance.loc[balance["covariate_type"] == "numeric", "covariate"].unique()
        )
        checked_categorical = sorted(
            balance.loc[
                balance["covariate_type"] == "categorical_level", "covariate"
            ].unique()
        )
        failed = balance[balance["pass_fail"] == "fail"]
        records = json.loads(balance.to_json(orient="records"))
        role_summary = {}
        if "analysis_role" in balance.columns:
            for role, frame in balance.groupby("analysis_role"):
                role_summary[str(role)] = {
                    "total_checks": int(len(frame)),
                    "passed_checks": int((frame["pass_fail"] == "pass").sum()),
                    "failed_checks": int((frame["pass_fail"] == "fail").sum()),
                    "overall_pass": bool((frame["pass_fail"] == "pass").all()),
                }
        confounder_summary = role_summary.get("pre_treatment_confounder_balance")
        payload = {
            "experiment_name": experiment_name,
            "assignment_strategy": "random_assignment",
            "confounding_control_rationale": (
                "Treatment/Control random assignment makes potential "
                "confounders independent of treatment in expectation, so "
                "pre-outcome differences should be sampling noise rather than "
                "systematic selection. The persisted balance check verifies "
                "the realized sample before interpreting treatment effects."
            ),
            "balance_check_method": (
                "Numeric covariates use standardized mean difference (SMD); "
                "persona and other categorical covariates are expanded to "
                "level indicators and checked with the same SMD formula."
            ),
            "imbalance_threshold": float(imbalance_threshold),
            "threshold_interpretation": {
                "pass": (
                    f"|SMD| <= {imbalance_threshold:.2f} means the realized "
                    "A/B split is balanced enough for unadjusted "
                    "difference-in-means analysis."
                ),
                "fail": (
                    f"|SMD| > {imbalance_threshold:.2f} flags practical "
                    "imbalance; investigate stratification, regression "
                    "adjustment, or rerandomization before causal "
                    "interpretation."
                ),
            },
            "covariates_checked": {
                "numeric": checked_numeric,
                "categorical": checked_categorical,
            },
            "group_sizes": {
                "treatment": (
                    int(balance["n_treatment"].iloc[0]) if not balance.empty else 0
                ),
                "control": (
                    int(balance["n_control"].iloc[0]) if not balance.empty else 0
                ),
            },
            "summary": {
                "total_checks": int(len(balance)),
                "passed_checks": int((balance["pass_fail"] == "pass").sum()),
                "failed_checks": int((balance["pass_fail"] == "fail").sum()),
                "max_abs_standardized_mean_difference": (
                    float(balance["abs_standardized_mean_difference"].max())
                    if not balance.empty else 0.0
                ),
                "overall_pass": bool(failed.empty),
                "overall_confounding_control_pass": (
                    bool(confounder_summary["overall_pass"])
                    if confounder_summary is not None
                    else bool(failed.empty)
                ),
            },
            "role_summary": role_summary,
            "failed_covariates": json.loads(failed.to_json(orient="records")),
            "results": records,
            "csv_path": str(csv_path),
            "json_path": str(json_path),
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        return payload

    # ------------------------------------------------------------------
    # Statistical significance testing
    # ------------------------------------------------------------------

    def _detect_metric_type(self, data: pd.DataFrame, metric: str) -> str:
        """Detect whether a metric is binary or continuous."""
        unique_vals = data[metric].dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            return "binary"
        return "continuous"

    def compute_significance(
        self,
        data: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
        test_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute statistical significance between treatment and control.

        Automatically selects the appropriate test:
        - Binary metrics: Z-test for proportions (with chi-square backup)
        - Continuous metrics: Welch's t-test

        Parameters
        ----------
        data : pd.DataFrame
            Experiment data with ``group`` and ``metric`` columns.
        metric : str
            Column name of the metric to test.
        alpha : float
            Significance level (default 0.05).
        test_type : str, optional
            Force test type: ``"ztest"``, ``"chi2"``, or ``"ttest"``.

        Returns
        -------
        dict
            Results with keys: ``test_statistic``, ``p_value``,
            ``is_significant``, ``test_used``, ``alpha``.
        """
        treatment = data[data["group"] == "treatment"][metric].dropna()
        control = data[data["group"] == "control"][metric].dropna()

        if test_type is None:
            metric_type = self._detect_metric_type(data, metric)
            test_type = "ztest" if metric_type == "binary" else "ttest"

        if test_type == "ztest":
            result = self._z_test_proportions(treatment, control)
        elif test_type == "chi2":
            result = self._chi_square_test(treatment, control)
        else:
            result = self._t_test(treatment, control)

        result["is_significant"] = result["p_value"] < alpha
        result["alpha"] = alpha

        return result

    def _z_test_proportions(
        self, treatment: pd.Series, control: pd.Series
    ) -> Dict[str, Any]:
        """Two-proportion Z-test."""
        n_t = len(treatment)
        n_c = len(control)
        p_t = treatment.mean()
        p_c = control.mean()

        # Pooled proportion
        p_pool = (treatment.sum() + control.sum()) / (n_t + n_c)

        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))

        if se == 0:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "test_used": "z_test_proportions",
            }

        z_stat = (p_t - p_c) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "test_statistic": float(z_stat),
            "p_value": float(p_value),
            "test_used": "z_test_proportions",
        }

    def _chi_square_test(
        self, treatment: pd.Series, control: pd.Series
    ) -> Dict[str, Any]:
        """Chi-square test for independence on binary outcomes."""
        t_success = int(treatment.sum())
        t_fail = len(treatment) - t_success
        c_success = int(control.sum())
        c_fail = len(control) - c_success

        contingency = np.array([[t_success, t_fail], [c_success, c_fail]])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

        return {
            "test_statistic": float(chi2),
            "p_value": float(p_value),
            "test_used": "chi_square",
        }

    def _t_test(
        self, treatment: pd.Series, control: pd.Series
    ) -> Dict[str, Any]:
        """Welch's t-test for continuous metrics."""
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        return {
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "test_used": "welch_t_test",
        }

    # ------------------------------------------------------------------
    # Analysis (effect size + confidence interval)
    # ------------------------------------------------------------------

    def analyze(
        self,
        data: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
    ) -> Dict[str, Any]:
        """Analyze an A/B test experiment.

        Computes effect size, confidence interval, and significance.

        Parameters
        ----------
        data : pd.DataFrame
            Experiment data.
        metric : str
            Metric column name.
        alpha : float
            Significance level.
        confidence_level : float
            Confidence level for CI (default 0.95).

        Returns
        -------
        dict
            Analysis results including ``effect_size``,
            ``confidence_interval``, ``p_value``, etc.
        """
        treatment = data[data["group"] == "treatment"][metric].dropna()
        control = data[data["group"] == "control"][metric].dropna()

        # Effect size = difference in means (treatment - control)
        effect_size = float(treatment.mean() - control.mean())

        # Confidence interval for the difference
        ci = self._confidence_interval(
            treatment, control, confidence_level=confidence_level
        )

        # Statistical significance
        sig_result = self.compute_significance(data, metric, alpha=alpha)

        result = {
            "effect_size": effect_size,
            "confidence_interval": ci,
            "treatment_mean": float(treatment.mean()),
            "control_mean": float(control.mean()),
            "treatment_size": len(treatment),
            "control_size": len(control),
            **sig_result,
        }

        return result

    def _confidence_interval(
        self,
        treatment: pd.Series,
        control: pd.Series,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute confidence interval for the difference in means.

        Uses the normal approximation for large samples.

        Returns
        -------
        tuple of (float, float)
            (lower_bound, upper_bound)
        """
        diff = treatment.mean() - control.mean()

        se = np.sqrt(
            treatment.var(ddof=1) / len(treatment)
            + control.var(ddof=1) / len(control)
        )

        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)

        lower = diff - z * se
        upper = diff + z * se

        return (float(lower), float(upper))

    # ------------------------------------------------------------------
    # Experiment summary
    # ------------------------------------------------------------------

    def get_summary(
        self,
        data: pd.DataFrame,
        metric: str,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Generate a summary report for an A/B test.

        Parameters
        ----------
        data : pd.DataFrame
            Experiment data.
        metric : str
            Metric column name.
        alpha : float
            Significance level.

        Returns
        -------
        dict
            Summary with group sizes, means, lift, significance, etc.
        """
        treatment = data[data["group"] == "treatment"][metric].dropna()
        control = data[data["group"] == "control"][metric].dropna()

        treatment_mean = float(treatment.mean())
        control_mean = float(control.mean())

        # Relative lift: (treatment - control) / control
        if control_mean != 0:
            relative_lift = (treatment_mean - control_mean) / abs(control_mean)
        else:
            relative_lift = float("inf") if treatment_mean != 0 else 0.0

        sig_result = self.compute_significance(data, metric, alpha=alpha)
        ci = self._confidence_interval(treatment, control)

        summary = {
            "treatment_size": len(treatment),
            "control_size": len(control),
            "treatment_mean": treatment_mean,
            "control_mean": control_mean,
            "absolute_difference": treatment_mean - control_mean,
            "relative_lift": float(relative_lift),
            "confidence_interval": ci,
            **sig_result,
        }

        return summary

    def to_dashboard_experiment(
        self,
        result: Dict[str, Any],
        *,
        name: Optional[str] = None,
        duration_days: Optional[int] = None,
        power: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Convert a single A/B result into dashboard detailed schema."""
        treatment_rate = result.get("treatment_churn_rate")
        control_rate = result.get("control_churn_rate")

        if treatment_rate is None:
            treatment_rate = result.get("treatment_mean", 0.0)
        if control_rate is None:
            control_rate = result.get("control_mean", 0.0)

        treatment_rate = _safe_rate_from_mean(float(treatment_rate))
        control_rate = _safe_rate_from_mean(float(control_rate))
        absolute_effect = float(treatment_rate - control_rate)

        lift = result.get("lift")
        if lift is None:
            relative = result.get("relative_lift")
            if relative is not None:
                lift = float(relative)
            elif control_rate != 0:
                lift = (control_rate - treatment_rate) / abs(control_rate)
            else:
                lift = 0.0

        ci = result.get("confidence_interval")
        if ci is None:
            ci = (absolute_effect, absolute_effect)
        effect_size = result.get("effect_size_cohens_h")
        if effect_size is None:
            p1 = np.clip(treatment_rate, 1e-6, 1 - 1e-6)
            p2 = np.clip(control_rate, 1e-6, 1 - 1e-6)
            effect_size = 2 * (
                np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))
            )

        statistically_significant = bool(result.get("is_significant", False))
        beneficial_significant = statistically_significant and treatment_rate < control_rate
        power_payload = result.get("power_analysis", {}) or {}
        design_power = float(
            power_payload.get("target_power", power_payload.get("design_power", 0.80))
        )
        required_sample_size = power_payload.get("required_sample_size")
        if required_sample_size is None:
            required_sample_size = self.power_analysis.required_sample_size(
                baseline_rate=control_rate if 0 < control_rate < 1 else 0.5,
                mde=max(abs(absolute_effect), 0.01),
                alpha=float(result.get("alpha", 0.05)),
                power=design_power,
            )
        required_sample_size = int(required_sample_size)
        observed_group_n = max(
            min(int(result.get("treatment_size", 0)), int(result.get("control_size", 0))),
            1,
        )
        derived_power = (
            power
            if power is not None
            else float(result.get("power", self.power_analysis.compute_power(
                n=observed_group_n,
                baseline_rate=control_rate if 0 < control_rate < 1 else 0.5,
                mde=max(abs(absolute_effect), 1e-6),
                alpha=float(result.get("alpha", 0.05)),
            )))
        )
        observed_power = float(np.clip(derived_power, 0.0, 1.0))
        is_underpowered = (
            observed_power < design_power
            or int(result.get("treatment_size", 0)) < required_sample_size
            or int(result.get("control_size", 0)) < required_sample_size
        )
        passes_power_gate = beneficial_significant and not is_underpowered

        return {
            "name": name or result.get("experiment_name") or result.get("name", "Experiment"),
            "treatment_size": int(result.get("treatment_size", 0)),
            "control_size": int(result.get("control_size", 0)),
            "treatment_churn_rate": treatment_rate,
            "control_churn_rate": control_rate,
            "lift": float(lift),
            "p_value": float(result.get("p_value", 1.0)),
            "is_significant": bool(passes_power_gate),
            "statistically_significant": bool(statistically_significant),
            "confidence_interval": [float(ci[0]), float(ci[1])],
            "effect_size_cohens_h": float(abs(effect_size)),
            "power": observed_power,
            "observed_power": observed_power,
            "design_power": design_power,
            "required_sample_size_per_group": required_sample_size,
            "required_total_sample_size": required_sample_size * 2,
            "is_underpowered": bool(is_underpowered),
            "power_status": "underpowered" if is_underpowered else "adequately_powered",
            "alpha": float(result.get("alpha", 0.05)),
            "absolute_effect": absolute_effect,
            "test_type": result.get("test_used", result.get("test_type", "unknown")),
            "duration_days": int(duration_days if duration_days is not None else result.get("duration_days", 14)),
        }

    def to_dashboard_detailed_results(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Convert framework or CLI results into detailed dashboard schema."""
        if isinstance(results, dict) and "experiments" in results:
            experiments = results["experiments"]
        elif isinstance(results, dict):
            experiments = [results]
        else:
            experiments = list(results)

        converted = [
            self.to_dashboard_experiment(exp, name=exp.get("name"))
            for exp in experiments
        ]
        best_experiment = None
        if converted:
            best_experiment = max(
                converted,
                key=lambda exp: (exp["is_significant"], exp["lift"]),
            )["name"]
        summary = {
            "total_experiments": len(converted),
            "significant_count": sum(1 for exp in converted if exp["is_significant"]),
            "best_experiment": best_experiment,
            "avg_lift": float(np.mean([exp["lift"] for exp in converted])) if converted else 0.0,
        }
        return {
            "experiments": converted,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save framework state to disk.

        Parameters
        ----------
        path : str
            File path (without extension). Saves as JSON.
        """
        save_path = Path(path).with_suffix(".json")
        state = {
            "treatment_ratio": self.treatment_ratio,
            "min_group_size": self.min_group_size,
            "random_seed": self.random_seed,
            "experiments": self.experiments,
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Saved ABTestFramework state to %s", save_path)

    @classmethod
    def load(cls, path: str) -> "ABTestFramework":
        """Load framework state from disk.

        Parameters
        ----------
        path : str
            File path (with or without .json extension).

        Returns
        -------
        ABTestFramework
            Restored framework instance.
        """
        load_path = Path(path)
        if not load_path.suffix:
            load_path = load_path.with_suffix(".json")

        with open(load_path, "r") as f:
            state = json.load(f)

        # Reconstruct a minimal config from saved state
        config = {
            "treatment": {
                "treatment_ratio": state["treatment_ratio"],
                "min_group_size": state["min_group_size"],
            },
            "simulation": {
                "random_seed": state["random_seed"],
            },
        }
        instance = cls(config)
        instance.experiments = state.get("experiments", {})
        return instance
