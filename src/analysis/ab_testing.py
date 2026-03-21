"""
A/B Testing Statistical Analysis Module.

Provides a comprehensive statistical testing framework for A/B experiments:

- StatisticalTestSuite: Unified interface for multiple statistical tests
  - Independent samples t-test (Welch's)
  - Chi-square test of independence
  - Mann-Whitney U test (non-parametric)
  - Z-test for proportions
- MultipleComparisonCorrection: p-value correction methods
  - Bonferroni correction
  - Benjamini-Hochberg FDR correction
  - Holm-Bonferroni step-down correction

Also re-exports ABTestFramework and PowerAnalysis from src.models.ab_testing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.models.ab_testing import ABTestFramework, PowerAnalysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Multiple Comparison Corrections
# ---------------------------------------------------------------------------


class MultipleComparisonCorrection:
    """Multiple comparison p-value correction methods.

    Provides Bonferroni, Benjamini-Hochberg (FDR), and Holm-Bonferroni
    corrections for controlling family-wise error rate or false discovery
    rate when running multiple statistical tests.
    """

    @staticmethod
    def bonferroni(
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Apply Bonferroni correction for multiple comparisons.

        Adjusts p-values by multiplying each by the number of tests.
        Controls family-wise error rate (FWER).

        Parameters
        ----------
        p_values : list of float
            Raw p-values from individual tests.
        alpha : float
            Desired significance level (default 0.05).

        Returns
        -------
        dict
            ``adjusted_p_values``: list of corrected p-values (capped at 1.0),
            ``rejected``: boolean mask of which hypotheses are rejected,
            ``alpha``: the significance level used,
            ``method``: ``"bonferroni"``,
            ``n_rejected``: number of rejected hypotheses.
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return {
                "adjusted_p_values": [],
                "rejected": [],
                "alpha": alpha,
                "method": "bonferroni",
                "n_rejected": 0,
            }

        adjusted = [min(p * n_tests, 1.0) for p in p_values]
        rejected = [p < alpha for p in adjusted]

        return {
            "adjusted_p_values": adjusted,
            "rejected": rejected,
            "alpha": alpha,
            "method": "bonferroni",
            "n_rejected": sum(rejected),
        }

    @staticmethod
    def fdr_bh(
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Apply Benjamini-Hochberg FDR correction.

        Controls the false discovery rate (FDR) — the expected proportion
        of false positives among rejected hypotheses.

        Parameters
        ----------
        p_values : list of float
            Raw p-values from individual tests.
        alpha : float
            Desired FDR level (default 0.05).

        Returns
        -------
        dict
            ``adjusted_p_values``: BH-adjusted p-values,
            ``rejected``: boolean mask of rejections,
            ``alpha``: significance level,
            ``method``: ``"fdr_bh"``,
            ``n_rejected``: count of rejected hypotheses.
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return {
                "adjusted_p_values": [],
                "rejected": [],
                "alpha": alpha,
                "method": "fdr_bh",
                "n_rejected": 0,
            }

        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # BH adjusted p-values: p_adj(i) = p(i) * m / rank(i)
        adjusted = np.zeros(n_tests)
        for i in range(n_tests):
            rank = i + 1
            adjusted[i] = sorted_p[i] * n_tests / rank

        # Enforce monotonicity (step-up): working backwards, ensure
        # each adjusted p-value is <= the one after it
        for i in range(n_tests - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1])

        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)

        # Map back to original order
        result_adjusted = np.zeros(n_tests)
        for i, orig_idx in enumerate(sorted_indices):
            result_adjusted[orig_idx] = adjusted[i]

        rejected = [p < alpha for p in result_adjusted]

        return {
            "adjusted_p_values": result_adjusted.tolist(),
            "rejected": rejected,
            "alpha": alpha,
            "method": "fdr_bh",
            "n_rejected": sum(rejected),
        }

    @staticmethod
    def holm_bonferroni(
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Apply Holm-Bonferroni step-down correction.

        A sequentially rejective procedure that is uniformly more powerful
        than Bonferroni while still controlling FWER.

        Parameters
        ----------
        p_values : list of float
            Raw p-values from individual tests.
        alpha : float
            Desired significance level (default 0.05).

        Returns
        -------
        dict
            ``adjusted_p_values``: Holm-adjusted p-values,
            ``rejected``: boolean mask of rejections,
            ``alpha``: significance level,
            ``method``: ``"holm_bonferroni"``,
            ``n_rejected``: count of rejected hypotheses.
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return {
                "adjusted_p_values": [],
                "rejected": [],
                "alpha": alpha,
                "method": "holm_bonferroni",
                "n_rejected": 0,
            }

        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # Holm adjusted: p_adj(i) = max(p(j) * (m - j)) for j <= i
        adjusted = np.zeros(n_tests)
        for i in range(n_tests):
            adjusted[i] = sorted_p[i] * (n_tests - i)

        # Enforce monotonicity (step-down): each must be >= previous
        for i in range(1, n_tests):
            adjusted[i] = max(adjusted[i], adjusted[i - 1])

        adjusted = np.minimum(adjusted, 1.0)

        # Map back to original order
        result_adjusted = np.zeros(n_tests)
        for i, orig_idx in enumerate(sorted_indices):
            result_adjusted[orig_idx] = adjusted[i]

        rejected = [p < alpha for p in result_adjusted]

        return {
            "adjusted_p_values": result_adjusted.tolist(),
            "rejected": rejected,
            "alpha": alpha,
            "method": "holm_bonferroni",
            "n_rejected": sum(rejected),
        }


# ---------------------------------------------------------------------------
# Statistical Test Suite
# ---------------------------------------------------------------------------


class StatisticalTestSuite:
    """Unified statistical testing interface for A/B experiments.

    Wraps scipy.stats tests with a consistent return format and
    integrates with MultipleComparisonCorrection for multi-metric tests.

    Parameters
    ----------
    alpha : float
        Default significance level (default 0.05).
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.correction = MultipleComparisonCorrection()
        logger.info("StatisticalTestSuite initialized (alpha=%.3f)", alpha)

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def t_test(
        self,
        group_a: Union[pd.Series, np.ndarray, List[float]],
        group_b: Union[pd.Series, np.ndarray, List[float]],
        equal_var: bool = False,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Independent samples t-test (Welch's by default).

        Compares means of two groups. Welch's t-test does not assume
        equal variances and is the recommended default.

        Parameters
        ----------
        group_a, group_b : array-like
            Sample observations for each group.
        equal_var : bool
            If True, perform Student's t-test assuming equal variance.
            If False (default), perform Welch's t-test.
        alpha : float, optional
            Significance level override.

        Returns
        -------
        dict
            ``test_statistic``, ``p_value``, ``is_significant``,
            ``test_used``, ``effect_size`` (Cohen's d),
            ``mean_a``, ``mean_b``, ``mean_diff``.
        """
        alpha = alpha if alpha is not None else self.alpha
        a = np.asarray(group_a, dtype=float)
        b = np.asarray(group_b, dtype=float)

        t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)

        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
            / (len(a) + len(b) - 2)
        )
        cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

        return {
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": bool(p_value < alpha),
            "test_used": "welch_t_test" if not equal_var else "student_t_test",
            "effect_size": float(cohens_d),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(a) - np.mean(b)),
            "alpha": alpha,
        }

    def chi_square_test(
        self,
        group_a: Union[pd.Series, np.ndarray, List[int]],
        group_b: Union[pd.Series, np.ndarray, List[int]],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Chi-square test of independence for binary outcomes.

        Tests whether the proportions of successes differ between
        two groups using a 2x2 contingency table.

        Parameters
        ----------
        group_a, group_b : array-like
            Binary (0/1) observations for each group.
        alpha : float, optional
            Significance level override.

        Returns
        -------
        dict
            ``test_statistic`` (chi2), ``p_value``, ``is_significant``,
            ``test_used``, ``dof``, ``effect_size`` (Cramér's V),
            ``proportion_a``, ``proportion_b``, ``contingency_table``.
        """
        alpha = alpha if alpha is not None else self.alpha
        a = np.asarray(group_a, dtype=int)
        b = np.asarray(group_b, dtype=int)

        # Build 2x2 contingency table
        a_success = int(np.sum(a))
        a_fail = len(a) - a_success
        b_success = int(np.sum(b))
        b_fail = len(b) - b_success

        contingency = np.array([[a_success, a_fail], [b_success, b_fail]])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Cramér's V effect size
        n = contingency.sum()
        cramers_v = np.sqrt(chi2 / n) if n > 0 else 0.0

        return {
            "test_statistic": float(chi2),
            "p_value": float(p_value),
            "is_significant": bool(p_value < alpha),
            "test_used": "chi_square",
            "dof": int(dof),
            "effect_size": float(cramers_v),
            "proportion_a": float(np.mean(a)),
            "proportion_b": float(np.mean(b)),
            "contingency_table": contingency.tolist(),
            "alpha": alpha,
        }

    def mann_whitney_u_test(
        self,
        group_a: Union[pd.Series, np.ndarray, List[float]],
        group_b: Union[pd.Series, np.ndarray, List[float]],
        alternative: str = "two-sided",
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Mann-Whitney U test (non-parametric).

        Tests whether one distribution is stochastically greater than
        the other. Does not assume normal distributions.

        Parameters
        ----------
        group_a, group_b : array-like
            Sample observations for each group.
        alternative : str
            ``"two-sided"``, ``"less"``, or ``"greater"``.
        alpha : float, optional
            Significance level override.

        Returns
        -------
        dict
            ``test_statistic`` (U), ``p_value``, ``is_significant``,
            ``test_used``, ``effect_size`` (rank-biserial correlation),
            ``median_a``, ``median_b``.
        """
        alpha = alpha if alpha is not None else self.alpha
        a = np.asarray(group_a, dtype=float)
        b = np.asarray(group_b, dtype=float)

        u_stat, p_value = stats.mannwhitneyu(a, b, alternative=alternative)

        # Rank-biserial correlation as effect size
        n_a, n_b = len(a), len(b)
        r = 1 - (2 * u_stat) / (n_a * n_b) if (n_a * n_b) > 0 else 0.0

        return {
            "test_statistic": float(u_stat),
            "p_value": float(p_value),
            "is_significant": bool(p_value < alpha),
            "test_used": "mann_whitney_u",
            "alternative": alternative,
            "effect_size": float(r),
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "alpha": alpha,
        }

    def z_test_proportions(
        self,
        group_a: Union[pd.Series, np.ndarray, List[int]],
        group_b: Union[pd.Series, np.ndarray, List[int]],
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Two-proportion Z-test.

        Tests whether two population proportions are equal using
        a pooled proportion under the null hypothesis.

        Parameters
        ----------
        group_a, group_b : array-like
            Binary (0/1) observations for each group.
        alpha : float, optional
            Significance level override.

        Returns
        -------
        dict
            ``test_statistic`` (z), ``p_value``, ``is_significant``,
            ``test_used``, ``effect_size`` (Cohen's h),
            ``proportion_a``, ``proportion_b``, ``proportion_diff``.
        """
        alpha = alpha if alpha is not None else self.alpha
        a = np.asarray(group_a, dtype=float)
        b = np.asarray(group_b, dtype=float)

        n_a, n_b = len(a), len(b)
        p_a = np.mean(a)
        p_b = np.mean(b)
        p_pool = (np.sum(a) + np.sum(b)) / (n_a + n_b)

        se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n_a + 1.0 / n_b))

        if se == 0:
            z_stat = 0.0
            p_value = 1.0
        else:
            z_stat = (p_a - p_b) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Cohen's h effect size
        cohens_h = 2 * (np.arcsin(np.sqrt(p_a)) - np.arcsin(np.sqrt(p_b)))

        return {
            "test_statistic": float(z_stat),
            "p_value": float(p_value),
            "is_significant": bool(p_value < alpha),
            "test_used": "z_test_proportions",
            "effect_size": float(cohens_h),
            "proportion_a": float(p_a),
            "proportion_b": float(p_b),
            "proportion_diff": float(p_a - p_b),
            "alpha": alpha,
        }

    # ------------------------------------------------------------------
    # Multi-metric testing with correction
    # ------------------------------------------------------------------

    def run_multiple_tests(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        group_column: str = "group",
        group_a_label: str = "treatment",
        group_b_label: str = "control",
        correction_method: str = "bonferroni",
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run statistical tests on multiple metrics with correction.

        Automatically selects the test type per metric (binary vs
        continuous) and applies the specified multiple comparison
        correction.

        Parameters
        ----------
        data : pd.DataFrame
            Experiment data with group column and metric columns.
        metrics : list of str
            Column names of metrics to test.
        group_column : str
            Name of the group assignment column.
        group_a_label, group_b_label : str
            Group labels to compare.
        correction_method : str
            One of ``"bonferroni"``, ``"fdr_bh"``, ``"holm_bonferroni"``.
        alpha : float, optional
            Significance level override.

        Returns
        -------
        dict
            ``individual_results``: per-metric test results,
            ``correction``: multiple comparison correction output,
            ``metrics``: list of metric names tested.
        """
        alpha = alpha if alpha is not None else self.alpha

        mask_a = data[group_column] == group_a_label
        mask_b = data[group_column] == group_b_label

        individual_results = {}
        raw_p_values = []

        for metric in metrics:
            a = data.loc[mask_a, metric].dropna()
            b = data.loc[mask_b, metric].dropna()

            # Detect metric type
            unique_vals = pd.concat([a, b]).unique()
            is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False})

            if is_binary:
                result = self.chi_square_test(a, b, alpha=alpha)
            else:
                result = self.t_test(a, b, alpha=alpha)

            individual_results[metric] = result
            raw_p_values.append(result["p_value"])

        # Apply multiple comparison correction
        correction_fn = {
            "bonferroni": MultipleComparisonCorrection.bonferroni,
            "fdr_bh": MultipleComparisonCorrection.fdr_bh,
            "fdr": MultipleComparisonCorrection.fdr_bh,
            "holm_bonferroni": MultipleComparisonCorrection.holm_bonferroni,
            "holm": MultipleComparisonCorrection.holm_bonferroni,
        }.get(correction_method, MultipleComparisonCorrection.bonferroni)

        correction_result = correction_fn(raw_p_values, alpha=alpha)

        # Update individual results with corrected p-values
        for i, metric in enumerate(metrics):
            individual_results[metric]["adjusted_p_value"] = (
                correction_result["adjusted_p_values"][i]
            )
            individual_results[metric]["is_significant_corrected"] = (
                correction_result["rejected"][i]
            )

        return {
            "individual_results": individual_results,
            "correction": correction_result,
            "metrics": metrics,
        }


# ---------------------------------------------------------------------------
# Experiment Manager
# ---------------------------------------------------------------------------


class ExperimentManager:
    """A/B test experiment manager with configuration, result tracking,
    sequential testing support, and summary reporting.

    Orchestrates the full lifecycle of A/B experiments: creation,
    configuration, data collection, sequential monitoring with optional
    early stopping, and final summary reporting.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically loaded from YAML).
        Reads keys from ``treatment`` and ``simulation`` sections.
    alpha : float
        Default significance level (default 0.05).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        alpha: float = 0.05,
    ) -> None:
        self.config = config or {}
        self.alpha = alpha

        # Underlying frameworks
        self.framework = ABTestFramework(self.config) if self.config else None
        self.stats = StatisticalTestSuite(alpha=alpha)
        self.power = PowerAnalysis()

        # Experiment registry: exp_id -> experiment metadata
        self._experiments: Dict[str, Dict[str, Any]] = {}

        # Result store: exp_id -> list of result snapshots (for sequential tracking)
        self._results: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            "ExperimentManager initialized (alpha=%.3f, n_experiments=0)",
            alpha,
        )

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        metrics: Optional[List[str]] = None,
        hypothesis: str = "",
        baseline_rate: Optional[float] = None,
        mde: Optional[float] = None,
        max_samples: Optional[int] = None,
        sequential: bool = False,
        spending_function: str = "obrien_fleming",
        n_interim_analyses: int = 5,
    ) -> str:
        """Create and configure a new A/B test experiment.

        Parameters
        ----------
        name : str
            Human-readable experiment name.
        description : str
            What the experiment tests.
        metrics : list of str, optional
            Metric column names to track.
        hypothesis : str
            The hypothesis being tested.
        baseline_rate : float, optional
            Expected baseline rate for power analysis.
        mde : float, optional
            Minimum detectable effect for power analysis.
        max_samples : int, optional
            Maximum sample size (triggers auto-stop for sequential tests).
        sequential : bool
            Whether to enable sequential testing / early stopping.
        spending_function : str
            Alpha-spending function for sequential tests.
            One of ``"obrien_fleming"``, ``"pocock"``, ``"linear"``.
        n_interim_analyses : int
            Number of planned interim analyses for sequential tests.

        Returns
        -------
        str
            Unique experiment identifier.
        """
        import uuid as _uuid

        exp_id = f"{name}_{_uuid.uuid4().hex[:8]}"

        # Compute required sample size if baseline_rate and mde are provided
        required_n = None
        if baseline_rate is not None and mde is not None and mde != 0:
            required_n = self.power.required_sample_size(
                baseline_rate=baseline_rate,
                mde=mde,
                alpha=self.alpha,
            )

        experiment = {
            "id": exp_id,
            "name": name,
            "description": description,
            "hypothesis": hypothesis,
            "metrics": metrics or [],
            "baseline_rate": baseline_rate,
            "mde": mde,
            "required_sample_size": required_n,
            "max_samples": max_samples,
            "sequential": sequential,
            "spending_function": spending_function,
            "n_interim_analyses": n_interim_analyses,
            "status": "created",
            "n_analyses_done": 0,
            "decision": None,
        }

        self._experiments[exp_id] = experiment
        self._results[exp_id] = []

        logger.info(
            "Created experiment '%s' (id=%s, sequential=%s)",
            name,
            exp_id,
            sequential,
        )
        return exp_id

    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        """Retrieve experiment configuration by ID.

        Parameters
        ----------
        exp_id : str
            Experiment identifier.

        Returns
        -------
        dict
            Full experiment configuration and current state.

        Raises
        ------
        KeyError
            If the experiment ID is not found.
        """
        if exp_id not in self._experiments:
            raise KeyError(f"Experiment '{exp_id}' not found")
        return self._experiments[exp_id]

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all registered experiments.

        Returns
        -------
        list of dict
            Experiment summaries with id, name, status.
        """
        return [
            {"id": e["id"], "name": e["name"], "status": e["status"]}
            for e in self._experiments.values()
        ]

    # ------------------------------------------------------------------
    # Result tracking
    # ------------------------------------------------------------------

    def record_result(
        self,
        exp_id: str,
        data: pd.DataFrame,
        metric: Optional[str] = None,
        group_column: str = "group",
        group_a_label: str = "treatment",
        group_b_label: str = "control",
    ) -> Dict[str, Any]:
        """Record an analysis result (snapshot) for an experiment.

        Runs statistical tests on the provided data and stores the
        result for tracking over time (useful for sequential analyses).

        Parameters
        ----------
        exp_id : str
            Experiment identifier.
        data : pd.DataFrame
            Experiment data with group and metric columns.
        metric : str, optional
            Metric to analyze. If None, uses the first configured metric.
        group_column : str
            Column containing group labels.
        group_a_label, group_b_label : str
            Labels for treatment and control groups.

        Returns
        -------
        dict
            Analysis result snapshot.

        Raises
        ------
        KeyError
            If the experiment ID is not found.
        ValueError
            If no metric is specified and none configured.
        """
        exp = self.get_experiment(exp_id)

        if metric is None:
            if exp["metrics"]:
                metric = exp["metrics"][0]
            else:
                raise ValueError(
                    "No metric specified and no default metrics configured"
                )

        # Compute group statistics
        mask_a = data[group_column] == group_a_label
        mask_b = data[group_column] == group_b_label
        a = data.loc[mask_a, metric].dropna()
        b = data.loc[mask_b, metric].dropna()

        # Detect metric type and run appropriate test
        unique_vals = pd.concat([a, b]).unique()
        is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False})

        if is_binary:
            test_result = self.stats.z_test_proportions(a.values, b.values, alpha=self.alpha)
        else:
            test_result = self.stats.t_test(a.values, b.values, alpha=self.alpha)

        snapshot = {
            "metric": metric,
            "n_treatment": int(len(a)),
            "n_control": int(len(b)),
            "n_total": int(len(a) + len(b)),
            "treatment_mean": float(a.mean()),
            "control_mean": float(b.mean()),
            "absolute_diff": float(a.mean() - b.mean()),
            **test_result,
        }

        self._results[exp_id].append(snapshot)
        exp["n_analyses_done"] += 1

        if exp["status"] == "created":
            exp["status"] = "running"

        logger.info(
            "Recorded result for experiment '%s' (analysis #%d, p=%.4f)",
            exp_id,
            exp["n_analyses_done"],
            snapshot["p_value"],
        )

        return snapshot

    def get_results(self, exp_id: str) -> List[Dict[str, Any]]:
        """Get all recorded result snapshots for an experiment.

        Parameters
        ----------
        exp_id : str
            Experiment identifier.

        Returns
        -------
        list of dict
            List of result snapshots in chronological order.
        """
        if exp_id not in self._results:
            raise KeyError(f"Experiment '{exp_id}' not found")
        return self._results[exp_id]

    # ------------------------------------------------------------------
    # Sequential testing
    # ------------------------------------------------------------------

    @staticmethod
    def _alpha_spending(
        spending_function: str,
        info_fraction: float,
        alpha: float,
    ) -> float:
        """Compute the cumulative alpha spent at a given information fraction.

        Parameters
        ----------
        spending_function : str
            One of ``"obrien_fleming"``, ``"pocock"``, ``"linear"``.
        info_fraction : float
            Fraction of total planned information (0 to 1).
        alpha : float
            Overall significance level.

        Returns
        -------
        float
            Cumulative alpha spent up to this fraction.
        """
        t = max(min(info_fraction, 1.0), 0.0)

        if spending_function == "obrien_fleming":
            # O'Brien-Fleming-like spending: very conservative early,
            # spends most alpha near the end
            if t == 0:
                return 0.0
            z = stats.norm.ppf(1 - alpha / 2)
            spent = 2 * (1 - stats.norm.cdf(z / np.sqrt(t)))
            return min(spent, alpha)

        elif spending_function == "pocock":
            # Pocock-like spending: alpha * ln(1 + (e-1)*t)
            if t == 0:
                return 0.0
            spent = alpha * np.log(1 + (np.e - 1) * t)
            return min(spent, alpha)

        elif spending_function == "linear":
            # Simple linear spending
            return alpha * t

        else:
            # Default to linear
            return alpha * t

    def sequential_test(
        self,
        exp_id: str,
        data: pd.DataFrame,
        metric: Optional[str] = None,
        group_column: str = "group",
        group_a_label: str = "treatment",
        group_b_label: str = "control",
    ) -> Dict[str, Any]:
        """Run a sequential test analysis with alpha-spending control.

        Records the result and evaluates whether the experiment should
        be stopped early (reject or accept null hypothesis) based on
        the alpha-spending boundary.

        Parameters
        ----------
        exp_id : str
            Experiment identifier (must be configured with sequential=True).
        data : pd.DataFrame
            Current experiment data.
        metric : str, optional
            Metric to analyze.
        group_column : str
            Column containing group labels.
        group_a_label, group_b_label : str
            Group labels.

        Returns
        -------
        dict
            Result snapshot augmented with sequential testing fields:
            ``information_fraction``, ``alpha_spent``, ``boundary_alpha``,
            ``stop_early``, ``decision``.
        """
        exp = self.get_experiment(exp_id)

        # Record the observation
        snapshot = self.record_result(
            exp_id=exp_id,
            data=data,
            metric=metric,
            group_column=group_column,
            group_a_label=group_a_label,
            group_b_label=group_b_label,
        )

        n_planned = exp["n_interim_analyses"]
        n_done = exp["n_analyses_done"]
        info_fraction = min(n_done / max(n_planned, 1), 1.0)

        # Compute alpha-spending boundary
        spending_fn = exp.get("spending_function", "obrien_fleming")
        cumulative_alpha = self._alpha_spending(
            spending_fn, info_fraction, self.alpha
        )

        # Compute incremental alpha for this look
        if n_done > 1:
            prev_fraction = (n_done - 1) / max(n_planned, 1)
            prev_alpha = self._alpha_spending(
                spending_fn, prev_fraction, self.alpha
            )
        else:
            prev_alpha = 0.0

        boundary_alpha = cumulative_alpha - prev_alpha

        # Decision logic
        stop_early = False
        decision = None

        if snapshot["p_value"] < boundary_alpha:
            stop_early = True
            decision = "reject_null"
            exp["status"] = "completed"
            exp["decision"] = "reject_null"
        elif info_fraction >= 1.0:
            # Final analysis — accept null if not significant
            stop_early = True
            if snapshot["p_value"] < self.alpha:
                decision = "reject_null"
            else:
                decision = "accept_null"
            exp["status"] = "completed"
            exp["decision"] = decision
        else:
            decision = "continue"

        # Check max_samples stopping
        max_samples = exp.get("max_samples")
        if max_samples is not None and snapshot["n_total"] >= max_samples:
            if not stop_early:
                stop_early = True
                if snapshot["p_value"] < self.alpha:
                    decision = "reject_null"
                else:
                    decision = "accept_null"
                exp["status"] = "completed"
                exp["decision"] = decision

        # Augment the snapshot
        seq_result = {
            **snapshot,
            "information_fraction": info_fraction,
            "cumulative_alpha_spent": cumulative_alpha,
            "boundary_alpha": boundary_alpha,
            "stop_early": stop_early,
            "decision": decision,
        }

        # Update the last recorded result with sequential info
        self._results[exp_id][-1] = seq_result

        logger.info(
            "Sequential test for '%s': info=%.2f, boundary_alpha=%.4f, "
            "p=%.4f, decision=%s",
            exp_id,
            info_fraction,
            boundary_alpha,
            snapshot["p_value"],
            decision,
        )

        return seq_result

    # ------------------------------------------------------------------
    # Summary reporting
    # ------------------------------------------------------------------

    def get_summary(self, exp_id: str) -> Dict[str, Any]:
        """Generate a comprehensive summary report for an experiment.

        Aggregates configuration, power analysis info, all recorded
        results, and the final decision (if sequential testing was used).

        Parameters
        ----------
        exp_id : str
            Experiment identifier.

        Returns
        -------
        dict
            Summary report with sections: ``experiment`` (config),
            ``results`` (all snapshots), ``latest_result``,
            ``power_analysis``, ``recommendation``.
        """
        exp = self.get_experiment(exp_id)
        results = self.get_results(exp_id)

        latest = results[-1] if results else None

        # Power analysis summary
        power_info = {}
        if exp.get("baseline_rate") and exp.get("mde"):
            power_info["required_sample_size"] = exp.get("required_sample_size")
            power_info["baseline_rate"] = exp["baseline_rate"]
            power_info["mde"] = exp["mde"]
            if latest:
                actual_n = latest.get("n_treatment", 0)
                if actual_n > 0 and exp["baseline_rate"] and exp["mde"]:
                    power_info["achieved_power"] = self.power.compute_power(
                        n=actual_n,
                        baseline_rate=exp["baseline_rate"],
                        mde=exp["mde"],
                        alpha=self.alpha,
                    )

        # Generate recommendation
        recommendation = self._generate_recommendation(exp, latest)

        summary = {
            "experiment": {
                "id": exp["id"],
                "name": exp["name"],
                "description": exp["description"],
                "hypothesis": exp["hypothesis"],
                "status": exp["status"],
                "sequential": exp["sequential"],
                "decision": exp.get("decision"),
                "n_analyses": exp["n_analyses_done"],
            },
            "results": results,
            "latest_result": latest,
            "power_analysis": power_info,
            "recommendation": recommendation,
        }

        return summary

    @staticmethod
    def _generate_recommendation(
        exp: Dict[str, Any],
        latest: Optional[Dict[str, Any]],
    ) -> str:
        """Generate a human-readable recommendation based on results.

        Parameters
        ----------
        exp : dict
            Experiment configuration.
        latest : dict or None
            Most recent analysis result.

        Returns
        -------
        str
            Recommendation text.
        """
        if latest is None:
            return "No results recorded yet. Collect data and run analysis."

        if exp.get("decision") == "reject_null":
            direction = "positive" if latest.get("absolute_diff", 0) > 0 else "negative"
            return (
                f"Statistically significant {direction} effect detected "
                f"(p={latest['p_value']:.4f}). Consider deploying the treatment."
            )
        elif exp.get("decision") == "accept_null":
            return (
                "No statistically significant effect detected. "
                "Consider whether the effect size is practically meaningful "
                "or increase sample size."
            )
        else:
            if latest.get("is_significant"):
                return (
                    f"Current data shows significance (p={latest['p_value']:.4f}) "
                    "but sequential boundary not yet crossed. Continue collecting data."
                )
            return (
                "Experiment is ongoing. No significant effect detected yet. "
                "Continue collecting data."
            )

    def complete_experiment(self, exp_id: str) -> Dict[str, Any]:
        """Mark an experiment as completed and return final summary.

        Parameters
        ----------
        exp_id : str
            Experiment identifier.

        Returns
        -------
        dict
            Final experiment summary.
        """
        exp = self.get_experiment(exp_id)
        results = self.get_results(exp_id)

        if results:
            latest = results[-1]
            if latest.get("is_significant"):
                exp["decision"] = "reject_null"
            else:
                exp["decision"] = "accept_null"
        else:
            exp["decision"] = "no_data"

        exp["status"] = "completed"

        logger.info(
            "Experiment '%s' completed with decision: %s",
            exp_id,
            exp["decision"],
        )

        return self.get_summary(exp_id)


__all__ = [
    "ABTestFramework",
    "PowerAnalysis",
    "StatisticalTestSuite",
    "MultipleComparisonCorrection",
    "ExperimentManager",
]
