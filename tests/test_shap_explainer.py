"""
TDD Tests for SHAP Explanations Module.

Tests cover:
- Global feature importance via SHAP values
- Individual prediction explanations (local SHAP)
- Summary plot generation (saves to file)
- Force plot generation for individual predictions
- Dependence plot generation
- Integration with MLChurnModel (XGBoost/LightGBM)
- Integration with EnsembleChurnModel
- Top-k feature extraction
- SHAP value shapes and consistency
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "simulator_config.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Load simulator configuration from YAML."""
    import yaml
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_data():
    """Create synthetic feature data with a clear signal for SHAP tests.

    500 samples, 15 features, binary churn_label with a linear signal
    so SHAP values are meaningful and testable.
    """
    np.random.seed(42)
    n = 500
    n_features = 15

    X = np.random.randn(n, n_features)

    # Create a clear signal from first 5 features
    signal = (
        0.8 * X[:, 0]
        - 0.6 * X[:, 1]
        + 0.5 * X[:, 2]
        - 0.4 * X[:, 3]
        + 0.3 * X[:, 4]
        + np.random.randn(n) * 0.3
    )
    prob = 1 / (1 + np.exp(-signal))
    churn_label = (prob > 0.5).astype(int)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["customer_id"] = [f"C{i:05d}" for i in range(n)]
    df["churn_label"] = churn_label
    dates = pd.date_range("2024-01-01", periods=n, freq="12h")
    df["reference_date"] = dates

    return df


@pytest.fixture
def trained_ml_model(config, sample_data):
    """Train an ML churn model on sample data."""
    from src.models.churn_model import MLChurnModel, time_based_split

    X_train, X_test, y_train, y_test = time_based_split(
        sample_data,
        train_months=config["pipeline"]["train_months"],
        test_months=config["pipeline"]["test_months"],
        date_column="reference_date",
    )
    feature_cols = [c for c in X_train.columns if c.startswith("feature_")]
    model = MLChurnModel(config)
    model.fit(X_train[feature_cols], y_train)
    return model, X_train[feature_cols], X_test[feature_cols], y_test


@pytest.fixture
def explainer_instance(config, trained_ml_model):
    """Create a ShapExplainer for the trained ML model."""
    from src.models.shap_explainer import ShapExplainer

    model, X_train, X_test, _ = trained_ml_model
    return ShapExplainer(model, X_train, config)


# ---------------------------------------------------------------------------
# ShapExplainer instantiation tests
# ---------------------------------------------------------------------------

class TestShapExplainerInstantiation:
    """Test ShapExplainer can be created from trained models."""

    def test_instantiation_from_ml_model(self, explainer_instance):
        """ShapExplainer must be instantiable from a trained ML model."""
        assert explainer_instance is not None

    def test_has_compute_shap_values(self, explainer_instance):
        """ShapExplainer must have compute_shap_values method."""
        assert hasattr(explainer_instance, "compute_shap_values")
        assert callable(explainer_instance.compute_shap_values)

    def test_has_global_importance(self, explainer_instance):
        """ShapExplainer must have global_feature_importance method."""
        assert hasattr(explainer_instance, "global_feature_importance")
        assert callable(explainer_instance.global_feature_importance)

    def test_has_explain_individual(self, explainer_instance):
        """ShapExplainer must have explain_individual method."""
        assert hasattr(explainer_instance, "explain_individual")
        assert callable(explainer_instance.explain_individual)

    def test_has_summary_plot(self, explainer_instance):
        """ShapExplainer must have save_summary_plot method."""
        assert hasattr(explainer_instance, "save_summary_plot")
        assert callable(explainer_instance.save_summary_plot)


# ---------------------------------------------------------------------------
# SHAP value computation tests
# ---------------------------------------------------------------------------

class TestShapValueComputation:
    """Test SHAP value computation correctness."""

    def test_shap_values_shape(self, explainer_instance, trained_ml_model):
        """SHAP values shape must match (n_samples, n_features)."""
        _, _, X_test, _ = trained_ml_model
        shap_values = explainer_instance.compute_shap_values(X_test)
        assert shap_values.shape == X_test.shape, (
            f"SHAP shape {shap_values.shape} != data shape {X_test.shape}"
        )

    def test_shap_values_not_all_zero(self, explainer_instance, trained_ml_model):
        """SHAP values must not be all zeros."""
        _, _, X_test, _ = trained_ml_model
        shap_values = explainer_instance.compute_shap_values(X_test)
        assert np.abs(shap_values).sum() > 0, "All SHAP values are zero"

    def test_shap_values_finite(self, explainer_instance, trained_ml_model):
        """SHAP values must be finite (no NaN or Inf)."""
        _, _, X_test, _ = trained_ml_model
        shap_values = explainer_instance.compute_shap_values(X_test)
        assert np.all(np.isfinite(shap_values)), "SHAP values contain NaN or Inf"

    def test_shap_values_for_subset(self, explainer_instance, trained_ml_model):
        """SHAP values can be computed for a subset of data."""
        _, _, X_test, _ = trained_ml_model
        subset = X_test.head(10)
        shap_values = explainer_instance.compute_shap_values(subset)
        assert shap_values.shape == (10, X_test.shape[1])


# ---------------------------------------------------------------------------
# Global feature importance tests
# ---------------------------------------------------------------------------

class TestGlobalFeatureImportance:
    """Test global feature importance via mean absolute SHAP values."""

    def test_global_importance_returns_dict(self, explainer_instance, trained_ml_model):
        """Global importance must return a dict mapping feature -> importance."""
        _, _, X_test, _ = trained_ml_model
        importance = explainer_instance.global_feature_importance(X_test)
        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_global_importance_all_features(self, explainer_instance, trained_ml_model):
        """Global importance must cover all features."""
        _, _, X_test, _ = trained_ml_model
        importance = explainer_instance.global_feature_importance(X_test)
        for col in X_test.columns:
            assert col in importance, f"Missing feature: {col}"

    def test_global_importance_sorted_descending(self, explainer_instance, trained_ml_model):
        """Global importance dict should be sorted by value descending."""
        _, _, X_test, _ = trained_ml_model
        importance = explainer_instance.global_feature_importance(X_test)
        values = list(importance.values())
        assert values == sorted(values, reverse=True), (
            "Global importance should be sorted descending"
        )

    def test_global_importance_nonnegative(self, explainer_instance, trained_ml_model):
        """Global importance values must be non-negative (mean |SHAP|)."""
        _, _, X_test, _ = trained_ml_model
        importance = explainer_instance.global_feature_importance(X_test)
        for feat, val in importance.items():
            assert val >= 0, f"Negative importance for {feat}: {val}"

    def test_top_k_features(self, explainer_instance, trained_ml_model):
        """get_top_features must return top-k most important features."""
        _, _, X_test, _ = trained_ml_model
        top_5 = explainer_instance.get_top_features(X_test, k=5)
        assert len(top_5) == 5
        assert isinstance(top_5, list)
        # Each element should be a tuple (feature_name, importance)
        for item in top_5:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], (int, float))

    def test_backward_compatible_top_features_alias(
        self, explainer_instance, trained_ml_model
    ):
        """Older top_features alias should still work."""
        _, _, X_test, _ = trained_ml_model
        top_3 = explainer_instance.top_features(X_test, n=3)
        assert len(top_3) == 3


# ---------------------------------------------------------------------------
# Individual prediction explanation tests
# ---------------------------------------------------------------------------

class TestIndividualExplanation:
    """Test individual prediction explanations via SHAP."""

    def test_explain_individual_returns_dict(self, explainer_instance, trained_ml_model):
        """explain_individual must return a dict with SHAP values per feature."""
        _, _, X_test, _ = trained_ml_model
        explanation = explainer_instance.explain_individual(X_test.iloc[0])
        assert isinstance(explanation, dict)

    def test_explain_individual_has_all_features(self, explainer_instance, trained_ml_model):
        """Individual explanation must include all features."""
        _, _, X_test, _ = trained_ml_model
        explanation = explainer_instance.explain_individual(X_test.iloc[0])
        for col in X_test.columns:
            assert col in explanation, f"Missing feature in explanation: {col}"

    def test_explain_individual_has_base_value(self, explainer_instance, trained_ml_model):
        """Individual explanation must include a base_value key."""
        _, _, X_test, _ = trained_ml_model
        explanation = explainer_instance.explain_individual(X_test.iloc[0])
        assert "base_value" in explanation

    def test_explain_individual_values_finite(self, explainer_instance, trained_ml_model):
        """Individual explanation SHAP values must be finite."""
        _, _, X_test, _ = trained_ml_model
        explanation = explainer_instance.explain_individual(X_test.iloc[0])
        for feat, val in explanation.items():
            if feat != "base_value":
                assert np.isfinite(val), f"Non-finite SHAP for {feat}: {val}"

    def test_explain_multiple_individuals(self, explainer_instance, trained_ml_model):
        """explain_individual should work for different samples."""
        _, _, X_test, _ = trained_ml_model
        exp1 = explainer_instance.explain_individual(X_test.iloc[0])
        exp2 = explainer_instance.explain_individual(X_test.iloc[1])
        # Different samples should generally produce different explanations
        vals1 = [exp1[c] for c in X_test.columns]
        vals2 = [exp2[c] for c in X_test.columns]
        assert not np.allclose(vals1, vals2, atol=1e-6), (
            "Different samples produced identical SHAP explanations"
        )


# ---------------------------------------------------------------------------
# Plot generation tests
# ---------------------------------------------------------------------------

class TestShapPlots:
    """Test SHAP plot generation (saves to file, does not display)."""

    def test_summary_plot_saves_file(self, explainer_instance, trained_ml_model, tmp_path):
        """save_summary_plot must create a PNG file."""
        _, _, X_test, _ = trained_ml_model
        output_path = tmp_path / "shap_summary.png"
        explainer_instance.save_summary_plot(X_test, str(output_path))
        assert output_path.exists(), "Summary plot file not created"
        assert output_path.stat().st_size > 0, "Summary plot file is empty"

    def test_backward_compatible_summary_plot_alias(
        self, explainer_instance, trained_ml_model, tmp_path
    ):
        """Older summary_plot alias should still save the figure."""
        _, _, X_test, _ = trained_ml_model
        output_path = tmp_path / "shap_summary_alias.png"
        explainer_instance.summary_plot(X_test, save_path=str(output_path))
        assert output_path.exists(), "Summary plot alias file not created"

    def test_export_top_features_saves_csv(
        self, explainer_instance, trained_ml_model, tmp_path
    ):
        """Top-features export helper should persist a report."""
        _, _, X_test, _ = trained_ml_model
        output_path = tmp_path / "top_features.csv"
        exported = explainer_instance.export_top_features(
            X_test, str(output_path), k=7
        )
        assert output_path.exists(), "Top-features CSV not created"
        assert len(exported) == 7
        assert list(exported.columns) == ["feature", "importance"]

    def test_force_plot_saves_file(self, explainer_instance, trained_ml_model, tmp_path):
        """save_force_plot must create a file for an individual prediction."""
        _, _, X_test, _ = trained_ml_model
        output_path = tmp_path / "shap_force.png"
        explainer_instance.save_force_plot(X_test.iloc[0], str(output_path))
        assert output_path.exists(), "Force plot file not created"
        assert output_path.stat().st_size > 0, "Force plot file is empty"

    def test_dependence_plot_saves_file(self, explainer_instance, trained_ml_model, tmp_path):
        """save_dependence_plot must create a PNG file for a given feature."""
        _, _, X_test, _ = trained_ml_model
        output_path = tmp_path / "shap_dependence.png"
        feature_name = X_test.columns[0]
        explainer_instance.save_dependence_plot(
            X_test, feature_name, str(output_path)
        )
        assert output_path.exists(), "Dependence plot file not created"
        assert output_path.stat().st_size > 0, "Dependence plot file is empty"

    def test_bar_plot_saves_file(self, explainer_instance, trained_ml_model, tmp_path):
        """save_bar_plot must create a global importance bar chart PNG."""
        _, _, X_test, _ = trained_ml_model
        output_path = tmp_path / "shap_bar.png"
        explainer_instance.save_bar_plot(X_test, str(output_path))
        assert output_path.exists(), "Bar plot file not created"
        assert output_path.stat().st_size > 0, "Bar plot file is empty"


# ---------------------------------------------------------------------------
# Integration with different model types
# ---------------------------------------------------------------------------

class TestShapModelIntegration:
    """Test SHAP works with different model backends."""

    def test_works_with_lightgbm_or_xgboost(self, config, sample_data):
        """ShapExplainer must work regardless of selected ML model type."""
        from src.models.churn_model import MLChurnModel, time_based_split
        from src.models.shap_explainer import ShapExplainer

        X_train, X_test, y_train, _ = time_based_split(
            sample_data,
            train_months=config["pipeline"]["train_months"],
            test_months=config["pipeline"]["test_months"],
            date_column="reference_date",
        )
        feature_cols = [c for c in X_train.columns if c.startswith("feature_")]

        model = MLChurnModel(config)
        model.fit(X_train[feature_cols], y_train)

        # Should work for whichever model was selected
        explainer = ShapExplainer(model, X_train[feature_cols], config)
        shap_values = explainer.compute_shap_values(X_test[feature_cols])
        assert shap_values.shape == X_test[feature_cols].shape

    def test_get_explanation_dataframe(self, explainer_instance, trained_ml_model):
        """get_explanation_dataframe must return a DataFrame of SHAP values."""
        _, _, X_test, _ = trained_ml_model
        df = explainer_instance.get_explanation_dataframe(X_test)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == X_test.shape
        assert list(df.columns) == list(X_test.columns)
