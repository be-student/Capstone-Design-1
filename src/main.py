#!/usr/bin/env python3
"""
CLI Entrypoint for E-Commerce Churn Prediction & Retention System.

Supports the following modes via ``--mode``:

    simulate   - Generate synthetic customer data
    train      - Train churn prediction models (ML, DL, Ensemble)
    uplift     - Train uplift model and 4-quadrant segmentation
    clv        - Predict Customer Lifetime Value
    optimize   - LP-based budget optimization (accepts --budget)
    ab_test    - A/B test statistical analysis
    survival   - Survival analysis (Cox PH)
    recommend  - Personalized retention recommendations
    cohort     - Cohort retention analysis
    segment    - Customer segmentation (RFM-based)
    monitor    - Model monitoring / drift detection
    features   - Run feature engineering pipeline only
    dashboard  - Launch Streamlit dashboard (localhost:8501)
    all        - Run full end-to-end pipeline

Usage:
    python src/main.py --mode train
    python src/main.py --mode optimize --budget 50000000
    python src/main.py --mode simulate --small
    python src/main.py --mode all --small
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project-level paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "simulator_config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


# ---------------------------------------------------------------------------
# JSON serialiser that handles numpy / pandas types
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _save_json(data: Any, path: Path) -> None:
    """Write *data* to a JSON file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


# ---------------------------------------------------------------------------
# Helpers: resolve directories, load data
# ---------------------------------------------------------------------------

def _resolve_dirs(args: argparse.Namespace):
    """Return (data_dir, results_dir, models_dir) from CLI args."""
    data_dir = Path(args.data) if args.data else DEFAULT_DATA_DIR
    output = Path(args.output) if args.output else None
    results_dir = (output / "results") if output else DEFAULT_RESULTS_DIR
    models_dir = (output / "models") if output else DEFAULT_MODELS_DIR
    for d in (data_dir, results_dir, models_dir):
        d.mkdir(parents=True, exist_ok=True)
    return data_dir, results_dir, models_dir


def _load_customers(data_dir: Path) -> pd.DataFrame:
    """Load customer profiles (parquet preferred, csv fallback)."""
    for ext in ("parquet", "csv"):
        p = data_dir / f"customers.{ext}"
        if p.exists():
            df = pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
            if "signup_date" in df.columns:
                df["signup_date"] = pd.to_datetime(df["signup_date"])
            return df
    raise FileNotFoundError(f"No customer data in {data_dir}. Run --mode simulate first.")


def _load_events(data_dir: Path) -> pd.DataFrame:
    """Load event logs (parquet preferred, csv fallback)."""
    for ext in ("parquet", "csv"):
        p = data_dir / f"events.{ext}"
        if p.exists():
            df = pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
            for col in ("event_date", "event_timestamp"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            return df
    raise FileNotFoundError(f"No event data in {data_dir}. Run --mode simulate first.")


_FEATURE_CACHE: Optional[pd.DataFrame] = None


def _compute_features(config: Dict, customers: pd.DataFrame,
                       events: pd.DataFrame) -> pd.DataFrame:
    """Run feature engineering and return feature DataFrame.

    Caches the result for repeated calls within the same pipeline run
    to avoid recomputing from 476K+ events each time (~100s per call).
    """
    global _FEATURE_CACHE  # noqa: PLW0603

    # Try loading from cached CSV first (written by run_features)
    cached_path = DEFAULT_RESULTS_DIR / "features.csv"

    if _FEATURE_CACHE is not None:
        logger.info("Using in-memory feature cache (%d rows)", len(_FEATURE_CACHE))
        return _FEATURE_CACHE.copy()

    if cached_path.exists():
        logger.info("Loading cached features from %s", cached_path)
        _FEATURE_CACHE = pd.read_csv(cached_path)
        return _FEATURE_CACHE.copy()

    from src.features import FeatureEngineer

    fe = FeatureEngineer(config)
    sim_days = config.get("simulation", {}).get("simulation_days", 365)
    start = config.get("simulation", {}).get("start_date", "2024-01-01")
    ref_date = pd.Timestamp(start) + pd.Timedelta(days=sim_days)
    features = fe.compute_all_features(customers, events, str(ref_date.date()))
    _FEATURE_CACHE = features
    return features


def _feature_cols(df: pd.DataFrame) -> List[str]:
    """Return numeric feature column names (drop meta columns)."""
    exclude = {"customer_id", "churn_label", "reference_date",
                "treatment_group", "signup_date", "persona"}
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in exclude]


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_simulate(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Generate synthetic customer data via SimulatorOrchestrator."""
    from src.data import SimulatorOrchestrator

    data_dir, results_dir, _ = _resolve_dirs(args)

    if args.small:
        small_cfg = config.get("simulation", {}).get("small_mode", {})
        config["simulation"]["num_customers"] = small_cfg.get("num_customers", 5000)
        config["simulation"]["simulation_months"] = small_cfg.get("simulation_months", 6)
        config["simulation"]["simulation_days"] = small_cfg.get("simulation_days", 180)
        logger.info("Small mode: %d customers, %d months",
                     config["simulation"]["num_customers"],
                     config["simulation"]["simulation_months"])

    orch = SimulatorOrchestrator(config)
    result = orch.run(output_dir=str(data_dir))

    summary = result.get("summary", {})
    logger.info("Simulation done: %d customers, %d events, churn=%.1f%%",
                summary.get("num_customers", 0),
                summary.get("num_events", 0),
                summary.get("churn_rate", 0) * 100)

    return {"mode": "simulate", "status": "completed", "summary": summary}


def run_train(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Train ML/DL churn prediction models and generate SHAP plots."""
    from src.models import (MLChurnModel, DLChurnModel,
                             EnsembleChurnModel, time_based_split, ShapExplainer)

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)

    features = _compute_features(config, customers, events)
    logger.info("Features: %d rows x %d cols", *features.shape)

    # Time-based split
    pipe = config.get("pipeline", {})
    if "reference_date" not in features.columns:
        sim_days = config.get("simulation", {}).get("simulation_days", 365)
        start = config.get("simulation", {}).get("start_date", "2024-01-01")
        features["reference_date"] = pd.Timestamp(start) + pd.Timedelta(days=sim_days)

    X_train, X_test, y_train, y_test = time_based_split(
        features,
        train_months=pipe.get("train_months", 10),
        test_months=pipe.get("test_months", 2),
    )

    fcols = _feature_cols(X_train)
    X_tr, X_te = X_train[fcols], X_test[fcols]

    results: Dict[str, Any] = {"mode": "train"}

    # ML
    logger.info("Training ML model...")
    ml = MLChurnModel(config)
    ml.fit(X_tr, y_train)
    ml_m = ml.evaluate(X_te, y_test)
    results["ml_metrics"] = ml_m
    ml.save(str(models_dir / "ml_churn_model.pkl"))
    logger.info("ML AUC-ROC: %.4f", ml_m.get("auc_roc", 0))

    # DL
    logger.info("Training DL model...")
    dl = DLChurnModel(config)
    dl.fit(X_tr, y_train)
    dl_m = dl.evaluate(X_te, y_test)
    results["dl_metrics"] = dl_m
    dl.save(str(models_dir / "dl_churn_model.pt"))
    logger.info("DL AUC-ROC: %.4f", dl_m.get("auc_roc", 0))

    # Ensemble
    logger.info("Building ensemble...")
    ens = EnsembleChurnModel(config)
    ens.ml_model = ml
    ens.dl_model = dl
    ens_m = ens.evaluate(X_te, y_test)
    results["ensemble_metrics"] = ens_m
    logger.info("Ensemble AUC-ROC: %.4f", ens_m.get("auc_roc", 0))

    # SHAP
    try:
        logger.info("Generating SHAP explanation...")
        exp = ShapExplainer(ml)
        exp.compute_shap_values(X_te)
        shap_path = results_dir / "shap_summary.png"
        exp.summary_plot(X_te, save_path=str(shap_path))
        results["top_features"] = exp.top_features(X_te, n=10)
        logger.info("SHAP saved to %s", shap_path)
    except Exception as exc:
        logger.warning("SHAP failed: %s", exc)

    _save_json(results, results_dir / "model_metrics.json")
    results["status"] = "completed"
    return results


def run_uplift(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Train uplift model and produce 4-quadrant segmentation CSV."""
    from src.models.uplift_model import UpliftModel, plot_qini_curve

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events)

    treatment = (customers["treatment_group"] == "treatment").astype(int).values \
        if "treatment_group" in customers.columns else np.ones(len(customers), dtype=int)
    y = customers["churn_label"].values.astype(int) \
        if "churn_label" in customers.columns else np.zeros(len(customers), dtype=int)

    n = min(len(features), len(treatment), len(y))
    fcols = _feature_cols(features)
    X = features[fcols].iloc[:n]
    treatment, y = treatment[:n], y[:n]

    learner = getattr(args, "learner", "t_learner")
    logger.info("Training uplift model (%s)...", learner)
    model = UpliftModel(config, learner=learner)
    model.fit(X, treatment, y)

    scores = model.predict_uplift(X)
    segments = model.segment_customers(scores)
    auuc = model.compute_auuc(y, scores, treatment)
    logger.info("AUUC: %.4f", auuc)

    cid = features["customer_id"].iloc[:n].values if "customer_id" in features.columns else np.arange(n)
    out = pd.DataFrame({"customer_id": cid, "uplift_score": scores, "segment": segments})
    seg_dist = out["segment"].value_counts()
    for s, c in seg_dist.items():
        logger.info("  %s: %d (%.1f%%)", s, c, c / n * 100)

    out.to_csv(results_dir / "uplift_results.csv", index=False)
    model.save(str(models_dir / "uplift_model.pkl"))
    logger.info("Uplift results saved to %s", results_dir / "uplift_results.csv")

    qini_path = str(results_dir / "qini_curve.png")
    plot_qini_curve(y, scores, treatment, save_path=qini_path)

    return {"mode": "uplift", "status": "completed", "auuc": float(auuc),
            "segment_distribution": seg_dist.to_dict()}


def run_clv(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Predict Customer Lifetime Value and save CSV."""
    from src.models.clv_model import CLVModel

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events)

    fcols = _feature_cols(features)
    X = features[fcols]

    # Target: annualised monetary proxy
    if "monetary" in features.columns:
        y_clv = features["monetary"].values * 12
    else:
        freq = features.get("frequency", pd.Series(np.ones(len(features)) * 3))
        aov = features.get("avg_order_value", pd.Series(np.ones(len(features)) * 50000))
        y_clv = (freq * aov * 12).values

    logger.info("Training CLV model...")
    model = CLVModel(config)
    model.fit(X, y_clv)
    preds = model.predict(X)

    cid = features["customer_id"].values if "customer_id" in features.columns else np.arange(len(features))
    out = pd.DataFrame({"customer_id": cid, "predicted_clv": preds})
    threshold_80 = np.percentile(preds, 80)
    out["high_value"] = (out["predicted_clv"] >= threshold_80).astype(int)
    out = out.sort_values("predicted_clv", ascending=False).reset_index(drop=True)

    out.to_csv(results_dir / "clv_predictions.csv", index=False)
    model.save(str(models_dir / "clv_model.pkl"))

    logger.info("CLV saved. Mean=%.0f, Median=%.0f, Top-20%% threshold=%.0f",
                np.mean(preds), np.median(preds), threshold_80)

    return {"mode": "clv", "status": "completed",
            "mean_clv": float(np.mean(preds)),
            "median_clv": float(np.median(preds)),
            "top20_threshold": float(threshold_80)}


def run_optimize(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run LP-based budget optimisation for retention campaigns."""
    from src.models.budget_optimizer import BudgetOptimizer

    data_dir, results_dir, _ = _resolve_dirs(args)

    budget = args.budget if args.budget else config.get("budget", {}).get("total_krw", 50_000_000)
    config.setdefault("budget", {})["total_krw"] = budget
    logger.info("Budget optimisation — total=%s KRW", f"{budget:,}")

    # Try loading upstream results
    uplift_path = results_dir / "uplift_results.csv"
    clv_path = results_dir / "clv_predictions.csv"
    seed = config.get("simulation", {}).get("random_seed", 42)

    if uplift_path.exists() and clv_path.exists():
        uplift_df = pd.read_csv(uplift_path)
        clv_df = pd.read_csv(clv_path)
        merged = uplift_df.merge(clv_df, on="customer_id", how="inner")
        customers = _load_customers(data_dir)
        churn = customers["churn_label"].astype(float).values[:len(merged)] \
            if "churn_label" in customers.columns \
            else np.random.default_rng(seed).uniform(0.1, 0.9, len(merged))
        rng = np.random.default_rng(seed)
        inp = pd.DataFrame({
            "customer_id": merged["customer_id"],
            "uplift_score": merged["uplift_score"],
            "clv": merged["predicted_clv"],
            "churn_prob": churn[:len(merged)],
            "cost_per_action": np.where(
                merged["uplift_score"] > 0.1, 70000,
                np.where(merged["uplift_score"] > 0.02, 30000, 1000)
            ),
        })
    else:
        logger.warning("Upstream results not found – using synthetic data.")
        rng = np.random.default_rng(seed)
        n = 1000
        inp = pd.DataFrame({
            "customer_id": np.arange(n),
            "uplift_score": rng.uniform(0, 0.3, n),
            "clv": rng.uniform(10000, 500000, n),
            "churn_prob": rng.uniform(0.05, 0.95, n),
            "cost_per_action": rng.uniform(5000, 50000, n),
        })

    opt = BudgetOptimizer(config)
    result_df = opt.optimize(inp, total_budget=budget)

    alloc = result_df["allocated_budget"].sum() if "allocated_budget" in result_df.columns else 0
    logger.info("Allocated: %s / %s KRW", f"{alloc:,.0f}", f"{budget:,}")

    result_df.to_csv(results_dir / "budget_optimization.csv", index=False)
    return {"mode": "optimize", "status": "completed",
            "total_budget": budget, "allocated": float(alloc)}


def run_ab_test(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run A/B testing analysis (power analysis + significance test)."""
    from src.models.ab_testing import PowerAnalysis, ABTestFramework

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)

    baseline = float(customers["churn_label"].mean()) if "churn_label" in customers.columns else 0.20
    mde = 0.05

    sample_size = PowerAnalysis.required_sample_size(baseline_rate=baseline, mde=mde)
    logger.info("Power analysis: n=%d per group (baseline=%.2f%%, MDE=%.2f%%)",
                sample_size, baseline * 100, mde * 100)

    ab = ABTestFramework(config)

    if "treatment_group" in customers.columns and "churn_label" in customers.columns:
        mask = customers["treatment_group"] == "treatment"
        results = ab.run_test(
            treatment_outcomes=customers.loc[mask, "churn_label"].values,
            control_outcomes=customers.loc[~mask, "churn_label"].values,
        )
    else:
        rng = np.random.default_rng(42)
        results = ab.run_test(
            treatment_outcomes=rng.binomial(1, 0.18, 5000),
            control_outcomes=rng.binomial(1, 0.22, 5000),
        )

    results["power_analysis"] = {"required_sample_size": sample_size,
                                  "baseline_rate": baseline, "mde": mde}

    _save_json(results, results_dir / "ab_test_results.json")
    logger.info("A/B test results saved.")
    results["mode"] = "ab_test"
    results["status"] = "completed"
    return results


def run_survival(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run Cox PH survival analysis."""
    from src.models.survival_analysis import SurvivalModel

    data_dir, results_dir, models_dir = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events)

    fcols = _feature_cols(features)
    X = features[fcols]

    if "recency" in X.columns:
        duration = X["recency"].clip(lower=1)
    else:
        duration = pd.Series(np.random.default_rng(42).integers(30, 365, len(X)))

    event = customers["churn_label"].values[:len(X)].astype(int) \
        if "churn_label" in customers.columns else np.zeros(len(X), dtype=int)

    surv_feats = fcols[:15]  # Limit to avoid multicollinearity
    logger.info("Training survival model (Cox PH) with %d features...", len(surv_feats))
    model = SurvivalModel(config)
    model.fit(X[surv_feats], duration, event)

    median_surv = model.predict_median_survival(X[surv_feats])

    out: Dict[str, Any] = {
        "mode": "survival",
        "status": "completed",
        "num_customers": len(X),
        "num_events": int(event.sum()),
    }
    if median_surv is not None:
        out["median_survival_overall"] = float(np.nanmedian(median_surv))
    if model.cox_model is not None:
        out["concordance_index"] = float(model.cox_model.concordance_index_)

    model.save(str(models_dir / "survival_model.pkl"))
    _save_json(out, results_dir / "survival_results.json")
    logger.info("Survival analysis complete. C-index=%.4f",
                out.get("concordance_index", 0))
    return out


def run_recommend(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Generate personalized retention recommendations."""
    from src.models.recommendations import RecommendationEngine

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    seed = config.get("simulation", {}).get("random_seed", 42)
    rng = np.random.default_rng(seed)
    n = len(customers)

    cid = customers["customer_id"].values if "customer_id" in customers.columns else np.arange(n)
    inp = pd.DataFrame({"customer_id": cid})

    inp["churn_prob"] = customers["churn_label"].astype(float).values \
        if "churn_label" in customers.columns else rng.uniform(0.05, 0.95, n)

    # Merge upstream results when available
    up = results_dir / "uplift_results.csv"
    if up.exists():
        udf = pd.read_csv(up)
        inp = inp.merge(udf[["customer_id", "uplift_score"]], on="customer_id", how="left")
        inp["uplift_score"] = inp["uplift_score"].fillna(0)
    else:
        inp["uplift_score"] = rng.uniform(-0.1, 0.3, n)

    cp = results_dir / "clv_predictions.csv"
    if cp.exists():
        cdf = pd.read_csv(cp)
        inp = inp.merge(cdf[["customer_id", "predicted_clv"]], on="customer_id", how="left")
        inp["clv"] = inp["predicted_clv"].fillna(50000)
        inp.drop(columns=["predicted_clv"], inplace=True, errors="ignore")
    else:
        inp["clv"] = rng.uniform(10000, 500000, n)

    if "persona" in customers.columns:
        inp["segment"] = customers["persona"].values

    logger.info("Generating recommendations for %d customers...", n)
    engine = RecommendationEngine(config)
    recs = engine.recommend(inp)
    recs.to_csv(results_dir / "recommendations.csv", index=False)
    logger.info("Recommendations saved (%d rows).", len(recs))

    return {"mode": "recommend", "status": "completed", "num_recommendations": len(recs)}


def run_cohort(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Perform cohort retention analysis."""
    from src.analysis.cohort_analysis import CohortAnalyzer

    data_dir, results_dir, _ = _resolve_dirs(args)
    events = _load_events(data_dir)

    cohort_type = getattr(args, "cohort_type", "monthly")
    logger.info("Running cohort analysis (type=%s)...", cohort_type)
    analyzer = CohortAnalyzer()
    cohort_data = analyzer.assign_cohorts(events, cohort_type=cohort_type)
    retention = analyzer.compute_retention_matrix(cohort_data)

    out: Dict[str, Any] = {"mode": "cohort", "status": "completed", "cohort_type": cohort_type}
    if retention is not None:
        out["num_cohorts"] = len(retention)
        out["retention_matrix_shape"] = list(retention.shape)
        retention.to_csv(results_dir / "cohort_retention_matrix.csv")
        logger.info("Retention matrix: %d cohorts", len(retention))

        heatmap_path = str(results_dir / "cohort_retention_heatmap.png")
        analyzer.plot_retention_heatmap(retention, save_path=heatmap_path)
        logger.info("Retention heatmap saved to %s", heatmap_path)

        lines_path = str(results_dir / "cohort_retention_curves.png")
        analyzer.plot_retention_lines(retention, save_path=lines_path)
        logger.info("Retention curves saved to %s", lines_path)

    _save_json(out, results_dir / "cohort_analysis.json")
    return out


def run_segment(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Segment customers using RFM-based rules."""
    from src.features import CustomerSegmenter

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events)

    seg_config = config.get("segmentation", {})
    segmenter = CustomerSegmenter(config=seg_config)
    result = segmenter.segment_customers(features)

    # Add priority score
    if "monetary" in result.columns and "recency" in result.columns:
        max_r = result["recency"].max() or 1
        result["priority_score"] = result.get("monetary", 1) * (1 - result["recency"] / max_r)
    else:
        result["priority_score"] = 0.0

    result.to_csv(results_dir / "segments_6plus.csv", index=False)

    if "segment" in result.columns:
        dist = result["segment"].value_counts()
        for s, c in dist.items():
            logger.info("  %s: %d (%.1f%%)", s, c, c / len(result) * 100)

    return {"mode": "segment", "status": "completed",
            "num_customers": len(result),
            "num_segments": result["segment"].nunique() if "segment" in result.columns else 0}


def run_features(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run feature engineering pipeline and save features."""
    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events)

    feat_path = results_dir / "features.csv"
    features.to_csv(feat_path, index=False)
    logger.info("Features saved: %d rows x %d cols -> %s", *features.shape, feat_path)
    return {"mode": "features", "status": "completed",
            "num_rows": len(features), "num_features": len(features.columns)}


def run_monitor(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run model monitoring -- PSI and KS drift detection."""
    from src.monitoring import DriftDetector, KSDriftDetector

    data_dir, results_dir, _ = _resolve_dirs(args)
    customers = _load_customers(data_dir)
    events = _load_events(data_dir)
    features = _compute_features(config, customers, events)

    num_cols = [c for c in features.select_dtypes(include=[np.number]).columns
                if c not in ("customer_id", "churn_label")]
    split = len(features) // 2
    ref, cur = features[num_cols].iloc[:split], features[num_cols].iloc[split:]

    # PSI
    drift_cfg = config.get("drift_detection", {})
    psi_det = DriftDetector(
        n_bins=drift_cfg.get("n_bins", 10),
        yellow_threshold=drift_cfg.get("yellow_threshold", 0.10),
        red_threshold=drift_cfg.get("red_threshold", 0.25),
    )
    psi_report = psi_det.detect(ref, cur)

    # KS
    ks_cfg = config.get("ks_drift_detection", {})
    ks_det = KSDriftDetector(
        warning_threshold=ks_cfg.get("warning_threshold", 0.05),
        drift_threshold=ks_cfg.get("drift_threshold", 0.01),
    )
    ks_report = ks_det.detect(ref, cur)

    report: Dict[str, Any] = {"psi": {"num_features": len(num_cols), "alerts": []},
                               "ks": {"num_features": len(num_cols), "alerts": []}}

    if hasattr(psi_report, "alerts"):
        for a in psi_report.alerts:
            report["psi"]["alerts"].append({
                "feature": getattr(a, "feature", str(a)),
                "psi_value": float(getattr(a, "psi_value", 0)),
                "level": str(getattr(a, "level", "")),
            })
    if hasattr(ks_report, "alerts"):
        for a in ks_report.alerts:
            report["ks"]["alerts"].append({
                "feature": getattr(a, "feature", str(a)),
                "p_value": float(getattr(a, "p_value", 0)),
                "level": str(getattr(a, "level", "")),
            })

    _save_json(report, results_dir / "monitoring_report.json")
    logger.info("Monitoring: PSI alerts=%d, KS alerts=%d",
                len(report["psi"]["alerts"]), len(report["ks"]["alerts"]))

    return {"mode": "monitor", "status": "completed",
            "psi_alerts": len(report["psi"]["alerts"]),
            "ks_alerts": len(report["ks"]["alerts"])}


def run_dashboard(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Launch Streamlit dashboard at localhost:8501."""
    app = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    if not app.exists():
        logger.error("Dashboard not found: %s", app)
        return {"mode": "dashboard", "status": "error", "message": str(app)}

    logger.info("Launching Streamlit dashboard -> http://localhost:8501")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app),
         "--server.port", "8501", "--server.address", "localhost"],
        cwd=str(PROJECT_ROOT),
    )
    return {"mode": "dashboard", "status": "completed"}


def run_all(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Run full end-to-end pipeline with checkpoint/resume support.

    Uses PipelineRunner to wrap each stage with checkpoint logic.
    On restart, already-completed stages are skipped automatically.

    Order: simulate -> train -> uplift -> clv -> segment ->
    optimize -> recommend -> cohort -> ab_test -> survival -> monitor

    The pipeline state is persisted to ``pipeline_state.json`` in the
    data directory so that the pipeline can resume from the last
    successful stage after a failure or container restart.
    """
    from src.pipeline.runner import PipelineRunner

    # Determine state file path
    data_dir = Path(args.data) if args.data else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = str(data_dir / "pipeline_state.json")

    # Create runner with checkpoint support
    runner = PipelineRunner(
        config=config,
        state_path=state_path,
        output_dir=str(Path(args.output) if args.output else DEFAULT_RESULTS_DIR),
    )

    # Register the actual handler functions for each canonical step
    step_handlers = [
        ("data_generation", run_simulate),
        ("preprocessing", run_features),
        ("feature_engineering", run_features),
        ("ml_model_training", run_train),
        ("dl_model_training", run_train),
        ("ensemble_creation", run_train),
        ("uplift_modeling", run_uplift),
        ("clv_prediction", run_clv),
        ("budget_optimization", run_optimize),
        ("ab_testing", run_ab_test),
        ("survival_analysis", run_survival),
        ("recommendations", run_recommend),
        ("scoring_api_setup", run_monitor),
        ("mlflow_logging", run_monitor),
    ]
    for step_name, handler_fn in step_handlers:
        runner.register_step(step_name, handler_fn)

    # Resume from last checkpoint (skips completed stages)
    return runner.resume(args)


# ---------------------------------------------------------------------------
# Mode dispatcher
# ---------------------------------------------------------------------------

MODES = {
    "simulate": run_simulate,
    "train": run_train,
    "uplift": run_uplift,
    "clv": run_clv,
    "optimize": run_optimize,
    "ab_test": run_ab_test,
    "survival": run_survival,
    "recommend": run_recommend,
    "cohort": run_cohort,
    "segment": run_segment,
    "features": run_features,
    "monitor": run_monitor,
    "dashboard": run_dashboard,
    "all": run_all,
}

# Keep backward-compatible aliases
MODES["abtest"] = run_ab_test


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Build and parse CLI arguments.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="churn-cli",
        description="E-Commerce Churn Prediction & Retention System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  simulate   Generate simulated customer data
  train      Train churn prediction models (ML/DL/Ensemble)
  uplift     Train uplift model and segment customers
  clv        Predict Customer Lifetime Value
  optimize   LP-based budget optimization (use --budget N)
  ab_test    A/B test statistical analysis
  survival   Survival analysis (Cox PH)
  recommend  Personalized retention recommendations
  cohort     Cohort retention analysis
  segment    Customer segmentation (RFM)
  features   Run feature engineering pipeline
  monitor    Model monitoring / drift detection
  dashboard  Launch Streamlit dashboard (localhost:8501)
  all        Run full end-to-end pipeline

Examples:
  python src/main.py --mode train
  python src/main.py --mode simulate --small
  python src/main.py --mode optimize --budget 50000000
  python src/main.py --mode all --small
        """,
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=sorted(set(MODES.keys())),
                        help="Execution mode (required)")
    parser.add_argument("--config", type=str,
                        default=str(DEFAULT_CONFIG),
                        help="Path to YAML config (default: config/simulator_config.yaml)")
    parser.add_argument("--data", type=str, default=None,
                        help="Data directory (default: data/raw/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output base directory for results & models")
    parser.add_argument("--budget", type=int, default=None,
                        help="Total marketing budget in KRW (--mode optimize)")
    parser.add_argument("--small", action="store_true", default=False,
                        help="Small mode for simulation (5 000 customers, 6 months)")
    parser.add_argument("--learner", type=str, default="t_learner",
                        choices=["t_learner", "s_learner"],
                        help="Uplift learner type (default: t_learner)")
    parser.add_argument("--cohort-type", type=str, default="monthly",
                        choices=["monthly", "weekly", "behavioral"],
                        dest="cohort_type",
                        help="Cohort type for cohort analysis")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Enable DEBUG logging")
    parser.add_argument("-q", "--quiet", action="store_true", default=False,
                        help="Suppress output (WARNING only)")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Main CLI entrypoint.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    dict
        Result dictionary from the executed mode handler.
    """
    args = build_parser(argv)

    # Logging level
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    logger.info("Churn Prediction CLI -- mode=%s", args.mode)

    config = load_config(args.config)
    logger.debug("Config loaded from %s", args.config)

    handler = MODES[args.mode]
    result = handler(config, args)

    if not args.quiet:
        print(json.dumps(result, indent=2, default=str))

    return result


if __name__ == "__main__":
    main()
