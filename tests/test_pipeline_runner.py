"""Tests for PipelineRunner checkpoint/resume logic."""

import json
import argparse
import pytest
from unittest.mock import MagicMock

import pandas as pd

import src.main as main_mod
from src.pipeline.artifact_validation import (
    ArtifactValidationError,
    sync_and_validate_artifacts,
    validate_artifact_mirror,
    validate_cohort_artifacts,
    validate_generation_summary,
)
from src.pipeline.runner import PipelineRunner, PIPELINE_STEP_ORDER


@pytest.fixture
def runner(tmp_path, config_dict):
    """Create a PipelineRunner with temp state file."""
    state_path = str(tmp_path / "pipeline_state.json")
    return PipelineRunner(
        config=config_dict,
        state_path=state_path,
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def config_dict():
    """Minimal config dict for testing."""
    return {
        "simulation": {"random_seed": 42},
        "pipeline": {
            "train_months": 10,
            "test_months": 2,
            "ensemble_weight_ml": 0.6,
            "ensemble_weight_dl": 0.4,
        },
    }


class TestPipelineRunnerInit:
    """Test PipelineRunner initialization."""

    def test_instantiation(self, runner):
        """PipelineRunner can be instantiated."""
        assert runner is not None

    def test_has_step_order(self, runner):
        """PipelineRunner exposes step order."""
        steps = runner.get_step_order()
        assert isinstance(steps, list)
        assert len(steps) >= 13

    def test_step_order_starts_with_data_generation(self, runner):
        """First step must be data_generation."""
        assert runner.get_step_order()[0] == "data_generation"

    def test_step_order_ends_with_mlflow_logging(self, runner):
        """Last step must be mlflow_logging."""
        assert runner.get_step_order()[-1] == "mlflow_logging"


class TestRunStep:
    """Test run_step with checkpoint wrapping."""

    def test_run_step_creates_pending_then_completed(self, runner, tmp_path):
        """run_step saves pending checkpoint, then completed on success."""
        ok_handler = MagicMock(return_value={"status": "completed"})
        runner.register_step("test_step", ok_handler)

        result = runner.run_step("test_step")
        assert result == {"status": "completed"}

        state = runner.get_state()
        assert state["stages"]["test_step"]["status"] == "completed"
        assert "duration_seconds" in state["stages"]["test_step"]["metadata"]

    def test_run_step_marks_failed_on_error(self, runner):
        """run_step marks stage failed if handler raises."""
        fail_handler = MagicMock(side_effect=ValueError("boom"))
        runner.register_step("fail_step", fail_handler)

        with pytest.raises(RuntimeError, match="boom"):
            runner.run_step("fail_step")

        state = runner.get_state()
        assert state["stages"]["fail_step"]["status"] == "failed"
        assert "boom" in state["stages"]["fail_step"]["metadata"]["error"]

    def test_run_step_unknown_raises(self, runner):
        """run_step raises ValueError for unknown step."""
        with pytest.raises(ValueError, match="Unknown pipeline step"):
            runner.run_step("nonexistent_step")


class TestResume:
    """Test resume from checkpoint."""

    def test_resume_skips_completed(self, runner):
        """resume() skips stages marked completed."""
        handler1 = MagicMock(return_value={"status": "ok"})
        handler2 = MagicMock(return_value={"status": "ok"})

        # Override step order to just 2 steps
        runner._step_order = ["step_a", "step_b"]
        runner._step_handlers = {
            "step_a": handler1,
            "step_b": handler2,
        }

        # Mark step_a as already completed
        runner._state.mark_complete("step_a")

        result = runner.resume()
        assert result["status"] == "completed"

        # step_a handler should NOT have been called
        handler1.assert_not_called()
        # step_b handler SHOULD have been called
        handler2.assert_called_once()

    def test_resume_all_completed(self, runner):
        """resume() returns immediately when all stages are done."""
        runner._step_order = ["step_a", "step_b"]
        runner._step_handlers = {"step_a": MagicMock(), "step_b": MagicMock()}
        runner._state.mark_complete("step_a")
        runner._state.mark_complete("step_b")

        result = runner.resume()
        assert result["status"] == "completed"

    def test_resume_retries_failed(self, runner):
        """resume() retries stages that previously failed."""
        handler = MagicMock(return_value={"status": "ok"})
        runner._step_order = ["step_a"]
        runner._step_handlers = {"step_a": handler}
        runner._state.mark_failed("step_a", error="previous error")

        result = runner.resume()
        assert result["status"] == "completed"
        handler.assert_called_once()

    def test_resume_fail_fast_does_not_run_downstream_steps(self, runner):
        """resume() must stop immediately when a stage fails."""
        fail_handler = MagicMock(side_effect=ValueError("boom"))
        downstream_handler = MagicMock(return_value={"status": "ok"})
        runner._step_order = ["step_a", "step_b"]
        runner._step_handlers = {
            "step_a": fail_handler,
            "step_b": downstream_handler,
        }

        with pytest.raises(RuntimeError, match="boom"):
            runner.resume()

        downstream_handler.assert_not_called()
        state = runner.get_state()
        assert state["stages"]["step_a"]["status"] == "failed"
        assert "step_b" not in state["stages"]


class TestRunForce:
    """Test run with force reset."""

    def test_run_force_resets_state(self, runner):
        """run(force=True) clears all checkpoints and re-runs."""
        handler = MagicMock(return_value={"status": "ok"})
        runner._step_order = ["step_a"]
        runner._step_handlers = {"step_a": handler}
        runner._state.mark_complete("step_a")

        result = runner.run(force=True)
        assert result["status"] == "completed"
        # Should have re-run since force=True
        handler.assert_called_once()


class TestSaveGetState:
    """Test state persistence methods."""

    def test_get_state_empty_initially(self, runner):
        """get_state returns empty stages initially."""
        state = runner.get_state()
        assert state["stages"] == {}

    def test_save_state_persists(self, runner):
        """save_state writes to disk."""
        runner.save_state({
            "stages": {"test": {"status": "completed", "timestamp": "now"}}
        })
        state = runner.get_state()
        assert "test" in state["stages"]

    def test_save_state_no_args_creates_file(self, runner, tmp_path):
        """save_state() with no args ensures file exists."""
        runner.save_state()
        state = runner.get_state()
        assert "stages" in state


class TestSeedTracking:
    """Test that pipeline state tracks seed for reproducibility."""

    def test_resume_records_seed(self, runner):
        """resume() records the random seed in state."""
        runner._step_order = []
        runner._step_handlers = {}

        runner.resume()
        state = runner.get_state()
        assert state["seed"] == 42


class TestCanonicalStepOrder:
    """Test PIPELINE_STEP_ORDER constant."""

    def test_has_16_steps(self):
        """Canonical order has 16 steps."""
        assert len(PIPELINE_STEP_ORDER) == 16

    def test_first_is_data_generation(self):
        assert PIPELINE_STEP_ORDER[0] == "data_generation"

    def test_last_is_mlflow_logging(self):
        assert PIPELINE_STEP_ORDER[-1] == "mlflow_logging"

    def test_training_after_features(self):
        fe_idx = PIPELINE_STEP_ORDER.index("feature_engineering")
        ml_idx = PIPELINE_STEP_ORDER.index("ml_model_training")
        assert ml_idx > fe_idx

    def test_ensemble_after_dl(self):
        dl_idx = PIPELINE_STEP_ORDER.index("dl_model_training")
        ens_idx = PIPELINE_STEP_ORDER.index("ensemble_creation")
        assert ens_idx > dl_idx

    def test_segment_after_clv(self):
        clv_idx = PIPELINE_STEP_ORDER.index("clv_prediction")
        segment_idx = PIPELINE_STEP_ORDER.index("customer_segmentation")
        assert segment_idx > clv_idx

    def test_cohort_before_ab_testing(self):
        cohort_idx = PIPELINE_STEP_ORDER.index("cohort_analysis")
        ab_idx = PIPELINE_STEP_ORDER.index("ab_testing")
        assert ab_idx > cohort_idx


class TestPipelineArtifactValidation:
    """Test full/small evidence and dashboard artifact trust checks."""

    def test_generation_summary_rejects_small_mode(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "generation_summary.json").write_text(
            json.dumps({
                "generation_mode": "small",
                "num_customers": 5000,
                "treatment_count": 2500,
                "control_count": 2500,
                "churn_rate": 0.2,
                "validation": {
                    "group_size_check": {"passed": False},
                    "target_churn_check": {"passed": True},
                },
            }),
            encoding="utf-8",
        )

        result = validate_generation_summary(data_dir)

        assert result["valid"] is False
        assert result["evidence_size"] == "small"
        assert result["reason"] == "full_mode_generation_required"

    def test_generation_summary_accepts_full_mode(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "generation_summary.json").write_text(
            json.dumps({
                "generation_mode": "full",
                "num_customers": 20000,
                "treatment_count": 10000,
                "control_count": 10000,
                "churn_rate": 0.2,
                "validation": {
                    "group_size_check": {"passed": True},
                    "target_churn_check": {"passed": True},
                },
            }),
            encoding="utf-8",
        )

        result = validate_generation_summary(data_dir)

        assert result["valid"] is True
        assert result["evidence_size"] == "full"

    def test_sync_and_validate_artifacts_copies_and_checks_hash_schema_rows(
        self, tmp_path
    ):
        results_dir = tmp_path / "results"
        artifacts_dir = tmp_path / "artifacts"
        results_dir.mkdir()
        pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "uplift_score": [0.1, 0.2],
        }).to_csv(results_dir / "uplift_results.csv", index=False)

        summary = sync_and_validate_artifacts(
            results_dir,
            artifacts_dir,
            ["uplift_results.csv"],
            required_columns={"uplift_results.csv": ["customer_id", "uplift_score"]},
            expected_row_counts={"uplift_results.csv": 2},
            strict=True,
        )

        row = summary["artifacts"][0]
        assert summary["valid"] is True
        assert row["hash_match"] is True
        assert row["results_row_count"] == 2
        assert (artifacts_dir / "uplift_results.csv").exists()

    def test_artifact_mirror_rejects_stale_dashboard_copy(self, tmp_path):
        results_dir = tmp_path / "results"
        artifacts_dir = tmp_path / "artifacts"
        results_dir.mkdir()
        artifacts_dir.mkdir()
        pd.DataFrame({"customer_id": ["C1", "C2"]}).to_csv(
            results_dir / "recommendations.csv", index=False
        )
        pd.DataFrame({"customer_id": ["C1"]}).to_csv(
            artifacts_dir / "recommendations.csv", index=False
        )

        result = validate_artifact_mirror(
            results_dir / "recommendations.csv",
            artifacts_dir / "recommendations.csv",
            expected_row_count=2,
        )

        assert result["valid"] is False
        assert result["reason"] == "mirror_hash_mismatch"

    def test_required_checklist_refreshes_stale_mirror_before_hashing(
        self, tmp_path, monkeypatch
    ):
        results_dir = tmp_path / "results"
        artifacts_dir = tmp_path / "artifacts"
        data_dir = tmp_path / "data"
        results_dir.mkdir()
        artifacts_dir.mkdir()
        data_dir.mkdir()
        pd.DataFrame({"customer_id": ["C1", "C2"]}).to_csv(
            results_dir / "recommendations.csv", index=False
        )
        pd.DataFrame({"customer_id": ["stale"]}).to_csv(
            artifacts_dir / "recommendations.csv", index=False
        )

        monkeypatch.setattr(
            main_mod, "REQUIRED_PIPELINE_ARTIFACTS", ["recommendations.csv"]
        )
        monkeypatch.setattr(
            main_mod,
            "_validate_generation_summary",
            lambda _: {"valid": True, "reason": "ok"},
        )
        monkeypatch.setattr(
            main_mod,
            "_validate_required_artifact",
            lambda *_args, **_kwargs: {"valid": True},
        )

        checklist = main_mod._write_artifact_checklist(
            {"dashboard": {"artifacts_dir": str(artifacts_dir)}},
            results_dir,
            data_dir,
        )

        row = checklist["artifacts"][0]
        assert row["mirror_hash_match"] is True
        assert row["satisfied"] is True
        assert checklist["full_submission_ready"] is True
        assert pd.read_csv(artifacts_dir / "recommendations.csv")[
            "customer_id"
        ].tolist() == ["C1", "C2"]

    def test_run_all_fails_when_checklist_not_full_ready_even_without_missing(
        self, tmp_path, monkeypatch
    ):
        class FakeRunner:
            def __init__(self, *args, **kwargs):
                self._state = MagicMock()
                self._state.reset = MagicMock()
                self._state._save_state = MagicMock()

            def get_step_order(self):
                return ["data_generation"]

            def get_state(self):
                return {"stages": {}, "run_context": None}

            def register_step(self, *_args, **_kwargs):
                return None

            def resume(self, _args):
                return {"status": "completed"}

        args = argparse.Namespace(
            small=False,
            data=str(tmp_path / "data"),
            output=str(tmp_path / "results"),
        )
        monkeypatch.setattr("src.pipeline.runner.PipelineRunner", FakeRunner)
        monkeypatch.setattr(
            main_mod,
            "_write_artifact_checklist",
            lambda *_args, **_kwargs: {
                "full_submission_ready": False,
                "missing": [],
                "generation_summary_validation": {
                    "valid": False,
                    "reason": "full_mode_generation_required",
                },
                "artifacts": [],
            },
        )

        with pytest.raises(RuntimeError, match="Full submission checklist failed"):
            main_mod.run_all({"simulation": {"num_customers": 20_000}}, args)

    def test_cohort_artifacts_reject_errors_null_milestone_and_short_sequences(
        self, tmp_path
    ):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "cohort_analysis.json").write_text(
            json.dumps({
                "status": "failed",
                "errors": ["retention_matrix_requires_multiple_periods"],
                "churn_sequences_saved": True,
                "pre_churn_events_saved": True,
                "journey_funnel_saved": True,
            }),
            encoding="utf-8",
        )
        pd.DataFrame({"cohort": ["2024-01"], "0": [1.0]}).to_csv(
            results_dir / "cohort_retention_matrix.csv", index=False
        )
        pd.DataFrame({
            "cohort": ["2024-01"],
            "M1": [None],
            "M3": [None],
            "M6": [None],
            "M12": [None],
        }).to_csv(results_dir / "cohort_milestones.csv", index=False)
        (results_dir / "churn_last30_sequences.json").write_text(
            json.dumps([["search -> purchase", 1]]),
            encoding="utf-8",
        )
        pd.DataFrame({"event_type": ["cs_contact"]}).to_csv(
            results_dir / "pre_churn_events.csv", index=False
        )
        pd.DataFrame({"stage": ["Signup"]}).to_csv(
            results_dir / "journey_funnel.csv", index=False
        )

        result = validate_cohort_artifacts(results_dir)

        assert result["valid"] is False
        assert "cohort_analysis_failed" in result["errors"]
        assert "retention_matrix_requires_multiple_periods" in result["errors"]
        assert "null_milestone_M1" in result["errors"]
        assert "churn_sequences_requires_top5_patterns" in result["errors"]
        with pytest.raises(ArtifactValidationError):
            validate_cohort_artifacts(results_dir, strict=True)

    def test_cohort_validation_rejects_fallback_milestones_and_weak_sequences(
        self, tmp_path
    ):
        data_dir = tmp_path / "data"
        results_dir = tmp_path / "results"
        data_dir.mkdir()
        results_dir.mkdir()
        pd.DataFrame({"customer_id": [f"C{i}" for i in range(10)]}).to_csv(
            data_dir / "customers.csv", index=False
        )
        (results_dir / "cohort_analysis.json").write_text(
            json.dumps({
                "status": "completed",
                "errors": [],
                "retention_matrix_shape": [1, 5],
                "exact_milestones": [1, 3],
                "fallback_milestones": ["M6", "M12"],
                "churn_sequences_saved": True,
                "pre_churn_events_saved": True,
                "journey_funnel_saved": True,
            }),
            encoding="utf-8",
        )
        pd.DataFrame({"cohort": ["2024-01"], "0": [1.0], "1": [0.8]}).to_csv(
            results_dir / "cohort_retention_matrix.csv", index=False
        )
        pd.DataFrame({
            "cohort": ["2024-01"],
            "M1": [0.8],
            "M3": [0.7],
            "M6": [0.7],
            "M12": [0.7],
        }).to_csv(results_dir / "cohort_milestones.csv", index=False)
        (results_dir / "churn_last30_sequences.json").write_text(
            json.dumps([
                ["a", 1], ["b", 1], ["c", 1], ["d", 1], ["e", 1],
            ]),
            encoding="utf-8",
        )
        pd.DataFrame({
            "event_type": ["cs_contact"],
            "churned_freq": [1.0],
            "active_freq": [0.2],
            "freq_ratio": [5.0],
        }).to_csv(results_dir / "pre_churn_events.csv", index=False)
        pd.DataFrame({
            "stage": [
                "Signup", "First Purchase", "Repeat Purchase", "Loyal", "Churned",
            ],
            "count": [9, 8, 5, 2, 1],
            "conversion_rate": [1.0, 0.8, 0.5, 0.2, 0.1],
            "drop_off_rate": [0.0, 0.2, 0.3, 0.3, 0.1],
        }).to_csv(results_dir / "journey_funnel.csv", index=False)

        result = validate_cohort_artifacts(results_dir, data_dir=data_dir)

        assert result["valid"] is False
        assert "missing_exact_retention_milestones: M6,M12" in result["errors"]
        assert (
            "fallback_retention_milestones_not_submission_evidence: M6,M12"
            in result["errors"]
        )
        assert "churn_sequences_observations_too_small: 5" in result["errors"]
        assert "journey_signup_count_mismatch: 9_expected_10" in result["errors"]

    def test_failed_cohort_run_invalidates_stale_required_checklist(
        self, tmp_path, monkeypatch
    ):
        data_dir = tmp_path / "data" / "raw"
        output_dir = tmp_path / "output"
        results_dir = output_dir / "results"
        artifacts_dir = tmp_path / "artifacts"
        results_dir.mkdir(parents=True)
        stale = {"full_submission_ready": True, "stale": True}
        (results_dir / "required_artifacts_checklist.json").write_text(
            json.dumps(stale),
            encoding="utf-8",
        )

        customers = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "churn_label": [1, 0],
        })
        events = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "event_type": ["purchase", "page_view"],
            "amount": [100.0, 0.0],
            "revenue": [100.0, 0.0],
        })
        args = argparse.Namespace(
            data=str(data_dir),
            output=str(output_dir),
            cohort_type="monthly",
        )
        config = {"dashboard": {"artifacts_dir": str(artifacts_dir)}}
        monkeypatch.setattr(main_mod, "_load_customers", lambda _data_dir: customers)
        monkeypatch.setattr(main_mod, "_load_events", lambda _data_dir: events)

        with pytest.raises(RuntimeError, match="Cohort analysis required outputs failed"):
            main_mod.run_cohort(config, args)

        checklist = json.loads(
            (results_dir / "required_artifacts_checklist.json").read_text(
                encoding="utf-8"
            )
        )
        assert checklist["full_submission_ready"] is False
        assert "cohort_analysis.json" in checklist["missing"]
        cohort_row = next(
            row for row in checklist["artifacts"]
            if row["artifact"] == "cohort_analysis.json"
        )
        assert cohort_row["validation"]["reason"] == "invalid_cohort_artifacts"
        assert (artifacts_dir / "required_artifacts_checklist.json").exists()
