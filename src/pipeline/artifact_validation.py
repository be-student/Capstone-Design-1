"""Pipeline artifact validation and results/dashboard mirror checks."""

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


FULL_CUSTOMER_MINIMUM = 20_000
FULL_GROUP_MINIMUM = 10_000
CHURN_RATE_MINIMUM = 0.15
CHURN_RATE_MAXIMUM = 0.25
MIN_CHURN_SEQUENCE_OBSERVATIONS = 6
REQUIRED_RETENTION_MILESTONES = ("M1", "M3", "M6", "M12")


class ArtifactValidationError(RuntimeError):
    """Raised when required pipeline artifact evidence is invalid."""


def file_sha256(path: Path) -> Optional[str]:
    """Return a SHA-256 digest for an existing file."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generation_summary(data_dir: Path) -> Dict[str, Any]:
    """Validate simulator evidence and explicitly distinguish full vs small."""
    path = data_dir / "generation_summary.json"
    if not path.exists():
        return {"valid": False, "reason": "missing_generation_summary"}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validation = payload.get("validation", {}) or {}
        group_check = validation.get("group_size_check", {}) or {}
        churn_check = validation.get("target_churn_check", {}) or {}
        mode = payload.get("generation_mode", validation.get("mode", "unknown"))
        num_customers = int(payload.get("num_customers", 0))
        treatment_count = int(payload.get("treatment_count", 0))
        control_count = int(payload.get("control_count", 0))
        churn_rate = float(payload.get("churn_rate", 0.0))
        group_size_passed = bool(group_check.get("passed", False))
        target_churn_passed = bool(
            churn_check.get(
                "passed", CHURN_RATE_MINIMUM <= churn_rate <= CHURN_RATE_MAXIMUM
            )
        )
    except Exception as exc:
        return {"valid": False, "reason": f"validation_error: {exc}"}

    is_small = str(mode).lower() == "small"
    full_thresholds_met = (
        num_customers >= FULL_CUSTOMER_MINIMUM
        and treatment_count >= FULL_GROUP_MINIMUM
        and control_count >= FULL_GROUP_MINIMUM
    )
    churn_rate_valid = CHURN_RATE_MINIMUM <= churn_rate <= CHURN_RATE_MAXIMUM
    valid = (
        not is_small
        and full_thresholds_met
        and churn_rate_valid
        and group_size_passed
        and target_churn_passed
    )
    return {
        "valid": valid,
        "mode": mode,
        "evidence_size": "small" if is_small else "full",
        "num_customers": num_customers,
        "treatment_count": treatment_count,
        "control_count": control_count,
        "churn_rate": churn_rate,
        "group_size_passed": group_size_passed,
        "target_churn_passed": target_churn_passed,
        "full_thresholds_met": full_thresholds_met,
        "reason": "ok" if valid else "full_mode_generation_required",
    }


def _read_csv_shape(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = sum(1 for _ in reader)
    return {"columns": fieldnames, "row_count": rows}


def validate_artifact_mirror(
    results_path: Path,
    artifact_path: Path,
    required_columns: Optional[Iterable[str]] = None,
    expected_row_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate hash, schema, and row-count parity for a mirrored artifact."""
    results_hash = file_sha256(results_path)
    artifact_hash = file_sha256(artifact_path)
    result: Dict[str, Any] = {
        "valid": False,
        "results_path": str(results_path),
        "artifact_path": str(artifact_path),
        "results_exists": results_path.exists(),
        "artifact_exists": artifact_path.exists(),
        "results_sha256": results_hash,
        "artifact_sha256": artifact_hash,
        "hash_match": results_hash is not None and results_hash == artifact_hash,
    }
    if not results_path.exists() or not artifact_path.exists():
        result["reason"] = "missing_mirror_file"
        return result
    if results_hash != artifact_hash:
        result["reason"] = "mirror_hash_mismatch"
        return result

    if results_path.suffix == ".csv":
        results_shape = _read_csv_shape(results_path)
        artifact_shape = _read_csv_shape(artifact_path)
        result["results_columns"] = results_shape["columns"]
        result["artifact_columns"] = artifact_shape["columns"]
        result["results_row_count"] = results_shape["row_count"]
        result["artifact_row_count"] = artifact_shape["row_count"]
        if results_shape != artifact_shape:
            result["reason"] = "mirror_schema_or_row_count_mismatch"
            return result
        missing = set(required_columns or []) - set(results_shape["columns"])
        if missing:
            result["reason"] = "missing_required_columns"
            result["missing_columns"] = sorted(missing)
            return result
        if (
            expected_row_count is not None
            and results_shape["row_count"] != expected_row_count
        ):
            result["reason"] = "unexpected_row_count"
            result["expected_row_count"] = expected_row_count
            return result

    result["valid"] = True
    result["reason"] = "ok"
    return result


def sync_and_validate_artifacts(
    results_dir: Path,
    artifacts_dir: Path,
    artifact_names: Sequence[str],
    required_columns: Optional[Dict[str, Iterable[str]]] = None,
    expected_row_counts: Optional[Dict[str, int]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Copy result artifacts to dashboard artifacts and validate parity."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in artifact_names:
        source = results_dir / name
        target = artifacts_dir / name
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        rows.append(
            validate_artifact_mirror(
                source,
                target,
                required_columns=(required_columns or {}).get(name),
                expected_row_count=(expected_row_counts or {}).get(name),
            )
        )

    invalid = [row for row in rows if not row["valid"]]
    summary = {
        "valid": not invalid,
        "checked_count": len(rows),
        "invalid_count": len(invalid),
        "invalid_artifacts": [Path(row["results_path"]).name for row in invalid],
        "artifacts": rows,
    }
    if strict and invalid:
        raise ArtifactValidationError(
            "Invalid mirrored artifacts: " + ", ".join(summary["invalid_artifacts"])
        )
    return summary


def _load_customer_count(data_dir: Optional[Path]) -> Optional[int]:
    if data_dir is None:
        return None
    customers_path = data_dir / "customers.csv"
    if not customers_path.exists():
        return None
    try:
        return _read_csv_shape(customers_path)["row_count"]
    except Exception:
        return None


def _sequence_observation_count(sequences: Any) -> int:
    total = 0
    if isinstance(sequences, dict):
        iterable = sequences.values()
    else:
        iterable = sequences if isinstance(sequences, list) else []
    for item in iterable:
        if isinstance(item, dict):
            count = item.get("count", item.get("frequency", 0))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            count = item[1]
        else:
            count = 0
        try:
            total += int(count)
        except (TypeError, ValueError):
            continue
    return total


def validate_cohort_artifacts(
    results_dir: Path,
    data_dir: Optional[Path] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate cohort and journey artifacts required by the submission."""
    errors: List[str] = []
    analysis_path = results_dir / "cohort_analysis.json"
    retention_path = results_dir / "cohort_retention_matrix.csv"
    milestones_path = results_dir / "cohort_milestones.csv"
    sequences_path = results_dir / "churn_last30_sequences.json"
    pre_churn_path = results_dir / "pre_churn_events.csv"
    journey_path = results_dir / "journey_funnel.csv"

    if not analysis_path.exists():
        errors.append("missing_cohort_analysis_json")
        payload: Dict[str, Any] = {}
    else:
        try:
            payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {}
            errors.append(f"cohort_analysis_unreadable: {exc}")
    if payload.get("status") == "failed":
        errors.append("cohort_analysis_failed")
    errors.extend(str(err) for err in payload.get("errors", []) or [])
    errors.extend(key for key in payload if key.endswith("_error"))
    exact_milestones = {int(value) for value in payload.get("exact_milestones", []) or []}
    missing_exact = [period for period in (1, 3, 6, 12) if period not in exact_milestones]
    if missing_exact:
        errors.append(
            "missing_exact_retention_milestones: "
            + ",".join(f"M{period}" for period in missing_exact)
        )
    fallback_milestones = payload.get("fallback_milestones", []) or []
    if fallback_milestones:
        errors.append(
            "fallback_retention_milestones_not_submission_evidence: "
            + ",".join(str(value) for value in fallback_milestones)
        )
    for flag in (
        "churn_sequences_saved",
        "pre_churn_events_saved",
        "journey_funnel_saved",
    ):
        if payload and not payload.get(flag):
            errors.append(f"missing_flag_{flag}")

    if not retention_path.exists():
        errors.append("missing_retention_matrix")
    else:
        shape = _read_csv_shape(retention_path)
        period_columns = [col for col in shape["columns"] if col != "cohort"]
        if len(period_columns) < 2:
            errors.append("retention_matrix_requires_multiple_periods")

    if not milestones_path.exists():
        errors.append("missing_cohort_milestones")
    else:
        with milestones_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = set(reader.fieldnames or [])
        for column in REQUIRED_RETENTION_MILESTONES:
            if column not in columns:
                errors.append(f"missing_milestone_{column}")
            elif rows and all(row.get(column, "") in ("", "nan", "NaN") for row in rows):
                errors.append(f"null_milestone_{column}")

    if not sequences_path.exists():
        errors.append("missing_churn_last30_sequences")
    else:
        try:
            sequences = json.loads(sequences_path.read_text(encoding="utf-8"))
            if len(sequences) < 5:
                errors.append("churn_sequences_requires_top5_patterns")
            observation_count = _sequence_observation_count(sequences)
            if observation_count < MIN_CHURN_SEQUENCE_OBSERVATIONS:
                errors.append(
                    "churn_sequences_observations_too_small: "
                    f"{observation_count}"
                )
        except Exception as exc:
            errors.append(f"churn_sequences_unreadable: {exc}")

    if not pre_churn_path.exists():
        errors.append("missing_pre_churn_events")
    elif _read_csv_shape(pre_churn_path)["row_count"] == 0:
        errors.append("empty_pre_churn_events")

    expected_customers = _load_customer_count(data_dir)
    if not journey_path.exists():
        errors.append("missing_journey_funnel")
    elif _read_csv_shape(journey_path)["row_count"] == 0:
        errors.append("empty_journey_funnel")
    else:
        with journey_path.open("r", encoding="utf-8", newline="") as f:
            journey_rows = list(csv.DictReader(f))
        signup_rows = [row for row in journey_rows if row.get("stage") == "Signup"]
        if not signup_rows:
            errors.append("journey_funnel_missing_signup_stage")
        elif expected_customers is not None:
            try:
                signup_count = int(float(signup_rows[0].get("count", "")))
            except (TypeError, ValueError):
                signup_count = -1
            if signup_count != expected_customers:
                errors.append(
                    "journey_signup_count_mismatch: "
                    f"{signup_count}_expected_{expected_customers}"
                )

    result = {"valid": not errors, "errors": errors}
    if expected_customers is not None:
        result["expected_customer_count"] = expected_customers
    if strict and errors:
        raise ArtifactValidationError(
            "Invalid cohort artifacts: " + "; ".join(errors)
        )
    return result
