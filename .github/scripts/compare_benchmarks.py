#!/usr/bin/env python3
"""Compare paired benchmark-process JSON as advisory-only evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
from typing import Any


COMPARISON_SCHEMA_VERSION = 1
MINIMUM_PROCESS_PAIRS = 5
REQUIRED_REPEATED_OBSERVATIONS = 3
ADVISORY_PERCENT = 10.0
ADVISORY_ABSOLUTE_NS = 1_000.0
CONFIDENCE_EXCLUSION_PERCENT = 5.0
FINGERPRINT_FIELDS = (
    "rust_version",
    "candle_version",
    "os",
    "architecture",
    "backend",
    "device",
    "driver",
)


class ComparisonInputError(ValueError):
    """The comparison input is malformed rather than merely incomparable."""


def _validate_sha(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ComparisonInputError(f"{name} must be an exact 40-character Git SHA")


def _load_document(path: Path, expected_sha: str, side: str) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonInputError(f"could not read {path}: {error}") from error
    if not isinstance(document, list):
        raise ComparisonInputError(f"{path} must contain a JSON array of benchmark records")
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(document):
        if not isinstance(record, dict):
            raise ComparisonInputError(f"{path} record {index} must be an object")
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ComparisonInputError(f"{path} record {index} has no scenario_id")
        if scenario_id in records:
            raise ComparisonInputError(f"{path} repeats scenario_id {scenario_id}")
        fingerprint = record.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise ComparisonInputError(f"{path} {scenario_id} has no fingerprint object")
        actual_sha = fingerprint.get("git_sha")
        if actual_sha != expected_sha:
            raise ComparisonInputError(
                f"{path} {scenario_id} does not contain the expected {side} SHA {expected_sha}"
            )
        workload = record.get("workload")
        estimate = record.get("library")
        if not isinstance(workload, dict) or not isinstance(estimate, dict):
            raise ComparisonInputError(f"{path} {scenario_id} lacks workload or library estimate")
        median_ns = estimate.get("median_ns")
        if (
            not isinstance(median_ns, (int, float))
            or isinstance(median_ns, bool)
            or not math.isfinite(float(median_ns))
            or float(median_ns) <= 0
        ):
            raise ComparisonInputError(f"{path} {scenario_id} has an invalid library median")
        records[scenario_id] = record
    return records


def _load_runs(paths: list[Path], expected_sha: str, side: str) -> list[dict[str, dict[str, Any]]]:
    return [_load_document(path, expected_sha, side) for path in paths]


def _constant(records: list[dict[str, Any]], key: str) -> Any | None:
    values = [record.get(key) for record in records]
    return values[0] if all(value == values[0] for value in values) else None


def _fingerprint_key(record: dict[str, Any]) -> tuple[Any, ...] | None:
    fingerprint = record.get("fingerprint")
    if not isinstance(fingerprint, dict) or any(field not in fingerprint for field in FINGERPRINT_FIELDS):
        return None
    return tuple(fingerprint[field] for field in FINGERPRINT_FIELDS)


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _paired_interval(values: list[float]) -> dict[str, float]:
    generator = random.Random(0x6840158760032202)
    bootstrap = []
    for _ in range(2_000):
        bootstrap.append(_median([values[generator.randrange(len(values))] for _ in values]))
    bootstrap.sort()
    return {
        "confidence_level": 0.95,
        "lower_bound_percent": bootstrap[49],
        "upper_bound_percent": bootstrap[1949],
    }


def classify_movement(
    median_percent: float,
    median_delta_ns: float,
    lower_percent: float,
    upper_percent: float,
) -> str:
    """Apply the advisory threshold without ever creating a gating result."""
    if (
        median_percent > ADVISORY_PERCENT
        and median_delta_ns > ADVISORY_ABSOLUTE_NS
        and lower_percent > CONFIDENCE_EXCLUSION_PERCENT
    ):
        return "advisory_regression"
    if (
        median_percent < -ADVISORY_PERCENT
        and median_delta_ns < -ADVISORY_ABSOLUTE_NS
        and upper_percent < -CONFIDENCE_EXCLUSION_PERCENT
    ):
        return "advisory_improvement"
    return "no_advisory"


def _incomparable(scenario_id: str, reason: str) -> dict[str, Any]:
    return {"scenario_id": scenario_id, "status": "incomparable", "reason": reason}


def _compare_scenario(
    scenario_id: str,
    base_records: list[dict[str, Any] | None],
    head_records: list[dict[str, Any] | None],
) -> dict[str, Any]:
    if all(record is None for record in base_records):
        return _incomparable(scenario_id, "missing_from_base")
    if all(record is None for record in head_records):
        return _incomparable(scenario_id, "missing_from_head")
    if any(record is None for record in base_records + head_records):
        return _incomparable(scenario_id, "missing_process_sample")
    base = [record for record in base_records if record is not None]
    head = [record for record in head_records if record is not None]

    base_schema = _constant(base, "schema_version")
    head_schema = _constant(head, "schema_version")
    if base_schema is None or head_schema is None or base_schema != head_schema:
        return _incomparable(scenario_id, "schema_mismatch")
    if base_schema != 1:
        return _incomparable(scenario_id, "unsupported_schema")

    base_workload = _constant(base, "workload")
    head_workload = _constant(head, "workload")
    if base_workload is None or head_workload is None or base_workload != head_workload:
        return _incomparable(scenario_id, "workload_mismatch")

    base_order = _constant(base, "sampling_order_policy")
    head_order = _constant(head, "sampling_order_policy")
    if base_order is None or head_order is None or base_order != head_order:
        return _incomparable(scenario_id, "sampling_policy_mismatch")

    base_fingerprints = [_fingerprint_key(record) for record in base]
    head_fingerprints = [_fingerprint_key(record) for record in head]
    if (
        None in base_fingerprints
        or None in head_fingerprints
        or len(set(base_fingerprints)) != 1
        or len(set(head_fingerprints)) != 1
        or base_fingerprints[0] != head_fingerprints[0]
    ):
        return _incomparable(scenario_id, "fingerprint_mismatch")

    base_medians = [float(record["library"]["median_ns"]) for record in base]
    head_medians = [float(record["library"]["median_ns"]) for record in head]
    deltas = [head_value - base_value for base_value, head_value in zip(base_medians, head_medians)]
    percentages = [100.0 * delta / base_value for base_value, delta in zip(base_medians, deltas)]
    median_delta = _median(deltas)
    median_percent = _median(percentages)
    interval = _paired_interval(percentages)
    status = classify_movement(
        median_percent,
        median_delta,
        interval["lower_bound_percent"],
        interval["upper_bound_percent"],
    )
    return {
        "scenario_id": scenario_id,
        "status": status,
        "workload": base_workload,
        "base_process_median_ns": _median(base_medians),
        "head_process_median_ns": _median(head_medians),
        "paired_median_delta_ns": median_delta,
        "paired_median_percent": median_percent,
        "paired_percent_confidence_interval": interval,
        "required_repeated_observations": REQUIRED_REPEATED_OBSERVATIONS,
        "automatic_action": False,
    }


def compare_files(
    base_paths: list[Path],
    head_paths: list[Path],
    base_sha: str,
    head_sha: str,
) -> dict[str, Any]:
    _validate_sha(base_sha, "base_sha")
    _validate_sha(head_sha, "head_sha")
    if len(base_paths) < MINIMUM_PROCESS_PAIRS or len(head_paths) < MINIMUM_PROCESS_PAIRS:
        raise ComparisonInputError("at least five base/head process pairs are required")
    if len(base_paths) != len(head_paths):
        raise ComparisonInputError("base and head process counts must match")
    base_runs = _load_runs(base_paths, base_sha, "base")
    head_runs = _load_runs(head_paths, head_sha, "head")
    scenario_ids = sorted(
        set().union(*(run.keys() for run in base_runs), *(run.keys() for run in head_runs))
    )
    scenarios = [
        _compare_scenario(
            scenario_id,
            [run.get(scenario_id) for run in base_runs],
            [run.get(scenario_id) for run in head_runs],
        )
        for scenario_id in scenario_ids
    ]
    counts = {
        status: sum(item["status"] == status for item in scenarios)
        for status in [
            "advisory_regression",
            "advisory_improvement",
            "no_advisory",
            "incomparable",
        ]
    }
    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "advisory_only": True,
        "can_fail_required_ci": False,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "process_pairs": len(base_paths),
        "thresholds": {
            "median_percent": ADVISORY_PERCENT,
            "median_absolute_ns": ADVISORY_ABSOLUTE_NS,
            "confidence_excludes_percent": CONFIDENCE_EXCLUSION_PERCENT,
        },
        "summary": counts,
        "scenarios": scenarios,
    }


def markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "## Advisory-only performance comparison",
        "",
        "This report cannot fail required CI, block a release, or trigger an automatic action.",
        "A regression requires the same observation in three independent workflow runs before filing.",
        "",
        "| Scenario | Status | Paired median | Absolute movement | 95% paired interval |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for item in report["scenarios"]:
        if item["status"] == "incomparable":
            lines.append(
                f"| `{item['scenario_id']}` | incomparable: {item['reason']} | — | — | — |"
            )
            continue
        interval = item["paired_percent_confidence_interval"]
        lines.append(
            "| `{scenario}` | {status} | {percent:+.2f}% | {delta:+.0f} ns | "
            "[{lower:+.2f}%, {upper:+.2f}%] |".format(
                scenario=item["scenario_id"],
                status=item["status"],
                percent=item["paired_median_percent"],
                delta=item["paired_median_delta_ns"],
                lower=interval["lower_bound_percent"],
                upper=interval["upper_bound_percent"],
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--base", type=Path, nargs="+", required=True)
    parser.add_argument("--head", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = compare_files(args.base, args.head, args.base_sha, args.head_sha)
    except ComparisonInputError as error:
        print(f"benchmark comparison input error: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = markdown_summary(report)
    print(summary, end="")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
