#!/usr/bin/env python3
"""Summarize repeated optimized library/reference benchmark processes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any


MINIMUM_PROCESSES = 5
MATERIAL_PERCENT = 10.0
MATERIAL_ABSOLUTE_NS = 1_000.0
CONFIDENCE_EXCLUSION_PERCENT = 5.0


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def paired_interval(values: list[float]) -> dict[str, float]:
    generator = random.Random(0x6840158760032202)
    bootstrap = [
        median([values[generator.randrange(len(values))] for _ in values])
        for _ in range(2_000)
    ]
    bootstrap.sort()
    return {
        "confidence_level": 0.95,
        "lower_bound_percent": bootstrap[49],
        "upper_bound_percent": bootstrap[1949],
    }


def classify(
    median_percent: float,
    median_delta_ns: float,
    lower_percent: float,
    upper_percent: float,
) -> str:
    if (
        median_percent > MATERIAL_PERCENT
        and median_delta_ns > MATERIAL_ABSOLUTE_NS
        and lower_percent > CONFIDENCE_EXCLUSION_PERCENT
    ):
        return "reference_gap"
    if (
        median_percent < -MATERIAL_PERCENT
        and median_delta_ns < -MATERIAL_ABSOLUTE_NS
        and upper_percent < -CONFIDENCE_EXCLUSION_PERCENT
    ):
        return "library_faster"
    return "parity"


def load_document(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 2:
        raise ValueError(f"{path} is not a schema v2 benchmark document")
    run = document.get("run")
    records = document.get("records")
    if not isinstance(run, dict) or not isinstance(records, list):
        raise ValueError(f"{path} has an invalid benchmark envelope")
    if run.get("build_profile") != "release":
        raise ValueError(f"{path} was not measured with the release profile")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{path} contains a non-object record")
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in indexed:
            raise ValueError(f"{path} contains an invalid or duplicate scenario id")
        for estimate_name in ("library", "reference"):
            estimate = record.get(estimate_name)
            value = estimate.get("median_ns") if isinstance(estimate, dict) else None
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise ValueError(f"{path} {scenario_id} has an invalid {estimate_name} median")
        indexed[scenario_id] = record
    return run, indexed


def summarize_scenario(scenario_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    contract_fields = ("workload", "sample_count", "sampling_order_policy")
    for field in contract_fields:
        if any(record.get(field) != records[0].get(field) for record in records[1:]):
            raise ValueError(f"{scenario_id} changes {field} between processes")
    if records[0].get("sampling_order_policy") != "alternating_library_then_reference":
        raise ValueError(f"{scenario_id} does not use deterministic alternating order")
    library = [float(record["library"]["median_ns"]) for record in records]
    reference = [float(record["reference"]["median_ns"]) for record in records]
    deltas = [library_ns - reference_ns for library_ns, reference_ns in zip(library, reference)]
    percentages = [
        100.0 * delta_ns / reference_ns
        for delta_ns, reference_ns in zip(deltas, reference)
    ]
    median_delta = median(deltas)
    median_percent = median(percentages)
    interval = paired_interval(percentages)
    return {
        "scenario_id": scenario_id,
        "status": classify(
            median_percent,
            median_delta,
            interval["lower_bound_percent"],
            interval["upper_bound_percent"],
        ),
        "workload": records[0]["workload"],
        "samples_per_process": records[0]["sample_count"],
        "library_process_median_ns": median(library),
        "reference_process_median_ns": median(reference),
        "median_delta_ns": median_delta,
        "median_percent": median_percent,
        "percent_interval": interval,
    }


def summarize_files(paths: list[Path]) -> dict[str, Any]:
    if len(paths) < MINIMUM_PROCESSES:
        raise ValueError("at least five independent process documents are required")
    loaded = [load_document(path) for path in paths]
    runs = [run for run, _ in loaded]
    if any(run != runs[0] for run in runs[1:]):
        raise ValueError("run identity changes between process documents")
    record_sets = [records for _, records in loaded]
    scenario_ids = set(record_sets[0])
    if not scenario_ids or any(set(records) != scenario_ids for records in record_sets[1:]):
        raise ValueError("scenario sets change between process documents")
    scenarios = [
        summarize_scenario(scenario_id, [records[scenario_id] for records in record_sets])
        for scenario_id in sorted(scenario_ids)
    ]
    statuses = ("reference_gap", "library_faster", "parity")
    return {
        "reference_gap_schema_version": 1,
        "process_count": len(paths),
        "run": runs[0],
        "thresholds": {
            "median_percent": MATERIAL_PERCENT,
            "median_absolute_ns": MATERIAL_ABSOLUTE_NS,
            "confidence_excludes_percent": CONFIDENCE_EXCLUSION_PERCENT,
        },
        "summary": {
            status: sum(scenario["status"] == status for scenario in scenarios)
            for status in statuses
        },
        "scenarios": scenarios,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = summarize_files(args.inputs)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())
