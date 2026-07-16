#!/usr/bin/env python3
"""Generate the committed performance report, data, and SVG figures."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import sys
from collections import Counter
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/candle-einops-matplotlib")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "candle-einops-performance-v1"
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "benchmarks/data/performance-2026-07-16.json"
REPORT_PATH = ROOT / "docs/performance.md"
OUTCOMES_PATH = ROOT / "docs/figures/performance-outcomes.svg"
HEATMAP_PATH = ROOT / "docs/figures/performance-heatmap.svg"

STATUS_ORDER = ["library_faster", "parity", "reference_gap", "skipped"]
STATUS_LABELS = {
    "library_faster": "Win",
    "parity": "Tie",
    "reference_gap": "Loss",
    "skipped": "Skipped",
}
STATUS_CODES = {
    "library_faster": "W",
    "parity": "T",
    "reference_gap": "L",
}
STATUS_COLORS = {
    "library_faster": "#238636",
    "parity": "#8c959f",
    "reference_gap": "#cf222e",
    "skipped": "#d8dee4",
}

PROVIDER_SPECS = [
    ("cpu-baseline", "CPU baseline"),
    ("cpu-accelerate", "CPU Accelerate"),
    ("metal", "Metal"),
    ("cuda", "CUDA"),
]


def scenario_ids(data: dict[str, Any]) -> list[str]:
    return sorted(
        {
            scenario
            for provider in data["providers"]
            for scenario in provider["scenarios"]
        }
    )


def outcome_counts(data: dict[str, Any], provider: dict[str, Any]) -> Counter[str]:
    counts = Counter(result["status"] for result in provider["scenarios"].values())
    counts["skipped"] = len(scenario_ids(data)) - len(provider["scenarios"])
    return counts


def total_outcome_counts(data: dict[str, Any]) -> Counter[str]:
    total: Counter[str] = Counter()
    for provider in data["providers"]:
        total.update(outcome_counts(data, provider))
    return total


def plural(count: int, singular: str) -> str:
    irregular = {"loss": "losses"}
    plural_form = irregular.get(singular, singular + "s")
    return f"{count} {singular if count == 1 else plural_form}"


def signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def result_cell(result: dict[str, Any] | None) -> str:
    if result is None:
        return "—"
    return (
        f"{STATUS_CODES[result['status']]} "
        f"{signed(result['median_percent'], 1)}% / "
        f"{signed(result['median_delta_ns'] / 1000.0, 2)} us"
    )


def normalize_summaries(paths: list[pathlib.Path], snapshot_date: str) -> dict[str, Any]:
    if len(paths) != len(PROVIDER_SPECS):
        raise ValueError("exactly four summaries are required in CPU, Accelerate, Metal, CUDA order")
    documents = [json.loads(path.read_text()) for path in paths]
    first = documents[0]
    process_counts = {document["process_count"] for document in documents}
    if process_counts != {5}:
        raise ValueError(f"expected five-process summaries, got {sorted(process_counts)}")
    thresholds = {json.dumps(document["thresholds"], sort_keys=True) for document in documents}
    if len(thresholds) != 1:
        raise ValueError("provider summaries use different classification thresholds")

    providers = []
    for (provider_id, label), document, source in zip(PROVIDER_SPECS, documents, paths):
        run = document["run"]
        scenarios = {}
        for result in sorted(document["scenarios"], key=lambda item: item["scenario_id"]):
            scenarios[result["scenario_id"]] = {
                "status": result["status"],
                "library_process_median_ns": result["library_process_median_ns"],
                "reference_process_median_ns": result["reference_process_median_ns"],
                "median_delta_ns": result["median_delta_ns"],
                "median_percent": result["median_percent"],
                "percent_interval": result["percent_interval"],
                "workload": result["workload"],
            }
        providers.append(
            {
                "id": provider_id,
                "label": label,
                "device": run["device_name"],
                "backend": run["backend"],
                "cpu_implementation": run["cpu_implementation"],
                "os": run["os"],
                "architecture": run["architecture"],
                "candle_version": run["candle_version"],
                "rust_version": run["rust_version"],
                "source_summary": str(source),
                "scenarios": scenarios,
            }
        )

    samples = {
        result["samples_per_process"]
        for document in documents
        for result in document["scenarios"]
    }
    if samples != {25}:
        raise ValueError(f"expected 25 samples per process, got {sorted(samples)}")
    return {
        "schema_version": 1,
        "snapshot": {
            "date": snapshot_date,
            "processes": 5,
            "samples_per_process": 25,
            "thresholds": first["thresholds"],
        },
        "providers": providers,
    }


def render_fingerprint(data: dict[str, Any]) -> str:
    source = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    generator = pathlib.Path(__file__).read_bytes()
    return hashlib.sha256(source + b"\0" + generator).hexdigest()


def accessible_svg(
    figure: plt.Figure,
    title: str,
    description: str,
    fingerprint: str,
) -> str:
    output = io.StringIO()
    figure.savefig(
        output,
        format="svg",
        bbox_inches="tight",
        metadata={"Date": None, "Title": title, "Description": description},
    )
    plt.close(figure)
    svg = output.getvalue()
    start = svg.index("<svg")
    end = svg.index(">", start)
    root = svg[start:end]
    root = root.replace(
        "<svg",
        '<svg role="img" aria-labelledby="performance-title performance-description" '
        f'data-render-fingerprint="{fingerprint}"',
        1,
    )
    accessible = (
        root
        + ">"
        + f'<title id="performance-title">{title}</title>'
        + f'<desc id="performance-description">{description}</desc>'
    )
    return svg[:start] + accessible + svg[end + 1 :]


def render_outcomes_svg(data: dict[str, Any]) -> str:
    rows = []
    for provider in data["providers"]:
        counts = outcome_counts(data, provider)
        for status in STATUS_ORDER:
            rows.append(
                {
                    "provider": provider["label"],
                    "status": status,
                    "count": counts[status],
                }
            )
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="provider", columns="status", values="count").fillna(0)
    labels = [provider["label"] for provider in data["providers"]]
    pivot = pivot.reindex(labels)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axis = plt.subplots(figsize=(10.5, 4.3))
    left = pd.Series(0.0, index=pivot.index)
    for status in STATUS_ORDER:
        values = pivot[status]
        bars = axis.barh(
            pivot.index,
            values,
            left=left,
            color=STATUS_COLORS[status],
            edgecolor="white",
            label=STATUS_LABELS[status],
        )
        for bar, value in zip(bars, values):
            if value:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(value)),
                    ha="center",
                    va="center",
                    color="white" if status != "skipped" else "#24292f",
                    fontsize=9,
                    fontweight="bold",
                )
        left += values
    axis.invert_yaxis()
    axis.set_title("candle-einops outcomes against direct Candle", loc="left", weight="bold")
    axis.set_xlabel("Tracked scenarios")
    axis.set_ylabel("")
    axis.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    sns.despine(ax=axis, left=True, bottom=True)
    figure.tight_layout()
    return accessible_svg(
        figure,
        "Performance outcome counts by provider",
        "Stacked horizontal bars show Win, Tie, Loss, and Skipped counts for CPU baseline, CPU Accelerate, Metal, and CUDA.",
        render_fingerprint(data),
    )


def render_heatmap_svg(data: dict[str, Any]) -> str:
    scenarios = scenario_ids(data)
    providers = data["providers"]
    numeric = []
    annotations = []
    values = {
        "reference_gap": -1,
        "parity": 0,
        "library_faster": 1,
    }
    for scenario in scenarios:
        numeric_row = []
        annotation_row = []
        for provider in providers:
            result = provider["scenarios"].get(scenario)
            if result is None:
                numeric_row.append(2)
                annotation_row.append("Skipped")
            else:
                numeric_row.append(values[result["status"]])
                annotation_row.append(
                    f"{STATUS_CODES[result['status']]}  {signed(result['median_percent'], 1)}%\n"
                    f"{signed(result['median_delta_ns'] / 1000.0, 2)} us"
                )
        numeric.append(numeric_row)
        annotations.append(annotation_row)

    frame = pd.DataFrame(
        numeric,
        index=scenarios,
        columns=[provider["label"] for provider in providers],
    )
    cmap = ListedColormap(
        [
            STATUS_COLORS["reference_gap"],
            STATUS_COLORS["parity"],
            STATUS_COLORS["library_faster"],
            STATUS_COLORS["skipped"],
        ]
    )
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)
    sns.set_theme(style="white", font_scale=0.8)
    figure, axis = plt.subplots(figsize=(14.5, max(18.0, len(scenarios) * 0.48)))
    sns.heatmap(
        frame,
        cmap=cmap,
        norm=norm,
        cbar=False,
        linewidths=0.75,
        linecolor="white",
        ax=axis,
    )
    for row, annotation_row in enumerate(annotations):
        for column, annotation in enumerate(annotation_row):
            value = numeric[row][column]
            axis.text(
                column + 0.5,
                row + 0.5,
                annotation,
                ha="center",
                va="center",
                fontsize=6.2,
                fontweight="bold",
                color="white" if value in {-1, 1} else "#24292f",
            )
    axis.set_title("Complete classified outcome matrix", loc="left", weight="bold", pad=12)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="y", labelsize=6.5)
    axis.tick_params(axis="x", labelrotation=0)
    axis.legend(
        handles=[
            Patch(facecolor=STATUS_COLORS[status], label=STATUS_LABELS[status])
            for status in STATUS_ORDER
        ],
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        frameon=False,
    )
    figure.tight_layout()
    return accessible_svg(
        figure,
        "Complete performance outcome heatmap",
        "Every tracked scenario shows its Win, Tie, Loss, or Skipped classification, median percentage delta, and median absolute delta for each provider.",
        render_fingerprint(data),
    )


def render_report(data: dict[str, Any]) -> str:
    totals = total_outcome_counts(data)
    snapshot = data["snapshot"]
    thresholds = snapshot["thresholds"]
    providers = data["providers"]
    scenarios = scenario_ids(data)
    lines = [
        "<!-- Generated by .github/scripts/generate_performance_report.py; do not edit by hand. -->",
        "# Performance",
        "",
        f"This report freezes the optimized {snapshot['date']} comparison of candle-einops against equivalent direct Candle 0.11 operations. It contains {plural(totals['library_faster'], 'win')}, {plural(totals['parity'], 'tie')}, and {plural(totals['reference_gap'], 'loss')} across {sum(totals[status] for status in STATUS_ORDER[:3])} executed provider/scenario combinations.",
        "",
        "Negative percentages and durations mean candle-einops is faster. `W`, `T`, and `L` mean classified win, statistical tie, and classified loss; `—` means the GPU construction-only scenario was skipped because a view enqueues no accelerator work.",
        "",
        "![Stacked outcome counts](figures/performance-outcomes.svg)",
        "",
        "![Complete outcome heatmap](figures/performance-heatmap.svg)",
        "",
        "## Snapshot",
        "",
        "| Provider | Device | Wins | Ties | Losses | Skipped |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for provider in providers:
        counts = outcome_counts(data, provider)
        lines.append(
            f"| {provider['label']} | {provider['device']} | {counts['library_faster']} | {counts['parity']} | {counts['reference_gap']} | {counts['skipped']} |"
        )

    losses = []
    wins = []
    for provider in providers:
        for scenario, result in provider["scenarios"].items():
            item = (provider, scenario, result)
            if result["status"] == "reference_gap":
                losses.append(item)
            elif result["status"] == "library_faster":
                wins.append(item)
    wins.sort(key=lambda item: item[2]["median_percent"])

    lines.extend(["", "## Classified losses", ""])
    if not losses:
        lines.append("There are no classified losses.")
    else:
        for provider, scenario, result in losses:
            lines.append(
                f"- `{scenario}` on {provider['label']}: {signed(result['median_percent'], 1)}% and {signed(result['median_delta_ns'] / 1000.0, 2)} us."
            )
    if any(scenario == "repeat/broadcast/single-axis/consume" for _, scenario, _ in losses):
        lines.extend(
            [
                "",
                "Both losses are the same deliberate repeat-view tradeoff. Direct Candle eagerly materializes `Tensor::repeat`; candle-einops returns a storage-sharing zero-stride view. Eager materialization is faster when this single leading repeat is immediately forced contiguous on baseline CPU or Metal, but it would discard the nearly allocation-free construction wins, the two-axis consumption wins, and the CUDA single-axis win. Callers that know they require an immediate contiguous result should benchmark direct `Tensor::repeat` for that path.",
            ]
        )

    lines.extend(["", "## Largest classified wins", ""])
    for provider, scenario, result in wins[:12]:
        lines.append(
            f"- `{scenario}` on {provider['label']}: {signed(result['median_percent'], 1)}% and {signed(result['median_delta_ns'] / 1000.0, 2)} us."
        )

    lines.extend(
        [
            "",
            "## Complete scenario matrix",
            "",
            "Each cell shows classification, median percentage delta, and median absolute delta.",
            "",
            "| Scenario | " + " | ".join(provider["label"] for provider in providers) + " |",
            "|---|" + "---:|" * len(providers),
        ]
    )
    for scenario in scenarios:
        cells = [result_cell(provider["scenarios"].get(scenario)) for provider in providers]
        lines.append(f"| `{scenario}` | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"Every executed cell uses an optimized release build, {snapshot['processes']} independent processes, and {snapshot['samples_per_process']} timed samples per process after warmup and device synchronization. A classified loss must exceed {thresholds['median_percent']:.0f}% and {thresholds['median_absolute_ns'] / 1000.0:.0f} us, with its 95% confidence interval excluding {thresholds['confidence_excludes_percent']:.0f}%. The same rules classify wins in the negative direction. Results inside those materiality and confidence boundaries are ties, even when a tiny operation has a large percentage delta.",
            "",
            "The benchmark compares complete public candle-einops paths with equivalent handwired Candle operations. It does not claim custom kernels: wins come from avoiding copies, reshapes, dispatches, or unfavorable operation ordering before invoking Candle's existing kernels.",
            "",
            "## Data and reproduction",
            "",
            "The normalized source data is committed at [`benchmarks/data/performance-2026-07-16.json`](../benchmarks/data/performance-2026-07-16.json). It includes provider metadata, process medians, confidence intervals, workload metadata, and classification thresholds.",
            "",
            "Regenerate the report and figures from committed data:",
            "",
            "```console",
            "uv run --project benchmarks/reporting python .github/scripts/generate_performance_report.py",
            "uv run --project benchmarks/reporting python .github/scripts/generate_performance_report.py --check",
            "```",
            "",
            "Refresh the normalized snapshot after collecting four new `gaps` summaries:",
            "",
            "```console",
            "uv run --project benchmarks/reporting python .github/scripts/generate_performance_report.py --import-summaries \\",
            "  target/benchmarks/final-complete-2/cpu-baseline/summary.json \\",
            "  target/benchmarks/final-complete-2/cpu-accelerate/summary.json \\",
            "  target/benchmarks/final-complete/metal/summary.json \\",
            "  target/benchmarks/final-complete-2/cuda/summary.json",
            "```",
            "",
            "Machine and driver metadata should always be reviewed before comparing snapshots across hosts.",
            "",
        ]
    )
    return "\n".join(lines)


def serialized_data(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=False) + "\n"


def generated_outputs(data: dict[str, Any]) -> dict[pathlib.Path, str]:
    return {
        REPORT_PATH: render_report(data),
        OUTCOMES_PATH: render_outcomes_svg(data),
        HEATMAP_PATH: render_heatmap_svg(data),
    }


def svg_is_current(existing: str, expected: str) -> bool:
    marker = 'data-render-fingerprint="'
    try:
        existing_start = existing.index(marker) + len(marker)
        expected_start = expected.index(marker) + len(marker)
    except ValueError:
        return False
    existing_fingerprint = existing[existing_start : existing.index('"', existing_start)]
    expected_fingerprint = expected[expected_start : expected.index('"', expected_start)]
    return (
        len(existing_fingerprint) == 64
        and existing_fingerprint == expected_fingerprint
        and 'role="img"' in existing
        and "<title" in existing
        and "<desc" in existing
    )


def write_or_check(outputs: dict[pathlib.Path, str], check: bool) -> int:
    stale = []
    for path, content in outputs.items():
        if check:
            existing = path.read_text() if path.exists() else ""
            current = (
                svg_is_current(existing, content)
                if path.suffix == ".svg"
                else existing == content
            )
            if not current:
                stale.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    if stale:
        print("performance report is stale: " + ", ".join(map(str, stale)), file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if committed outputs differ")
    parser.add_argument(
        "--import-summaries",
        nargs=4,
        type=pathlib.Path,
        metavar=("CPU", "ACCELERATE", "METAL", "CUDA"),
        help="normalize four gap summary files before rendering",
    )
    parser.add_argument("--snapshot-date", default="2026-07-16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.import_summaries:
        data = normalize_summaries(args.import_summaries, args.snapshot_date)
    else:
        data = json.loads(DATA_PATH.read_text())
    outputs = generated_outputs(data)
    if args.import_summaries:
        outputs[DATA_PATH] = serialized_data(data)
    return write_or_check(outputs, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
