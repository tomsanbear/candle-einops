#!/usr/bin/env python3
"""JSONL bridge to the public Python einops API."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from typing import Any, TextIO

import einops
import numpy as np


PROTOCOL_VERSION = 1
SERVICE_NAME = "candle-einops-python-oracle"


def _json_number(value: Any) -> int | float | str:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, float):
        if math.isnan(scalar):
            return "nan"
        if math.isinf(scalar):
            return "+inf" if scalar > 0 else "-inf"
    return scalar


def _error(case_id: Any, error: Exception) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "ok": False,
        "error": {
            "kind": type(error).__name__,
            "message": " ".join(str(error).split()),
        },
    }


def evaluate_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one request with einops, returning strict-JSON data."""

    case_id = request.get("case_id")
    try:
        shape = tuple(int(extent) for extent in request["shape"])
        dtype = np.dtype(request.get("dtype", "float64"))
        tensor = np.asarray(request["values"], dtype=dtype).reshape(shape)
        pattern = str(request["pattern"])
        axes_lengths = {
            str(name): int(length)
            for name, length in request.get("axes_lengths", {}).items()
        }
        operation = request["operation"]

        if operation == "rearrange":
            result = einops.rearrange(tensor, pattern, **axes_lengths)
        elif operation == "repeat":
            result = einops.repeat(tensor, pattern, **axes_lengths)
        elif operation == "reduce":
            result = einops.reduce(
                tensor,
                pattern,
                request["reduction"],
                **axes_lengths,
            )
        else:
            raise ValueError(f"unsupported operation: {operation!r}")

        array = np.asarray(result)
        return {
            "case_id": case_id,
            "ok": True,
            "shape": list(array.shape),
            "values": [_json_number(value) for value in array.reshape(-1)],
        }
    except Exception as error:  # The protocol reports oracle failures as data.
        return _error(case_id, error)


def serve(stdin: TextIO, stdout: TextIO) -> None:
    """Serve ordered JSONL requests until EOF."""

    stdout.write(
        json.dumps(
            {
                "kind": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "service": SERVICE_NAME,
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    stdout.flush()
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = evaluate_request(request)
        except Exception as error:
            response = _error(None, error)
        stdout.write(json.dumps(response, allow_nan=False, separators=(",", ":")) + "\n")
        stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        help="evaluate and print one JSON request instead of serving JSONL",
    )
    arguments = parser.parse_args()
    if arguments.replay is not None:
        print(
            json.dumps(
                evaluate_request(json.loads(arguments.replay)),
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        return
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
