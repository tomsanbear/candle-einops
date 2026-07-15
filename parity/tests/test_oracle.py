import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from hypothesis import given, settings, strategies as st


PARITY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARITY_ROOT))

from oracle import evaluate_request, serve  # noqa: E402


class OracleContractTests(unittest.TestCase):
    def test_rearrange_uses_einops_and_preserves_case_identity(self) -> None:
        response = evaluate_request(
            {
                "case_id": "transpose-7",
                "operation": "rearrange",
                "pattern": "rows columns -> columns rows",
                "shape": [2, 3],
                "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "axes_lengths": {},
            }
        )

        self.assertEqual(
            response,
            {
                "case_id": "transpose-7",
                "ok": True,
                "shape": [3, 2],
                "values": [0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
            },
        )

    def test_repeat_and_reduce_call_the_public_python_api(self) -> None:
        repeated = evaluate_request(
            {
                "case_id": "repeat-3",
                "operation": "repeat",
                "pattern": "rows columns -> rows copies columns",
                "shape": [1, 2],
                "values": [2.0, 5.0],
                "axes_lengths": {"copies": 3},
            }
        )
        reduced = evaluate_request(
            {
                "case_id": "mean-4",
                "operation": "reduce",
                "pattern": "rows columns -> rows",
                "reduction": "mean",
                "shape": [2, 2],
                "values": [1.0, 3.0, 5.0, 7.0],
                "axes_lengths": {},
            }
        )

        self.assertEqual(repeated["shape"], [1, 3, 2])
        self.assertEqual(repeated["values"], [2.0, 5.0, 2.0, 5.0, 2.0, 5.0])
        self.assertEqual(reduced["shape"], [2])
        self.assertEqual(reduced["values"], [2.0, 6.0])

    def test_zero_sized_shapes_and_scalar_results_are_serializable(self) -> None:
        response = evaluate_request(
            {
                "case_id": "empty-sum",
                "operation": "reduce",
                "pattern": "rows columns -> rows",
                "reduction": "sum",
                "shape": [2, 0],
                "values": [],
                "axes_lengths": {},
            }
        )

        self.assertEqual(response["shape"], [2])
        self.assertEqual(response["values"], [0.0, 0.0])

    def test_einops_failures_are_normalized_without_tracebacks(self) -> None:
        response = evaluate_request(
            {
                "case_id": "invalid-shape",
                "operation": "rearrange",
                "pattern": "rows columns -> columns rows",
                "shape": [2, 3],
                "values": [1.0],
                "axes_lengths": {},
            }
        )

        self.assertEqual(response["case_id"], "invalid-shape")
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"]["kind"], "ValueError")
        self.assertNotIn("Traceback", response["error"]["message"])

    def test_jsonl_server_preserves_request_order(self) -> None:
        requests = [
            {
                "case_id": f"case-{index}",
                "operation": "rearrange",
                "pattern": "row column -> column row",
                "shape": [1, 1],
                "values": [float(index)],
                "axes_lengths": {},
            }
            for index in range(3)
        ]
        stdin = io.StringIO("".join(json.dumps(case) + "\n" for case in requests))
        stdout = io.StringIO()

        serve(stdin, stdout)

        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([item["case_id"] for item in responses], ["case-0", "case-1", "case-2"])

    @settings(max_examples=64, derandomize=True, deadline=None)
    @given(
        rows=st.integers(min_value=0, max_value=5),
        columns=st.integers(min_value=0, max_value=5),
        data=st.data(),
    )
    def test_randomized_rearrange_contract(
        self,
        rows: int,
        columns: int,
        data,
    ) -> None:
        values = data.draw(
            st.lists(
                st.integers(min_value=-32, max_value=32),
                min_size=rows * columns,
                max_size=rows * columns,
            )
        )
        response = evaluate_request(
            {
                "case_id": f"hypothesis-{rows}-{columns}",
                "operation": "rearrange",
                "pattern": "rows columns -> columns rows",
                "shape": [rows, columns],
                "values": values,
                "axes_lengths": {},
            }
        )
        expected = np.asarray(values, dtype=np.float64).reshape(rows, columns).T

        self.assertTrue(response["ok"], response)
        self.assertEqual(response["shape"], list(expected.shape))
        self.assertEqual(response["values"], expected.reshape(-1).tolist())


if __name__ == "__main__":
    unittest.main()
