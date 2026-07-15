use std::cell::{Cell, RefCell};

use candle_core::{Device, Tensor};
use candle_einops::einsum;
use candle_einops_parity_runner::{
    EinsumOperand, EinsumRequest, NormalizedResponse, Operation, OracleClient, OracleValue,
    ParityConfig, PatternId, persist_replay,
};
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestRunner};

const FAILURE_REPLAY: &str = "parity/regressions/einsum-last-failure.jsonl";

#[derive(Clone, Copy, Debug)]
enum Pattern {
    UnaryPermutation,
    UnaryReduction,
    ElementwiseBroadcast,
    OuterProduct,
    MatrixProduct,
    BatchedProduct,
    EllipsisRetained,
    EllipsisOmitted,
    Diagonal,
    Trace,
    HigherDiagonal,
    ThreeOperand,
    FourOperand,
    Scalar,
    InvalidShape,
    InvalidRank,
}

impl Pattern {
    const ALL: [Self; 16] = [
        Self::UnaryPermutation,
        Self::UnaryReduction,
        Self::ElementwiseBroadcast,
        Self::OuterProduct,
        Self::MatrixProduct,
        Self::BatchedProduct,
        Self::EllipsisRetained,
        Self::EllipsisOmitted,
        Self::Diagonal,
        Self::Trace,
        Self::HigherDiagonal,
        Self::ThreeOperand,
        Self::FourOperand,
        Self::Scalar,
        Self::InvalidShape,
        Self::InvalidRank,
    ];

    const fn id(self) -> &'static str {
        match self {
            Self::UnaryPermutation => "einsum/unary-permutation-v1",
            Self::UnaryReduction => "einsum/unary-reduction-v1",
            Self::ElementwiseBroadcast => "einsum/elementwise-broadcast-v1",
            Self::OuterProduct => "einsum/outer-product-v1",
            Self::MatrixProduct => "einsum/matrix-product-v1",
            Self::BatchedProduct => "einsum/batched-product-v1",
            Self::EllipsisRetained => "einsum/ellipsis-retained-v1",
            Self::EllipsisOmitted => "einsum/ellipsis-omitted-v1",
            Self::Diagonal => "einsum/diagonal-v1",
            Self::Trace => "einsum/trace-v1",
            Self::HigherDiagonal => "einsum/higher-diagonal-v1",
            Self::ThreeOperand => "einsum/three-operand-v1",
            Self::FourOperand => "einsum/four-operand-v1",
            Self::Scalar => "einsum/scalar-v1",
            Self::InvalidShape => "einsum/invalid-shape-v1",
            Self::InvalidRank => "einsum/invalid-rank-v1",
        }
    }

    const fn python_pattern(self) -> &'static str {
        match self {
            Self::UnaryPermutation => "a b -> b a",
            Self::UnaryReduction => "a k -> a",
            Self::ElementwiseBroadcast => "a b, a b -> a b",
            Self::OuterProduct => "a, b -> a b",
            Self::MatrixProduct | Self::InvalidShape => "a k, k b -> a b",
            Self::BatchedProduct => "batch a k, batch k b -> batch a b",
            Self::EllipsisRetained => "... a k, ... k -> ... a",
            Self::EllipsisOmitted => "... a k -> a",
            Self::Diagonal => "i i -> i",
            Self::Trace => "i i ->",
            Self::HigherDiagonal => "i i i -> i",
            Self::ThreeOperand => "a k, k b, b c -> a c",
            Self::FourOperand => "a k, k b, b c, c d -> a d",
            Self::Scalar => ", a -> a",
            Self::InvalidRank => "a k, k -> a",
        }
    }

    const fn expects_success(self) -> bool {
        !matches!(self, Self::InvalidShape | Self::InvalidRank)
    }
}

#[derive(Clone, Debug)]
struct Operand {
    shape: Vec<usize>,
    values: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Case {
    pattern: Pattern,
    operands: Vec<Operand>,
    operation_scale: usize,
}

fn values(shape: &[usize], salt: i8, stream: usize) -> Vec<f32> {
    let count = shape.iter().product();
    (0..count)
        .map(|index| {
            let value = i16::from(salt) + (index as i16 * 5) + (stream as i16 * 3);
            f32::from((value.rem_euclid(9) - 4) as i8)
        })
        .collect()
}

fn operand(shape: Vec<usize>, salt: i8, stream: usize) -> Operand {
    Operand {
        values: values(&shape, salt, stream),
        shape,
    }
}

fn build_case(pattern: Pattern, dimensions: [u8; 6], salt: i8, extent: usize) -> Case {
    let free = |index: usize| usize::from(dimensions[index]) % (extent + 1);
    let contracted = |index: usize| usize::from(dimensions[index]) % extent + 1;
    let a = free(0);
    let b = free(1);
    let batch = free(2);
    let k = contracted(3);
    let c = contracted(4);
    let d = free(5);
    let shapes = match pattern {
        Pattern::UnaryPermutation => vec![vec![a, b]],
        Pattern::UnaryReduction => vec![vec![a, k]],
        Pattern::ElementwiseBroadcast => vec![vec![a, b], vec![1, b]],
        Pattern::OuterProduct => vec![vec![a], vec![b]],
        Pattern::MatrixProduct => vec![vec![a, k], vec![k, b]],
        Pattern::BatchedProduct => vec![vec![batch, a, k], vec![1, k, b]],
        Pattern::EllipsisRetained => vec![vec![batch, a, k], vec![1, k]],
        Pattern::EllipsisOmitted => vec![vec![contracted(2), a, k]],
        Pattern::Diagonal | Pattern::Trace => vec![vec![a, a]],
        Pattern::HigherDiagonal => vec![vec![a, a, a]],
        Pattern::ThreeOperand => vec![vec![a, k], vec![k, b], vec![b, c]],
        Pattern::FourOperand => vec![vec![a, k], vec![k, b], vec![b, c], vec![c, d]],
        Pattern::Scalar => vec![vec![], vec![a]],
        Pattern::InvalidShape => vec![vec![a, k], vec![k + 1, b]],
        Pattern::InvalidRank => vec![vec![a], vec![k]],
    };
    let operation_scale = match pattern {
        Pattern::UnaryReduction | Pattern::MatrixProduct | Pattern::BatchedProduct => k,
        Pattern::EllipsisRetained => k,
        Pattern::EllipsisOmitted => contracted(2) * k,
        Pattern::Trace => a.max(1),
        Pattern::ThreeOperand => k * b.max(1),
        Pattern::FourOperand => k * b.max(1) * c,
        _ => 1,
    };
    Case {
        pattern,
        operands: shapes
            .into_iter()
            .enumerate()
            .map(|(stream, shape)| operand(shape, salt, stream))
            .collect(),
        operation_scale,
    }
}

fn case_strategy(max_elements: usize) -> impl Strategy<Value = Case> {
    let mut extent = 1usize;
    while (extent + 1).pow(3) <= max_elements {
        extent += 1;
    }
    let extent = extent.min(3);
    (0usize..Pattern::ALL.len(), any::<[u8; 6]>(), any::<i8>()).prop_map(
        move |(pattern, dimensions, salt)| {
            build_case(Pattern::ALL[pattern], dimensions, salt, extent)
        },
    )
}

fn explicit_edges() -> Vec<Case> {
    Pattern::ALL
        .into_iter()
        .enumerate()
        .map(|(index, pattern)| {
            let dimensions = if matches!(pattern, Pattern::OuterProduct | Pattern::Diagonal) {
                [0, 1, 0, 0, 0, 0]
            } else {
                [1, 1, 1, 0, 0, 1]
            };
            build_case(pattern, dimensions, index as i8, 2)
        })
        .collect()
}

fn request(case: &Case, case_id: String) -> EinsumRequest {
    EinsumRequest {
        case_id,
        pattern_id: PatternId::new(case.pattern.id()).expect("static pattern id"),
        operation: Operation::Einsum,
        pattern: case.pattern.python_pattern().to_string(),
        operands: case
            .operands
            .iter()
            .map(|operand| EinsumOperand {
                dtype: "float32".to_string(),
                shape: operand.shape.clone(),
                values: operand
                    .values
                    .iter()
                    .map(|value| OracleValue::from(f64::from(*value)))
                    .collect(),
            })
            .collect(),
    }
}

fn rust_result(case: &Case, operands: &[Tensor]) -> candle_core::Result<Tensor> {
    match case.pattern {
        Pattern::UnaryPermutation => einsum!("a b -> b a", &operands[0]),
        Pattern::UnaryReduction => einsum!("a k -> a", &operands[0]),
        Pattern::ElementwiseBroadcast => einsum!("a b, a b -> a b", &operands[0], &operands[1]),
        Pattern::OuterProduct => einsum!("a, b -> a b", &operands[0], &operands[1]),
        Pattern::MatrixProduct | Pattern::InvalidShape => {
            einsum!("a k, k b -> a b", &operands[0], &operands[1])
        }
        Pattern::BatchedProduct => einsum!(
            "batch a k, batch k b -> batch a b",
            &operands[0],
            &operands[1]
        ),
        Pattern::EllipsisRetained => {
            einsum!(".. a k, .. k -> .. a", &operands[0], &operands[1])
        }
        Pattern::EllipsisOmitted => einsum!(".. a k -> a", &operands[0]),
        Pattern::Diagonal => einsum!("i i -> i", &operands[0]),
        Pattern::Trace => einsum!("i i ->", &operands[0]),
        Pattern::HigherDiagonal => einsum!("i i i -> i", &operands[0]),
        Pattern::ThreeOperand => einsum!(
            "a k, k b, b c -> a c",
            &operands[0],
            &operands[1],
            &operands[2]
        ),
        Pattern::FourOperand => einsum!(
            "a k, k b, b c, c d -> a d",
            &operands[0],
            &operands[1],
            &operands[2],
            &operands[3]
        ),
        Pattern::Scalar => einsum!(", a -> a", &operands[0], &operands[1]),
        Pattern::InvalidRank => einsum!("a k, k -> a", &operands[0], &operands[1]),
    }
}

fn fail(request: &EinsumRequest, message: impl std::fmt::Display) -> TestCaseError {
    let replay = match persist_replay(FAILURE_REPLAY, request) {
        Ok(()) => FAILURE_REPLAY.to_string(),
        Err(error) => format!("could not persist replay: {error}"),
    };
    TestCaseError::fail(format!("{message}; minimized request replay={replay}"))
}

fn compare(client: &mut OracleClient, case: &Case, case_id: String) -> TestCaseResult {
    let request = request(case, case_id);
    let response = client
        .evaluate_einsum(std::slice::from_ref(&request))
        .map_err(|error| fail(&request, error))?
        .pop()
        .expect("one response");
    let operands = case
        .operands
        .iter()
        .map(|operand| {
            Tensor::from_vec(
                operand.values.clone(),
                operand.shape.as_slice(),
                &Device::Cpu,
            )
        })
        .collect::<candle_core::Result<Vec<_>>>()
        .map_err(|error| fail(&request, error))?;

    match (rust_result(case, &operands), response) {
        (Err(_), NormalizedResponse::Error(_)) if !case.pattern.expects_success() => Ok(()),
        (Ok(actual), NormalizedResponse::Success(expected)) if case.pattern.expects_success() => {
            if actual.dims() != expected.shape {
                return Err(fail(
                    &request,
                    format_args!(
                        "shape mismatch: Rust={:?}, Python={:?}",
                        actual.dims(),
                        expected.shape
                    ),
                ));
            }
            let actual = actual
                .flatten_all()
                .and_then(|value| value.to_vec1::<f32>())
                .map_err(|error| fail(&request, error))?;
            let expected = expected
                .values
                .into_iter()
                .map(|value| match value {
                    OracleValue::Number(value) => Ok(value as f32),
                    OracleValue::Symbol(value) => {
                        Err(fail(&request, format_args!("non-finite {value}")))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let scale = case.operation_scale.max(1) as f32;
            let relative = f32::EPSILON * scale * 32.;
            let absolute = f32::EPSILON * scale * 16.;
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                let tolerance = absolute + relative * expected.abs();
                if (actual - expected).abs() > tolerance {
                    return Err(fail(
                        &request,
                        format_args!(
                            "value {index}: Rust={actual}, Python={expected}, tolerance={tolerance}"
                        ),
                    ));
                }
            }
            Ok(())
        }
        (rust, python) => Err(fail(
            &request,
            format_args!("acceptance mismatch: Rust={rust:?}, Python={python:?}"),
        )),
    }
}

#[test]
fn randomized_einsum_matches_locked_python_einops() {
    let config = ParityConfig::from_env().expect("valid bounded parity configuration");
    let mut proptest = config.proptest_config();
    proptest.failure_persistence = None;
    let mut runner = TestRunner::new(proptest);
    let oracle = RefCell::new(OracleClient::spawn_uv().expect("locked Python oracle starts"));
    let sequence = Cell::new(0u64);

    for case in explicit_edges() {
        let current = sequence.get();
        sequence.set(current + 1);
        compare(&mut oracle.borrow_mut(), &case, format!("edge-{current}"))
            .expect("explicit einsum semantic matrix matches");
    }
    runner
        .run(&case_strategy(config.max_elements), |case| {
            let current = sequence.get();
            sequence.set(current + 1);
            compare(
                &mut oracle.borrow_mut(),
                &case,
                format!("generated-{current}"),
            )
        })
        .unwrap_or_else(|error| {
            panic!(
                "einsum parity failed with seed {} after {} cases: {error}",
                config.seed, config.cases
            )
        });

    let status = oracle
        .into_inner()
        .shutdown()
        .expect("Python oracle exits cleanly");
    assert!(status.success());
}
