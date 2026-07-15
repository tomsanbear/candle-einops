use std::cell::RefCell;
use std::collections::BTreeMap;

use candle_core::{Device, Shape, Tensor};
use candle_einops::einops;
use candle_einops_parity_runner::{
    NormalizedResponse, Operation, OracleClient, OracleRequest, OracleValue, ParityConfig,
    PatternId, persist_replay,
};
use proptest::prelude::*;
use proptest::strategy::BoxedStrategy;
use proptest::test_runner::{TestCaseError, TestCaseResult, TestRunner};

const FAILURE_REPLAY: &str = "parity/regressions/rearrange-last-failure.jsonl";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RearrangePattern {
    Permute2d,
    Permute3d,
    Compose,
    RuntimeDecompose,
    SqueezeSingletons,
    Ellipsis0,
    Ellipsis1,
    Ellipsis2,
    Ellipsis3,
    NonContiguous,
    InvalidDecompose,
    InvalidSqueeze,
}

impl RearrangePattern {
    const fn id(self) -> &'static str {
        match self {
            Self::Permute2d => "rearrange/permute-2d-v1",
            Self::Permute3d => "rearrange/permute-3d-v1",
            Self::Compose => "rearrange/compose-v1",
            Self::RuntimeDecompose => "rearrange/runtime-decompose-v1",
            Self::SqueezeSingletons => "rearrange/squeeze-singletons-v1",
            Self::Ellipsis0 => "rearrange/ellipsis-0-v1",
            Self::Ellipsis1 => "rearrange/ellipsis-1-v1",
            Self::Ellipsis2 => "rearrange/ellipsis-2-v1",
            Self::Ellipsis3 => "rearrange/ellipsis-3-v1",
            Self::NonContiguous => "rearrange/non-contiguous-v1",
            Self::InvalidDecompose => "rearrange/invalid-decompose-v1",
            Self::InvalidSqueeze => "rearrange/invalid-squeeze-v1",
        }
    }

    const fn python_pattern(self) -> &'static str {
        match self {
            Self::Permute2d => "rows columns -> columns rows",
            Self::Permute3d => "a b c -> c a b",
            Self::Compose => "a b c -> (a b) c",
            Self::RuntimeDecompose | Self::InvalidDecompose => "(a factor) c -> factor a c",
            Self::SqueezeSingletons | Self::InvalidSqueeze => "1 a 1 -> a",
            Self::Ellipsis0 | Self::Ellipsis1 | Self::Ellipsis2 | Self::Ellipsis3 => {
                "a ... z -> z a ..."
            }
            Self::NonContiguous => "c a b -> (a b) c",
        }
    }

    const fn expects_success(self) -> bool {
        !matches!(self, Self::InvalidDecompose | Self::InvalidSqueeze)
    }
}

#[derive(Clone, Debug)]
struct RearrangeCase {
    pattern: RearrangePattern,
    dimensions: Vec<usize>,
    values: Vec<f32>,
    factor: Option<usize>,
}

impl RearrangeCase {
    fn deterministic(
        pattern: RearrangePattern,
        dimensions: &[usize],
        factor: Option<usize>,
    ) -> Self {
        let count = dimensions.iter().product();
        let values = (0..count)
            .map(|index| index as f32 - count as f32 / 2.)
            .collect();
        Self {
            pattern,
            dimensions: dimensions.to_vec(),
            values,
            factor,
        }
    }

    fn input(&self) -> candle_core::Result<Tensor> {
        let contiguous = Tensor::from_vec(
            self.values.clone(),
            Shape::from_dims(&self.dimensions),
            &Device::Cpu,
        )?;
        if self.pattern == RearrangePattern::NonContiguous {
            contiguous.permute((2, 0, 1))
        } else {
            Ok(contiguous)
        }
    }

    fn request(&self, case_id: String, input: &Tensor) -> candle_core::Result<OracleRequest> {
        let values = input
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| OracleValue::from(f64::from(value)))
            .collect();
        let mut axes_lengths = BTreeMap::new();
        if let Some(factor) = self.factor {
            axes_lengths.insert("factor".to_string(), factor);
        }
        Ok(OracleRequest {
            case_id,
            pattern_id: PatternId::new(self.pattern.id()).expect("static pattern id is valid"),
            operation: Operation::Rearrange,
            pattern: self.pattern.python_pattern().to_string(),
            reduction: None,
            dtype: "float32".to_string(),
            shape: input.dims().to_vec(),
            values,
            axes_lengths,
        })
    }
}

fn finite_values(count: usize) -> BoxedStrategy<Vec<f32>> {
    prop::collection::vec(-32_i16..=32, count)
        .prop_map(|values| values.into_iter().map(f32::from).collect())
        .boxed()
}

fn shaped_case(
    pattern: RearrangePattern,
    dimensions: Vec<usize>,
    factor: Option<usize>,
) -> BoxedStrategy<RearrangeCase> {
    let count = dimensions.iter().product();
    finite_values(count)
        .prop_map(move |values| RearrangeCase {
            pattern,
            dimensions: dimensions.clone(),
            values,
            factor,
        })
        .boxed()
}

fn permute_2d() -> BoxedStrategy<RearrangeCase> {
    (0_usize..=3, 0_usize..=3)
        .prop_flat_map(|(rows, columns)| {
            shaped_case(RearrangePattern::Permute2d, vec![rows, columns], None)
        })
        .boxed()
}

fn permute_3d() -> BoxedStrategy<RearrangeCase> {
    (0_usize..=3, 0_usize..=3, 0_usize..=3)
        .prop_flat_map(|(a, b, c)| shaped_case(RearrangePattern::Permute3d, vec![a, b, c], None))
        .boxed()
}

fn compose() -> BoxedStrategy<RearrangeCase> {
    (0_usize..=3, 0_usize..=3, 0_usize..=3)
        .prop_flat_map(|(a, b, c)| shaped_case(RearrangePattern::Compose, vec![a, b, c], None))
        .boxed()
}

fn runtime_decompose() -> BoxedStrategy<RearrangeCase> {
    (0_usize..=3, 1_usize..=3, 0_usize..=3)
        .prop_flat_map(|(a, factor, c)| {
            shaped_case(
                RearrangePattern::RuntimeDecompose,
                vec![a * factor, c],
                Some(factor),
            )
        })
        .boxed()
}

fn squeeze_singletons() -> BoxedStrategy<RearrangeCase> {
    (0_usize..=3)
        .prop_flat_map(|a| shaped_case(RearrangePattern::SqueezeSingletons, vec![1, a, 1], None))
        .boxed()
}

fn ellipsis(pattern: RearrangePattern, captures: usize) -> BoxedStrategy<RearrangeCase> {
    prop::collection::vec(0_usize..=3, captures + 2)
        .prop_flat_map(move |dimensions| shaped_case(pattern, dimensions, None))
        .boxed()
}

fn non_contiguous() -> BoxedStrategy<RearrangeCase> {
    (0_usize..=3, 0_usize..=3, 0_usize..=3)
        .prop_flat_map(|(a, b, c)| {
            shaped_case(RearrangePattern::NonContiguous, vec![a, b, c], None)
        })
        .boxed()
}

fn invalid_decompose() -> BoxedStrategy<RearrangeCase> {
    (0_usize..=3, 2_usize..=3, 1_usize..=3)
        .prop_flat_map(|(a, factor, c)| {
            shaped_case(
                RearrangePattern::InvalidDecompose,
                vec![a * factor + 1, c],
                Some(factor),
            )
        })
        .boxed()
}

fn invalid_squeeze() -> BoxedStrategy<RearrangeCase> {
    (2_usize..=3, 1_usize..=3)
        .prop_flat_map(|(not_singleton, a)| {
            shaped_case(
                RearrangePattern::InvalidSqueeze,
                vec![not_singleton, a, 1],
                None,
            )
        })
        .boxed()
}

fn round_strategy() -> BoxedStrategy<Vec<RearrangeCase>> {
    (
        (
            permute_2d(),
            permute_3d(),
            compose(),
            runtime_decompose(),
            squeeze_singletons(),
            ellipsis(RearrangePattern::Ellipsis0, 0),
        ),
        (
            ellipsis(RearrangePattern::Ellipsis1, 1),
            ellipsis(RearrangePattern::Ellipsis2, 2),
            ellipsis(RearrangePattern::Ellipsis3, 3),
            non_contiguous(),
            invalid_decompose(),
            invalid_squeeze(),
        ),
    )
        .prop_map(
            |(
                (permute2, permute3, compose, decompose, squeeze, ellipsis0),
                (
                    ellipsis1,
                    ellipsis2,
                    ellipsis3,
                    non_contiguous,
                    invalid_decompose,
                    invalid_squeeze,
                ),
            )| {
                vec![
                    permute2,
                    permute3,
                    compose,
                    decompose,
                    squeeze,
                    ellipsis0,
                    ellipsis1,
                    ellipsis2,
                    ellipsis3,
                    non_contiguous,
                    invalid_decompose,
                    invalid_squeeze,
                ]
            },
        )
        .boxed()
}

fn explicit_edge_cases() -> Vec<RearrangeCase> {
    vec![
        RearrangeCase::deterministic(RearrangePattern::Permute2d, &[0, 3], None),
        RearrangeCase::deterministic(RearrangePattern::Permute3d, &[1, 2, 1], None),
        RearrangeCase::deterministic(RearrangePattern::Compose, &[2, 0, 3], None),
        RearrangeCase::deterministic(RearrangePattern::RuntimeDecompose, &[0, 2], Some(3)),
        RearrangeCase::deterministic(RearrangePattern::SqueezeSingletons, &[1, 0, 1], None),
        RearrangeCase::deterministic(RearrangePattern::Ellipsis0, &[0, 1], None),
        RearrangeCase::deterministic(RearrangePattern::Ellipsis1, &[1, 0, 2], None),
        RearrangeCase::deterministic(RearrangePattern::Ellipsis2, &[1, 2, 0, 1], None),
        RearrangeCase::deterministic(RearrangePattern::Ellipsis3, &[1, 1, 2, 0, 3], None),
        RearrangeCase::deterministic(RearrangePattern::NonContiguous, &[2, 0, 3], None),
        RearrangeCase::deterministic(RearrangePattern::InvalidDecompose, &[3, 2], Some(2)),
        RearrangeCase::deterministic(RearrangePattern::InvalidSqueeze, &[2, 3, 1], None),
    ]
}

fn run_rust_case(case: &RearrangeCase, input: &Tensor) -> candle_core::Result<Tensor> {
    let factor = case.factor.unwrap_or(1);
    match case.pattern {
        RearrangePattern::Permute2d => einops!("rows columns -> columns rows", input),
        RearrangePattern::Permute3d => einops!("a b c -> c a b", input),
        RearrangePattern::Compose => einops!("a b c -> (a b) c", input),
        RearrangePattern::RuntimeDecompose | RearrangePattern::InvalidDecompose => {
            einops!("(a {factor}) c -> {factor} a c", input)
        }
        RearrangePattern::SqueezeSingletons | RearrangePattern::InvalidSqueeze => {
            einops!("1 a 1 -> a", input)
        }
        RearrangePattern::Ellipsis0
        | RearrangePattern::Ellipsis1
        | RearrangePattern::Ellipsis2
        | RearrangePattern::Ellipsis3 => einops!("a .. z -> z a ..", input),
        RearrangePattern::NonContiguous => einops!("c a b -> (a b) c", input),
    }
}

fn compare_batch(client: &mut OracleClient, cases: &[RearrangeCase]) -> TestCaseResult {
    let prepared = cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let input = case.input().map_err(|error| {
                TestCaseError::fail(format!("build {} input: {error}", case.pattern.id()))
            })?;
            let request = case
                .request(format!("{}-{index}", case.pattern.id()), &input)
                .map_err(|error| {
                    TestCaseError::fail(format!("build {} request: {error}", case.pattern.id()))
                })?;
            Ok((case, input, request))
        })
        .collect::<Result<Vec<_>, TestCaseError>>()?;
    let requests = prepared
        .iter()
        .map(|(_, _, request)| request.clone())
        .collect::<Vec<_>>();
    let responses = client
        .evaluate(&requests)
        .map_err(|error| TestCaseError::fail(format!("Python oracle batch: {error}")))?;

    for ((case, input, request), response) in prepared.into_iter().zip(responses) {
        let rust = run_rust_case(case, &input);
        match (rust, response) {
            (Ok(actual), NormalizedResponse::Success(expected)) => {
                if !case.pattern.expects_success() {
                    return parity_failure(
                        &request,
                        format!(
                            "{} unexpectedly succeeded in both runtimes",
                            case.pattern.id()
                        ),
                    );
                }
                let actual_shape = actual.dims().to_vec();
                let actual_values = actual
                    .flatten_all()
                    .and_then(|tensor| tensor.to_vec1::<f32>())
                    .map_err(|error| {
                        TestCaseError::fail(format!(
                            "read {} Rust output: {error}",
                            case.pattern.id()
                        ))
                    })?;
                let expected_values = expected
                    .values
                    .into_iter()
                    .map(|value| match value {
                        OracleValue::Number(value) => Ok(value as f32),
                        OracleValue::Symbol(value) => Err(TestCaseError::fail(format!(
                            "{} returned non-finite symbol {value}",
                            case.pattern.id()
                        ))),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if actual_shape != expected.shape || actual_values != expected_values {
                    return parity_failure(
                        &request,
                        format!(
                            "{} mismatch: Rust shape/value {:?}/{actual_values:?}, Python {:?}/{expected_values:?}",
                            case.pattern.id(),
                            actual_shape,
                            expected.shape,
                        ),
                    );
                }
            }
            (Err(_), NormalizedResponse::Error(_)) if !case.pattern.expects_success() => {}
            (Err(error), NormalizedResponse::Error(expected)) => {
                return parity_failure(
                    &request,
                    format!(
                        "{} valid case failed in both runtimes: Rust={error}; Python={}: {}",
                        case.pattern.id(),
                        expected.error.kind,
                        expected.error.message
                    ),
                );
            }
            (Ok(_), NormalizedResponse::Error(error)) => {
                return parity_failure(
                    &request,
                    format!(
                        "{} acceptance mismatch: Rust accepted, Python rejected {}: {}",
                        case.pattern.id(),
                        error.error.kind,
                        error.error.message
                    ),
                );
            }
            (Err(error), NormalizedResponse::Success(_)) => {
                return parity_failure(
                    &request,
                    format!(
                        "{} acceptance mismatch: Rust rejected {error}, Python accepted",
                        case.pattern.id()
                    ),
                );
            }
        }
    }
    Ok(())
}

fn parity_failure(request: &OracleRequest, message: String) -> TestCaseResult {
    let replay_note = match persist_replay(FAILURE_REPLAY, request) {
        Ok(()) => format!("; replay={FAILURE_REPLAY}"),
        Err(error) => format!("; could not persist replay: {error}"),
    };
    Err(TestCaseError::fail(format!("{message}{replay_note}")))
}

#[test]
fn randomized_rearrange_matches_locked_python_einops() {
    let settings = ParityConfig::from_env().expect("valid bounded parity configuration");
    let mut proptest_config = settings.proptest_config();
    proptest_config.failure_persistence = None;
    let mut runner = TestRunner::new(proptest_config);
    let client = RefCell::new(OracleClient::spawn_uv().expect("locked Python oracle starts"));

    compare_batch(&mut client.borrow_mut(), &explicit_edge_cases())
        .expect("explicit edge matrix matches");
    runner
        .run(&round_strategy(), |cases| {
            compare_batch(&mut client.borrow_mut(), &cases)
        })
        .unwrap_or_else(|error| {
            panic!(
                "rearrange parity failed with seed {} after {} cases: {error}; replay with CANDLE_EINOPS_PARITY_SEED={}",
                settings.seed, settings.cases, settings.seed
            )
        });

    let status = client
        .into_inner()
        .shutdown()
        .expect("Python oracle exits cleanly");
    assert!(status.success());
}
