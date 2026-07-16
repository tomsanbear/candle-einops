use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

use candle_core::{Device, Tensor};
use candle_einops::einops;
use candle_einops_parity_runner::{
    NormalizedResponse, Operation, OracleClient, OracleRequest, OracleValue, ParityConfig,
    PatternId,
};
use proptest::collection;
use proptest::prelude::*;
use proptest::test_runner::{TestCaseError, TestRunner};

#[derive(Clone, Copy, Debug)]
enum RepeatTemplate {
    NewAxis,
    Leading,
    Trailing,
    Grouped,
    Ellipsis,
}

impl RepeatTemplate {
    fn stable_id(self) -> &'static str {
        match self {
            Self::NewAxis => "repeat/new-axis-v1",
            Self::Leading => "repeat/leading-axis-v1",
            Self::Trailing => "repeat/trailing-axis-v1",
            Self::Grouped => "repeat/grouped-axis-v1",
            Self::Ellipsis => "repeat/ellipsis-axis-v1",
        }
    }

    fn python_pattern(self) -> &'static str {
        match self {
            Self::NewAxis => "a b -> a copies b",
            Self::Leading => "a b -> copies a b",
            Self::Trailing => "a b -> a b copies",
            Self::Grouped => "a b -> (a copies) b",
            Self::Ellipsis => "a ... -> a ... copies",
        }
    }
}

#[derive(Clone, Debug)]
struct RepeatCase {
    template: RepeatTemplate,
    shape: Vec<usize>,
    copies: usize,
    values: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
enum Reduction {
    Sum,
    Mean,
    Min,
    Max,
    Product,
}

impl Reduction {
    fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Min => "min",
            Self::Max => "max",
            Self::Product => "prod",
        }
    }

    fn exact(self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }
}

#[derive(Clone, Copy, Debug)]
enum ReductionLayout {
    One,
    Consecutive,
    All,
    Ellipsis,
    Grouped,
}

impl ReductionLayout {
    fn name(self) -> &'static str {
        match self {
            Self::One => "one",
            Self::Consecutive => "consecutive",
            Self::All => "all",
            Self::Ellipsis => "ellipsis",
            Self::Grouped => "grouped",
        }
    }
}

#[derive(Clone, Debug)]
struct ReductionCase {
    layout: ReductionLayout,
    reduction: Reduction,
    a: usize,
    b: usize,
    c: usize,
    values: Vec<f32>,
}

impl ReductionCase {
    fn shape(&self) -> Vec<usize> {
        match self.layout {
            ReductionLayout::Grouped => vec![self.a, self.b * 2],
            _ => vec![self.a, self.b, self.c],
        }
    }

    fn reduction_len(&self) -> usize {
        match self.layout {
            ReductionLayout::One | ReductionLayout::Grouped => self.b,
            ReductionLayout::Consecutive | ReductionLayout::Ellipsis => self.b * self.c,
            ReductionLayout::All => self.a * self.b * self.c,
        }
    }

    fn stable_id(&self) -> String {
        format!("reduce/{}-{}-v1", self.reduction.name(), self.layout.name())
    }

    fn python_pattern(&self) -> &'static str {
        match self.layout {
            ReductionLayout::One => "a b c -> a c",
            ReductionLayout::Consecutive => "a b c -> a",
            ReductionLayout::All => "a b c ->",
            ReductionLayout::Ellipsis => "a ... -> a",
            ReductionLayout::Grouped => "a (b c) -> a c",
        }
    }
}

fn max_extent(max_elements: usize) -> usize {
    let mut extent = 1usize;
    while (extent + 1).pow(3) <= max_elements {
        extent += 1;
    }
    extent.min(4)
}

fn repeat_strategy(template: RepeatTemplate, max_elements: usize) -> BoxedStrategy<RepeatCase> {
    let extent = max_extent(max_elements);
    (0..=extent, 0..=extent, 0..=extent)
        .prop_flat_map(move |(a, b, copies)| {
            let shape = match template {
                RepeatTemplate::Ellipsis => vec![a, b, extent],
                _ => vec![a, b],
            };
            let len: usize = shape.iter().product();
            collection::vec(-8i16..=8, len).prop_map(move |values| RepeatCase {
                template,
                shape: shape.clone(),
                copies,
                values: values.into_iter().map(f32::from).collect(),
            })
        })
        .boxed()
}

fn reduction_strategy(
    layout: ReductionLayout,
    reduction: Reduction,
    max_elements: usize,
) -> BoxedStrategy<ReductionCase> {
    let extent = max_extent(max_elements);
    let extents = if matches!(reduction, Reduction::Sum | Reduction::Product) {
        (0..=extent, 0..=extent, 0..=extent).boxed()
    } else {
        (1..=extent, 1..=extent, 1..=extent).boxed()
    };
    extents
        .prop_flat_map(move |(a, b, c)| {
            let len = match layout {
                ReductionLayout::Grouped => a * b * 2,
                _ => a * b * c,
            };
            collection::vec(-2i16..=2, len).prop_map(move |values| ReductionCase {
                layout,
                reduction,
                a,
                b,
                c,
                values: values.into_iter().map(f32::from).collect(),
            })
        })
        .boxed()
}

fn oracle_values(values: &[f32]) -> Vec<OracleValue> {
    values
        .iter()
        .map(|value| OracleValue::from(f64::from(*value)))
        .collect()
}

fn repeat_request(case_id: String, case: &RepeatCase) -> OracleRequest {
    OracleRequest {
        case_id,
        pattern_id: PatternId::new(case.template.stable_id()).expect("stable repeat id"),
        operation: Operation::Repeat,
        pattern: case.template.python_pattern().to_string(),
        reduction: None,
        dtype: "float32".to_string(),
        shape: case.shape.clone(),
        values: oracle_values(&case.values),
        axes_lengths: BTreeMap::from([("copies".to_string(), case.copies)]),
    }
}

fn reduction_request(case_id: String, case: &ReductionCase) -> OracleRequest {
    let axes_lengths = matches!(case.layout, ReductionLayout::Grouped)
        .then(|| BTreeMap::from([("c".to_string(), 2)]))
        .unwrap_or_default();
    OracleRequest {
        case_id,
        pattern_id: PatternId::new(case.stable_id()).expect("stable reduction id"),
        operation: Operation::Reduce,
        pattern: case.python_pattern().to_string(),
        reduction: Some(case.reduction.name().to_string()),
        dtype: "float32".to_string(),
        shape: case.shape(),
        values: oracle_values(&case.values),
        axes_lengths,
    }
}

fn rust_repeat(case: &RepeatCase, input: &Tensor) -> candle_core::Result<Tensor> {
    let copies = case.copies;
    match case.template {
        RepeatTemplate::NewAxis => einops!("a b -> a {copies} b", input),
        RepeatTemplate::Leading => einops!("a b -> {copies} a b", input),
        RepeatTemplate::Trailing => einops!("a b -> a b {copies}", input),
        RepeatTemplate::Grouped => einops!("a b -> (a {copies}) b", input),
        RepeatTemplate::Ellipsis => einops!("a .. -> a .. {copies}", input),
    }
}

fn rust_reduce(case: &ReductionCase, input: &Tensor) -> candle_core::Result<Tensor> {
    match (case.reduction, case.layout) {
        (Reduction::Sum, ReductionLayout::One) => einops!("a sum(b) c -> a c", input),
        (Reduction::Sum, ReductionLayout::Consecutive) => einops!("a sum(b c) -> a", input),
        (Reduction::Sum, ReductionLayout::All) => einops!("sum(..) ->", input),
        (Reduction::Sum, ReductionLayout::Ellipsis) => einops!("a sum(..) -> a", input),
        (Reduction::Sum, ReductionLayout::Grouped) => {
            einops!("a (sum(b) c:2) -> a c", input)
        }
        (Reduction::Mean, ReductionLayout::One) => einops!("a mean(b) c -> a c", input),
        (Reduction::Mean, ReductionLayout::Consecutive) => einops!("a mean(b c) -> a", input),
        (Reduction::Mean, ReductionLayout::All) => einops!("mean(..) ->", input),
        (Reduction::Mean, ReductionLayout::Ellipsis) => einops!("a mean(..) -> a", input),
        (Reduction::Mean, ReductionLayout::Grouped) => {
            einops!("a (mean(b) c:2) -> a c", input)
        }
        (Reduction::Min, ReductionLayout::One) => einops!("a min(b) c -> a c", input),
        (Reduction::Min, ReductionLayout::Consecutive) => einops!("a min(b c) -> a", input),
        (Reduction::Min, ReductionLayout::All) => einops!("min(..) ->", input),
        (Reduction::Min, ReductionLayout::Ellipsis) => einops!("a min(..) -> a", input),
        (Reduction::Min, ReductionLayout::Grouped) => {
            einops!("a (min(b) c:2) -> a c", input)
        }
        (Reduction::Max, ReductionLayout::One) => einops!("a max(b) c -> a c", input),
        (Reduction::Max, ReductionLayout::Consecutive) => einops!("a max(b c) -> a", input),
        (Reduction::Max, ReductionLayout::All) => einops!("max(..) ->", input),
        (Reduction::Max, ReductionLayout::Ellipsis) => einops!("a max(..) -> a", input),
        (Reduction::Max, ReductionLayout::Grouped) => {
            einops!("a (max(b) c:2) -> a c", input)
        }
        (Reduction::Product, ReductionLayout::One) => einops!("a prod(b) c -> a c", input),
        (Reduction::Product, ReductionLayout::Consecutive) => einops!("a prod(b c) -> a", input),
        (Reduction::Product, ReductionLayout::All) => einops!("prod(..) ->", input),
        (Reduction::Product, ReductionLayout::Ellipsis) => einops!("a prod(..) -> a", input),
        (Reduction::Product, ReductionLayout::Grouped) => {
            einops!("a (prod(b) c:2) -> a c", input)
        }
    }
}

fn success(
    response: NormalizedResponse,
    request: &OracleRequest,
) -> Result<(Vec<usize>, Vec<f32>), TestCaseError> {
    let NormalizedResponse::Success(response) = response else {
        return Err(TestCaseError::fail(format!(
            "Python oracle rejected valid case; replay={} ",
            serde_json::to_string(request).expect("request serializes")
        )));
    };
    let mut values = Vec::with_capacity(response.values.len());
    for value in response.values {
        match value {
            OracleValue::Number(value) => values.push(value as f32),
            OracleValue::Symbol(value) => {
                return Err(TestCaseError::fail(format!(
                    "unexpected symbolic oracle value `{value}`"
                )));
            }
        }
    }
    Ok((response.shape, values))
}

fn actual_values(tensor: &Tensor) -> Result<Vec<f32>, TestCaseError> {
    tensor
        .flatten_all()
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|error| TestCaseError::fail(error.to_string()))
}

fn with_replay(request: &OracleRequest, error: impl std::fmt::Display) -> TestCaseError {
    TestCaseError::fail(format!(
        "{error}; minimized replay={}",
        serde_json::to_string(request).expect("request serializes")
    ))
}

fn test_runner(config: ParityConfig) -> TestRunner {
    let mut proptest = config.proptest_config();
    // The bridge prints the fully minimized JSON request on failure, which is
    // directly replayable by the runner and more useful than a source seed file.
    proptest.failure_persistence = None;
    TestRunner::new(proptest)
}

fn assert_values(
    actual: &[f32],
    expected: &[f32],
    exact: bool,
    reduction_len: usize,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(actual.len(), expected.len());
    let scale = reduction_len.max(1) as f32;
    let (relative, absolute) = if exact {
        (0., 0.)
    } else {
        (f32::EPSILON * scale * 16., f32::EPSILON * scale * 8.)
    };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = absolute + relative * expected.abs();
        prop_assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
    Ok(())
}

#[test]
fn randomized_repeat_matches_locked_python_einops() {
    let config = ParityConfig::from_env().expect("valid deterministic parity config");
    let mut runner = test_runner(config);
    let oracle = RefCell::new(OracleClient::spawn_uv().expect("locked Python oracle starts"));
    let sequence = Cell::new(0u64);

    for template in [
        RepeatTemplate::NewAxis,
        RepeatTemplate::Leading,
        RepeatTemplate::Trailing,
        RepeatTemplate::Grouped,
        RepeatTemplate::Ellipsis,
    ] {
        runner
            .run(&repeat_strategy(template, config.max_elements), |case| {
                let current = sequence.get();
                sequence.set(current + 1);
                let request = repeat_request(format!("repeat-{current}"), &case);
                let response = oracle
                    .borrow_mut()
                    .evaluate(std::slice::from_ref(&request))
                    .map_err(|error| TestCaseError::fail(error.to_string()))?
                    .pop()
                    .expect("one response");
                let (expected_shape, expected_values) = success(response, &request)?;
                let input =
                    Tensor::from_vec(case.values.clone(), case.shape.as_slice(), &Device::Cpu)
                        .map_err(|error| TestCaseError::fail(error.to_string()))?;
                let actual =
                    rust_repeat(&case, &input).map_err(|error| with_replay(&request, error))?;
                if actual.dims() != expected_shape {
                    return Err(with_replay(
                        &request,
                        format_args!(
                            "shape mismatch: Rust={:?}, Python={expected_shape:?}",
                            actual.dims()
                        ),
                    ));
                }
                assert_values(&actual_values(&actual)?, &expected_values, true, 1)
                    .map_err(|error| with_replay(&request, error))
            })
            .expect("repeat parity; rerun with CANDLE_EINOPS_PARITY_SEED to replay");
    }
}

#[test]
fn randomized_reductions_match_locked_python_einops() {
    let config = ParityConfig::from_env().expect("valid deterministic parity config");
    let mut runner = test_runner(config);
    let oracle = RefCell::new(OracleClient::spawn_uv().expect("locked Python oracle starts"));
    let sequence = Cell::new(0u64);

    for layout in [
        ReductionLayout::One,
        ReductionLayout::Consecutive,
        ReductionLayout::All,
        ReductionLayout::Ellipsis,
        ReductionLayout::Grouped,
    ] {
        for reduction in [
            Reduction::Sum,
            Reduction::Mean,
            Reduction::Min,
            Reduction::Max,
            Reduction::Product,
        ] {
            runner
                .run(
                    &reduction_strategy(layout, reduction, config.max_elements),
                    |case| {
                        let current = sequence.get();
                        sequence.set(current + 1);
                        let request = reduction_request(format!("reduce-{current}"), &case);
                        let response = oracle
                            .borrow_mut()
                            .evaluate(std::slice::from_ref(&request))
                            .map_err(|error| TestCaseError::fail(error.to_string()))?
                            .pop()
                            .expect("one response");
                        let (expected_shape, expected_values) = success(response, &request)?;
                        let input = Tensor::from_vec(
                            case.values.clone(),
                            case.shape().as_slice(),
                            &Device::Cpu,
                        )
                        .map_err(|error| TestCaseError::fail(error.to_string()))?;
                        let actual = rust_reduce(&case, &input)
                            .map_err(|error| with_replay(&request, error))?;
                        if actual.dims() != expected_shape {
                            return Err(with_replay(
                                &request,
                                format_args!(
                                    "shape mismatch: Rust={:?}, Python={expected_shape:?}",
                                    actual.dims()
                                ),
                            ));
                        }
                        assert_values(
                            &actual_values(&actual)?,
                            &expected_values,
                            case.reduction.exact(),
                            case.reduction_len(),
                        )
                        .map_err(|error| with_replay(&request, error))
                    },
                )
                .expect("reduction parity; rerun with CANDLE_EINOPS_PARITY_SEED to replay");
        }
    }
}

#[test]
fn empty_min_and_max_are_rejected_by_both_contracts() {
    for reduction in [Reduction::Min, Reduction::Max] {
        let case = ReductionCase {
            layout: ReductionLayout::One,
            reduction,
            a: 2,
            b: 0,
            c: 3,
            values: Vec::new(),
        };
        let request = reduction_request(format!("empty-{}", reduction.name()), &case);
        let mut oracle = OracleClient::spawn_uv().expect("locked Python oracle starts");
        let python = oracle
            .evaluate(std::slice::from_ref(&request))
            .expect("normalized Python response")
            .pop()
            .expect("one response");
        assert!(matches!(python, NormalizedResponse::Error(_)));

        let input = Tensor::zeros(case.shape(), candle_core::DType::F32, &Device::Cpu)
            .expect("empty input");
        assert!(rust_reduce(&case, &input).is_err());
    }
}

// Empty mean intentionally is not a parity case: locked NumPy returns NaNs with
// a RuntimeWarning, while that backend policy is not part of the einops contract.
