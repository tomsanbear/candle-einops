use std::collections::BTreeMap;

use candle_core::{Device, Result, Tensor};
use candle_einops_parity_runner::{
    NormalizedResponse, Operation, OracleClient, OracleRequest, OracleValue, PatternId,
};

fn request(
    case_id: &str,
    pattern_id: &str,
    operation: Operation,
    pattern: &str,
    reduction: Option<&str>,
    shape: &[usize],
    values: &[f32],
    axes_lengths: BTreeMap<String, usize>,
) -> OracleRequest {
    OracleRequest {
        case_id: case_id.to_string(),
        pattern_id: PatternId::new(pattern_id).expect("stable pattern id"),
        operation,
        pattern: pattern.to_string(),
        reduction: reduction.map(str::to_string),
        dtype: "float32".to_string(),
        shape: shape.to_vec(),
        values: values
            .iter()
            .map(|value| OracleValue::from(f64::from(*value)))
            .collect(),
        axes_lengths,
    }
}

fn rust_repeat_contract(_input: &Tensor, _copies: usize) -> Result<Tensor> {
    unimplemented!("stable repeat-pattern dispatch is not implemented")
}

fn rust_reduce_contract(_input: &Tensor) -> Result<Tensor> {
    unimplemented!("stable reduce-pattern dispatch is not implemented")
}

#[test]
fn repeat_contract_uses_python_einops_as_the_oracle() {
    let values = [1f32, 2., 3., 4., 5., 6.];
    let mut axes_lengths = BTreeMap::new();
    axes_lengths.insert("copies".to_string(), 2);
    let request = request(
        "red-repeat",
        "repeat/new-axis-v1",
        Operation::Repeat,
        "rows columns -> rows copies columns",
        None,
        &[2, 3],
        &values,
        axes_lengths,
    );
    let mut oracle = OracleClient::spawn_uv().expect("locked Python oracle starts");
    let response = oracle.evaluate(&[request]).expect("oracle response");
    let NormalizedResponse::Success(expected) = &response[0] else {
        panic!("Python repeat failed: {:?}", response[0]);
    };
    assert_eq!(expected.shape, [2, 2, 3]);

    let input = Tensor::from_vec(values.to_vec(), (2, 3), &Device::Cpu).expect("input");
    let actual = rust_repeat_contract(&input, 2).expect("Rust repeat succeeds");
    assert_eq!(actual.dims(), expected.shape);
}

#[test]
fn reduce_contract_uses_python_einops_as_the_oracle() {
    let values = [1f32, 2., 3., 4., 5., 6.];
    let request = request(
        "red-reduce",
        "reduce/sum-one-v1",
        Operation::Reduce,
        "rows columns -> rows",
        Some("sum"),
        &[2, 3],
        &values,
        BTreeMap::new(),
    );
    let mut oracle = OracleClient::spawn_uv().expect("locked Python oracle starts");
    let response = oracle.evaluate(&[request]).expect("oracle response");
    let NormalizedResponse::Success(expected) = &response[0] else {
        panic!("Python reduction failed: {:?}", response[0]);
    };
    assert_eq!(expected.values, [6f64.into(), 15f64.into()]);

    let input = Tensor::from_vec(values.to_vec(), (2, 3), &Device::Cpu).expect("input");
    let actual = rust_reduce_contract(&input).expect("Rust reduction succeeds");
    assert_eq!(actual.dims(), expected.shape);
}
