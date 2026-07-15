//! Independent host reference for the frozen explicit-label einsum contract.
//!
//! This deliberately uses only `Vec<f64>` indexing. It does not call Candle or
//! any production contraction, reshape, or matrix-multiplication helper.

use std::collections::HashMap;

type OracleResult<T> = Result<T, String>;

#[derive(Clone, Debug, PartialEq)]
struct HostTensor {
    shape: Vec<usize>,
    values: Vec<f64>,
}

impl HostTensor {
    fn new(shape: &[usize], values: &[f64]) -> OracleResult<Self> {
        let expected = element_count(shape)?;
        if values.len() != expected {
            return Err(format!(
                "shape {shape:?} requires {expected} values, got {}",
                values.len()
            ));
        }
        Ok(Self {
            shape: shape.to_vec(),
            values: values.to_vec(),
        })
    }
}

#[derive(Debug)]
struct Equation<'a> {
    inputs: Vec<Vec<&'a str>>,
    output: Vec<&'a str>,
}

fn element_count(shape: &[usize]) -> OracleResult<usize> {
    shape.iter().try_fold(1_usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| format!("shape element count overflows: {shape:?}"))
    })
}

fn parse_equation(equation: &str) -> OracleResult<Equation<'_>> {
    let Some((input_text, output_text)) = equation.split_once("->") else {
        return Err("equation requires an explicit `->`".into());
    };
    if output_text.contains("->") {
        return Err("equation must contain exactly one `->`".into());
    }

    let inputs = input_text
        .split(',')
        .map(|axes| parse_axes(axes, true))
        .collect::<OracleResult<Vec<_>>>()?;
    let output = parse_axes(output_text, false)?;

    let mut known = Vec::new();
    for axes in &inputs {
        let mut local = Vec::new();
        for &label in axes {
            if local.contains(&label) {
                return Err(format!(
                    "repeated input label `{label}` is reserved for diagonal support"
                ));
            }
            local.push(label);
            if !known.contains(&label) {
                known.push(label);
            }
        }
    }

    let mut output_seen = Vec::new();
    for &label in &output {
        if output_seen.contains(&label) {
            return Err(format!("duplicate output label `{label}`"));
        }
        if !known.contains(&label) {
            return Err(format!("output label `{label}` does not occur in an input"));
        }
        output_seen.push(label);
    }

    Ok(Equation { inputs, output })
}

fn parse_axes(text: &str, input: bool) -> OracleResult<Vec<&str>> {
    text.split_whitespace()
        .map(|label| {
            if label == ".." {
                return Err("`..` is reserved for ellipsis support".into());
            }
            let mut chars = label.chars();
            let valid_start = chars
                .next()
                .is_some_and(|character| character == '_' || character.is_alphabetic());
            if !valid_start
                || !chars.all(|character| character == '_' || character.is_alphanumeric())
            {
                return Err(format!("invalid axis label `{label}`"));
            }
            Ok(label)
        })
        .collect::<OracleResult<Vec<_>>>()
        .and_then(|axes| {
            if !input && text.contains(',') {
                Err("output must be a single whitespace-delimited axis list".into())
            } else {
                Ok(axes)
            }
        })
}

fn evaluate(equation: &str, operands: &[HostTensor]) -> OracleResult<HostTensor> {
    let equation = parse_equation(equation)?;
    if equation.inputs.len() != operands.len() {
        return Err(format!(
            "equation has {} inputs but received {} operands",
            equation.inputs.len(),
            operands.len()
        ));
    }

    let mut dimensions = HashMap::new();
    let mut input_order = Vec::new();
    for (operand_index, (labels, operand)) in equation.inputs.iter().zip(operands).enumerate() {
        if labels.len() != operand.shape.len() {
            return Err(format!(
                "operand {operand_index} has rank {}, expected {}",
                operand.shape.len(),
                labels.len()
            ));
        }
        for (&label, &extent) in labels.iter().zip(&operand.shape) {
            if !input_order.contains(&label) {
                input_order.push(label);
            }
            match dimensions.get(label).copied() {
                None => {
                    dimensions.insert(label, extent);
                }
                Some(previous) if previous == extent || extent == 1 => {}
                Some(1) => {
                    dimensions.insert(label, extent);
                }
                Some(previous) => {
                    return Err(format!(
                        "label `{label}` cannot broadcast extents {previous} and {extent}"
                    ));
                }
            }
        }
    }

    let output_shape = equation
        .output
        .iter()
        .map(|label| dimensions[label])
        .collect::<Vec<_>>();
    let contracted = input_order
        .into_iter()
        .filter(|label| !equation.output.contains(label))
        .collect::<Vec<_>>();
    let contracted_shape = contracted
        .iter()
        .map(|label| dimensions[label])
        .collect::<Vec<_>>();

    let output_count = element_count(&output_shape)?;
    let contraction_count = element_count(&contracted_shape)?;
    let mut values = Vec::with_capacity(output_count);
    for output_flat in 0..output_count {
        let mut coordinates = HashMap::new();
        for (&label, coordinate) in equation
            .output
            .iter()
            .zip(unravel(output_flat, &output_shape))
        {
            coordinates.insert(label, coordinate);
        }

        let mut sum = 0_f64;
        for contraction_flat in 0..contraction_count {
            for (&label, coordinate) in contracted
                .iter()
                .zip(unravel(contraction_flat, &contracted_shape))
            {
                coordinates.insert(label, coordinate);
            }

            let mut product = 1_f64;
            for (labels, operand) in equation.inputs.iter().zip(operands) {
                let local_coordinates = labels
                    .iter()
                    .zip(&operand.shape)
                    .map(
                        |(label, &extent)| {
                            if extent == 1 { 0 } else { coordinates[label] }
                        },
                    )
                    .collect::<Vec<_>>();
                product *= operand.values[flatten(&local_coordinates, &operand.shape)];
            }
            sum += product;
        }
        values.push(sum);
    }

    HostTensor::new(&output_shape, &values)
}

fn evaluate_once<'a>(
    equation: &str,
    operand_expressions: Vec<Box<dyn FnOnce() -> HostTensor + 'a>>,
) -> OracleResult<HostTensor> {
    let operands = operand_expressions
        .into_iter()
        .map(|expression| expression())
        .collect::<Vec<_>>();
    evaluate(equation, &operands)
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for (coordinate, &extent) in coordinates.iter_mut().rev().zip(shape.iter().rev()) {
        *coordinate = flat % extent;
        flat /= extent;
    }
    coordinates
}

fn flatten(coordinates: &[usize], shape: &[usize]) -> usize {
    coordinates
        .iter()
        .zip(shape)
        .fold(0, |flat, (&coordinate, &extent)| flat * extent + coordinate)
}

struct Case {
    name: &'static str,
    equation: &'static str,
    operands: Vec<HostTensor>,
    expected: HostTensor,
}

fn tensor(shape: &[usize], values: &[f64]) -> HostTensor {
    HostTensor::new(shape, values).expect("valid checked-in host tensor")
}

fn corpus() -> Vec<Case> {
    vec![
        Case {
            name: "unary transpose",
            equation: "rows columns -> columns rows",
            operands: vec![tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.])],
            expected: tensor(&[3, 2], &[1., 4., 2., 5., 3., 6.]),
        },
        Case {
            name: "unary reduction",
            equation: "rows columns -> rows",
            operands: vec![tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.])],
            expected: tensor(&[2], &[6., 15.]),
        },
        Case {
            name: "dot product",
            equation: "feature, feature ->",
            operands: vec![tensor(&[3], &[1., 2., 3.]), tensor(&[3], &[4., 5., 6.])],
            expected: tensor(&[], &[32.]),
        },
        Case {
            name: "outer product",
            equation: "row, column -> row column",
            operands: vec![tensor(&[2], &[1., 2.]), tensor(&[3], &[10., 20., 30.])],
            expected: tensor(&[2, 3], &[10., 20., 30., 20., 40., 60.]),
        },
        Case {
            name: "retained-label Hadamard broadcast",
            equation: "batch feature, batch feature -> batch feature",
            operands: vec![
                tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
                tensor(&[1, 3], &[10., 20., 30.]),
            ],
            expected: tensor(&[2, 3], &[10., 40., 90., 40., 100., 180.]),
        },
        Case {
            name: "matrix vector product",
            equation: "row feature, feature -> row",
            operands: vec![
                tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
                tensor(&[3], &[1., 2., 3.]),
            ],
            expected: tensor(&[2], &[14., 32.]),
        },
        Case {
            name: "matrix multiplication",
            equation: "row inner, inner column -> row column",
            operands: vec![
                tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
                tensor(&[3, 2], &[1., 2., 3., 4., 5., 6.]),
            ],
            expected: tensor(&[2, 2], &[22., 28., 49., 64.]),
        },
        Case {
            name: "batched contraction with batch broadcast",
            equation: "batch row inner, batch inner column -> batch row column",
            operands: vec![
                tensor(
                    &[2, 2, 3],
                    &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                ),
                tensor(&[1, 3, 2], &[1., 2., 3., 4., 5., 6.]),
            ],
            expected: tensor(&[2, 2, 2], &[22., 28., 49., 64., 76., 100., 103., 136.]),
        },
        Case {
            name: "scalar identity",
            equation: " -> ",
            operands: vec![tensor(&[], &[7.])],
            expected: tensor(&[], &[7.]),
        },
        Case {
            name: "zero-sized transpose",
            equation: "empty feature -> feature empty",
            operands: vec![tensor(&[0, 2], &[])],
            expected: tensor(&[2, 0], &[]),
        },
        Case {
            name: "zero-sized reduction uses additive identity",
            equation: "row empty -> row",
            operands: vec![tensor(&[2, 0], &[])],
            expected: tensor(&[2], &[0., 0.]),
        },
    ]
}

#[test]
fn deterministic_corpus_matches_checked_in_expected_values() {
    for case in corpus() {
        let actual = evaluate(case.equation, &case.operands)
            .unwrap_or_else(|error| panic!("{} failed: {error}", case.name));
        assert_eq!(actual, case.expected, "{}", case.name);
    }
}

#[test]
fn rejects_invalid_equations_ranks_shapes_and_storage() {
    let matrix = tensor(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
    let vector = tensor(&[3], &[1., 2., 3.]);

    assert!(evaluate("row feature", std::slice::from_ref(&matrix)).is_err());
    assert!(evaluate("row -> row -> row", std::slice::from_ref(&vector)).is_err());
    assert!(evaluate("row, row -> row", std::slice::from_ref(&vector)).is_err());
    assert!(evaluate("row feature -> row", std::slice::from_ref(&vector)).is_err());
    assert!(evaluate("row -> missing", std::slice::from_ref(&vector)).is_err());
    assert!(evaluate("row -> row row", std::slice::from_ref(&vector)).is_err());
    assert!(evaluate("row row -> row", std::slice::from_ref(&matrix)).is_err());
    assert!(evaluate(".. row -> row", std::slice::from_ref(&matrix)).is_err());
    assert!(HostTensor::new(&[2, 2], &[1., 2., 3.]).is_err());

    let incompatible = tensor(&[4], &[1., 2., 3., 4.]);
    assert!(evaluate("feature, feature -> feature", &[vector, incompatible]).is_err());
}

#[test]
fn operand_expressions_are_evaluated_once_from_left_to_right() {
    use std::cell::RefCell;

    let evaluation_order = RefCell::new(Vec::new());
    let output = evaluate_once(
        "feature, feature ->",
        vec![
            Box::new(|| {
                evaluation_order.borrow_mut().push("left");
                tensor(&[2], &[2., 3.])
            }),
            Box::new(|| {
                evaluation_order.borrow_mut().push("right");
                tensor(&[2], &[5., 7.])
            }),
        ],
    )
    .expect("valid dot product");

    assert_eq!(evaluation_order.into_inner(), ["left", "right"]);
    assert_eq!(output, tensor(&[], &[31.]));
}
