use candle_core::{DType, Device, Result, Shape, Tensor, Var};
use candle_einops::einsum;

fn flat_f64(tensor: &Tensor) -> Result<Vec<f64>> {
    tensor.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f64, context: &str) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{context}: shape");
    let actual = flat_f64(actual)?;
    let expected = flat_f64(expected)?;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        let allowed = tolerance * (1. + expected.abs());
        assert!(
            (actual - expected).abs() <= allowed,
            "{context}[{index}]: actual={actual}, expected={expected}, allowed={allowed}"
        );
    }
    Ok(())
}

#[test]
fn floating_forward_matrix_follows_candle_matmul_support() -> Result<()> {
    for (dtype, tolerance) in [
        (DType::BF16, 2e-2),
        (DType::F16, 2e-3),
        (DType::F32, 1e-5),
        (DType::F64, 1e-10),
    ] {
        let left = Tensor::new(&[[1_f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?.to_dtype(dtype)?;
        let right =
            Tensor::new(&[[1_f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?.to_dtype(dtype)?;

        let transposed = einsum!("row column -> column row", &left)?;
        assert_eq!(transposed.dtype(), dtype);
        assert!(transposed.device().same_device(&Device::Cpu));
        assert_close(
            &transposed,
            &left.transpose(0, 1)?,
            tolerance,
            &format!("{dtype:?} unary"),
        )?;

        match left.matmul(&right) {
            Ok(expected) => {
                let actual = einsum!("row inner, inner column -> row column", &left, &right)?;
                assert_eq!(actual.dtype(), dtype);
                assert!(actual.device().same_device(&Device::Cpu));
                assert_close(&actual, &expected, tolerance, &format!("{dtype:?} matmul"))?;
            }
            Err(_) => {
                let error = einsum!("row inner, inner column -> row column", &left, &right)
                    .expect_err("einsum must not cast around Candle dtype rejection");
                let message = error.to_string();
                assert!(
                    message.contains("einsum binary B/M/K/N matmul"),
                    "{message}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn integer_permutations_are_supported_and_contractions_are_rejected() -> Result<()> {
    for dtype in [DType::U8, DType::U32, DType::I64] {
        let matrix = Tensor::new(&[[1_i64, 2], [3, 4]], &Device::Cpu)?.to_dtype(dtype)?;
        let transposed = einsum!("row column -> column row", &matrix)?;
        assert_eq!(transposed.dtype(), dtype);
        assert_eq!(flat_f64(&transposed)?, [1., 3., 2., 4.]);

        let vector = Tensor::new(&[1_i64, 2], &Device::Cpu)?.to_dtype(dtype)?;
        let direct = vector.reshape((1, 2))?.matmul(&vector.reshape((2, 1))?);
        assert!(
            direct.is_err(),
            "Candle unexpectedly added {dtype:?} matmul support"
        );
        let error = einsum!("feature, feature ->", &vector, &vector)
            .expect_err("integer contraction must preserve Candle's rejection");
        let message = error.to_string();
        assert!(
            message.contains("einsum binary B/M/K/N matmul"),
            "{message}"
        );
    }
    Ok(())
}

#[test]
fn non_contraction_multiplication_follows_candle_dtype_support() -> Result<()> {
    for dtype in [DType::U8, DType::U32, DType::I64, DType::BF16] {
        let left = Tensor::new(&[2_i64, 3], &Device::Cpu)?.to_dtype(dtype)?;
        let right = Tensor::new(&[5_i64, 7], &Device::Cpu)?.to_dtype(dtype)?;
        let scalar = Tensor::new(4_i64, &Device::Cpu)?.to_dtype(dtype)?;

        let elementwise = einsum!("feature, feature -> feature", &left, &right)?;
        let outer = einsum!("row, column -> row column", &left, &right)?;
        let scaled = einsum!(", feature -> feature", &scalar, &left)?;

        assert_eq!(elementwise.dtype(), dtype);
        assert_eq!(outer.dtype(), dtype);
        assert_eq!(scaled.dtype(), dtype);
        assert_eq!(flat_f64(&elementwise)?, [10., 21.]);
        assert_eq!(flat_f64(&outer)?, [10., 14., 15., 21.]);
        assert_eq!(flat_f64(&scaled)?, [8., 12.]);
    }
    Ok(())
}

#[test]
fn nan_and_infinity_propagate_without_sanitizing_or_casting() -> Result<()> {
    let finite = Tensor::new(&[1_f64, 2.], &Device::Cpu)?;
    let positive = Tensor::new(&[f64::INFINITY, 3.], &Device::Cpu)?;
    let negative = Tensor::new(&[f64::NEG_INFINITY, 3.], &Device::Cpu)?;
    let nan = Tensor::new(&[f64::NAN, 3.], &Device::Cpu)?;

    assert!(
        einsum!("feature, feature ->", &positive, &finite)?
            .to_scalar::<f64>()?
            .is_infinite()
    );
    assert!(
        einsum!("feature, feature ->", &negative, &finite)?
            .to_scalar::<f64>()?
            .is_infinite()
    );
    assert!(
        einsum!("feature, feature ->", &nan, &finite)?
            .to_scalar::<f64>()?
            .is_nan()
    );
    assert!(einsum!("feature -> feature", &nan)?.to_vec1::<f64>()?[0].is_nan());
    Ok(())
}

struct GradientInput<'a> {
    shape: &'a [usize],
    values: &'a [f64],
}

fn check_gradients(
    name: &str,
    inputs: &[GradientInput<'_>],
    einsum_loss: impl Fn(&[Tensor]) -> Result<Tensor>,
    direct_loss: impl Fn(&[Tensor]) -> Result<Tensor>,
) -> Result<()> {
    let einsum_vars = inputs
        .iter()
        .map(|input| {
            Var::from_vec(
                input.values.to_vec(),
                Shape::from_dims(input.shape),
                &Device::Cpu,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let direct_vars = inputs
        .iter()
        .map(|input| {
            Var::from_vec(
                input.values.to_vec(),
                Shape::from_dims(input.shape),
                &Device::Cpu,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let einsum_tensors = einsum_vars
        .iter()
        .map(|variable| variable.as_tensor().clone())
        .collect::<Vec<_>>();
    let direct_tensors = direct_vars
        .iter()
        .map(|variable| variable.as_tensor().clone())
        .collect::<Vec<_>>();

    let einsum_value = einsum_loss(&einsum_tensors)?;
    let direct_value = direct_loss(&direct_tensors)?;
    assert_close(
        &einsum_value,
        &direct_value,
        1e-10,
        &format!("{name} forward"),
    )?;
    let einsum_gradients = einsum_value.backward()?;
    let direct_gradients = direct_value.backward()?;

    for (operand, (einsum_var, direct_var)) in einsum_vars.iter().zip(&direct_vars).enumerate() {
        let einsum_gradient = einsum_gradients
            .get(einsum_var.as_tensor())
            .unwrap_or_else(|| panic!("{name}: missing einsum gradient for operand {operand}"));
        let direct_gradient = direct_gradients
            .get(direct_var.as_tensor())
            .unwrap_or_else(|| panic!("{name}: missing direct gradient for operand {operand}"));
        assert_close(
            einsum_gradient,
            direct_gradient,
            1e-9,
            &format!("{name} operand {operand} analytic"),
        )?;

        let analytic = flat_f64(einsum_gradient)?;
        for element in 0..inputs[operand].values.len() {
            let epsilon = 1e-5;
            let mut plus = inputs
                .iter()
                .map(|input| input.values.to_vec())
                .collect::<Vec<_>>();
            let mut minus = plus.clone();
            plus[operand][element] += epsilon;
            minus[operand][element] -= epsilon;
            let plus_tensors = make_tensors(inputs, &plus)?;
            let minus_tensors = make_tensors(inputs, &minus)?;
            let plus_value = einsum_loss(&plus_tensors)?.to_scalar::<f64>()?;
            let minus_value = einsum_loss(&minus_tensors)?.to_scalar::<f64>()?;
            let finite_difference = (plus_value - minus_value) / (2. * epsilon);
            let allowed = 2e-5 * (1. + finite_difference.abs());
            assert!(
                (analytic[element] - finite_difference).abs() <= allowed,
                "{name} operand {operand}[{element}]: analytic={}, finite={finite_difference}, allowed={allowed}",
                analytic[element]
            );
        }
    }
    Ok(())
}

fn make_tensors(inputs: &[GradientInput<'_>], values: &[Vec<f64>]) -> Result<Vec<Tensor>> {
    inputs
        .iter()
        .zip(values)
        .map(|(input, values)| {
            Tensor::from_vec(values.clone(), Shape::from_dims(input.shape), &Device::Cpu)
        })
        .collect()
}

#[test]
fn unary_and_ellipsis_gradients_match_direct_candle_and_finite_differences() -> Result<()> {
    let unary_values = [1., -2., 3., 4., -5., 6.];
    check_gradients(
        "unary reduction",
        &[GradientInput {
            shape: &[2, 3],
            values: &unary_values,
        }],
        |operands| {
            let weights = Tensor::new(&[0.5_f64, -2., 3.], &Device::Cpu)?;
            einsum!("row feature -> feature", &operands[0])?
                .mul(&weights)?
                .sum_all()
        },
        |operands| {
            let weights = Tensor::new(&[0.5_f64, -2., 3.], &Device::Cpu)?;
            operands[0].sum(0)?.mul(&weights)?.sum_all()
        },
    )?;

    let ellipsis_values = [1., 2., 3., 4., 5., 6., -1., -2., -3., -4., -5., -6.];
    check_gradients(
        "ellipsis reduction",
        &[GradientInput {
            shape: &[2, 2, 3],
            values: &ellipsis_values,
        }],
        |operands| {
            let weights = Tensor::new(&[1.5_f64, -0.5, 2.], &Device::Cpu)?;
            einsum!(".. feature -> feature", &operands[0])?
                .mul(&weights)?
                .sum_all()
        },
        |operands| {
            let weights = Tensor::new(&[1.5_f64, -0.5, 2.], &Device::Cpu)?;
            operands[0].sum(0)?.sum(0)?.mul(&weights)?.sum_all()
        },
    )
}

#[test]
fn broadcast_and_zero_sensitive_gradients_match_direct_and_finite_differences() -> Result<()> {
    let left = [1., -2., 3., 4., -5., 6.];
    let right = [0.5, -1.5, 2.];
    check_gradients(
        "binary retained broadcast",
        &[
            GradientInput {
                shape: &[2, 3],
                values: &left,
            },
            GradientInput {
                shape: &[1, 3],
                values: &right,
            },
        ],
        |operands| {
            let weights = Tensor::new(&[[1_f64, 2., -1.], [0.5, -3., 4.]], &Device::Cpu)?;
            einsum!(
                "batch feature, batch feature -> batch feature",
                &operands[0],
                &operands[1]
            )?
            .mul(&weights)?
            .sum_all()
        },
        |operands| {
            let weights = Tensor::new(&[[1_f64, 2., -1.], [0.5, -3., 4.]], &Device::Cpu)?;
            operands[0]
                .broadcast_mul(&operands[1])?
                .mul(&weights)?
                .sum_all()
        },
    )?;

    let zero_left = [0., 2., -3.];
    let zero_right = [4., 0., -2.];
    check_gradients(
        "zero-sensitive dot",
        &[
            GradientInput {
                shape: &[3],
                values: &zero_left,
            },
            GradientInput {
                shape: &[3],
                values: &zero_right,
            },
        ],
        |operands| einsum!("feature, feature ->", &operands[0], &operands[1]),
        |operands| operands[0].mul(&operands[1])?.sum_all(),
    )?;

    let empty_left = Tensor::zeros((2, 0), DType::F64, &Device::Cpu)?;
    let empty_right = Tensor::zeros((0, 3), DType::F64, &Device::Cpu)?;
    let empty = einsum!(
        "row inner, inner column -> row column",
        &empty_left,
        &empty_right
    )?;
    assert_eq!(empty.dims(), &[2, 3]);
    assert_eq!(flat_f64(&empty)?, [0.; 6]);
    Ok(())
}

#[test]
fn interleaved_multiple_diagonal_gradients_match_index_selection_and_finite_differences()
-> Result<()> {
    let values = (0..36)
        .map(|value| f64::from(value) / 7. - 2.)
        .collect::<Vec<_>>();
    check_gradients(
        "interleaved i j i j diagonal",
        &[GradientInput {
            shape: &[2, 3, 2, 3],
            values: &values,
        }],
        |operands| {
            let weights = Tensor::new(&[[1_f64, -2., 0.5], [3., -1., 4.]], &Device::Cpu)?;
            einsum!("i j i j -> i j", &operands[0])?
                .mul(&weights)?
                .sum_all()
        },
        |operands| {
            let indices = Tensor::new(&[0_u32, 7, 14, 21, 28, 35], &Device::Cpu)?;
            let weights = Tensor::new(&[[1_f64, -2., 0.5], [3., -1., 4.]], &Device::Cpu)?;
            operands[0]
                .flatten_all()?
                .index_select(&indices, 0)?
                .reshape((2, 3))?
                .mul(&weights)?
                .sum_all()
        },
    )
}

#[test]
fn nary_gradients_match_direct_candle_and_finite_differences() -> Result<()> {
    let left = [1., -2., 3., 4., 0.5, -1.];
    let middle = [2., -1., 0.5, 3., -2., 4.];
    let tail = [1.5, -0.25];
    check_gradients(
        "n-ary contraction",
        &[
            GradientInput {
                shape: &[2, 3],
                values: &left,
            },
            GradientInput {
                shape: &[3, 2],
                values: &middle,
            },
            GradientInput {
                shape: &[2],
                values: &tail,
            },
        ],
        |operands| {
            let weights = Tensor::new(&[0.75_f64, -2.], &Device::Cpu)?;
            einsum!(
                "row inner, inner column, column -> row",
                &operands[0],
                &operands[1],
                &operands[2]
            )?
            .mul(&weights)?
            .sum_all()
        },
        |operands| {
            let weights = Tensor::new(&[0.75_f64, -2.], &Device::Cpu)?;
            operands[0]
                .matmul(&operands[1])?
                .broadcast_mul(&operands[2].unsqueeze(0)?)?
                .sum(1)?
                .mul(&weights)?
                .sum_all()
        },
    )
}

fn accelerator_smoke(name: &str, device: Result<Device>) -> Result<()> {
    let Ok(device) = device else {
        eprintln!("skipping {name} einsum smoke: backend unavailable");
        return Ok(());
    };
    let left = Tensor::new(&[[1_f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?.to_device(&device)?;
    let right =
        Tensor::new(&[[1_f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?.to_device(&device)?;
    let output =
        einsum!("row inner, inner column -> row column", &left, &right)?.to_device(&Device::Cpu)?;
    let expected = Tensor::new(&[[22_f32, 28.], [49., 64.]], &Device::Cpu)?;
    assert_close(&output, &expected, 1e-4, &format!("{name} smoke"))
}

#[test]
fn available_accelerators_match_cpu_forward_values() -> Result<()> {
    accelerator_smoke("Metal", Device::new_metal(0))?;
    accelerator_smoke("CUDA", Device::new_cuda(0))
}
