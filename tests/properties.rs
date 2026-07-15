use std::panic::{AssertUnwindSafe, catch_unwind};

use candle_core::{DType, Device, Result, Tensor};
use candle_einops::{Backend, Operation, einops};

fn values_u32(tensor: &Tensor) -> Result<Vec<u32>> {
    tensor.flatten_all()?.to_vec1::<u32>()
}

fn values_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

#[test]
fn bounded_permutations_match_candle_and_invert() -> Result<()> {
    for a in 0..=3 {
        for b in 0..=3 {
            for c in 0..=3 {
                let len = a * b * c;
                let input = Tensor::arange(0u32, len as u32, &Device::Cpu)?.reshape(&[a, b, c])?;

                let permuted = einops!("a b c -> c a b", &input)?;
                let expected = input.permute((2, 0, 1))?;
                assert_eq!(permuted.dims(), expected.dims(), "shape seed: {a}x{b}x{c}");
                assert_eq!(
                    values_u32(&permuted)?,
                    values_u32(&expected)?,
                    "value seed: {a}x{b}x{c}"
                );

                let inverted = einops!("c a b -> a b c", &permuted)?;
                assert_eq!(inverted.dims(), input.dims(), "inverse seed: {a}x{b}x{c}");
                assert_eq!(
                    values_u32(&inverted)?,
                    values_u32(&input)?,
                    "inverse seed: {a}x{b}x{c}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn bounded_composition_decomposition_round_trips() -> Result<()> {
    for a in 0..=3 {
        for factor in 1..=3 {
            for c in 0..=3 {
                let len = a * factor * c;
                let input =
                    Tensor::arange(0u32, len as u32, &Device::Cpu)?.reshape(&[a, factor, c])?;

                let composed = einops!("a factor c -> (a factor) c", &input)?;
                let expected = Tensor::reshape(&input, &[a * factor, c])?;
                assert_eq!(composed.dims(), expected.dims());
                assert_eq!(values_u32(&composed)?, values_u32(&expected)?);

                let decomposed = einops!("(a {factor}) c -> a {factor} c", &composed)?;
                assert_eq!(decomposed.dims(), input.dims(), "seed: {a}x{factor}x{c}");
                assert_eq!(
                    values_u32(&decomposed)?,
                    values_u32(&input)?,
                    "seed: {a}x{factor}x{c}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn bounded_repeat_matches_host_indexing() -> Result<()> {
    for rows in 0..=3 {
        for columns in 0..=3 {
            for copies in 0..=3 {
                let len = rows * columns;
                let input =
                    Tensor::arange(0u32, len as u32, &Device::Cpu)?.reshape(&[rows, columns])?;

                let output = einops!("rows columns -> rows {copies} columns", &input)?;
                let mut expected = Vec::with_capacity(rows * copies * columns);
                for row in 0..rows {
                    for _ in 0..copies {
                        for column in 0..columns {
                            expected.push((row * columns + column) as u32);
                        }
                    }
                }

                assert_eq!(
                    output.dims(),
                    &[rows, copies, columns],
                    "seed: {rows}x{columns}x{copies}"
                );
                assert_eq!(
                    values_u32(&output)?,
                    expected,
                    "seed: {rows}x{columns}x{copies}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn zero_repeat_length_produces_an_empty_axis() -> Result<()> {
    let input = Tensor::new(&[[7u32]], &Device::Cpu)?;
    let copies = 0;

    let output = einops!("rows columns -> rows {copies} columns", &input)?;

    assert_eq!(output.dims(), &[1, 0, 1]);
    assert!(values_u32(&output)?.is_empty());
    Ok(())
}

#[test]
fn ellipsis_matches_explicit_axes_for_zero_one_and_two_captures() -> Result<()> {
    let rank_two = Tensor::arange(0u32, 6, &Device::Cpu)?.reshape(&[2, 3])?;
    let ellipsis = einops!("a .. z -> z a ..", &rank_two)?;
    let explicit = rank_two.permute((1, 0))?;
    assert_eq!(ellipsis.dims(), explicit.dims());
    assert_eq!(values_u32(&ellipsis)?, values_u32(&explicit)?);

    let rank_three = Tensor::arange(0u32, 24, &Device::Cpu)?.reshape(&[2, 3, 4])?;
    let ellipsis = einops!("a .. z -> z a ..", &rank_three)?;
    let explicit = rank_three.permute((2, 0, 1))?;
    assert_eq!(ellipsis.dims(), explicit.dims());
    assert_eq!(values_u32(&ellipsis)?, values_u32(&explicit)?);

    let rank_four = Tensor::arange(0u32, 120, &Device::Cpu)?.reshape(&[2, 3, 4, 5])?;
    let ellipsis = einops!("a .. z -> z a ..", &rank_four)?;
    let explicit = rank_four.permute((3, 0, 1, 2))?;
    assert_eq!(ellipsis.dims(), explicit.dims());
    assert_eq!(values_u32(&ellipsis)?, values_u32(&explicit)?);
    Ok(())
}

#[test]
fn scalar_empty_singleton_and_non_contiguous_inputs_match_oracles() -> Result<()> {
    let scalar = Tensor::new(7f32, &Device::Cpu)?;
    let scalar_sum = einops!("sum(..) -> ", &scalar)?;
    assert!(scalar_sum.dims().is_empty());
    assert_eq!(scalar_sum.to_scalar::<f32>()?, 7.);

    let empty = Tensor::zeros((2, 0, 3), DType::F32, &Device::Cpu)?;
    let sum = einops!("a sum(b) c -> a c", &empty)?;
    assert_eq!(sum.dims(), &[2, 3]);
    assert_eq!(values_f32(&sum)?, vec![0.; 6]);
    let product = einops!("a prod(b) c -> a c", &empty)?;
    assert_eq!(product.dims(), &[2, 3]);
    assert_eq!(values_f32(&product)?, vec![1.; 6]);

    for singleton in 1usize..=3 {
        let input =
            Tensor::arange(0u32, singleton as u32, &Device::Cpu)?.reshape(&[1, singleton, 1])?;
        let output = einops!("1 middle 1 -> middle", &input)?;
        assert_eq!(output.dims(), &[singleton as usize]);
        assert_eq!(values_u32(&output)?, values_u32(&input)?);
    }

    let contiguous = Tensor::arange(0u32, 24, &Device::Cpu)?.reshape(&[2, 3, 4])?;
    let non_contiguous = contiguous.permute((2, 0, 1))?;
    let output = einops!("c a b -> (a b) c", &non_contiguous)?;
    let expected = non_contiguous.permute((1, 2, 0))?.reshape(&[6, 4])?;
    assert_eq!(output.dims(), expected.dims());
    assert_eq!(values_u32(&output)?, values_u32(&expected)?);
    Ok(())
}

#[test]
fn dtype_behavior_matches_candle() -> Result<()> {
    let dtypes = [
        DType::U8,
        DType::U32,
        DType::I64,
        DType::BF16,
        DType::F16,
        DType::F32,
        DType::F64,
    ];

    for dtype in dtypes {
        let input = Tensor::arange(1f32, 7., &Device::Cpu)?
            .reshape(&[2, 3])?
            .to_dtype(dtype)?;

        let rearranged = einops!("rows columns -> columns rows", &input)?;
        let expected = input.permute((1, 0))?;
        assert_eq!(rearranged.dtype(), dtype);
        assert_eq!(
            values_f32(&rearranged)?,
            values_f32(&expected)?,
            "dtype: {dtype:?}"
        );

        let macro_sum = einops!("rows sum(columns) -> rows", &input);
        let candle_sum = input.sum(1);
        match (macro_sum, candle_sum) {
            (Ok(actual), Ok(expected)) => {
                assert_eq!(actual.dtype(), expected.dtype());
                assert_eq!(
                    values_f32(&actual)?,
                    values_f32(&expected)?,
                    "dtype: {dtype:?}"
                );
            }
            (Err(_), Err(_)) => {}
            (actual, expected) => panic!(
                "sum support differs for {dtype:?}: macro={}, candle={}",
                actual.is_ok(),
                expected.is_ok()
            ),
        }

        let macro_mean = einops!("rows mean(columns) -> rows", &input);
        let candle_mean = input.mean(1);
        match (macro_mean, candle_mean) {
            (Ok(actual), Ok(expected)) => {
                assert_eq!(actual.dtype(), expected.dtype());
                assert_eq!(
                    values_f32(&actual)?,
                    values_f32(&expected)?,
                    "dtype: {dtype:?}"
                );
            }
            (Err(_), Err(_)) => {}
            (actual, expected) => panic!(
                "mean support differs for {dtype:?}: macro={}, candle={}",
                actual.is_ok(),
                expected.is_ok()
            ),
        }
    }
    Ok(())
}

fn assert_returns_error_without_unwinding(call: impl FnOnce() -> Result<Tensor>) {
    let outcome = catch_unwind(AssertUnwindSafe(call));
    assert!(outcome.is_ok(), "invalid metadata unwound");
    assert!(outcome.unwrap().is_err(), "invalid metadata was accepted");
}

#[test]
fn invalid_backend_metadata_returns_errors_without_unwinding() -> Result<()> {
    let input = Tensor::arange(0u32, 6, &Device::Cpu)?.reshape(&[2, 3])?;

    assert_returns_error_without_unwinding(|| Backend::transpose(&input, &[0]));
    assert_returns_error_without_unwinding(|| Backend::transpose(&input, &[0, 0]));
    assert_returns_error_without_unwinding(|| Backend::transpose(&input, &[0, 2]));

    assert_returns_error_without_unwinding(|| {
        let mut reductions = [(2, Operation::Sum)];
        Backend::reduce_axes(&input, &mut reductions)
    });
    assert_returns_error_without_unwinding(|| Backend::add_axes(&input, 3, &[(1, 2), (1, 3)]));
    assert_returns_error_without_unwinding(|| Backend::add_axes(&input, 2, &[(0, 2)]));
    Ok(())
}

#[test]
fn duplicate_reduction_axes_are_rejected_without_unwinding() -> Result<()> {
    let input = Tensor::arange(0u32, 6, &Device::Cpu)?.reshape(&[2, 3])?;

    assert_returns_error_without_unwinding(|| {
        let mut reductions = [(0, Operation::Sum), (0, Operation::Sum)];
        Backend::reduce_axes(&input, &mut reductions)
    });
    Ok(())
}
