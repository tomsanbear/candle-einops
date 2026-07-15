use candle_core::{Device, Result, Tensor};
use candle_einops_benchmarks::balanced_product_axis;

#[test]
fn portable_candidate_matches_product_contract() -> Result<()> {
    let input = Tensor::new(
        &[[1f32, -2., 0., 4.], [5., 6., -7., 8.]],
        &Device::Cpu,
    )?;
    let output = balanced_product_axis(&input, 1)?;
    assert_eq!(output.to_vec1::<f32>()?, [0., -1680.]);

    let empty = Tensor::zeros((2, 0), candle_core::DType::F32, &Device::Cpu)?;
    assert_eq!(balanced_product_axis(&empty, 1)?.to_vec1::<f32>()?, [1., 1.]);
    Ok(())
}
