use candle_core::{Device, Result, Tensor, Var};
use candle_einops_benchmarks::balanced_product_axis;

#[test]
fn portable_candidate_matches_product_contract() -> Result<()> {
    let input = Tensor::new(&[[1f32, -2., 0., 4.], [5., 6., -7., 8.]], &Device::Cpu)?;
    let output = balanced_product_axis(&input, 1)?;
    assert_eq!(output.to_vec1::<f32>()?, [0., -1680.]);

    let empty = Tensor::zeros((2, 0), candle_core::DType::F32, &Device::Cpu)?;
    assert_eq!(
        balanced_product_axis(&empty, 1)?.to_vec1::<f32>()?,
        [1., 1.]
    );

    let variable = Var::from_tensor(&input)?;
    let product = balanced_product_axis(variable.as_tensor(), 1)?;
    let gradients = product.sum_all()?.backward()?;
    assert_eq!(
        gradients
            .get(variable.as_tensor())
            .expect("portable candidate keeps autograd")
            .to_vec2::<f32>()?,
        [[0., 0., -8., 0.], [-336., -280., 240., -210.]],
    );
    Ok(())
}
