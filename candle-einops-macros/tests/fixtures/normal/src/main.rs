use candle_core::{Device, Result, Tensor};
use candle_einops::{einops, einsum};

fn main() -> Result<()> {
    let vector = Tensor::arange(0f32, 6f32, &Device::Cpu)?;
    let repeated = einops!("(a:2 b:3) -> b a copy:2", &vector)?;
    assert_eq!(repeated.dims(), &[3, 2, 2]);

    let matrix = vector.reshape((2, 3))?;
    let reduced = einops!("a sum(b) -> a", &matrix)?;
    assert_eq!(reduced.dims(), &[2]);

    let composed = einops!("a b -> (a b)", &matrix)?;
    assert_eq!(composed.dims(), &[6]);

    let transposed = einsum!("rows columns -> columns rows", &matrix)?;
    assert_eq!(transposed.to_vec2::<f32>()?, [[0., 3.], [1., 4.], [2., 5.]]);
    let product = einsum!("row inner, inner column -> row column", &matrix, &transposed)?;
    assert_eq!(product.to_vec2::<f32>()?, [[5., 14.], [14., 50.]]);
    let reduced = einsum!(".. column -> column", &matrix)?;
    assert_eq!(reduced.to_vec1::<f32>()?, [3., 5., 7.]);
    let square = Tensor::arange(0f32, 9f32, &Device::Cpu)?.reshape((3, 3))?;
    let diagonal = einsum!("index index -> index", &square)?;
    assert_eq!(diagonal.to_vec1::<f32>()?, [0., 4., 8.]);
    let weights = Tensor::new(&[1f32, 1.], &Device::Cpu)?;
    let nary = einsum!(
        "row inner, inner column, column -> row",
        &matrix,
        &transposed,
        &weights,
    )?;
    assert_eq!(nary.to_vec1::<f32>()?, [19., 64.]);
    Ok(())
}
