use r#match::{Device, Result, Tensor};
use r#type::{einops, einsum};

fn main() -> Result<()> {
    let matrix = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let transposed = einops!("rows columns -> columns rows", &matrix)?;
    assert_eq!(transposed.dims(), &[3, 2]);
    let einsum_transposed = einsum!("rows columns -> columns rows", &matrix)?;
    assert_eq!(einsum_transposed.dims(), &[3, 2]);
    let product = einsum!(
        "row inner, inner column -> row column",
        &matrix,
        &einsum_transposed
    )?;
    assert_eq!(product.dims(), &[2, 2]);
    let ellipsis = einsum!(".. columns -> columns ..", &matrix)?;
    assert_eq!(ellipsis.dims(), &[3, 2]);
    let square = Tensor::arange(0f32, 9f32, &Device::Cpu)?.reshape((3, 3))?;
    let diagonal = einsum!("index index -> index", &square)?;
    assert_eq!(diagonal.to_vec1::<f32>()?, [0., 4., 8.]);
    let weights = Tensor::new(&[1f32, 1.], &Device::Cpu)?;
    let nary = einsum!(
        "row inner, inner column, column -> row",
        &matrix,
        &einsum_transposed,
        &weights
    )?;
    assert_eq!(nary.dims(), &[2]);
    Ok(())
}
