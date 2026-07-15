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
    Ok(())
}
