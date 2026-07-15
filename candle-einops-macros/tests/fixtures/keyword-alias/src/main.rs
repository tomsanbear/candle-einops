use r#match::{Device, Result, Tensor};
use r#type::einops;

fn main() -> Result<()> {
    let matrix = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let transposed = einops!("rows columns -> columns rows", &matrix)?;
    assert_eq!(transposed.dims(), &[3, 2]);
    Ok(())
}
