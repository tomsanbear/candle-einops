use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_einops::einsum;
#[test]
fn simple() -> Result<()> {
    let device = Device::Cpu;
    let x = &Tensor::new(&[1, 2, 3, 4], &device)?;
    let y = &Tensor::new(&[1, 2, 3, 4], &device)?;
    einsum!("a b c, d e f -> a e f", x, y)?;

    Ok(())
}
