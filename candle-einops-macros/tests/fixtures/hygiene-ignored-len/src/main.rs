use candle_core::{Device, Result, Tensor};
use candle_einops::einops;

fn main() -> Result<()> {
    let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let input_ignored_len = 5usize;
    let repeated = einops!("rows .. -> rows .. {input_ignored_len}", &tensor)?;
    assert_eq!(repeated.dims(), &[2, 3, 5]);
    Ok(())
}
