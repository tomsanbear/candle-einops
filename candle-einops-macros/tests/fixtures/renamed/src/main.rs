use candle_core::{Device, Result, Tensor};
use einops_runtime::einops;

fn main() -> Result<()> {
    let vector = Tensor::arange(0f32, 6f32, &Device::Cpu)?;
    let repeated = einops!("(a:2 b:3) -> b a copy:2", &vector);
    assert_eq!(repeated.dims(), &[3, 2, 2]);

    let matrix = vector.reshape((2, 3))?;
    let reduced = einops!("a sum(b) -> a", &matrix);
    assert_eq!(reduced.dims(), &[2]);

    let composed = einops!("a b -> (a b)", &matrix);
    assert_eq!(composed.dims(), &[6]);
    Ok(())
}
