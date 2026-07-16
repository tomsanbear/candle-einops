use ::candle_core::{Device, Result, Tensor};
use ::candle_einops::einops;

struct Vec;

mod std {
    pub mod iter {}
}

fn main() -> Result<()> {
    let _shadowed_vec = Vec;
    let tensor = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;

    let input = 2usize;
    let repeated = einops!("rows columns -> rows columns {input}", &tensor)?;
    assert_eq!(repeated.dims(), &[2, 3, 2]);

    let input_shape = 4usize;
    let repeated = einops!(
        "rows columns -> rows columns {input_shape}",
        &tensor
    )?;
    assert_eq!(repeated.dims(), &[2, 3, 4]);

    let input_ignored_len = 5usize;
    let repeated = einops!("rows .. -> rows .. {input_ignored_len}", &tensor)?;
    assert_eq!(repeated.dims(), &[2, 3, 5]);

    let permuted = einops!("rows .. columns -> columns .. rows", &tensor)?;
    assert_eq!(permuted.dims(), &[3, 2]);

    let reduced = einops!("sum(..) ->", &tensor)?;
    assert_eq!(reduced.to_scalar::<f32>()?, 15f32);

    Ok(())
}
