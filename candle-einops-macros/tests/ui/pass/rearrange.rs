extern crate self as candle_einops;

use candle_einops_macros::einops;

pub struct Tensor;

pub trait Backend {
    type Output;

    fn transpose(self, axes: &[usize]) -> Self::Output;
}

impl Backend for &Tensor {
    type Output = Tensor;

    fn transpose(self, axes: &[usize]) -> Self::Output {
        assert_eq!(axes, &[1, 0]);
        Tensor
    }
}

fn main() {
    let tensor = Tensor;
    let _ = einops!("a b -> b a", &tensor);
}
