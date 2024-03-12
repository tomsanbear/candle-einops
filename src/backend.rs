mod candle;

use crate::Operation;

pub trait Backend {
    type Output;
    fn shape(self) -> Vec<usize>;
    fn reshape(self, shape: &[usize]) -> Self::Output;
    fn transpose(self, axes: &[usize]) -> Self::Output;
    fn reduce_axes(self, axes_operations: &mut [(usize, Operation)]) -> Self::Output;
    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Self::Output;
}
