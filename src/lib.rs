mod backend;

pub use backend::Backend;
pub use candle_einops_macros::einops;
pub use candle_einops_macros::einsum;

/// Specifies the operation used to reduce an axis
#[derive(Copy, Clone, Debug)]
pub enum Operation {
    /// Take the minimum value
    Min,
    /// Take the maximum value
    Max,
    /// Sum all elements
    Sum,
    /// Get the mean value
    Mean,
}
