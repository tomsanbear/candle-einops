mod backend;

/// The result type returned by [`einops!`] and [`Backend`] transformations.
pub use candle_core::Result;
pub use candle_einops_macros::einops;

pub use backend::Backend;

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
    /// Multiply all elements
    Prod,
}
