//! Compile-time einops-style tensor transformations for Candle.
//!
//! The [`einops!`] macro combines rearrange, reduce, repeat, composition, and
//! decomposition operations. Backend failures are returned as Candle errors.
//!
//! ```
//! use candle_core::{Device, Result, Tensor};
//! use candle_einops::einops;
//!
//! # fn main() -> Result<()> {
//! let input = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let output = einops!("rows columns -> columns rows", &input)?;
//! assert_eq!(output.dims(), &[3, 2]);
//! # Ok(())
//! # }
//! ```

extern crate self as candle_einops;

mod backend;
mod einsum;

/// The result type returned by [`einops!`] and [`Backend`] transformations.
pub use candle_core::Result;
pub use candle_einops_macros::einops;

pub use backend::Backend;

/// Implementation details used by macros generated for this crate.
///
/// This module is not a stable public API.
#[doc(hidden)]
pub mod __private {
    pub use crate::einsum::{UnaryEinsumSpec, execute_unary_einsum};
}

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
