//! Compile-time einops-style tensor transformations for Candle.
//!
//! The [`einops!`] macro combines rearrange, reduce, repeat, composition, and
//! decomposition operations. [`einsum!`] provides explicit-output unary
//! permutations and reductions. Backend failures are returned as Candle errors.
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
//!
//! Unary einsum equations use whitespace-delimited named axes. Axes omitted
//! from the explicit output are summed:
//!
//! ```
//! use candle_core::{Device, Result, Tensor};
//! use candle_einops::einsum;
//!
//! # fn main() -> Result<()> {
//! let input = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let columns = einsum!("rows columns -> columns", &input)?;
//! assert_eq!(columns.to_vec1::<f32>()?, [3., 5., 7.]);
//! # Ok(())
//! # }
//! ```

extern crate self as candle_einops;

mod backend;
mod einsum;

/// The result type returned by [`einops!`] and [`Backend`] transformations.
pub use candle_core::Result;
pub use candle_einops_macros::{einops, einsum};

pub use backend::Backend;

/// Implementation details used by macros generated for this crate.
///
/// This module is not a stable public API.
#[doc(hidden)]
pub mod __private {
    pub use crate::einsum::{
        BinaryEinsumSpec, UnaryEinsumSpec, execute_binary_einsum, execute_unary_einsum,
    };
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
