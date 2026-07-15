//! Compile-time einops-style tensor transformations for Candle.
//!
//! The [`einops!`] macro combines rearrange, reduce, repeat, composition, and
//! decomposition operations. [`einsum!`] provides explicit-output,
//! arbitrary-arity Einstein summation. Backend failures are returned as Candle
//! errors.
//!
//! Einsum equations require exactly one `->`, use whitespace-delimited named
//! axes, and have one comma-separated input list per operand. Axes omitted from
//! the output are summed, `..` captures right-aligned runtime axes, and repeated
//! labels select diagonals. See the repository's `docs/einsum-contract.md` for
//! the complete supported contract.
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
//! A `..` captures zero or more axes. Captures from multiple operands align
//! from the right and broadcast, while omitting `..` from the output reduces
//! those axes:
//!
//! ```
//! use candle_core::{Device, Result, Tensor};
//! use candle_einops::einsum;
//!
//! # fn main() -> Result<()> {
//! let input = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((2, 2, 3))?;
//! let reduced = einsum!(".. feature -> feature", &input)?;
//! assert_eq!(reduced.to_vec1::<f32>()?, [18., 22., 26.]);
//! # Ok(())
//! # }
//! ```
//!
//! Contractions lower through Candle matrix multiplication, including batch
//! broadcasting. Equations with more than two operands use deterministic,
//! shape-aware greedy pair selection:
//!
//! ```
//! use candle_core::{Device, Result, Tensor};
//! use candle_einops::einsum;
//!
//! # fn main() -> Result<()> {
//! let left = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
//! let right = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;
//! let output = einsum!("row inner, inner column -> row column", &left, &right)?;
//! assert_eq!(output.to_vec2::<f32>()?, [[22., 28.], [49., 64.]]);
//! let weights = Tensor::new(&[1f32, 1.], &Device::Cpu)?;
//! let projected = einsum!(
//!     "row inner, inner column, column -> row",
//!     &left,
//!     &right,
//!     &weights,
//! )?;
//! assert_eq!(projected.to_vec1::<f32>()?, [50., 113.]);
//! # Ok(())
//! # }
//! ```
//!
//! Repeating a label within one operand extracts its diagonal. Omitting that
//! label from the output computes a trace:
//!
//! ```
//! use candle_core::{Device, Result, Tensor};
//! use candle_einops::einsum;
//!
//! # fn main() -> Result<()> {
//! let matrix = Tensor::arange(0f32, 9f32, &Device::Cpu)?.reshape((3, 3))?;
//! let diagonal = einsum!("index index -> index", &matrix)?;
//! assert_eq!(diagonal.to_vec1::<f32>()?, [0., 4., 8.]);
//! let trace = einsum!("index index ->", &matrix)?;
//! assert_eq!(trace.to_scalar::<f32>()?, 12.);
//! # Ok(())
//! # }
//! ```
//!
//! ## Dtypes, devices, and gradients
//!
//! `einsum!` never casts operands or transfers them between devices. Every
//! operand in a multi-operand equation must have the same dtype and reside on
//! the same device; mismatches return a contextual [`candle_core::Error`].
//! Unary permutations preserve every dtype supported by the corresponding
//! Candle operation. Binary equations without contracted labels use Candle
//! multiplication, including its integer and BF16 support. True contractions
//! lower through Candle matrix multiplication and therefore inherit its dtype
//! and device support: unsupported combinations return an error rather than
//! being silently converted.
//!
//! Einsum execution is assembled from tracked public Candle operations, so
//! floating-point inputs participate in Candle autograd. Accelerator execution
//! likewise follows the features and devices made available by Candle; results
//! remain on the operands' original device.
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
    #[cfg(feature = "benchmark-internals")]
    pub use crate::einsum::{
        BenchmarkBinaryGraphEstimate, benchmark_binary_graph_estimate,
        benchmark_nary_planner_selects_exact, benchmark_pack_canonical_operand,
    };
    pub use crate::einsum::{
        BinaryEinsumSpec, EinsumAxisPattern, EllipsisEinsumSpec, UnaryEinsumSpec,
        einsum_operand_ref, execute_binary_einsum, execute_binary_ellipsis_einsum,
        execute_binary_multiply, execute_canonical_binary_einsum, execute_nary_einsum,
        execute_unary_einsum, execute_unary_ellipsis_einsum,
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
