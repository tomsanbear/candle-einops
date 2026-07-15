mod einops;

/// Macro to perform tensor transformations using simple expressions
///
/// # Example
///
/// ```ignore
/// use candle_core::{Result, Tensor};
/// use candle_einops::einops;
///
/// fn channels_first(input: &Tensor) -> Result<Tensor> {
///     einops!("height width channels -> channels height width", input)
/// }
/// ```
#[proc_macro]
pub fn einops(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    einops::einops(input.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
