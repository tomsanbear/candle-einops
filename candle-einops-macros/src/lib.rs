mod einops;

/// Macro to perform tensor transformations using simple expressions
///
/// This macro is re-exported as `candle_einops::einops`; the runtime crate's
/// documentation contains a complete runnable example.
#[proc_macro]
pub fn einops(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    einops::einops(input.into())
        .unwrap_or_else(|e| e.to_compile_error())
        .into()
}
