mod einops;
mod einsum;

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

/// Evaluates an explicit-output Einstein summation equation.
///
/// Supports one or two operands with unique, named axes and at most one `..`
/// per axis list. Repeated input labels are reserved for a later syntax slice.
#[proc_macro]
pub fn einsum(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    einsum::einsum(input.into())
        .unwrap_or_else(|error| error.to_compile_error())
        .into()
}
