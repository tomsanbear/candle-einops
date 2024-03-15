use itertools::Itertools;
use proc_macro2::{Ident, Literal, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::{Ref, Token};
use syn::{braced, token, Expr, Field, LitStr, Token};

pub fn einsum(input: proc_macro2::TokenStream) -> syn::Result<TokenStream> {
    let parsed_expression: ParsedExpression = syn::parse2(input)?;
    let code = quote! {
        #parsed_expression
    };
    Ok(code)
}

/// Parses syntax for an einsum expression
/// einsum!("a b c, d e f -> a b c f", &x, &y)
/// where each comma delimited string before the arrow is an input tensor and the string after the arrow is the output tensor
struct ParsedExpression {}

impl Parse for ParsedExpression {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // gets the first argument, this determines how many input tensors we need
        if !input.peek(LitStr) {
            return Err(input.error("first argument must be a string literal"));
        }
        let expression: LitStr = input.parse()?;

        // Iterate across the stream until we collect all tensors
        let mut tensors: Vec<Ident> = vec![];
        for _ in 0.. {
            // advance past the comma
            if !input.peek(Token![,]) {
                break;
            }
            input.parse::<Token![,]>()?;

            // parse the next tensor, should be an identifier or reference to an identifier
            tensors.push(input.parse()?);
        }

        // Extract LHS and RHS of the expression, split on ->
        let expression_str = expression.value();
        let (lhs, rhs) = expression_str
            .split("->")
            .collect_tuple()
            .ok_or_else(|| input.error("expected expression to contain ->"))?;
        let lhs = lhs.trim();
        let rhs = rhs.trim();

        // Gather the tensors information from the lhs and rhs
        let input_tensors = lhs.split(',').into_iter().map(|s| s.trim()).collect_vec();
        let output_tensor = rhs.trim();

        // Ensure the number of tensors provided matches the number of tensors in the expression
        if tensors.len() != input_tensors.len() {
            return Err(input.error(format!(
                "expected {} input tensors, got {}",
                input_tensors.len(),
                tensors.len()
            )));
        }

        // Get the shape of each input

        return Err(input.error(format!(
            "lhs {:?} rhs {:?} tensor count {:?}",
            lhs,
            rhs,
            input_tensors.len()
        )));
    }
}

impl quote::ToTokens for ParsedExpression {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let expression = quote! { "a b c, d e f -> a b c f" };
        tokens.extend(quote! {
            let expression = #expression;
        });
    }
}
