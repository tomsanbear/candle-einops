use std::collections::HashMap;

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};

pub fn einsum(input: TokenStream) -> syn::Result<TokenStream> {
    let invocation = syn::parse2::<Invocation>(input)?;
    Ok(quote!(#invocation))
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AxisId(usize);

#[derive(Debug)]
struct Operand {
    axes: Vec<AxisId>,
}

#[derive(Debug)]
struct Equation {
    input: Operand,
    output: Vec<AxisId>,
}

impl Equation {
    fn parse(literal: &syn::LitStr) -> syn::Result<Self> {
        let text = literal.value();
        if text.matches("->").count() != 1 {
            return Err(syn::Error::new(
                literal.span(),
                "einsum equation requires exactly one explicit `->`",
            ));
        }
        let (input_text, output_text) = text
            .split_once("->")
            .ok_or_else(|| syn::Error::new(literal.span(), "missing einsum output"))?;

        let input_lists = input_text.split(',').collect::<Vec<_>>();
        if input_lists.len() != 1 {
            return Err(syn::Error::new(
                literal.span(),
                format!(
                    "unary einsum supports exactly one input axis list; equation contains {}",
                    input_lists.len()
                ),
            ));
        }
        if output_text.contains(',') {
            return Err(syn::Error::new(
                literal.span(),
                "einsum output must be one whitespace-delimited axis list",
            ));
        }

        let input_labels = parse_labels(input_lists[0], literal.span())?;
        let output_labels = parse_labels(output_text, literal.span())?;
        let mut interned = HashMap::new();
        let mut input_axes = Vec::with_capacity(input_labels.len());
        for label in input_labels {
            if interned.contains_key(&label) {
                return Err(syn::Error::new(
                    literal.span(),
                    format!(
                        "repeated einsum input label `{label}` is reserved for diagonal support"
                    ),
                ));
            }
            let axis = AxisId(input_axes.len());
            interned.insert(label, axis);
            input_axes.push(axis);
        }

        let mut output = Vec::with_capacity(output_labels.len());
        let mut output_names = Vec::with_capacity(output_labels.len());
        for label in output_labels {
            if output_names.contains(&label) {
                return Err(syn::Error::new(
                    literal.span(),
                    format!("duplicate einsum output label `{label}`"),
                ));
            }
            let axis = interned.get(&label).copied().ok_or_else(|| {
                syn::Error::new(
                    literal.span(),
                    format!("einsum output label `{label}` does not occur in the input"),
                )
            })?;
            output_names.push(label);
            output.push(axis);
        }

        Ok(Self {
            input: Operand { axes: input_axes },
            output,
        })
    }

    fn permutation(&self) -> Vec<usize> {
        self.output
            .iter()
            .chain(
                self.input
                    .axes
                    .iter()
                    .filter(|axis| !self.output.contains(axis)),
            )
            .map(|axis| axis.0)
            .collect()
    }
}

fn parse_labels(text: &str, span: Span) -> syn::Result<Vec<String>> {
    text.split_whitespace()
        .map(|label| {
            if label == ".." {
                return Err(syn::Error::new(
                    span,
                    "einsum `..` is reserved for ellipsis support",
                ));
            }
            let mut characters = label.chars();
            let valid_start = characters
                .next()
                .is_some_and(|character| character == '_' || character.is_alphabetic());
            if !valid_start
                || !characters.all(|character| character == '_' || character.is_alphanumeric())
            {
                return Err(syn::Error::new(
                    span,
                    format!("invalid einsum axis label `{label}`"),
                ));
            }
            Ok(label.to_owned())
        })
        .collect()
}

struct Invocation {
    runtime_crate: syn::Path,
    operand: syn::Expr,
    operand_ident: Ident,
    equation: Equation,
}

impl Parse for Invocation {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let literal = input.parse::<syn::LitStr>()?;
        let equation = Equation::parse(&literal)?;
        input.parse::<syn::Token![,]>()?;
        if input.is_empty() {
            return Err(syn::Error::new(
                literal.span(),
                "unary einsum accepts exactly one operand expression",
            ));
        }
        let operand = input.parse::<syn::Expr>()?;
        if !input.is_empty() {
            let span = input.span();
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            }
            return Err(syn::Error::new(
                span,
                "unary einsum accepts exactly one operand expression",
            ));
        }

        Ok(Self {
            runtime_crate: runtime_crate_path()?,
            operand,
            operand_ident: private_ident("operand"),
            equation,
        })
    }
}

impl ToTokens for Invocation {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self {
            runtime_crate,
            operand,
            operand_ident,
            equation,
        } = self;
        let input_rank = equation.input.axes.len();
        let output_rank = equation.output.len();
        let permutation = equation.permutation();
        quote!({
            let #operand_ident = #operand;
            #runtime_crate::__private::execute_unary_einsum(
                &#operand_ident,
                #runtime_crate::__private::UnaryEinsumSpec::new(
                    #input_rank,
                    #output_rank,
                    &[#(#permutation),*],
                ),
            )
        })
        .to_tokens(tokens);
    }
}

fn runtime_crate_path() -> syn::Result<syn::Path> {
    match crate_name("candle-einops") {
        Ok(FoundCrate::Itself) => Ok(syn::parse_quote!(::candle_einops)),
        Ok(FoundCrate::Name(name)) => syn::parse_str(&format!("::r#{name}")).map_err(|_| {
            syn::Error::new(
                Span::call_site(),
                format!("dependency alias `{name}` cannot be used as a Rust crate path"),
            )
        }),
        Err(error) => Err(syn::Error::new(
            Span::call_site(),
            format!("could not resolve the `candle-einops` runtime crate: {error}"),
        )),
    }
}

fn private_ident(name: &str) -> Ident {
    Ident::new(&format!("__candle_einsum_{name}"), Span::mixed_site())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_retained_axes_before_reductions() {
        let literal: syn::LitStr = syn::parse_quote!("a b c -> c a");
        let equation = Equation::parse(&literal).expect("valid unary equation");
        assert_eq!(equation.permutation(), [2, 0, 1]);
    }
}
