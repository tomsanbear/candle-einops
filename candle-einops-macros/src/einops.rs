mod parse;
mod tokens;

use proc_macro2::{Ident, Span};
use quote::{format_ident, quote};
use syn::parse::ParseStream;

use parse::{
    parse_composition_permute_repeat, parse_decomposition, parse_reduce, Composition,
    Decomposition, Index, Operation, Shape,
};
use tokens::{
    to_tokens_composition, to_tokens_decomposition, to_tokens_permute, to_tokens_reduce,
    to_tokens_repeat,
};

pub fn einops(input: proc_macro2::TokenStream) -> syn::Result<proc_macro2::TokenStream> {
    let parsed_expression: ParsedExpression = syn::parse2(input)?;
    let code = quote! { #parsed_expression };
    Ok(code)
}

#[derive(Debug)]
struct ParsedExpression {
    tensor: syn::Ident,
    tensor_expression: proc_macro2::TokenStream,
    expression: Expression,
}

impl syn::parse::Parse for ParsedExpression {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let expression: Expression = input.parse::<syn::LitStr>()?.parse()?;

        input.parse::<syn::Token![,]>()?;

        let (tensor_ident, tensor_tokens) = {
            let tensor_ident = Ident::new("input", Span::call_site());
            let expr = input.parse::<syn::Expr>()?;
            let tensor_tokens = quote!(let #tensor_ident = #expr;);
            (tensor_ident, tensor_tokens)
        };

        Ok(Self {
            tensor: tensor_ident,
            tensor_expression: tensor_tokens,
            expression,
        })
    }
}

#[derive(Debug)]
struct Expression {
    // A bool that is 'true' if,
    // - A new dimension is derived
    // - Dimensions of size 1 need squeezing
    requires_decomposition: bool,
    // Step 1, Where a dimension can be exploded or decomposed
    decomposition: Vec<Decomposition>,
    // Step 2, Reducing dimensions with operations like min, max, ..
    reduce: Vec<(Index, Operation)>,
    // Step 3, Transposing dimensions
    permute: Vec<Index>,
    // Step 4, Tiling or repeating dimensions
    repeat: Vec<(Index, Shape)>,
    // Step 5, Combining dimensions into a single dimension
    composition: Vec<Composition>,
}

impl syn::parse::Parse for Expression {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let (decomposition, requires_decomposition) = parse_decomposition(input)?;

        let reduce = parse_reduce(&decomposition);

        let (composition, permute, repeat) =
            parse_composition_permute_repeat(input, &decomposition)?;

        Ok(Expression {
            requires_decomposition,
            decomposition,
            reduce,
            permute,
            repeat,
            composition,
        })
    }
}

impl quote::ToTokens for ParsedExpression {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ParsedExpression {
            tensor: ref tensor_ident,
            tensor_expression: ref tensor_tokens,
            ref expression,
        } = self;
        let Expression {
            requires_decomposition,
            ref decomposition,
            ref reduce,
            ref permute,
            ref repeat,
            ref composition,
        } = expression;

        // Variable to store the shape slice
        let shape_ident = format_ident!("{}_{}", tensor_ident, "shape");

        // Variable that stores the length of dimensions ignored
        // in the expression using '..' symbol
        let ignored_len_ident = format_ident!("{}_{}", tensor_ident, "ignored_len");

        // If needed we generate tokens for decomposing the tensor
        let decomposition_tokens = if *requires_decomposition {
            to_tokens_decomposition(
                decomposition,
                tensor_ident,
                &ignored_len_ident,
                &shape_ident,
            )
        } else {
            proc_macro2::TokenStream::new()
        };
        let last_unknown_index = decomposition
            .last()
            .map(|expression| match expression {
                Decomposition::Named {
                    index: Index::Unknown(i),
                    ..
                }
                | Decomposition::Derived {
                    index: Index::Unknown(i),
                    ..
                }
                | Decomposition::Named {
                    index: Index::Range(i),
                    ..
                } => Some(*i),
                _ => None,
            })
            .flatten();
        let decomposition_ignored_len =
            !decomposition_tokens.is_empty() && last_unknown_index.is_some();

        // If needed we generate tokens for reducing the tensor
        let (reduce_tokens, reduce_ignored_len) = if !reduce.is_empty() {
            let requires_ignored_len = reduce
                .iter()
                .any(|(index, _)| matches!(index, Index::Range(_) | Index::Unknown(_)));
            let tokens = to_tokens_reduce(reduce, tensor_ident, &ignored_len_ident);
            (tokens, requires_ignored_len)
        } else {
            (proc_macro2::TokenStream::new(), false)
        };

        // If needed we generate tokens for transposing the tensor
        let (permute_tokens, permute_ignored_len) = if permute.windows(2).any(|w| w[0] > w[1]) {
            let requires_ignored_len = permute
                .iter()
                .any(|expression| matches!(expression, Index::Range(_) | Index::Unknown(_)));
            let tokens = to_tokens_permute(permute, tensor_ident, &ignored_len_ident);
            (tokens, requires_ignored_len)
        } else {
            (proc_macro2::TokenStream::new(), false)
        };

        // If needed we generate tokens for repeating the tensor
        let (repeat_tokens, repeat_ignored_len) = if !repeat.is_empty() {
            let requires_ignored_len = repeat.iter().any(|(i, _)| matches!(i, Index::Unknown(_)));
            let tokens = to_tokens_repeat(repeat, tensor_ident, &ignored_len_ident, &shape_ident);
            (tokens, requires_ignored_len)
        } else {
            (proc_macro2::TokenStream::new(), false)
        };

        // If needed we generate tokens for combining dimensions of the tensor
        let (composition_tokens, composition_ignored_len) = if composition
            .iter()
            .any(|expression| matches!(expression, Composition::Combined { .. }))
        {
            let requires_ignored_len = composition.iter().any(|expression| {
                matches!(
                    expression,
                    Composition::Combined {
                        from: Index::Unknown(_) | Index::Range(_),
                        to: Some(Index::Unknown(_) | Index::Range(_)) | None
                    } | Composition::Individual(Index::Range(_) | Index::Unknown(_))
                )
            });
            let tokens =
                to_tokens_composition(composition, tensor_ident, &ignored_len_ident, &shape_ident);
            (tokens, requires_ignored_len)
        } else {
            (proc_macro2::TokenStream::new(), false)
        };

        let ignored_len_tokens = if decomposition_ignored_len
            || reduce_ignored_len
            || permute_ignored_len
            || repeat_ignored_len
            || composition_ignored_len
        {
            let index = last_unknown_index.unwrap();
            quote!(let #ignored_len_ident = #shape_ident.len() - #index;)
        } else {
            proc_macro2::TokenStream::new()
        };

        // NOTE Do not change the order
        let tokens_empty = [
            decomposition_tokens.is_empty(),
            reduce_tokens.is_empty(),
            permute_tokens.is_empty(),
            repeat_tokens.is_empty(),
            composition_tokens.is_empty(),
        ];

        let error_tokens = if tokens_empty.iter().all(|x| *x) {
            // If transformations are applied, we raise a compile time error
            quote!(compile_error!(
                "No transformations applied, no need for einops"
            );)
        } else {
            proc_macro2::TokenStream::new()
        };

        // We need the shape of the tensor, if either of ignored length calculation, or
        // decomposition operation has to take place
        let shape_tokens = match (
            ignored_len_tokens.is_empty(),
            decomposition_tokens.is_empty(),
            // If all of the shapes of input tensor is known, we can skip
            // calculating the shape of the tensor
            decomposition.iter().any(|expression| {
                matches!(
                    expression,
                    Decomposition::Derived { .. } | Decomposition::Named { shape: None, .. }
                )
            }),
        ) {
            (true, true, _) => proc_macro2::TokenStream::new(),
            (false, _, _) | (_, false, true) => {
                quote!(let #shape_ident = ::candle_einops::Backend::shape(&#tensor_ident);)
            }
            (_, false, false) => proc_macro2::TokenStream::new(),
        };

        // We have to recalculate the shape of the tensor before repeat transformation
        let repeat_shape_tokens = if repeat_tokens.is_empty() ||
            // We can skip it if non of the first three transformations happen,
            // and we already have the shape slice from the ignored length calculation
            (tokens_empty.iter().take(3).all(|x| *x) && !ignored_len_tokens.is_empty())
        {
            proc_macro2::TokenStream::new()
        } else {
            quote!(let #shape_ident = ::candle_einops::Backend::shape(&#tensor_ident);)
        };

        // We have to recalculate the shape of the tensor before composition transformation
        let composition_shape_tokens = if composition_tokens.is_empty() ||
            // We can skip it if non of the first four transformations happen,
            // and we already have the shape slice from the ignored length calculation
            (tokens_empty.iter().take(4).all(|x| *x) && !ignored_len_tokens.is_empty())
        {
            proc_macro2::TokenStream::new()
        } else {
            quote!(let #shape_ident = ::candle_einops::Backend::shape(&#tensor_ident);)
        };

        let code = quote! {{
            #error_tokens

            #tensor_tokens

            #shape_tokens

            #ignored_len_tokens

            #decomposition_tokens

            #reduce_tokens

            #permute_tokens

            #repeat_shape_tokens
            #repeat_tokens

            #composition_shape_tokens
            #composition_tokens

            #tensor_ident
        }};

        code.to_tokens(tokens);
    }
}
