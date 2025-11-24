use crate::einops::{Composition, Decomposition, Index, Operation, Shape};

use quote::quote;

pub fn to_tokens_composition(
    right_expression: &[Composition],
    tensor_ident: &syn::Ident,
    ignored_len_ident: &syn::Ident,
    shape_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let (before_ignored, ignored, after_ignored, _) = right_expression.iter().fold(
        (
            Vec::new(),
            proc_macro2::TokenStream::new(),
            Vec::new(),
            false,
        ),
        |(mut before_ignored, mut ignored, mut after_ignored, mut is_after_ignored), expression| {
            let mut insert_shape = |shape| {
                if is_after_ignored {
                    after_ignored.push(shape);
                } else {
                    before_ignored.push(shape);
                }
            };
            match expression {
                Composition::Individual(Index::Known(index))
                | Composition::Combined {
                    from: Index::Known(index),
                    to: None,
                } => {
                    let shape = quote!(#shape_ident[#index]);
                    insert_shape(shape);
                }
                Composition::Individual(Index::Unknown(index))
                | Composition::Combined {
                    from: Index::Unknown(index),
                    to: None,
                } => {
                    let shape = quote!(
                        #shape_ident[#index + #ignored_len_ident - 1]
                    );
                    insert_shape(shape);
                }
                Composition::Individual(Index::Range(index)) => {
                    ignored = quote!(
                        (#index..(#index + #ignored_len_ident))
                            .into_iter().map(|i| #shape_ident[i])
                    );
                    is_after_ignored = true;
                }
                Composition::Combined {
                    from: Index::Range(index),
                    to: None,
                } => {
                    let shape = quote!(
                        (#index..(#index + #ignored_len_ident))
                            .into_iter().map(|i| #shape_ident[i]).product::<usize>()
                    );
                    insert_shape(shape);
                }
                Composition::Combined {
                    from: Index::Known(from_index),
                    to: Some(Index::Known(to_index)),
                } => {
                    let shape = quote!(
                        (#from_index..=#to_index)
                            .into_iter().map(|i| #shape_ident[i]).product::<usize>()
                    );
                    insert_shape(shape);
                }
                Composition::Combined {
                    from: Index::Known(from_index),
                    to: Some(Index::Unknown(to_index)),
                }
                | Composition::Combined {
                    from: Index::Known(from_index),
                    to: Some(Index::Range(to_index)),
                } => {
                    let shape = quote!(
                        (#from_index..(#to_index + #ignored_len_ident))
                            .into_iter().map(|i| #shape_ident[i]).product::<usize>()
                    );
                    insert_shape(shape);
                }
                Composition::Combined {
                    from: Index::Range(from_index),
                    to: Some(Index::Unknown(to_index)),
                } => {
                    let shape = quote!(
                        (#from_index..=(#to_index + #ignored_len_ident))
                            .into_iter().map(|i| #shape_ident[i]).product::<usize>()
                    );
                    insert_shape(shape);
                }
                Composition::Combined {
                    from: Index::Unknown(from_index),
                    to: Some(Index::Unknown(to_index)),
                } => {
                    let shape = quote!(
                        ((#from_index + #ignored_len_ident - 1)..(#to_index + #ignored_len_ident))
                            .into_iter().map(|i| #shape_ident[i]).product::<usize>()
                    );
                    insert_shape(shape);
                }
                _ => unreachable!(),
            }
            (before_ignored, ignored, after_ignored, is_after_ignored)
        },
    );

    let composition_shape = match (
        before_ignored.is_empty(),
        ignored.is_empty(),
        after_ignored.is_empty(),
    ) {
        (false, true, true) => quote!([#(#before_ignored),*]),
        (false, false, true) => {
            let before_len = before_ignored.len();
            quote!({
                let mut shape: Vec<usize> = Vec::with_capacity(#before_len + #ignored.len());
                shape.extend([#(#before_ignored),*]);
                shape.extend(#ignored);
                shape
            })
        },
        (false, false, false) => {
            let before_len = before_ignored.len();
            let after_len = after_ignored.len();
            quote!({
                let mut shape: Vec<usize> = Vec::with_capacity(#before_len + #ignored.len() + #after_len);
                shape.extend([#(#before_ignored),*]);
                shape.extend(#ignored);
                shape.extend([#(#after_ignored),*]);
                shape
            })
        },
        (true, false, false) => {
            let after_len = after_ignored.len();
            quote!({
                let mut shape: Vec<usize> = Vec::with_capacity(#ignored.len() + #after_len);
                shape.extend(#ignored);
                shape.extend([#(#after_ignored),*]);
                shape
            })
        },
        _ => unreachable!(),
    };

    quote!(
        let #tensor_ident = ::candle_einops::Backend::reshape(#tensor_ident, &#composition_shape);
    )
}

pub fn to_tokens_repeat(
    repeat: &[(Index, Shape)],
    tensor_ident: &syn::Ident,
    ignored_len_ident: &syn::Ident,
    shape_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let n_repeats = repeat.len();
    let repeat_pos_len = repeat.iter().map(|expression| match expression {
        (Index::Known(index), Shape::Lit(len)) => quote!((#index, #len)),
        (Index::Unknown(index), Shape::Lit(len)) => quote!((#index + #ignored_len_ident - 1, #len)),
        (Index::Known(index), Shape::Expr(expr)) => quote!((#index, #expr)),
        (Index::Unknown(index), Shape::Expr(expr)) => {
            quote!((#index + #ignored_len_ident - 1, #expr))
        }
        _ => unreachable!(),
    });

    quote!(
        let #tensor_ident = ::candle_einops::Backend::add_axes(
            #tensor_ident, #shape_ident.len() + #n_repeats, &[#(#repeat_pos_len),*]
        );
    )
}

pub fn to_tokens_permute(
    permute: &[Index],
    tensor_ident: &syn::Ident,
    ignored_len_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let (before_ignored, ignored_permute, after_ignored, _) = permute.iter().fold(
        (
            Vec::new(),
            proc_macro2::TokenStream::new(),
            Vec::new(),
            false,
        ),
        |(mut before_ignored, mut ignored_permute, mut after_ignored, mut is_after_ignored), p| {
            let mut insert_index = |index| {
                if is_after_ignored {
                    after_ignored.push(index);
                } else {
                    before_ignored.push(index);
                }
            };
            match p {
                Index::Known(index) => {
                    insert_index(quote!(#index));
                }
                Index::Range(index) => {
                    is_after_ignored = true;
                    ignored_permute = quote!(
                        (#index..(#index + #ignored_len_ident)).into_iter()
                    )
                }
                Index::Unknown(index) => {
                    insert_index(quote!(#index + #ignored_len_ident - 1));
                }
            };
            (
                before_ignored,
                ignored_permute,
                after_ignored,
                is_after_ignored,
            )
        },
    );

    let permute_indices = match (
        before_ignored.is_empty(),
        ignored_permute.is_empty(),
        after_ignored.is_empty(),
    ) {
        (false, true, true) => quote!([#(#before_ignored),*]),
        (false, false, true) => {
            let before_len = before_ignored.len();
            quote!({
                let mut indices: Vec<usize> = Vec::with_capacity(#before_len + #ignored_permute.len());
                indices.extend([#(#before_ignored),*]);
                indices.extend(#ignored_permute);
                indices
            })
        },
        (false, false, false) => {
            let before_len = before_ignored.len();
            let after_len = after_ignored.len();
            quote!({
                let mut indices: Vec<usize> = Vec::with_capacity(#before_len + #ignored_permute.len() + #after_len);
                indices.extend([#(#before_ignored),*]);
                indices.extend(#ignored_permute);
                indices.extend([#(#after_ignored),*]);
                indices
            })
        },
        (true, false, false) => {
            let after_len = after_ignored.len();
            quote!({
                let mut indices: Vec<usize> = Vec::with_capacity(#ignored_permute.len() + #after_len);
                indices.extend(#ignored_permute);
                indices.extend([#(#after_ignored),*]);
                indices
            })
        },
        _ => unreachable!(),
    };

    quote!(
        let #tensor_ident = ::candle_einops::Backend::transpose(#tensor_ident, &#permute_indices);
    )
}

pub fn to_tokens_reduce(
    reduce: &[(Index, Operation)],
    tensor_ident: &syn::Ident,
    ignored_len_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let (reduce_indices, reduce_operations, ignored_indices, ignored_operations) =
        reduce.iter().fold(
            (Vec::new(), Vec::new(), None, None),
            |(
                mut reduce_indices,
                mut reduce_operations,
                mut ignored_indices,
                mut ignored_operations,
            ),
             expression| {
                let (index, operation) = expression;
                let operation = match operation {
                    Operation::Min => quote!(::candle_einops::Operation::Min),
                    Operation::Max => quote!(::candle_einops::Operation::Max),
                    Operation::Sum => quote!(::candle_einops::Operation::Sum),
                    Operation::Mean => quote!(::candle_einops::Operation::Mean),
                    Operation::Prod => quote!(::candle_einops::Operation::Prod),
                };
                match index {
                    Index::Known(i) => {
                        reduce_indices.push(quote!(#i));
                        reduce_operations.push(operation);
                    }
                    Index::Unknown(i) => {
                        reduce_indices.push(quote!(#i + #ignored_len_ident - 1));
                        reduce_operations.push(operation);
                    }
                    Index::Range(i) => {
                        ignored_indices = Some(quote!((#i..(#i + #ignored_len_ident)).into_iter()));
                        ignored_operations =
                            Some(quote!(std::iter::repeat(#operation).take(#ignored_len_ident)));
                    }
                }
                (
                    reduce_indices,
                    reduce_operations,
                    ignored_indices,
                    ignored_operations,
                )
            },
        );

    match (
        ignored_indices,
        ignored_operations,
        reduce_indices.is_empty(),
    ) {
        (Some(ignored_indices), Some(ignored_operations), true) => {
            quote!(
                let #tensor_ident = ::candle_einops::Backend::reduce_axes(
                    #tensor_ident,
                    &mut #ignored_indices
                        .zip(#ignored_operations)
                        .collect::<Vec<(_, _)>>()
                );
            )
        }

        (Some(ignored_indices), Some(ignored_operations), false) => {
            let reduce_len = reduce_indices.len();
            quote!(
                let #tensor_ident = ::candle_einops::Backend::reduce_axes(
                    #tensor_ident,
                    &mut {
                        let mut axes: Vec<(usize, ::candle_einops::Operation)> = Vec::with_capacity(#reduce_len + #ignored_indices.len());
                        axes.extend([#(#reduce_indices),*].into_iter().zip([#(#reduce_operations),*]));
                        axes.extend(#ignored_indices.zip(#ignored_operations));
                        axes
                    }
                );
            )
        }

        (None, None, false) => {
            quote!(
                let #tensor_ident = ::candle_einops::Backend::reduce_axes(
                    #tensor_ident, &mut [#((#reduce_indices, #reduce_operations)),*]
                );
            )
        }
        _ => unreachable!(),
    }
}

pub fn to_tokens_decomposition(
    left_expression: &[Decomposition],
    tensor_ident: &syn::Ident,
    ignored_len_ident: &syn::Ident,
    shape_ident: &syn::Ident,
) -> proc_macro2::TokenStream {
    let (known_indices, ignored_indices, unknown_indices) = left_expression.iter().fold(
        (Vec::new(), proc_macro2::TokenStream::new(), Vec::new()),
        |(mut known_indices, mut ignored_indices, mut unknown_indices), expression| {
            match expression {
                Decomposition::Named {
                    index: Index::Known(_),
                    shape: Some(Shape::Lit(size)),
                    ..
                } => known_indices.push(quote!(#size)),
                Decomposition::Named {
                    index: Index::Known(_),
                    shape: Some(Shape::Expr(size)),
                    ..
                } => known_indices.push(quote!(#size)),
                Decomposition::Named {
                    index: Index::Known(i),
                    ..
                } => known_indices.push(quote!(#shape_ident[#i])),
                Decomposition::Derived {
                    index: Index::Known(i),
                    shape_calc,
                    ..
                } => known_indices.push(quote!(#shape_ident[#i] / #shape_calc)),
                Decomposition::Named {
                    index: Index::Range(i),
                    ..
                } => {
                    ignored_indices = quote!(
                        (#i..(#i + #ignored_len_ident)).into_iter().map(|i| #shape_ident[i])
                    );
                }
                Decomposition::Named {
                    index: Index::Unknown(_),
                    shape: Some(Shape::Lit(size)),
                    ..
                } => unknown_indices.push(quote!(#size)),
                Decomposition::Named {
                    index: Index::Unknown(_),
                    shape: Some(Shape::Expr(size)),
                    ..
                } => unknown_indices.push(quote!(#size)),
                Decomposition::Named {
                    index: Index::Unknown(i),
                    ..
                } => unknown_indices.push(quote!(#shape_ident[#i + #ignored_len_ident - 1])),
                Decomposition::Derived {
                    index: Index::Unknown(i),
                    shape_calc,
                    ..
                } => unknown_indices
                    .push(quote!(#shape_ident[#i + #ignored_len_ident - 1] / #shape_calc)),
                _ => unreachable!(),
            }
            (known_indices, ignored_indices, unknown_indices)
        },
    );

    let decomposition_shape = match (
        known_indices.is_empty(),
        ignored_indices.is_empty(),
        unknown_indices.is_empty(),
    ) {
        (false, true, true) => {
            quote!([#(#known_indices),*])
        }
        (false, false, true) => {
            let known_len = known_indices.len();
            quote!({
                let mut shape: Vec<usize> = Vec::with_capacity(#known_len + #ignored_indices.len());
                shape.extend([#(#known_indices),*]);
                shape.extend(#ignored_indices);
                shape
            })
        },
        (false, false, false) => {
            let known_len = known_indices.len();
            let unknown_len = unknown_indices.len();
            quote!({
                let mut shape: Vec<usize> = Vec::with_capacity(#known_len + #ignored_indices.len() + #unknown_len);
                shape.extend([#(#known_indices),*]);
                shape.extend(#ignored_indices);
                shape.extend([#(#unknown_indices),*]);
                shape
            })
        },
        (true, false, false) => {
            let unknown_len = unknown_indices.len();
            quote!({
                let mut shape: Vec<usize> = Vec::with_capacity(#ignored_indices.len() + #unknown_len);
                shape.extend(#ignored_indices);
                shape.extend([#(#unknown_indices),*]);
                shape
            })
        },
        (true, false, true) => quote!(
            #ignored_indices.collect::<Vec<_>>()
        ),
        _ => unreachable!(),
    };

    quote!(
        let #tensor_ident = ::candle_einops::Backend::reshape(#tensor_ident, &#decomposition_shape);
    )
}
