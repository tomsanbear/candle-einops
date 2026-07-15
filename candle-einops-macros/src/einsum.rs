use std::collections::HashMap;

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, quote};
use syn::parse::{Parse, ParseStream};

#[cfg(test)]
mod properties;

pub fn einsum(input: TokenStream) -> syn::Result<TokenStream> {
    let invocation = syn::parse2::<Invocation>(input)?;
    Ok(quote!(#invocation))
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AxisId(usize);

#[derive(Debug)]
struct Operand {
    axes: Vec<AxisId>,
    ellipsis_position: Option<usize>,
}

#[derive(Debug)]
struct Equation {
    operands: Vec<Operand>,
    output: Vec<AxisId>,
    output_ellipsis_position: Option<usize>,
    names: Vec<String>,
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
        if output_text.contains(',') {
            return Err(syn::Error::new(
                literal.span(),
                "einsum output must be one whitespace-delimited axis list",
            ));
        }

        let mut interned = HashMap::new();
        let mut names = Vec::new();
        let mut operands = Vec::with_capacity(input_lists.len());
        for input in input_lists {
            let axis_list = parse_axis_list(input, literal.span(), "operand axis list")?;
            let mut axes = Vec::with_capacity(axis_list.labels.len());
            for label in axis_list.labels {
                let axis = *interned.entry(label.clone()).or_insert_with(|| {
                    let axis = AxisId(names.len());
                    names.push(label.clone());
                    axis
                });
                axes.push(axis);
            }
            operands.push(Operand {
                axes,
                ellipsis_position: axis_list.ellipsis_position,
            });
        }

        let output_axis_list = parse_axis_list(output_text, literal.span(), "output axis list")?;
        if output_axis_list.ellipsis_position.is_some()
            && !operands
                .iter()
                .any(|operand| operand.ellipsis_position.is_some())
        {
            return Err(syn::Error::new(
                literal.span(),
                "einsum output `..` requires an input `..`",
            ));
        }
        let mut output = Vec::with_capacity(output_axis_list.labels.len());
        let mut output_names = Vec::with_capacity(output_axis_list.labels.len());
        for label in output_axis_list.labels {
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
            operands,
            output,
            output_ellipsis_position: output_axis_list.ellipsis_position,
            names,
        })
    }

    fn has_ellipsis(&self) -> bool {
        self.output_ellipsis_position.is_some()
            || self
                .operands
                .iter()
                .any(|operand| operand.ellipsis_position.is_some())
    }

    fn has_repeated_input_labels(&self) -> bool {
        self.operands.iter().any(|operand| {
            operand
                .axes
                .iter()
                .enumerate()
                .any(|(index, axis)| operand.axes[..index].contains(axis))
        })
    }

    fn requires_runtime_normalization(&self) -> bool {
        self.has_ellipsis() || self.has_repeated_input_labels()
    }

    fn unary_permutation(&self) -> Vec<usize> {
        self.output
            .iter()
            .chain(
                self.operands[0]
                    .axes
                    .iter()
                    .filter(|axis| !self.output.contains(axis)),
            )
            .map(|axis| axis.0)
            .collect()
    }

    fn binary_plan(&self) -> BinaryPlan {
        let left = &self.operands[0].axes;
        let right = &self.operands[1].axes;
        let in_output = |axis: &AxisId| self.output.contains(axis);
        let in_left = |axis: &AxisId| left.contains(axis);
        let in_right = |axis: &AxisId| right.contains(axis);
        let all_axes = (0..self.names.len()).map(AxisId).collect::<Vec<_>>();

        let batch = all_axes
            .iter()
            .copied()
            .filter(|axis| in_left(axis) && in_right(axis) && in_output(axis))
            .collect::<Vec<_>>();
        let left_free = all_axes
            .iter()
            .copied()
            .filter(|axis| in_left(axis) && !in_right(axis) && in_output(axis))
            .collect::<Vec<_>>();
        let contracted = all_axes
            .iter()
            .copied()
            .filter(|axis| in_left(axis) && in_right(axis) && !in_output(axis))
            .collect::<Vec<_>>();
        let right_free = all_axes
            .iter()
            .copied()
            .filter(|axis| !in_left(axis) && in_right(axis) && in_output(axis))
            .collect::<Vec<_>>();
        let left_reduction_axes = left
            .iter()
            .enumerate()
            .filter_map(|(index, axis)| (!in_right(axis) && !in_output(axis)).then_some(index))
            .collect::<Vec<_>>();
        let right_reduction_axes = right
            .iter()
            .enumerate()
            .filter_map(|(index, axis)| (!in_left(axis) && !in_output(axis)).then_some(index))
            .collect::<Vec<_>>();

        let left_remaining = left
            .iter()
            .copied()
            .filter(|axis| in_right(axis) || in_output(axis))
            .collect::<Vec<_>>();
        let right_remaining = right
            .iter()
            .copied()
            .filter(|axis| in_left(axis) || in_output(axis))
            .collect::<Vec<_>>();
        let left_canonical = batch
            .iter()
            .chain(&left_free)
            .chain(&contracted)
            .copied()
            .collect::<Vec<_>>();
        let right_canonical = batch
            .iter()
            .chain(&contracted)
            .chain(&right_free)
            .copied()
            .collect::<Vec<_>>();
        let left_permutation = permutation_from(&left_remaining, &left_canonical);
        let right_permutation = permutation_from(&right_remaining, &right_canonical);

        let canonical_output = batch
            .iter()
            .chain(&left_free)
            .chain(&right_free)
            .copied()
            .collect::<Vec<_>>();
        let output_permutation = permutation_from(&canonical_output, &self.output);
        let batch_labels = batch
            .iter()
            .map(|axis| self.names[axis.0].clone())
            .collect();
        let contracted_labels = contracted
            .iter()
            .map(|axis| self.names[axis.0].clone())
            .collect();

        BinaryPlan {
            input_ranks: [left.len(), right.len()],
            reduction_axes: [left_reduction_axes, right_reduction_axes],
            permutations: [left_permutation, right_permutation],
            batch_labels,
            left_free_rank: left_free.len(),
            contracted_labels,
            right_free_rank: right_free.len(),
            output_permutation,
        }
    }
}

fn permutation_from(current: &[AxisId], desired: &[AxisId]) -> Vec<usize> {
    desired
        .iter()
        .map(|axis| {
            current
                .iter()
                .position(|candidate| candidate == axis)
                .expect("equation classification must retain every desired axis")
        })
        .collect()
}

#[derive(Debug, PartialEq)]
struct BinaryPlan {
    input_ranks: [usize; 2],
    reduction_axes: [Vec<usize>; 2],
    permutations: [Vec<usize>; 2],
    batch_labels: Vec<String>,
    left_free_rank: usize,
    contracted_labels: Vec<String>,
    right_free_rank: usize,
    output_permutation: Vec<usize>,
}

struct AxisList {
    labels: Vec<String>,
    ellipsis_position: Option<usize>,
}

fn parse_axis_list(text: &str, span: Span, kind: &str) -> syn::Result<AxisList> {
    let mut labels = Vec::new();
    let mut ellipsis_position = None;
    for label in text.split_whitespace() {
        if label == ".." {
            if ellipsis_position.replace(labels.len()).is_some() {
                return Err(syn::Error::new(
                    span,
                    format!("einsum {kind} contains more than one `..`"),
                ));
            }
        } else {
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
            labels.push(label.to_owned());
        }
    }
    Ok(AxisList {
        labels,
        ellipsis_position,
    })
}

struct Invocation {
    runtime_crate: syn::Path,
    operands: Vec<syn::Expr>,
    operand_idents: Vec<Ident>,
    equation: Equation,
}

impl Parse for Invocation {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let literal = input.parse::<syn::LitStr>()?;
        let equation = Equation::parse(&literal)?;
        input.parse::<syn::Token![,]>()?;
        let mut operands = Vec::new();
        loop {
            if input.is_empty() {
                break;
            }
            operands.push(input.parse::<syn::Expr>()?);
            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
        }
        if operands.len() != equation.operands.len() {
            return Err(syn::Error::new(
                literal.span(),
                format!(
                    "einsum equation has {} inputs but received {} operand expressions",
                    equation.operands.len(),
                    operands.len()
                ),
            ));
        }

        let operand_idents = (0..operands.len())
            .map(|index| private_ident(&format!("operand_{index}")))
            .collect();
        Ok(Self {
            runtime_crate: runtime_crate_path()?,
            operands,
            operand_idents,
            equation,
        })
    }
}

impl ToTokens for Invocation {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let Self {
            runtime_crate,
            operands,
            operand_idents,
            equation,
        } = self;
        let bindings = operand_idents
            .iter()
            .zip(operands)
            .map(|(ident, operand)| quote!(let #ident = #operand;));
        let execution = if equation.operands.len() > 2 || equation.requires_runtime_normalization()
        {
            let patterns = equation.operands.iter().map(|pattern| {
                let labels = pattern
                    .axes
                    .iter()
                    .map(|axis| equation.names[axis.0].clone())
                    .collect::<Vec<_>>();
                let position = option_tokens(pattern.ellipsis_position);
                quote!(#runtime_crate::__private::EinsumAxisPattern::new(
                    &[#(#labels),*],
                    #position,
                ))
            });
            let output_labels = equation
                .output
                .iter()
                .map(|axis| equation.names[axis.0].clone())
                .collect::<Vec<_>>();
            let output_position = option_tokens(equation.output_ellipsis_position);
            let spec = quote!(#runtime_crate::__private::EllipsisEinsumSpec::new(
                &[#(#patterns),*],
                #runtime_crate::__private::EinsumAxisPattern::new(
                    &[#(#output_labels),*],
                    #output_position,
                ),
            ));
            if equation.operands.len() == 1 {
                let operand = &operand_idents[0];
                quote!(#runtime_crate::__private::execute_unary_ellipsis_einsum(
                    &#operand,
                    #spec,
                ))
            } else if equation.operands.len() == 2 {
                let left = &operand_idents[0];
                let right = &operand_idents[1];
                quote!(#runtime_crate::__private::execute_binary_ellipsis_einsum(
                    &#left,
                    &#right,
                    #spec,
                ))
            } else {
                let operand_refs = operand_idents.iter().map(
                    |operand| quote!(#runtime_crate::__private::einsum_operand_ref(&#operand)),
                );
                quote!(#runtime_crate::__private::execute_nary_einsum(
                    &[#(#operand_refs),*],
                    #spec,
                ))
            }
        } else if equation.operands.len() == 1 {
            let operand = &operand_idents[0];
            let input_rank = equation.operands[0].axes.len();
            let output_rank = equation.output.len();
            let permutation = equation.unary_permutation();
            quote!(
                #runtime_crate::__private::execute_unary_einsum(
                    &#operand,
                    #runtime_crate::__private::UnaryEinsumSpec::new(
                        #input_rank,
                        #output_rank,
                        &[#(#permutation),*],
                    ),
                )
            )
        } else {
            let left = &operand_idents[0];
            let right = &operand_idents[1];
            let plan = equation.binary_plan();
            let [left_rank, right_rank] = plan.input_ranks;
            let [left_reductions, right_reductions] = plan.reduction_axes;
            let [left_permutation, right_permutation] = plan.permutations;
            let batch_rank = plan.batch_labels.len();
            let left_free_rank = plan.left_free_rank;
            let contracted_rank = plan.contracted_labels.len();
            let right_free_rank = plan.right_free_rank;
            let batch_labels = plan.batch_labels;
            let contracted_labels = plan.contracted_labels;
            let output_permutation = plan.output_permutation;
            quote!(
                #runtime_crate::__private::execute_binary_einsum(
                    &#left,
                    &#right,
                    #runtime_crate::__private::BinaryEinsumSpec::new(
                        [#left_rank, #right_rank],
                        [&[#(#left_reductions),*], &[#(#right_reductions),*]],
                        [&[#(#left_permutation),*], &[#(#right_permutation),*]],
                        #batch_rank,
                        #left_free_rank,
                        #contracted_rank,
                        #right_free_rank,
                        &[#(#batch_labels),*],
                        &[#(#contracted_labels),*],
                        &[#(#output_permutation),*],
                    ),
                )
            )
        };
        quote!({
            #(#bindings)*
            #execution
        })
        .to_tokens(tokens);
    }
}

fn option_tokens(value: Option<usize>) -> TokenStream {
    match value {
        Some(value) => quote!(::core::option::Option::Some(#value)),
        None => quote!(::core::option::Option::None),
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
    fn plans_retained_axes_before_unary_reductions() {
        let literal: syn::LitStr = syn::parse_quote!("a b c -> c a");
        let equation = Equation::parse(&literal).expect("valid unary equation");
        assert_eq!(equation.unary_permutation(), [2, 0, 1]);
    }

    #[test]
    fn classifies_and_canonicalizes_binary_labels() {
        let literal: syn::LitStr = syn::parse_quote!(
            "private batch row inner, batch inner column extra -> column batch row"
        );
        let equation = Equation::parse(&literal).expect("valid binary equation");
        let plan = equation.binary_plan();
        assert_eq!(plan.input_ranks, [4, 4]);
        assert_eq!(plan.reduction_axes, [vec![0], vec![3]]);
        assert_eq!(plan.permutations, [vec![0, 1, 2], vec![0, 1, 2]]);
        assert_eq!(plan.batch_labels, ["batch"]);
        assert_eq!(plan.left_free_rank, 1);
        assert_eq!(plan.contracted_labels, ["inner"]);
        assert_eq!(plan.right_free_rank, 1);
        assert_eq!(plan.output_permutation, [2, 0, 1]);
    }

    #[test]
    fn records_ellipsis_positions_without_changing_named_axes() {
        let literal: syn::LitStr =
            syn::parse_quote!("row .. inner, .. inner column -> row .. column");
        let equation = Equation::parse(&literal).expect("valid ellipsis equation");
        assert_eq!(equation.operands[0].ellipsis_position, Some(1));
        assert_eq!(equation.operands[1].ellipsis_position, Some(0));
        assert_eq!(equation.output_ellipsis_position, Some(1));
        assert_eq!(equation.operands[0].axes.len(), 2);
    }

    #[test]
    fn retains_repeated_labels_for_runtime_diagonal_normalization() {
        let literal: syn::LitStr = syn::parse_quote!("batch i i i -> batch i");
        let equation = Equation::parse(&literal).expect("valid diagonal equation");
        assert!(equation.has_repeated_input_labels());
        assert_eq!(equation.operands[0].axes.len(), 4);
        assert_eq!(equation.operands[0].axes[1], equation.operands[0].axes[2]);
        assert_eq!(equation.operands[0].axes[2], equation.operands[0].axes[3]);
    }
}
