//! Bounded decision model for the permute-plus-composition layout spike.
//!
//! This module deliberately models layouts without constructing tensors from
//! custom strides. Candle exposes layout inspection, but its safe public tensor
//! constructors cannot attach an arbitrary layout to shared storage.

use candle_core::{Result as CandleResult, Tensor};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayoutSpec {
    dims: Vec<usize>,
    strides: Vec<usize>,
    start_offset: usize,
}

impl LayoutSpec {
    pub fn new(
        dims: &[usize],
        strides: &[usize],
        start_offset: usize,
    ) -> Result<Self, ClassifierError> {
        if dims.len() != strides.len() {
            return Err(ClassifierError::RankMismatch {
                dims: dims.len(),
                strides: strides.len(),
            });
        }
        Ok(Self {
            dims: dims.to_vec(),
            strides: strides.to_vec(),
            start_offset,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CollapseDecision {
    /// Exact dimensions are already handled by the runtime identity-reshape
    /// elision and must not motivate another fused API.
    ExistingIdentityElision,
    /// Candle's existing `permute` result is C-contiguous under its singleton
    /// rules, so its public `reshape` already returns a storage-sharing view.
    ExistingPublicReshapeView {
        output_dims: Vec<usize>,
        output_strides: Vec<usize>,
        start_offset: usize,
    },
    /// The requested groups have a valid affine layout, but Candle 0.11 has no
    /// safe public API that attaches these strides to aliased tensor storage.
    LayoutOnlyCandidate {
        output_dims: Vec<usize>,
        output_strides: Vec<usize>,
        start_offset: usize,
    },
    CopyRequired(CopyReason),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CopyReason {
    NonCollapsibleBoundary {
        outer_axis: usize,
        inner_axis: usize,
    },
    /// Empty layouts have no observable element order. The spike refuses to
    /// infer a novel view from vacuous equality; existing identity behavior is
    /// still recognized before reaching this fallback.
    ZeroExtentConservative,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClassifierError {
    RankMismatch { dims: usize, strides: usize },
    InvalidPermutation,
    InvalidGroups,
    ElementCountOverflow,
    StrideOverflow,
    TooManyGroups,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicFusionPlan {
    ExistingIdentityElision,
    ExistingRouteView {
        expanded_permutation: Vec<usize>,
        output_dims: Vec<usize>,
    },
    ReorderedPublicViews {
        pre_permutation: Vec<usize>,
        reshape_dims: Vec<usize>,
        post_permutation: Vec<usize>,
    },
    Fallback {
        expanded_permutation: Vec<usize>,
        output_dims: Vec<usize>,
    },
}

/// Finds a safe public-operation plan for the requested output groups.
///
/// Each group contains original input axes in the logical flatten order. A
/// candidate is viable only when some ordering of whole groups makes the actual
/// input layout C-contiguous under Candle's exact singleton/zero rules. The
/// plan then permutes whole groups back after reshaping; it never constructs a
/// tensor from arbitrary strides.
pub fn classify_public_fusion(
    input: &LayoutSpec,
    desired_groups: &[Vec<usize>],
) -> Result<PublicFusionPlan, ClassifierError> {
    validate_original_axis_groups(input.dims.len(), desired_groups)?;
    let expanded_permutation = desired_groups.iter().flatten().copied().collect::<Vec<_>>();
    let output_dims = group_products(&input.dims, desired_groups)?;
    if expanded_permutation.iter().copied().eq(0..input.dims.len())
        && desired_groups.iter().all(|group| group.len() == 1)
    {
        return Ok(PublicFusionPlan::ExistingIdentityElision);
    }
    let desired_dims = expanded_permutation
        .iter()
        .map(|&axis| input.dims[axis])
        .collect::<Vec<_>>();
    let desired_strides = expanded_permutation
        .iter()
        .map(|&axis| input.strides[axis])
        .collect::<Vec<_>>();
    if is_c_contiguous(&desired_dims, &desired_strides)? {
        return Ok(PublicFusionPlan::ExistingRouteView {
            expanded_permutation,
            output_dims,
        });
    }
    if desired_groups.len() > 8 {
        return Err(ClassifierError::TooManyGroups);
    }

    for group_order in permutations(desired_groups.len()) {
        let pre_permutation = group_order
            .iter()
            .flat_map(|&group| desired_groups[group].iter().copied())
            .collect::<Vec<_>>();
        let dims = pre_permutation
            .iter()
            .map(|&axis| input.dims[axis])
            .collect::<Vec<_>>();
        let strides = pre_permutation
            .iter()
            .map(|&axis| input.strides[axis])
            .collect::<Vec<_>>();
        if !is_c_contiguous(&dims, &strides)? {
            continue;
        }
        let reshape_dims = group_order
            .iter()
            .map(|&group| output_dims[group])
            .collect::<Vec<_>>();
        let post_permutation = (0..desired_groups.len())
            .map(|desired| {
                group_order
                    .iter()
                    .position(|&group| group == desired)
                    .expect("group order is a permutation")
            })
            .collect::<Vec<_>>();
        return Ok(PublicFusionPlan::ReorderedPublicViews {
            pre_permutation,
            reshape_dims,
            post_permutation,
        });
    }

    Ok(PublicFusionPlan::Fallback {
        expanded_permutation,
        output_dims,
    })
}

pub fn run_public_fusion_prototype(
    input: &Tensor,
    desired_groups: &[Vec<usize>],
) -> CandleResult<(Tensor, PublicFusionPlan)> {
    let spec = LayoutSpec::new(input.dims(), input.stride(), input.layout().start_offset())
        .map_err(|error| candle_core::Error::Msg(format!("invalid spike layout: {error:?}")))?;
    let plan = classify_public_fusion(&spec, desired_groups)
        .map_err(|error| candle_core::Error::Msg(format!("invalid fusion request: {error:?}")))?;
    let output = match &plan {
        PublicFusionPlan::ExistingIdentityElision => input.clone(),
        PublicFusionPlan::ExistingRouteView {
            expanded_permutation,
            output_dims,
        }
        | PublicFusionPlan::Fallback {
            expanded_permutation,
            output_dims,
        } => input
            .permute(expanded_permutation.as_slice())?
            .reshape(output_dims.as_slice())?,
        PublicFusionPlan::ReorderedPublicViews {
            pre_permutation,
            reshape_dims,
            post_permutation,
        } => input
            .permute(pre_permutation.as_slice())?
            .reshape(reshape_dims.as_slice())?
            .permute(post_permutation.as_slice())?,
    };
    Ok((output, plan))
}

fn validate_original_axis_groups(
    rank: usize,
    groups: &[Vec<usize>],
) -> Result<(), ClassifierError> {
    if groups.iter().any(Vec::is_empty) {
        return Err(ClassifierError::InvalidGroups);
    }
    let mut axes = groups.iter().flatten().copied().collect::<Vec<_>>();
    axes.sort_unstable();
    if axes != (0..rank).collect::<Vec<_>>() {
        return Err(ClassifierError::InvalidGroups);
    }
    Ok(())
}

fn group_products(dims: &[usize], groups: &[Vec<usize>]) -> Result<Vec<usize>, ClassifierError> {
    groups
        .iter()
        .map(|group| {
            group.iter().try_fold(1usize, |product, &axis| {
                product
                    .checked_mul(dims[axis])
                    .ok_or(ClassifierError::ElementCountOverflow)
            })
        })
        .collect()
}

fn permutations(count: usize) -> Vec<Vec<usize>> {
    fn extend(prefix: &mut Vec<usize>, remaining: &mut Vec<usize>, output: &mut Vec<Vec<usize>>) {
        if remaining.is_empty() {
            output.push(prefix.clone());
            return;
        }
        for index in 0..remaining.len() {
            let item = remaining.remove(index);
            prefix.push(item);
            extend(prefix, remaining, output);
            prefix.pop();
            remaining.insert(index, item);
        }
    }
    let mut output = Vec::new();
    extend(
        &mut Vec::with_capacity(count),
        &mut (0..count).collect(),
        &mut output,
    );
    output
}

pub fn classify_permute_compose(
    input: &LayoutSpec,
    permutation: &[usize],
    groups: &[Vec<usize>],
) -> Result<CollapseDecision, ClassifierError> {
    let rank = input.dims.len();
    if permutation.len() != rank
        || !(0..rank).all(|axis| permutation.iter().filter(|&&item| item == axis).count() == 1)
    {
        return Err(ClassifierError::InvalidPermutation);
    }
    let flattened_groups = groups.iter().flatten().copied().collect::<Vec<_>>();
    if groups.iter().any(Vec::is_empty) || flattened_groups != (0..rank).collect::<Vec<_>>() {
        return Err(ClassifierError::InvalidGroups);
    }

    if permutation.iter().copied().eq(0..rank)
        && groups
            .iter()
            .enumerate()
            .all(|(axis, group)| group.as_slice() == [axis])
    {
        return Ok(CollapseDecision::ExistingIdentityElision);
    }

    let dims = permutation
        .iter()
        .map(|&axis| input.dims[axis])
        .collect::<Vec<_>>();
    let strides = permutation
        .iter()
        .map(|&axis| input.strides[axis])
        .collect::<Vec<_>>();
    if dims.contains(&0) {
        return Ok(CollapseDecision::CopyRequired(
            CopyReason::ZeroExtentConservative,
        ));
    }

    let mut output_dims = Vec::with_capacity(groups.len());
    let mut output_strides = Vec::with_capacity(groups.len());
    for group in groups {
        let output_dim = group.iter().try_fold(1usize, |product, &axis| {
            product
                .checked_mul(dims[axis])
                .ok_or(ClassifierError::ElementCountOverflow)
        })?;
        let non_singleton = group
            .iter()
            .copied()
            .filter(|&axis| dims[axis] > 1)
            .collect::<Vec<_>>();
        for pair in non_singleton.windows(2) {
            let outer = pair[0];
            let inner = pair[1];
            let expected =
                dims[outer + 1..=inner]
                    .iter()
                    .try_fold(strides[inner], |stride, &dim| {
                        stride
                            .checked_mul(dim)
                            .ok_or(ClassifierError::StrideOverflow)
                    })?;
            if strides[outer] != expected {
                return Ok(CollapseDecision::CopyRequired(
                    CopyReason::NonCollapsibleBoundary {
                        outer_axis: outer,
                        inner_axis: inner,
                    },
                ));
            }
        }
        output_dims.push(output_dim);
        output_strides.push(non_singleton.last().map_or_else(
            || strides[*group.last().expect("groups are non-empty")],
            |&axis| strides[axis],
        ));
    }

    if is_c_contiguous(&dims, &strides)? {
        output_strides = contiguous_strides(&output_dims)?;
        Ok(CollapseDecision::ExistingPublicReshapeView {
            output_dims,
            output_strides,
            start_offset: input.start_offset,
        })
    } else {
        Ok(CollapseDecision::LayoutOnlyCandidate {
            output_dims,
            output_strides,
            start_offset: input.start_offset,
        })
    }
}

fn is_c_contiguous(dims: &[usize], strides: &[usize]) -> Result<bool, ClassifierError> {
    let mut expected = 1usize;
    for (&dim, &stride) in dims.iter().zip(strides).rev() {
        if dim > 1 && stride != expected {
            return Ok(false);
        }
        expected = expected
            .checked_mul(dim)
            .ok_or(ClassifierError::StrideOverflow)?;
    }
    Ok(true)
}

fn contiguous_strides(dims: &[usize]) -> Result<Vec<usize>, ClassifierError> {
    let mut output = vec![0; dims.len()];
    let mut stride = 1usize;
    for (axis, &dim) in dims.iter().enumerate().rev() {
        output[axis] = stride;
        stride = stride
            .checked_mul(dim)
            .ok_or(ClassifierError::StrideOverflow)?;
    }
    Ok(output)
}
