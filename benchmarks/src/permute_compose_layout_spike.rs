//! Bounded decision model for the permute-plus-composition layout spike.
//!
//! This module deliberately models layouts without constructing tensors from
//! custom strides. Candle exposes layout inspection, but its safe public tensor
//! constructors cannot attach an arbitrary layout to shared storage.

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
