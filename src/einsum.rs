use candle_core::{Result, Tensor};

/// Validated compile-time plan for the unary explicit-output einsum slice.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct UnaryEinsumSpec<'a> {
    input_rank: usize,
    output_rank: usize,
    permutation: &'a [usize],
}

impl<'a> UnaryEinsumSpec<'a> {
    /// Constructs a plan emitted by `candle-einops-macros`.
    #[doc(hidden)]
    pub const fn new(input_rank: usize, output_rank: usize, permutation: &'a [usize]) -> Self {
        Self {
            input_rank,
            output_rank,
            permutation,
        }
    }
}

/// Executes a unary explicit-output einsum plan.
#[doc(hidden)]
pub fn execute_unary_einsum<T>(operand: &T, spec: UnaryEinsumSpec<'_>) -> Result<Tensor>
where
    T: AsRef<Tensor> + ?Sized,
{
    let operand = operand.as_ref();
    if operand.rank() != spec.input_rank {
        candle_core::bail!(
            "einsum operand 0 has rank {}, expected {} for the equation input",
            operand.rank(),
            spec.input_rank,
        )
    }
    if spec.output_rank > spec.input_rank {
        candle_core::bail!(
            "invalid unary einsum plan: output rank {} exceeds input rank {}",
            spec.output_rank,
            spec.input_rank,
        )
    }
    if spec.permutation.len() != spec.input_rank {
        candle_core::bail!(
            "invalid unary einsum plan: permutation has {} axes, expected {}",
            spec.permutation.len(),
            spec.input_rank,
        )
    }

    let mut seen = vec![false; spec.input_rank];
    for &axis in spec.permutation {
        if axis >= spec.input_rank {
            candle_core::bail!(
                "invalid unary einsum plan: permutation axis {axis} is out of range for rank {}",
                spec.input_rank,
            )
        }
        if seen[axis] {
            candle_core::bail!(
                "invalid unary einsum plan: permutation contains axis {axis} more than once"
            )
        }
        seen[axis] = true;
    }

    let is_identity = spec.permutation.iter().copied().eq(0..spec.input_rank);
    let mut output = if is_identity {
        operand.clone()
    } else {
        operand
            .permute(spec.permutation)
            .map_err(|error| error.context("einsum unary permutation"))?
    };

    if spec.output_rank < spec.input_rank {
        let reduction_axes = (spec.output_rank..spec.input_rank).collect::<Vec<_>>();
        output = output
            .sum(reduction_axes.as_slice())
            .map_err(|error| error.context("einsum unary reduction"))?;
    }

    Ok(output)
}

/// Validated compile-time plan for a binary explicit-output einsum.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct BinaryEinsumSpec<'a> {
    input_ranks: [usize; 2],
    reduction_axes: [&'a [usize]; 2],
    permutations: [&'a [usize]; 2],
    batch_rank: usize,
    left_free_rank: usize,
    contracted_rank: usize,
    right_free_rank: usize,
    batch_labels: &'a [&'a str],
    contracted_labels: &'a [&'a str],
    output_permutation: &'a [usize],
}

impl<'a> BinaryEinsumSpec<'a> {
    /// Constructs a plan emitted by `candle-einops-macros`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        input_ranks: [usize; 2],
        reduction_axes: [&'a [usize]; 2],
        permutations: [&'a [usize]; 2],
        batch_rank: usize,
        left_free_rank: usize,
        contracted_rank: usize,
        right_free_rank: usize,
        batch_labels: &'a [&'a str],
        contracted_labels: &'a [&'a str],
        output_permutation: &'a [usize],
    ) -> Self {
        Self {
            input_ranks,
            reduction_axes,
            permutations,
            batch_rank,
            left_free_rank,
            contracted_rank,
            right_free_rank,
            batch_labels,
            contracted_labels,
            output_permutation,
        }
    }
}

/// Executes a binary GEMM-lowered explicit-output einsum plan.
#[doc(hidden)]
pub fn execute_binary_einsum<L, R>(
    left: &L,
    right: &R,
    spec: BinaryEinsumSpec<'_>,
) -> Result<Tensor>
where
    L: AsRef<Tensor> + ?Sized,
    R: AsRef<Tensor> + ?Sized,
{
    let left = left.as_ref();
    let right = right.as_ref();
    for (index, (operand, expected_rank)) in
        [left, right].into_iter().zip(spec.input_ranks).enumerate()
    {
        if operand.rank() != expected_rank {
            candle_core::bail!(
                "einsum operand {index} has rank {}, expected {expected_rank} for the equation input",
                operand.rank(),
            )
        }
    }
    if left.dtype() != right.dtype() {
        candle_core::bail!(
            "einsum operands have different dtypes: left {:?}, right {:?}",
            left.dtype(),
            right.dtype(),
        )
    }
    if !left.device().same_device(right.device()) {
        candle_core::bail!(
            "einsum operands are on different devices: left {:?}, right {:?}",
            left.device(),
            right.device(),
        )
    }

    let left = prepare_operand(
        left,
        0,
        spec.input_ranks[0],
        spec.reduction_axes[0],
        spec.permutations[0],
    )?;
    let right = prepare_operand(
        right,
        1,
        spec.input_ranks[1],
        spec.reduction_axes[1],
        spec.permutations[1],
    )?;

    let left_expected_rank = spec
        .batch_rank
        .checked_add(spec.left_free_rank)
        .and_then(|rank| rank.checked_add(spec.contracted_rank))
        .ok_or_else(|| {
            candle_core::Error::msg("einsum binary canonical left rank overflows usize")
        })?;
    let right_expected_rank = spec
        .batch_rank
        .checked_add(spec.contracted_rank)
        .and_then(|rank| rank.checked_add(spec.right_free_rank))
        .ok_or_else(|| {
            candle_core::Error::msg("einsum binary canonical right rank overflows usize")
        })?;
    if left.rank() != left_expected_rank || right.rank() != right_expected_rank {
        candle_core::bail!(
            "invalid binary einsum plan: canonical ranks are left {}, right {}, expected left {left_expected_rank}, right {right_expected_rank}",
            left.rank(),
            right.rank(),
        )
    }
    if spec.batch_labels.len() != spec.batch_rank
        || spec.contracted_labels.len() != spec.contracted_rank
    {
        candle_core::bail!(
            "invalid binary einsum plan: shared-label metadata does not match canonical ranks"
        )
    }

    let left_dims = left.dims().to_vec();
    let right_dims = right.dims().to_vec();
    let mut batch_dims = Vec::with_capacity(spec.batch_rank);
    for axis in 0..spec.batch_rank {
        batch_dims.push(resolve_extent(
            spec.batch_labels[axis],
            left_dims[axis],
            right_dims[axis],
        )?);
    }
    let left_free_start = spec.batch_rank;
    let contracted_left_start = left_free_start + spec.left_free_rank;
    let contracted_right_start = spec.batch_rank;
    let mut contracted_dims = Vec::with_capacity(spec.contracted_rank);
    for axis in 0..spec.contracted_rank {
        contracted_dims.push(resolve_extent(
            spec.contracted_labels[axis],
            left_dims[contracted_left_start + axis],
            right_dims[contracted_right_start + axis],
        )?);
    }
    let left_free_dims = &left_dims[left_free_start..contracted_left_start];
    let right_free_start = contracted_right_start + spec.contracted_rank;
    let right_free_dims = &right_dims[right_free_start..];

    let mut left_shape = batch_dims.clone();
    left_shape.extend_from_slice(left_free_dims);
    left_shape.extend_from_slice(&contracted_dims);
    let mut right_shape = batch_dims.clone();
    right_shape.extend_from_slice(&contracted_dims);
    right_shape.extend_from_slice(right_free_dims);

    let b = checked_product(&batch_dims, "batch (B)")?;
    let m = checked_product(left_free_dims, "left-free (M)")?;
    let k = checked_product(&contracted_dims, "contracted (K)")?;
    let n = checked_product(right_free_dims, "right-free (N)")?;

    let left = left
        .broadcast_as(left_shape)
        .map_err(|error| error.context("einsum binary left broadcast"))?
        .reshape((b, m, k))
        .map_err(|error| error.context("einsum binary left B/M/K reshape"))?;
    let right = right
        .broadcast_as(right_shape)
        .map_err(|error| error.context("einsum binary right broadcast"))?
        .reshape((b, k, n))
        .map_err(|error| error.context("einsum binary right B/K/N reshape"))?;
    let output = left
        .matmul(&right)
        .map_err(|error| error.context("einsum binary B/M/K/N matmul"))?;

    let mut canonical_output_shape = batch_dims;
    canonical_output_shape.extend_from_slice(left_free_dims);
    canonical_output_shape.extend_from_slice(right_free_dims);
    validate_permutation(
        spec.output_permutation,
        canonical_output_shape.len(),
        "output",
    )?;
    let output = output
        .reshape(canonical_output_shape)
        .map_err(|error| error.context("einsum binary canonical output reshape"))?;
    if spec
        .output_permutation
        .iter()
        .copied()
        .eq(0..spec.output_permutation.len())
    {
        Ok(output)
    } else {
        output
            .permute(spec.output_permutation)
            .map_err(|error| error.context("einsum binary explicit output permutation"))
    }
}

fn prepare_operand(
    operand: &Tensor,
    operand_index: usize,
    input_rank: usize,
    reduction_axes: &[usize],
    permutation: &[usize],
) -> Result<Tensor> {
    let mut reduced = vec![false; input_rank];
    for &axis in reduction_axes {
        if axis >= input_rank {
            candle_core::bail!(
                "invalid binary einsum plan: operand {operand_index} reduction axis {axis} is out of range for rank {input_rank}"
            )
        }
        if std::mem::replace(&mut reduced[axis], true) {
            candle_core::bail!(
                "invalid binary einsum plan: operand {operand_index} reduction axis {axis} occurs more than once"
            )
        }
    }
    let remaining_rank = input_rank - reduction_axes.len();
    validate_permutation(permutation, remaining_rank, "operand")?;
    let operand = if reduction_axes.is_empty() {
        operand.clone()
    } else {
        operand.sum(reduction_axes).map_err(|error| {
            error.context(format!("einsum operand {operand_index} pre-reduction"))
        })?
    };
    if permutation.iter().copied().eq(0..remaining_rank) {
        Ok(operand)
    } else {
        operand.permute(permutation).map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} canonical permutation"
            ))
        })
    }
}

fn validate_permutation(permutation: &[usize], rank: usize, context: &str) -> Result<()> {
    if permutation.len() != rank {
        candle_core::bail!(
            "invalid binary einsum plan: {context} permutation has {} axes, expected {rank}",
            permutation.len(),
        )
    }
    let mut seen = vec![false; rank];
    for &axis in permutation {
        if axis >= rank || std::mem::replace(&mut seen[axis], true) {
            candle_core::bail!(
                "invalid binary einsum plan: {context} permutation is not a permutation of 0..{rank}"
            )
        }
    }
    Ok(())
}

fn resolve_extent(label: &str, left: usize, right: usize) -> Result<usize> {
    match (left, right) {
        (left, right) if left == right => Ok(left),
        (1, right) => Ok(right),
        (left, 1) => Ok(left),
        _ => {
            candle_core::bail!("einsum label `{label}` cannot broadcast extents {left} and {right}")
        }
    }
}

fn checked_product(dimensions: &[usize], category: &str) -> Result<usize> {
    dimensions.iter().try_fold(1_usize, |product, &extent| {
        product.checked_mul(extent).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "einsum binary {category} flattened extent overflows usize"
            ))
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn rejects_invalid_runtime_specs_without_panicking() -> Result<()> {
        let input = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;

        assert!(execute_unary_einsum(&input, UnaryEinsumSpec::new(3, 2, &[0, 1, 2])).is_err());
        assert!(execute_unary_einsum(&input, UnaryEinsumSpec::new(2, 3, &[0, 1])).is_err());
        assert!(execute_unary_einsum(&input, UnaryEinsumSpec::new(2, 2, &[0])).is_err());
        assert!(execute_unary_einsum(&input, UnaryEinsumSpec::new(2, 2, &[0, 2])).is_err());
        assert!(execute_unary_einsum(&input, UnaryEinsumSpec::new(2, 2, &[0, 0])).is_err());

        Ok(())
    }

    #[test]
    fn rejects_invalid_binary_specs_and_checked_shape_overflow() -> Result<()> {
        let left = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        let right = Tensor::zeros((3, 4), DType::F32, &Device::Cpu)?;
        let valid = BinaryEinsumSpec::new(
            [2, 2],
            [&[], &[]],
            [&[0, 1], &[0, 1]],
            0,
            1,
            1,
            1,
            &[],
            &["inner"],
            &[0, 1],
        );
        assert_eq!(execute_binary_einsum(&left, &right, valid)?.dims(), &[2, 4]);

        let invalid = BinaryEinsumSpec::new(
            [2, 2],
            [&[], &[]],
            [&[0, 0], &[0, 1]],
            0,
            1,
            1,
            1,
            &[],
            &["inner"],
            &[0, 1],
        );
        assert!(execute_binary_einsum(&left, &right, invalid).is_err());
        assert!(checked_product(&[usize::MAX, 2], "test").is_err());
        Ok(())
    }
}
