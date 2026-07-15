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

/// One compile-time axis-list pattern containing at most one runtime ellipsis.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct EinsumAxisPattern<'a> {
    labels: &'a [&'a str],
    ellipsis_position: Option<usize>,
}

impl<'a> EinsumAxisPattern<'a> {
    /// Constructs an axis pattern emitted by `candle-einops-macros`.
    #[doc(hidden)]
    pub const fn new(labels: &'a [&'a str], ellipsis_position: Option<usize>) -> Self {
        Self {
            labels,
            ellipsis_position,
        }
    }
}

/// Runtime-normalized plan for an equation containing ellipses, repeated
/// labels, or more than two operands.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct EllipsisEinsumSpec<'a> {
    operands: &'a [EinsumAxisPattern<'a>],
    output: EinsumAxisPattern<'a>,
}

impl<'a> EllipsisEinsumSpec<'a> {
    /// Constructs an ellipsis plan emitted by `candle-einops-macros`.
    #[doc(hidden)]
    pub const fn new(operands: &'a [EinsumAxisPattern<'a>], output: EinsumAxisPattern<'a>) -> Self {
        Self { operands, output }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BinaryExecution {
    Multiply,
    CanonicalMatmul,
    General,
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
    execute_binary_with(left, right, spec, BinaryExecution::General)
}

/// Executes a binary equation with no contracted labels as broadcast multiplication.
#[doc(hidden)]
pub fn execute_binary_multiply<L, R>(
    left: &L,
    right: &R,
    spec: BinaryEinsumSpec<'_>,
) -> Result<Tensor>
where
    L: AsRef<Tensor> + ?Sized,
    R: AsRef<Tensor> + ?Sized,
{
    execute_binary_with(left, right, spec, BinaryExecution::Multiply)
}

/// Executes a canonical rank-two or rank-three contraction with direct matmul.
#[doc(hidden)]
pub fn execute_canonical_binary_einsum<L, R>(
    left: &L,
    right: &R,
    spec: BinaryEinsumSpec<'_>,
) -> Result<Tensor>
where
    L: AsRef<Tensor> + ?Sized,
    R: AsRef<Tensor> + ?Sized,
{
    execute_binary_with(left, right, spec, BinaryExecution::CanonicalMatmul)
}

fn execute_binary_with<L, R>(
    left: &L,
    right: &R,
    spec: BinaryEinsumSpec<'_>,
    requested_execution: BinaryExecution,
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

    let mut canonical_output_shape = batch_dims.clone();
    canonical_output_shape.extend_from_slice(left_free_dims);
    canonical_output_shape.extend_from_slice(right_free_dims);
    validate_permutation(
        spec.output_permutation,
        canonical_output_shape.len(),
        "output",
    )?;

    let execution = if spec.contracted_rank == 0 {
        BinaryExecution::Multiply
    } else {
        requested_execution
    };

    if execution == BinaryExecution::CanonicalMatmul {
        let canonical = spec.batch_rank <= 1
            && spec.left_free_rank == 1
            && spec.contracted_rank == 1
            && spec.right_free_rank == 1
            && spec.reduction_axes.iter().all(|axes| axes.is_empty())
            && spec
                .permutations
                .iter()
                .all(|permutation| permutation.iter().copied().eq(0..permutation.len()));
        if !canonical {
            candle_core::bail!("invalid binary einsum plan: direct matmul path is not canonical")
        }
    }

    if execution != BinaryExecution::Multiply && (b == 0 || m == 0 || k == 0 || n == 0) {
        let output = graph_preserving_zero(&left, &right, &canonical_output_shape)?;
        return apply_output_permutation(output, spec.output_permutation);
    }

    if execution == BinaryExecution::Multiply {
        if spec.contracted_rank != 0 {
            candle_core::bail!(
                "invalid binary einsum plan: multiply fast path received contracted axes"
            )
        }
        let mut left = left;
        for _ in 0..spec.right_free_rank {
            left = left.unsqueeze(left.rank()).map_err(|error| {
                error.context("einsum binary multiply left free-axis alignment")
            })?;
        }
        let mut right = right;
        for _ in 0..spec.left_free_rank {
            right = right.unsqueeze(spec.batch_rank).map_err(|error| {
                error.context("einsum binary multiply right free-axis alignment")
            })?;
        }
        let output = left
            .broadcast_mul(&right)
            .map_err(|error| error.context("einsum binary broadcast multiplication"))?;
        return apply_output_permutation(output, spec.output_permutation);
    }

    if execution == BinaryExecution::CanonicalMatmul {
        let left = broadcast_if_needed(&left, &left_shape, "einsum binary left broadcast")?;
        let right = broadcast_if_needed(&right, &right_shape, "einsum binary right broadcast")?;
        let output = left
            .matmul(&right)
            .map_err(|error| error.context("einsum binary B/M/K/N matmul"))?;
        return apply_output_permutation(output, spec.output_permutation);
    }

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
    let output = output
        .reshape(canonical_output_shape)
        .map_err(|error| error.context("einsum binary canonical output reshape"))?;
    apply_output_permutation(output, spec.output_permutation)
}

fn graph_preserving_zero(left: &Tensor, right: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let zero_anchor = |operand: &Tensor, side: &'static str| {
        operand
            .unsqueeze(0)
            .and_then(|operand| operand.narrow(0, 0, 0))
            .and_then(|operand| operand.sum_all())
            .map_err(|error| error.context(format!("einsum binary {side} zero anchor")))
    };
    zero_anchor(left, "left")?
        .add(&zero_anchor(right, "right")?)
        .map_err(|error| error.context("einsum binary zero anchors"))?
        .broadcast_as(shape)
        .map_err(|error| error.context("einsum binary zero output broadcast"))
}

/// Expands and executes a unary equation containing an ellipsis.
#[doc(hidden)]
pub fn execute_unary_ellipsis_einsum<T>(operand: &T, spec: EllipsisEinsumSpec<'_>) -> Result<Tensor>
where
    T: AsRef<Tensor> + ?Sized,
{
    if spec.operands.len() != 1 {
        candle_core::bail!(
            "invalid ellipsis einsum plan: unary execution received {} operand patterns",
            spec.operands.len()
        )
    }
    let operand = operand.as_ref();
    let capture = ellipsis_capture(operand, 0, spec.operands[0])?;
    let normalized = normalize_ellipsis_operand(operand, spec.operands[0], capture, capture, 0)?;
    let input_axes = expand_axis_pattern(spec.operands[0], capture, true);
    let (normalized, input_axes) = normalize_repeated_axes(normalized, input_axes, 0)?;
    let output_axes = expand_axis_pattern(spec.output, capture, false);
    validate_expanded_output(&[&input_axes], &output_axes)?;
    let permutation = output_axes
        .iter()
        .chain(input_axes.iter().filter(|axis| !output_axes.contains(axis)))
        .map(|axis| {
            input_axes
                .iter()
                .position(|candidate| candidate == axis)
                .expect("validated unary ellipsis axis")
        })
        .collect::<Vec<_>>();
    execute_unary_einsum(
        &normalized,
        UnaryEinsumSpec::new(input_axes.len(), output_axes.len(), &permutation),
    )
}

/// Expands and executes a binary equation containing an ellipsis.
#[doc(hidden)]
pub fn execute_binary_ellipsis_einsum<L, R>(
    left: &L,
    right: &R,
    spec: EllipsisEinsumSpec<'_>,
) -> Result<Tensor>
where
    L: AsRef<Tensor> + ?Sized,
    R: AsRef<Tensor> + ?Sized,
{
    if spec.operands.len() != 2 {
        candle_core::bail!(
            "invalid ellipsis einsum plan: binary execution received {} operand patterns",
            spec.operands.len()
        )
    }
    let left = left.as_ref();
    let right = right.as_ref();
    let captures = [
        ellipsis_capture(left, 0, spec.operands[0])?,
        ellipsis_capture(right, 1, spec.operands[1])?,
    ];
    let maximum_capture = captures[0].max(captures[1]);
    let left = normalize_ellipsis_operand(left, spec.operands[0], captures[0], maximum_capture, 0)?;
    let right =
        normalize_ellipsis_operand(right, spec.operands[1], captures[1], maximum_capture, 1)?;
    let left_axes = expand_axis_pattern(spec.operands[0], maximum_capture, true);
    let right_axes = expand_axis_pattern(spec.operands[1], maximum_capture, true);
    let (left, left_axes) = normalize_repeated_axes(left, left_axes, 0)?;
    let (right, right_axes) = normalize_repeated_axes(right, right_axes, 1)?;
    let output_axes = expand_axis_pattern(spec.output, maximum_capture, false);
    validate_expanded_output(&[&left_axes, &right_axes], &output_axes)?;
    execute_expanded_binary(&left, &right, &left_axes, &right_axes, &output_axes)
}

/// Converts one generated operand binding to the tensor reference used by the
/// arbitrary-arity runtime ABI.
#[doc(hidden)]
pub fn einsum_operand_ref<T>(operand: &T) -> &Tensor
where
    T: AsRef<Tensor> + ?Sized,
{
    operand.as_ref()
}

/// Normalizes and greedily contracts an arbitrary number of operands.
#[doc(hidden)]
pub fn execute_nary_einsum(operands: &[&Tensor], spec: EllipsisEinsumSpec<'_>) -> Result<Tensor> {
    if operands.is_empty() {
        candle_core::bail!("invalid n-ary einsum plan: at least one operand is required")
    }
    if spec.operands.len() != operands.len() {
        candle_core::bail!(
            "invalid n-ary einsum plan: received {} tensors but {} operand patterns",
            operands.len(),
            spec.operands.len()
        )
    }
    let first = operands[0];
    for (index, operand) in operands.iter().copied().enumerate().skip(1) {
        if operand.dtype() != first.dtype() {
            candle_core::bail!(
                "einsum operands have different dtypes: operand 0 {:?}, operand {index} {:?}",
                first.dtype(),
                operand.dtype()
            )
        }
        if !operand.device().same_device(first.device()) {
            candle_core::bail!(
                "einsum operands are on different devices: operand 0 {:?}, operand {index} {:?}",
                first.device(),
                operand.device()
            )
        }
    }

    let captures = operands
        .iter()
        .zip(spec.operands)
        .enumerate()
        .map(|(index, (operand, pattern))| ellipsis_capture(operand, index, *pattern))
        .collect::<Result<Vec<_>>>()?;
    let maximum_capture = captures.iter().copied().max().unwrap_or(0);
    let mut planned = Vec::with_capacity(operands.len());
    for (index, ((operand, pattern), capture)) in
        operands.iter().zip(spec.operands).zip(captures).enumerate()
    {
        let normalized =
            normalize_ellipsis_operand(operand, *pattern, capture, maximum_capture, index)?;
        let axes = expand_axis_pattern(*pattern, maximum_capture, true);
        let (tensor, axes) = normalize_repeated_axes(normalized, axes, index)?;
        planned.push(PlannedOperand {
            tensor,
            axes,
            stable_ordinal: index,
        });
    }
    let output_axes = expand_axis_pattern(spec.output, maximum_capture, false);
    let input_axes = planned
        .iter()
        .map(|operand| operand.axes.as_slice())
        .collect::<Vec<_>>();
    validate_expanded_output(&input_axes, &output_axes)?;
    validate_nary_broadcasts(&planned)?;
    let global_axis_order = stable_axis_order(&planned);

    while planned.len() > 1 {
        let selected = select_nary_pair_with_order(&planned, &output_axes, &global_axis_order)?;
        let right = planned.remove(selected.right);
        let left = planned.remove(selected.left);
        let tensor = execute_expanded_binary(
            &left.tensor,
            &right.tensor,
            &left.axes,
            &right.axes,
            &selected.output_axes,
        )?;
        planned.insert(
            selected.left,
            PlannedOperand {
                tensor,
                axes: selected.output_axes,
                stable_ordinal: left.stable_ordinal.min(right.stable_ordinal),
            },
        );
    }

    let final_operand = planned
        .pop()
        .expect("non-empty n-ary plan retains one operand");
    let permutation = output_axes
        .iter()
        .chain(
            final_operand
                .axes
                .iter()
                .filter(|axis| !output_axes.contains(axis)),
        )
        .map(|axis| {
            final_operand
                .axes
                .iter()
                .position(|candidate| candidate == axis)
                .expect("validated final n-ary output axis")
        })
        .collect::<Vec<_>>();
    execute_unary_einsum(
        &final_operand.tensor,
        UnaryEinsumSpec::new(final_operand.axes.len(), output_axes.len(), &permutation),
    )
}

fn execute_expanded_binary(
    left: &Tensor,
    right: &Tensor,
    left_axes: &[ExpandedAxis<'_>],
    right_axes: &[ExpandedAxis<'_>],
    output_axes: &[ExpandedAxis<'_>],
) -> Result<Tensor> {
    let plan = classify_expanded_binary(left_axes, right_axes, output_axes);
    let batch_label_storage = plan
        .batch
        .iter()
        .map(ExpandedAxis::display_name)
        .collect::<Vec<_>>();
    let contracted_label_storage = plan
        .contracted
        .iter()
        .map(ExpandedAxis::display_name)
        .collect::<Vec<_>>();
    let batch_labels = batch_label_storage
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let contracted_labels = contracted_label_storage
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    execute_binary_einsum(
        left,
        right,
        BinaryEinsumSpec::new(
            [left_axes.len(), right_axes.len()],
            [&plan.left_reductions, &plan.right_reductions],
            [&plan.left_permutation, &plan.right_permutation],
            plan.batch.len(),
            plan.left_free.len(),
            plan.contracted.len(),
            plan.right_free.len(),
            &batch_labels,
            &contracted_labels,
            &plan.output_permutation,
        ),
    )
}

struct PlannedOperand<'a> {
    tensor: Tensor,
    axes: Vec<ExpandedAxis<'a>>,
    stable_ordinal: usize,
}

impl<'a> PlannedOperand<'a> {
    #[cfg(test)]
    fn new_for_test(
        shape: (usize, usize),
        labels: &[&'a str],
        stable_ordinal: usize,
    ) -> Result<Self> {
        Ok(Self {
            tensor: Tensor::zeros(shape, candle_core::DType::F32, &candle_core::Device::Cpu)?,
            axes: labels.iter().copied().map(ExpandedAxis::Named).collect(),
            stable_ordinal,
        })
    }
}

#[derive(Debug)]
struct PairEstimate<'a> {
    left: usize,
    right: usize,
    output_axes: Vec<ExpandedAxis<'a>>,
    output_elements: u128,
    flops: u128,
}

fn stable_axis_order<'a>(operands: &[PlannedOperand<'a>]) -> Vec<ExpandedAxis<'a>> {
    let mut axes = Vec::new();
    let mut ordered = operands.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|operand| operand.stable_ordinal);
    for operand in ordered {
        for &axis in &operand.axes {
            if !axes.contains(&axis) {
                axes.push(axis);
            }
        }
    }
    axes
}

fn validate_nary_broadcasts(operands: &[PlannedOperand<'_>]) -> Result<()> {
    let mut dimensions = Vec::<(ExpandedAxis<'_>, usize)>::new();
    for operand in operands {
        for (&axis, &extent) in operand.axes.iter().zip(operand.tensor.dims()) {
            if let Some((_, resolved)) = dimensions
                .iter_mut()
                .find(|(candidate, _)| *candidate == axis)
            {
                *resolved = resolve_extent(&axis.display_name(), *resolved, extent)?;
            } else {
                dimensions.push((axis, extent));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn select_nary_pair<'a>(
    operands: &[PlannedOperand<'a>],
    final_output: &[ExpandedAxis<'a>],
) -> Result<PairEstimate<'a>> {
    let global_axis_order = stable_axis_order(operands);
    select_nary_pair_with_order(operands, final_output, &global_axis_order)
}

fn select_nary_pair_with_order<'a>(
    operands: &[PlannedOperand<'a>],
    final_output: &[ExpandedAxis<'a>],
    global_axis_order: &[ExpandedAxis<'a>],
) -> Result<PairEstimate<'a>> {
    if operands.len() < 2 {
        candle_core::bail!("invalid n-ary einsum planner state: fewer than two operands")
    }
    let mut best: Option<PairEstimate<'a>> = None;
    for left in 0..operands.len() - 1 {
        for right in left + 1..operands.len() {
            let candidate =
                estimate_pair_with_order(operands, left, right, final_output, global_axis_order)?;
            let candidate_key = (
                candidate.output_elements,
                candidate.flops,
                operands[left].stable_ordinal,
                operands[right].stable_ordinal,
                left,
                right,
            );
            let replace = best.as_ref().is_none_or(|current| {
                candidate_key
                    < (
                        current.output_elements,
                        current.flops,
                        operands[current.left].stable_ordinal,
                        operands[current.right].stable_ordinal,
                        current.left,
                        current.right,
                    )
            });
            if replace {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| candle_core::Error::msg("n-ary einsum planner found no operand pair"))
}

#[cfg(test)]
fn estimate_pair<'a>(
    operands: &[PlannedOperand<'a>],
    left: usize,
    right: usize,
    final_output: &[ExpandedAxis<'a>],
) -> Result<PairEstimate<'a>> {
    let global_axis_order = stable_axis_order(operands);
    estimate_pair_with_order(operands, left, right, final_output, &global_axis_order)
}

fn estimate_pair_with_order<'a>(
    operands: &[PlannedOperand<'a>],
    left: usize,
    right: usize,
    final_output: &[ExpandedAxis<'a>],
    global_axis_order: &[ExpandedAxis<'a>],
) -> Result<PairEstimate<'a>> {
    let left_operand = operands.get(left).ok_or_else(|| {
        candle_core::Error::msg(format!(
            "n-ary einsum planner left index {left} is out of range"
        ))
    })?;
    let right_operand = operands.get(right).ok_or_else(|| {
        candle_core::Error::msg(format!(
            "n-ary einsum planner right index {right} is out of range"
        ))
    })?;
    if left >= right {
        candle_core::bail!("n-ary einsum planner pair must be ordered, received ({left}, {right})")
    }

    let mut pair_axes = Vec::new();
    for &axis in left_operand.axes.iter().chain(&right_operand.axes) {
        if !pair_axes.contains(&axis) {
            pair_axes.push(axis);
        }
    }
    let mut live_axes = final_output.to_vec();
    for (index, operand) in operands.iter().enumerate() {
        if index != left && index != right {
            for &axis in &operand.axes {
                if !live_axes.contains(&axis) {
                    live_axes.push(axis);
                }
            }
        }
    }
    let output_axes = global_axis_order
        .iter()
        .copied()
        .filter(|axis| pair_axes.contains(axis) && live_axes.contains(axis))
        .collect::<Vec<_>>();
    let output_extents = output_axes
        .iter()
        .map(|axis| pair_axis_extent(left_operand, right_operand, *axis))
        .collect::<Result<Vec<_>>>()?;
    let flop_extents = pair_axes
        .iter()
        .map(|axis| pair_axis_extent(left_operand, right_operand, *axis))
        .collect::<Result<Vec<_>>>()?;
    let output_elements = checked_nary_product(&output_extents).map_err(|error| {
        error.context(format!(
            "einsum n-ary pair ({left}, {right}) intermediate estimate"
        ))
    })?;
    let flops = checked_nary_product(&flop_extents).map_err(|error| {
        error.context(format!("einsum n-ary pair ({left}, {right}) FLOP estimate"))
    })?;
    Ok(PairEstimate {
        left,
        right,
        output_axes,
        output_elements,
        flops,
    })
}

fn pair_axis_extent(
    left: &PlannedOperand<'_>,
    right: &PlannedOperand<'_>,
    axis: ExpandedAxis<'_>,
) -> Result<usize> {
    let left_extent = left
        .axes
        .iter()
        .position(|candidate| *candidate == axis)
        .map(|position| left.tensor.dims()[position]);
    let right_extent = right
        .axes
        .iter()
        .position(|candidate| *candidate == axis)
        .map(|position| right.tensor.dims()[position]);
    match (left_extent, right_extent) {
        (Some(left), Some(right)) => resolve_extent(&axis.display_name(), left, right),
        (Some(extent), None) | (None, Some(extent)) => Ok(extent),
        (None, None) => candle_core::bail!(
            "invalid n-ary einsum planner axis `{}` is absent from both operands",
            axis.display_name()
        ),
    }
}

fn checked_nary_product(extents: &[usize]) -> Result<u128> {
    if extents.contains(&0) {
        return Ok(0);
    }
    extents.iter().try_fold(1_u128, |product, &extent| {
        product
            .checked_mul(extent as u128)
            .ok_or_else(|| candle_core::Error::msg("einsum n-ary cost estimate overflows u128"))
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpandedAxis<'a> {
    Named(&'a str),
    Ellipsis(usize),
}

impl ExpandedAxis<'_> {
    fn display_name(&self) -> String {
        match self {
            Self::Named(name) => (*name).to_owned(),
            Self::Ellipsis(index) => format!("..[{index}]"),
        }
    }
}

fn ellipsis_capture(
    operand: &Tensor,
    operand_index: usize,
    pattern: EinsumAxisPattern<'_>,
) -> Result<usize> {
    if let Some(position) = pattern.ellipsis_position {
        if position > pattern.labels.len() {
            candle_core::bail!(
                "invalid ellipsis einsum plan: operand {operand_index} ellipsis position {position} exceeds {} explicit labels",
                pattern.labels.len()
            )
        }
        operand.rank().checked_sub(pattern.labels.len()).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "einsum operand {operand_index} has rank {}, but {} explicit axes leave no valid ellipsis capture",
                operand.rank(),
                pattern.labels.len()
            ))
        })
    } else if operand.rank() == pattern.labels.len() {
        Ok(0)
    } else {
        candle_core::bail!(
            "einsum operand {operand_index} has rank {}, expected {} because its axis list has no ellipsis",
            operand.rank(),
            pattern.labels.len()
        )
    }
}

fn normalize_ellipsis_operand(
    operand: &Tensor,
    pattern: EinsumAxisPattern<'_>,
    capture: usize,
    maximum_capture: usize,
    operand_index: usize,
) -> Result<Tensor> {
    let missing = maximum_capture.checked_sub(capture).ok_or_else(|| {
        candle_core::Error::msg("invalid ellipsis einsum plan: capture exceeds maximum")
    })?;
    let insertion = pattern.ellipsis_position.unwrap_or(0);
    let mut normalized = operand.clone();
    for _ in 0..missing {
        normalized = normalized.unsqueeze(insertion).map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} ellipsis right-alignment"
            ))
        })?;
    }
    Ok(normalized)
}

#[derive(Debug, PartialEq, Eq)]
enum RepeatedAxisLoweringPlan {
    Sequential,
    OriginalFlatGather {
        output_shape: Vec<usize>,
        offsets: Vec<u32>,
    },
}

fn validate_repeated_extents(
    dims: &[usize],
    axes: &[ExpandedAxis<'_>],
    operand_index: usize,
) -> Result<()> {
    for (position, axis) in axes.iter().copied().enumerate() {
        let Some(previous) = axes[..position]
            .iter()
            .position(|candidate| *candidate == axis)
        else {
            continue;
        };
        let extent = dims[previous];
        let other = dims[position];
        if other != extent {
            candle_core::bail!(
                "einsum operand {operand_index} repeated label `{}` has unequal extents {extent} and {other}",
                axis.display_name()
            )
        }
    }
    Ok(())
}

fn checked_contiguous_strides(dims: &[usize]) -> Option<Vec<usize>> {
    let mut product = 1_usize;
    let mut strides = vec![0; dims.len()];
    for (position, &extent) in dims.iter().enumerate().rev() {
        strides[position] = product;
        product = product.checked_mul(extent)?;
    }
    Some(strides)
}

fn simulated_contiguous(dims: &[usize], strides: &[usize]) -> Option<bool> {
    let mut expected = 1_usize;
    for (&extent, &stride) in dims.iter().zip(strides).rev() {
        if extent > 1 && stride != expected {
            return Some(false);
        }
        expected = expected.checked_mul(extent)?;
    }
    Some(true)
}

fn sequential_diagonal_would_materialize(
    original_dims: &[usize],
    original_axes: &[ExpandedAxis<'_>],
) -> Option<bool> {
    let mut dims = original_dims.to_vec();
    let mut axes = original_axes.to_vec();
    let mut strides = checked_contiguous_strides(&dims)?;
    loop {
        let Some(repeated_axis) = axes
            .iter()
            .copied()
            .find(|axis| axes.iter().filter(|candidate| **candidate == *axis).count() > 1)
        else {
            return Some(false);
        };
        let positions = axes
            .iter()
            .enumerate()
            .filter_map(|(position, axis)| (*axis == repeated_axis).then_some(position))
            .collect::<Vec<_>>();
        let other_positions = (0..axes.len())
            .filter(|position| !positions.contains(position))
            .collect::<Vec<_>>();
        let permutation = positions
            .iter()
            .chain(&other_positions)
            .copied()
            .collect::<Vec<_>>();
        let adjacent_dims = permutation
            .iter()
            .map(|&position| dims[position])
            .collect::<Vec<_>>();
        let adjacent_strides = permutation
            .iter()
            .map(|&position| strides[position])
            .collect::<Vec<_>>();
        if !simulated_contiguous(&adjacent_dims, &adjacent_strides)? {
            return Some(true);
        }

        let selected_dims = std::iter::once(dims[positions[0]])
            .chain(other_positions.iter().map(|&position| dims[position]))
            .collect::<Vec<_>>();
        let selected_strides = checked_contiguous_strides(&selected_dims)?;
        let result_order = std::iter::once(positions[0])
            .chain(other_positions.iter().copied())
            .collect::<Vec<_>>();
        let desired_order = (0..axes.len())
            .filter(|position| !positions[1..].contains(position))
            .collect::<Vec<_>>();
        let restoration = desired_order
            .iter()
            .map(|position| {
                result_order
                    .iter()
                    .position(|candidate| candidate == position)
            })
            .collect::<Option<Vec<_>>>()?;
        dims = restoration
            .iter()
            .map(|&position| selected_dims[position])
            .collect();
        strides = restoration
            .iter()
            .map(|&position| selected_strides[position])
            .collect();
        axes = desired_order
            .iter()
            .map(|&position| axes[position])
            .collect();
    }
}

fn original_flat_gather_offsets(
    dims: &[usize],
    axes: &[ExpandedAxis<'_>],
    operand_index: usize,
) -> Result<Option<(Vec<usize>, Vec<u32>)>> {
    let mut unique_axes = Vec::new();
    let mut unique_positions = Vec::with_capacity(axes.len());
    for axis in axes.iter().copied() {
        let position =
            if let Some(position) = unique_axes.iter().position(|candidate| *candidate == axis) {
                position
            } else {
                unique_axes.push(axis);
                unique_axes.len() - 1
            };
        unique_positions.push(position);
    }
    let output_shape = unique_axes
        .iter()
        .map(|axis| {
            let position = axes
                .iter()
                .position(|candidate| candidate == axis)
                .expect("a unique diagonal axis came from the original axes");
            dims[position]
        })
        .collect::<Vec<_>>();

    if output_shape.contains(&0) {
        return Ok(Some((output_shape, Vec::new())));
    }
    let Some(original_strides) = checked_contiguous_strides(dims) else {
        return Ok(None);
    };
    let Some(output_elements) = output_shape
        .iter()
        .try_fold(1_usize, |product, &extent| product.checked_mul(extent))
    else {
        return Ok(None);
    };
    let mut maximum_offset = 0_usize;
    for (position, &stride) in original_strides.iter().enumerate() {
        let coordinate = output_shape[unique_positions[position]] - 1;
        let Some(contribution) = coordinate.checked_mul(stride) else {
            return Ok(None);
        };
        let Some(next) = maximum_offset.checked_add(contribution) else {
            return Ok(None);
        };
        maximum_offset = next;
    }
    if u32::try_from(maximum_offset).is_err() {
        return Ok(None);
    }

    let Some(output_strides) = checked_contiguous_strides(&output_shape) else {
        return Ok(None);
    };
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(output_elements)
        .map_err(|error| {
            candle_core::Error::msg(format!(
                "einsum operand {operand_index} diagonal offset allocation failed: {error}"
            ))
        })?;
    for output_index in 0..output_elements {
        let mut offset = 0_usize;
        for (position, &stride) in original_strides.iter().enumerate() {
            let unique_position = unique_positions[position];
            let coordinate =
                (output_index / output_strides[unique_position]) % output_shape[unique_position];
            offset += coordinate * stride;
        }
        offsets.push(u32::try_from(offset).expect("maximum checked original-layout offset"));
    }
    Ok(Some((output_shape, offsets)))
}

fn plan_repeated_axis_lowering(
    dims: &[usize],
    axes: &[ExpandedAxis<'_>],
    original_contiguous: bool,
    operand_index: usize,
) -> Result<RepeatedAxisLoweringPlan> {
    validate_repeated_extents(dims, axes, operand_index)?;
    if !original_contiguous || sequential_diagonal_would_materialize(dims, axes) != Some(true) {
        return Ok(RepeatedAxisLoweringPlan::Sequential);
    }
    let Some((output_shape, offsets)) = original_flat_gather_offsets(dims, axes, operand_index)?
    else {
        return Ok(RepeatedAxisLoweringPlan::Sequential);
    };
    Ok(RepeatedAxisLoweringPlan::OriginalFlatGather {
        output_shape,
        offsets,
    })
}

fn original_flat_diagonal_gather(
    operand: &Tensor,
    output_shape: &[usize],
    offsets: Vec<u32>,
    operand_index: usize,
) -> Result<Tensor> {
    let index_count = offsets.len();
    let indices = Tensor::from_vec(offsets, index_count, operand.device()).map_err(|error| {
        error.context(format!(
            "einsum operand {operand_index} device-local combined diagonal indices"
        ))
    })?;
    operand
        .flatten_all()
        .map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} original-layout diagonal flatten"
            ))
        })?
        .index_select(&indices, 0)
        .map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} differentiable combined diagonal selection"
            ))
        })?
        .reshape(output_shape)
        .map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} combined diagonal reshape"
            ))
        })
}

fn normalize_repeated_axes<'a>(
    mut operand: Tensor,
    mut axes: Vec<ExpandedAxis<'a>>,
    operand_index: usize,
) -> Result<(Tensor, Vec<ExpandedAxis<'a>>)> {
    match plan_repeated_axis_lowering(
        operand.dims(),
        &axes,
        operand.is_contiguous(),
        operand_index,
    )? {
        RepeatedAxisLoweringPlan::Sequential => {}
        RepeatedAxisLoweringPlan::OriginalFlatGather {
            output_shape,
            offsets,
        } => {
            let mut unique_axes = Vec::new();
            for axis in axes {
                if !unique_axes.contains(&axis) {
                    unique_axes.push(axis);
                }
            }
            return Ok((
                original_flat_diagonal_gather(&operand, &output_shape, offsets, operand_index)?,
                unique_axes,
            ));
        }
    }
    loop {
        let Some(repeated_axis) = axes
            .iter()
            .copied()
            .find(|axis| axes.iter().filter(|candidate| **candidate == *axis).count() > 1)
        else {
            return Ok((operand, axes));
        };
        let positions = axes
            .iter()
            .enumerate()
            .filter_map(|(position, axis)| (*axis == repeated_axis).then_some(position))
            .collect::<Vec<_>>();
        let extent = operand.dims()[positions[0]];
        for &position in &positions[1..] {
            let other = operand.dims()[position];
            if other != extent {
                candle_core::bail!(
                    "einsum operand {operand_index} repeated label `{}` has unequal extents {extent} and {other}",
                    repeated_axis.display_name()
                )
            }
        }

        let other_positions = (0..axes.len())
            .filter(|position| !positions.contains(position))
            .collect::<Vec<_>>();
        let adjacency_permutation = positions
            .iter()
            .chain(&other_positions)
            .copied()
            .collect::<Vec<_>>();
        let device = operand.device().clone();
        let adjacent = if adjacency_permutation.iter().copied().eq(0..axes.len()) {
            operand
        } else {
            operand
                .permute(adjacency_permutation.as_slice())
                .map_err(|error| {
                    error.context(format!(
                        "einsum operand {operand_index} diagonal adjacency permutation"
                    ))
                })?
        };

        let (repeated_flat_extent, diagonal_stride) =
            checked_diagonal_layout(extent, positions.len(), operand_index)?;
        let mut flattened_shape = vec![repeated_flat_extent];
        flattened_shape.extend_from_slice(&adjacent.dims()[positions.len()..]);
        let flattened = adjacent.reshape(flattened_shape).map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} repeated-axis flatten"
            ))
        })?;

        let mut diagonal_indices = Vec::with_capacity(extent);
        for coordinate in 0..extent {
            let index = coordinate.checked_mul(diagonal_stride).ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "einsum operand {operand_index} diagonal index overflows usize"
                ))
            })?;
            diagonal_indices.push(u32::try_from(index).map_err(|_| {
                candle_core::Error::msg(format!(
                    "einsum operand {operand_index} diagonal index {index} exceeds u32"
                ))
            })?);
        }
        let indices = Tensor::from_vec(diagonal_indices, extent, &device).map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} device-local diagonal indices"
            ))
        })?;
        let selected = flattened.index_select(&indices, 0).map_err(|error| {
            error.context(format!(
                "einsum operand {operand_index} differentiable diagonal selection"
            ))
        })?;

        let result_order = std::iter::once(positions[0])
            .chain(other_positions.iter().copied())
            .collect::<Vec<_>>();
        let desired_order = (0..axes.len())
            .filter(|position| !positions[1..].contains(position))
            .collect::<Vec<_>>();
        let restoration = desired_order
            .iter()
            .map(|position| {
                result_order
                    .iter()
                    .position(|candidate| candidate == position)
                    .expect("diagonal result order contains every surviving axis")
            })
            .collect::<Vec<_>>();
        operand = if restoration.iter().copied().eq(0..restoration.len()) {
            selected
        } else {
            selected.permute(restoration.as_slice()).map_err(|error| {
                error.context(format!(
                    "einsum operand {operand_index} diagonal axis restoration"
                ))
            })?
        };
        axes = desired_order
            .iter()
            .map(|&position| axes[position])
            .collect();
    }
}

fn checked_diagonal_layout(
    extent: usize,
    multiplicity: usize,
    operand_index: usize,
) -> Result<(usize, usize)> {
    let mut diagonal_stride = 0_usize;
    let mut power = 1_usize;
    for _ in 0..multiplicity {
        diagonal_stride = diagonal_stride.checked_add(power).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "einsum operand {operand_index} diagonal stride overflows usize"
            ))
        })?;
        power = power.checked_mul(extent).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "einsum operand {operand_index} repeated-axis extent product overflows usize"
            ))
        })?;
    }
    Ok((power, diagonal_stride))
}

fn expand_axis_pattern<'a>(
    pattern: EinsumAxisPattern<'a>,
    ellipsis_rank: usize,
    implicit_operand_ellipsis: bool,
) -> Vec<ExpandedAxis<'a>> {
    let synthetic = (0..ellipsis_rank).map(ExpandedAxis::Ellipsis);
    match pattern.ellipsis_position {
        Some(position) => pattern.labels[..position]
            .iter()
            .copied()
            .map(ExpandedAxis::Named)
            .chain(synthetic)
            .chain(
                pattern.labels[position..]
                    .iter()
                    .copied()
                    .map(ExpandedAxis::Named),
            )
            .collect(),
        None if implicit_operand_ellipsis => synthetic
            .chain(pattern.labels.iter().copied().map(ExpandedAxis::Named))
            .collect(),
        None => pattern
            .labels
            .iter()
            .copied()
            .map(ExpandedAxis::Named)
            .collect(),
    }
}

fn validate_expanded_output(
    inputs: &[&[ExpandedAxis<'_>]],
    output: &[ExpandedAxis<'_>],
) -> Result<()> {
    for axis in output {
        if !inputs.iter().any(|input| input.contains(axis)) {
            candle_core::bail!(
                "invalid ellipsis einsum plan: output axis `{}` does not occur in an input",
                axis.display_name()
            )
        }
    }
    Ok(())
}

struct ExpandedBinaryPlan<'a> {
    batch: Vec<ExpandedAxis<'a>>,
    left_free: Vec<ExpandedAxis<'a>>,
    contracted: Vec<ExpandedAxis<'a>>,
    right_free: Vec<ExpandedAxis<'a>>,
    left_reductions: Vec<usize>,
    right_reductions: Vec<usize>,
    left_permutation: Vec<usize>,
    right_permutation: Vec<usize>,
    output_permutation: Vec<usize>,
}

fn classify_expanded_binary<'a>(
    left: &[ExpandedAxis<'a>],
    right: &[ExpandedAxis<'a>],
    output: &[ExpandedAxis<'a>],
) -> ExpandedBinaryPlan<'a> {
    let mut all = Vec::new();
    for axis in left.iter().chain(right) {
        if !all.contains(axis) {
            all.push(*axis);
        }
    }
    let batch = all
        .iter()
        .copied()
        .filter(|axis| left.contains(axis) && right.contains(axis) && output.contains(axis))
        .collect::<Vec<_>>();
    let left_free = all
        .iter()
        .copied()
        .filter(|axis| left.contains(axis) && !right.contains(axis) && output.contains(axis))
        .collect::<Vec<_>>();
    let contracted = all
        .iter()
        .copied()
        .filter(|axis| left.contains(axis) && right.contains(axis) && !output.contains(axis))
        .collect::<Vec<_>>();
    let right_free = all
        .iter()
        .copied()
        .filter(|axis| !left.contains(axis) && right.contains(axis) && output.contains(axis))
        .collect::<Vec<_>>();
    let left_reductions = left
        .iter()
        .enumerate()
        .filter_map(|(index, axis)| {
            (!right.contains(axis) && !output.contains(axis)).then_some(index)
        })
        .collect();
    let right_reductions = right
        .iter()
        .enumerate()
        .filter_map(|(index, axis)| {
            (!left.contains(axis) && !output.contains(axis)).then_some(index)
        })
        .collect();
    let left_remaining = left
        .iter()
        .copied()
        .filter(|axis| right.contains(axis) || output.contains(axis))
        .collect::<Vec<_>>();
    let right_remaining = right
        .iter()
        .copied()
        .filter(|axis| left.contains(axis) || output.contains(axis))
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
    let canonical_output = batch
        .iter()
        .chain(&left_free)
        .chain(&right_free)
        .copied()
        .collect::<Vec<_>>();
    ExpandedBinaryPlan {
        batch,
        left_free,
        contracted,
        right_free,
        left_reductions,
        right_reductions,
        left_permutation: dynamic_permutation(&left_remaining, &left_canonical),
        right_permutation: dynamic_permutation(&right_remaining, &right_canonical),
        output_permutation: dynamic_permutation(&canonical_output, output),
    }
}

fn dynamic_permutation<T: Eq>(current: &[T], desired: &[T]) -> Vec<usize> {
    desired
        .iter()
        .map(|axis| {
            current
                .iter()
                .position(|candidate| candidate == axis)
                .expect("validated ellipsis classification")
        })
        .collect()
}

fn broadcast_if_needed(tensor: &Tensor, shape: &[usize], context: &'static str) -> Result<Tensor> {
    if tensor.dims() == shape {
        Ok(tensor.clone())
    } else {
        tensor
            .broadcast_as(shape)
            .map_err(|error| error.context(context))
    }
}

fn apply_output_permutation(output: Tensor, permutation: &[usize]) -> Result<Tensor> {
    if permutation.iter().copied().eq(0..permutation.len()) {
        Ok(output)
    } else {
        output
            .permute(permutation)
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

    #[test]
    fn ellipsis_runtime_validates_rank_and_normalizes_right_alignment() -> Result<()> {
        let vector = Tensor::zeros(1, DType::F32, &Device::Cpu)?;
        let invalid_operands = [EinsumAxisPattern::new(&["row", "inner"], Some(0))];
        let invalid =
            EllipsisEinsumSpec::new(&invalid_operands, EinsumAxisPattern::new(&["row"], Some(0)));
        assert!(execute_unary_ellipsis_einsum(&vector, invalid).is_err());

        let left = Tensor::ones((2, 1, 2, 3), DType::F32, &Device::Cpu)?;
        let right = Tensor::ones((4, 3, 2), DType::F32, &Device::Cpu)?;
        let valid_operands = [
            EinsumAxisPattern::new(&["row", "inner"], Some(0)),
            EinsumAxisPattern::new(&["inner", "column"], Some(0)),
        ];
        let valid = EllipsisEinsumSpec::new(
            &valid_operands,
            EinsumAxisPattern::new(&["row", "column"], Some(1)),
        );
        assert_eq!(
            execute_binary_ellipsis_einsum(&left, &right, valid)?.dims(),
            &[2, 2, 4, 2]
        );
        Ok(())
    }

    #[test]
    fn diagonal_layout_arithmetic_is_checked() {
        assert!(checked_diagonal_layout(usize::MAX, 2, 0).is_err());
        assert_eq!(checked_diagonal_layout(3, 3, 0).unwrap(), (27, 13));
        assert_eq!(checked_diagonal_layout(0, 3, 0).unwrap(), (0, 1));
    }

    fn named_axes(labels: &[&'static str]) -> Vec<ExpandedAxis<'static>> {
        labels.iter().copied().map(ExpandedAxis::Named).collect()
    }

    #[test]
    fn diagonal_plan_selects_only_when_sequential_flatten_would_copy() -> Result<()> {
        assert_eq!(
            plan_repeated_axis_lowering(&[3, 3], &named_axes(&["i", "i"]), true, 0)?,
            RepeatedAxisLoweringPlan::Sequential
        );
        assert_eq!(
            plan_repeated_axis_lowering(&[2, 3, 3], &named_axes(&["batch", "i", "i"]), true, 0,)?,
            RepeatedAxisLoweringPlan::OriginalFlatGather {
                output_shape: vec![2, 3],
                offsets: vec![0, 4, 8, 9, 13, 17],
            }
        );
        assert_eq!(
            plan_repeated_axis_lowering(
                &[2, 3, 2, 3],
                &named_axes(&["i", "j", "i", "j"]),
                true,
                0,
            )?,
            RepeatedAxisLoweringPlan::OriginalFlatGather {
                output_shape: vec![2, 3],
                offsets: vec![0, 7, 14, 21, 28, 35],
            }
        );
        Ok(())
    }

    #[test]
    fn diagonal_plan_preserves_fallbacks_and_checks_zero_before_offsets() -> Result<()> {
        assert_eq!(
            plan_repeated_axis_lowering(
                &[2, 3, 2, 3],
                &named_axes(&["i", "j", "i", "j"]),
                false,
                0,
            )?,
            RepeatedAxisLoweringPlan::Sequential
        );
        assert_eq!(
            plan_repeated_axis_lowering(
                &[2, 65_536, 65_536],
                &named_axes(&["batch", "i", "i"]),
                true,
                0,
            )?,
            RepeatedAxisLoweringPlan::Sequential
        );
        assert_eq!(
            plan_repeated_axis_lowering(&[2, 0, 0], &named_axes(&["batch", "i", "i"]), true, 0,)?,
            RepeatedAxisLoweringPlan::OriginalFlatGather {
                output_shape: vec![2, 0],
                offsets: vec![],
            }
        );
        Ok(())
    }

    #[test]
    fn diagonal_plan_validates_every_repeated_extent_before_arithmetic() {
        let error = plan_repeated_axis_lowering(
            &[usize::MAX, usize::MAX, 2, 3],
            &named_axes(&["i", "i", "j", "j"]),
            true,
            0,
        )
        .expect_err("the later unequal group must be rejected first");
        assert!(error.to_string().contains("repeated label `j`"));
        assert!(error.to_string().contains("unequal extents 2 and 3"));
    }

    #[test]
    fn nary_planner_avoids_a_pathological_left_to_right_intermediate() -> Result<()> {
        let operands = vec![
            PlannedOperand::new_for_test((100, 2), &["a", "b"], 0)?,
            PlannedOperand::new_for_test((2, 100), &["b", "c"], 1)?,
            PlannedOperand::new_for_test((100, 2), &["c", "d"], 2)?,
        ];
        let output = [ExpandedAxis::Named("a"), ExpandedAxis::Named("d")];
        let left_to_right = estimate_pair(&operands, 0, 1, &output)?;
        let selected = select_nary_pair(&operands, &output)?;
        assert_eq!((selected.left, selected.right), (1, 2));
        assert_eq!(left_to_right.output_elements, 10_000);
        assert_eq!(selected.output_elements, 4);
        Ok(())
    }

    #[test]
    fn nary_planner_breaks_equal_cost_ties_by_operand_order() -> Result<()> {
        let operands = vec![
            PlannedOperand::new_for_test((2, 2), &["a", "b"], 0)?,
            PlannedOperand::new_for_test((2, 2), &["b", "c"], 1)?,
            PlannedOperand::new_for_test((2, 2), &["c", "d"], 2)?,
        ];
        let output = [ExpandedAxis::Named("a"), ExpandedAxis::Named("d")];
        let selected = select_nary_pair(&operands, &output)?;
        assert_eq!((selected.left, selected.right), (0, 1));
        assert_eq!((selected.output_elements, selected.flops), (4, 8));
        Ok(())
    }

    #[test]
    fn nary_cost_estimate_overflow_is_checked() {
        assert!(checked_nary_product(&[usize::MAX; 5]).is_err());
        assert_eq!(
            checked_nary_product(&[usize::MAX, usize::MAX, 0]).unwrap(),
            0
        );
    }
}
