use candle_core::{Result, Shape, Tensor};

use crate::Operation;

#[derive(Debug)]
struct ReductionRun {
    axes: Vec<usize>,
    operation: Operation,
}

fn operations_are_fusible(left: Operation, right: Operation) -> bool {
    matches!(
        (left, right),
        (Operation::Sum, Operation::Sum)
            | (Operation::Mean, Operation::Mean)
            | (Operation::Min, Operation::Min)
            | (Operation::Max, Operation::Max)
    )
}

fn plan_reduction_runs(axes_operations: &mut [(usize, Operation)]) -> Vec<ReductionRun> {
    axes_operations.sort_by_key(|(axis, _)| *axis);
    let mut runs: Vec<ReductionRun> = Vec::new();
    for &(axis, operation) in axes_operations.iter().rev() {
        if let Some(run) = runs.last_mut()
            && operations_are_fusible(run.operation, operation)
            && (matches!(operation, Operation::Sum | Operation::Mean)
                || run
                    .axes
                    .last()
                    .is_some_and(|&previous| axis + 1 == previous))
        {
            run.axes.push(axis);
        } else {
            runs.push(ReductionRun {
                axes: vec![axis],
                operation,
            });
        }
    }
    runs
}

#[cfg(test)]
std::thread_local! {
    static BACKEND_REDUCTION_CALL_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_backend_reduction_call() {
    BACKEND_REDUCTION_CALL_COUNT.set(BACKEND_REDUCTION_CALL_COUNT.get() + 1);
}

#[cfg(not(test))]
fn record_backend_reduction_call() {}

#[cfg(test)]
fn reset_backend_reduction_call_count() {
    BACKEND_REDUCTION_CALL_COUNT.set(0);
}

#[cfg(test)]
fn backend_reduction_call_count() -> usize {
    BACKEND_REDUCTION_CALL_COUNT.get()
}

/// Tensor operations used by [`crate::einops!`].
///
/// Transformations return Candle [`Result`] values so backend failures retain
/// their original error and context.
pub trait Backend {
    type Output;
    fn shape(self) -> Vec<usize>;
    /// Reshapes a tensor while preserving its storage and layout when the
    /// requested dimensions are already exact.
    ///
    /// Call [`Tensor::contiguous`] explicitly when contiguous storage is
    /// required; an identity reshape no longer provides accidental
    /// materialization for non-contiguous inputs.
    fn reshape(self, shape: &[usize]) -> Result<Self::Output>;
    fn transpose(self, axes: &[usize]) -> Result<Self::Output>;
    /// Composes adjacent logical axis groups after any preceding permutation.
    ///
    /// The default retains the historical reshape sequence. Tensor backends
    /// may recover a storage-sharing layout before falling back to that copy.
    fn compose_axes(self, output_shape: &[usize], _group_lengths: &[usize]) -> Result<Self::Output>
    where
        Self: Sized,
    {
        self.reshape(output_shape)
    }
    /// Applies a permutation followed immediately by axis composition.
    ///
    /// Backends may specialize this boundary. The default preserves the
    /// historical operation sequence for third-party implementations.
    fn permute_and_compose(
        self,
        permutation: &[usize],
        output_shape: &[usize],
        _group_lengths: &[usize],
    ) -> Result<<Self::Output as Backend>::Output>
    where
        Self: Sized,
        Self::Output: Backend,
    {
        let output = self.transpose(permutation)?;
        Backend::reshape(output, output_shape)
    }
    fn reduce_axes(self, axes_operations: &mut [(usize, Operation)]) -> Result<Self::Output>;
    /// Inserts new axes as broadcast views.
    ///
    /// The returned tensor can be non-contiguous and can alias the input.
    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Result<Self::Output>;
}

impl<T: AsRef<Tensor>> Backend for T {
    type Output = Tensor;

    fn shape(self) -> Vec<usize> {
        self.as_ref().dims().to_vec()
    }

    fn reshape(self, shape: &[usize]) -> Result<Self::Output> {
        let input = self.as_ref();
        let shape = Shape::from_dims(shape);
        if shape.elem_count() != input.elem_count() {
            return input.reshape(shape);
        }
        if input.dims() == shape.dims() {
            Ok(input.clone())
        } else {
            input.reshape(shape)
        }
    }

    fn transpose(self, axes: &[usize]) -> Result<Self::Output> {
        self.as_ref().permute(axes)
    }

    fn compose_axes(self, output_shape: &[usize], group_lengths: &[usize]) -> Result<Self::Output> {
        execute_tensor_compose_axes(self.as_ref(), output_shape, group_lengths)
    }

    fn permute_and_compose(
        self,
        permutation: &[usize],
        output_shape: &[usize],
        group_lengths: &[usize],
    ) -> Result<<Self::Output as Backend>::Output>
    where
        Self::Output: Backend,
    {
        execute_tensor_permute_and_compose(self.as_ref(), permutation, output_shape, group_lengths)
    }

    fn reduce_axes(self, axes_operations: &mut [(usize, Operation)]) -> Result<Self::Output> {
        let mut output = self.as_ref().clone();
        let mut occupied = vec![false; output.rank()];

        for &(axis, _) in axes_operations.iter() {
            if axis >= occupied.len() {
                candle_core::bail!(
                    "reduce_axes: axis {axis} out of range for rank {}",
                    occupied.len()
                )
            }
            if occupied[axis] {
                candle_core::bail!("reduce_axes: duplicate axis {axis}")
            }
            occupied[axis] = true;
        }

        for run in plan_reduction_runs(axes_operations) {
            output = match run.operation {
                Operation::Min | Operation::Max if run.axes.len() > 1 => {
                    if let Some(collapsed) =
                        collapse_extrema_run(&output, &run.axes, run.operation)?
                    {
                        record_backend_reduction_call();
                        collapsed
                    } else {
                        let mut reduced = output;
                        for axis in run.axes {
                            record_backend_reduction_call();
                            reduced = match run.operation {
                                Operation::Min => reduced.min(axis)?,
                                Operation::Max => reduced.max(axis)?,
                                _ => unreachable!("extrema run operation"),
                            };
                        }
                        reduced
                    }
                }
                Operation::Min => {
                    record_backend_reduction_call();
                    output.min(run.axes[0])?
                }
                Operation::Max => {
                    record_backend_reduction_call();
                    output.max(run.axes[0])?
                }
                Operation::Sum => {
                    record_backend_reduction_call();
                    output.sum(run.axes.as_slice())?
                }
                Operation::Mean => {
                    record_backend_reduction_call();
                    output.mean(run.axes.as_slice())?
                }
                Operation::Prod => {
                    record_backend_reduction_call();
                    let axis = run.axes[0];
                    let axis_len = output.dim(axis)?;
                    if axis_len == 0 {
                        let mut shape = output.dims().to_vec();
                        shape.remove(axis);
                        Tensor::ones(Shape::from_dims(&shape), output.dtype(), output.device())?
                    } else {
                        let mut product = output.narrow(axis, 0, 1)?.squeeze(axis)?;
                        for index in 1..axis_len {
                            let factor = output.narrow(axis, index, 1)?.squeeze(axis)?;
                            product = product.mul(&factor)?;
                        }
                        product
                    }
                }
            };
        }

        Ok(output)
    }

    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Result<Self::Output> {
        let input = self.as_ref();

        let expected_naxes = input.rank() + pos2len.len();
        if naxes != expected_naxes {
            candle_core::bail!("add_axes: expected final rank {expected_naxes}, got {naxes}")
        }

        let mut inserted_lengths = vec![1; naxes];
        let mut occupied = vec![false; naxes];

        for &(axis_pos, axis_len) in pos2len {
            if axis_pos >= naxes {
                candle_core::bail!(
                    "add_axes: axis position {axis_pos} out of range for final rank {naxes}"
                )
            }
            if occupied[axis_pos] {
                candle_core::bail!("add_axes: duplicate axis position {axis_pos}")
            }
            occupied[axis_pos] = true;
            inserted_lengths[axis_pos] = axis_len;
        }

        let mut singleton_shape = Vec::with_capacity(naxes);
        let mut final_shape = Vec::with_capacity(naxes);
        let mut input_axis = 0;
        for axis in 0..naxes {
            if occupied[axis] {
                singleton_shape.push(1);
                final_shape.push(inserted_lengths[axis]);
            } else {
                let length = input.dims()[input_axis];
                singleton_shape.push(length);
                final_shape.push(length);
                input_axis += 1;
            }
        }

        let expanded = if input.is_contiguous() {
            input.reshape(Shape::from_dims(&singleton_shape))?
        } else {
            let mut output = input.clone();
            let mut positions = pos2len
                .iter()
                .map(|&(axis_pos, _)| axis_pos)
                .collect::<Vec<_>>();
            positions.sort_unstable();
            for axis_pos in positions {
                output = output.unsqueeze(axis_pos)?;
            }
            output
        };

        expanded.broadcast_as(Shape::from_dims(&final_shape))
    }
}

fn collapse_extrema_run(
    input: &Tensor,
    descending_axes: &[usize],
    operation: Operation,
) -> Result<Option<Tensor>> {
    let start = *descending_axes
        .last()
        .expect("an extrema run contains at least one axis");
    let mut group_lengths = vec![1; start];
    group_lengths.push(descending_axes.len());
    group_lengths.extend(std::iter::repeat_n(
        1,
        input.rank() - start - descending_axes.len(),
    ));
    let identity = (0..input.rank()).collect::<Vec<_>>();
    if plan_permute_compose_group_order(input.dims(), input.stride(), &identity, &group_lengths)?
        .is_none()
    {
        return Ok(None);
    }
    let mut collapsed_shape = input.dims()[..start].to_vec();
    let collapsed_extent = input.dims()[start..start + descending_axes.len()]
        .iter()
        .try_fold(1usize, |product, &extent| product.checked_mul(extent))
        .ok_or_else(|| candle_core::Error::msg("extrema collapsed extent overflows usize"))?;
    collapsed_shape.push(collapsed_extent);
    collapsed_shape.extend_from_slice(&input.dims()[start + descending_axes.len()..]);
    let collapsed = execute_tensor_compose_axes(input, &collapsed_shape, &group_lengths)?;
    match operation {
        Operation::Min => collapsed.min(start).map(Some),
        Operation::Max => collapsed.max(start).map(Some),
        _ => unreachable!("collapsed extrema operation"),
    }
}

pub(crate) fn execute_tensor_permute_and_compose(
    input: &Tensor,
    permutation: &[usize],
    output_shape: &[usize],
    group_lengths: &[usize],
) -> Result<Tensor> {
    let rank = input.rank();
    if permutation.len() != rank {
        candle_core::bail!(
            "permute_and_compose: permutation rank {} does not match input rank {rank}",
            permutation.len()
        )
    }
    let mut seen = vec![false; rank];
    for &axis in permutation {
        if axis >= rank || seen[axis] {
            candle_core::bail!("permute_and_compose: invalid permutation")
        }
        seen[axis] = true;
    }
    if group_lengths.len() != output_shape.len()
        || group_lengths.contains(&0)
        || group_lengths
            .iter()
            .try_fold(0usize, |sum, &length| sum.checked_add(length))
            != Some(rank)
    {
        candle_core::bail!("permute_and_compose: invalid group metadata")
    }

    let mut groups = Vec::with_capacity(group_lengths.len());
    let mut start = 0;
    for (&length, &expected_output) in group_lengths.iter().zip(output_shape) {
        let end = start + length;
        let group = permutation[start..end].to_vec();
        let product = checked_axis_product(input.dims(), &group)?;
        if product != expected_output {
            candle_core::bail!(
                "permute_and_compose: group product {product} does not match output extent {expected_output}"
            )
        }
        groups.push(group);
        start = end;
    }

    if let Some(order) =
        plan_permute_compose_group_order(input.dims(), input.stride(), permutation, group_lengths)?
    {
        let pre_permutation = order
            .iter()
            .flat_map(|&group| groups[group].iter().copied())
            .collect::<Vec<_>>();
        let reshape_dims = order
            .iter()
            .map(|&group| output_shape[group])
            .collect::<Vec<_>>();
        let post_permutation = (0..groups.len())
            .map(|desired| {
                order
                    .iter()
                    .position(|&group| group == desired)
                    .expect("group order is a permutation")
            })
            .collect::<Vec<_>>();
        let permuted = input.permute(pre_permutation)?;
        let reshaped = permuted.reshape(&reshape_dims)?;
        return reshaped.permute(post_permutation);
    }

    input.permute(permutation)?.reshape(output_shape)
}

pub(crate) fn execute_tensor_compose_axes(
    input: &Tensor,
    output_shape: &[usize],
    group_lengths: &[usize],
) -> Result<Tensor> {
    if output_shape.len() != group_lengths.len() {
        candle_core::bail!("compose_axes: output and group ranks differ")
    }
    let mut nonempty_shape = Vec::with_capacity(output_shape.len());
    let mut nonempty_lengths = Vec::with_capacity(group_lengths.len());
    for (&extent, &length) in output_shape.iter().zip(group_lengths) {
        if length == 0 {
            if extent != 1 {
                candle_core::bail!("compose_axes: an empty group must have extent one")
            }
        } else {
            nonempty_shape.push(extent);
            nonempty_lengths.push(length);
        }
    }
    let mut output = if nonempty_lengths.is_empty() {
        input.clone()
    } else {
        execute_tensor_permute_and_compose(
            input,
            &(0..input.rank()).collect::<Vec<_>>(),
            &nonempty_shape,
            &nonempty_lengths,
        )?
    };
    for (axis, &length) in group_lengths.iter().enumerate() {
        if length == 0 {
            output = output.unsqueeze(axis)?;
        }
    }
    Ok(output)
}

fn checked_axis_product(dims: &[usize], axes: &[usize]) -> Result<usize> {
    axes.iter().try_fold(1usize, |product, &axis| {
        product.checked_mul(dims[axis]).ok_or_else(|| {
            candle_core::Error::msg("permute_and_compose: group product overflows usize")
        })
    })
}

pub(crate) fn plan_permute_compose_group_order(
    dims: &[usize],
    strides: &[usize],
    permutation: &[usize],
    group_lengths: &[usize],
) -> Result<Option<Vec<usize>>> {
    if dims.len() != strides.len()
        || permutation.len() != dims.len()
        || group_lengths.contains(&0)
        || group_lengths
            .iter()
            .try_fold(0usize, |sum, &length| sum.checked_add(length))
            != Some(dims.len())
    {
        candle_core::bail!("permute_and_compose: invalid layout metadata")
    }
    if group_lengths.len() > 8 {
        return Ok(None);
    }
    if permutation_is_c_contiguous(dims, strides, permutation)? {
        return Ok(Some((0..group_lengths.len()).collect()));
    }

    let mut starts = Vec::with_capacity(group_lengths.len());
    let mut start = 0;
    for &length in group_lengths {
        starts.push(start);
        start += length;
    }
    fn search(
        dims: &[usize],
        strides: &[usize],
        permutation: &[usize],
        starts: &[usize],
        lengths: &[usize],
        order: &mut Vec<usize>,
        used: &mut [bool],
    ) -> Result<Option<Vec<usize>>> {
        if order.len() == lengths.len() {
            let mut expected = 1usize;
            for &group in order.iter().rev() {
                let start = starts[group];
                let end = start + lengths[group];
                for &axis in permutation[start..end].iter().rev() {
                    let extent = dims[axis];
                    if extent > 1 && strides[axis] != expected {
                        return Ok(None);
                    }
                    expected = expected.checked_mul(extent).ok_or_else(|| {
                        candle_core::Error::msg(
                            "permute_and_compose: stride product overflows usize",
                        )
                    })?;
                }
            }
            return Ok(Some(order.clone()));
        }
        for group in 0..lengths.len() {
            if used[group] {
                continue;
            }
            used[group] = true;
            order.push(group);
            if let Some(plan) = search(dims, strides, permutation, starts, lengths, order, used)? {
                return Ok(Some(plan));
            }
            order.pop();
            used[group] = false;
        }
        Ok(None)
    }
    search(
        dims,
        strides,
        permutation,
        &starts,
        group_lengths,
        &mut Vec::with_capacity(group_lengths.len()),
        &mut vec![false; group_lengths.len()],
    )
}

fn permutation_is_c_contiguous(
    dims: &[usize],
    strides: &[usize],
    permutation: &[usize],
) -> Result<bool> {
    let mut expected = 1usize;
    for &axis in permutation.iter().rev() {
        let extent = dims[axis];
        if extent > 1 && strides[axis] != expected {
            return Ok(false);
        }
        expected = expected.checked_mul(extent).ok_or_else(|| {
            candle_core::Error::msg("permute_and_compose: stride product overflows usize")
        })?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result};

    #[test]
    fn plans_only_adjacent_homogeneous_sum_and_mean_runs() {
        let mut sums = [
            (0, Operation::Sum),
            (1, Operation::Sum),
            (2, Operation::Sum),
        ];
        let runs = plan_reduction_runs(&mut sums);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].axes, [2, 1, 0]);
        assert!(matches!(runs[0].operation, Operation::Sum));

        let mut mixed = [
            (0, Operation::Sum),
            (1, Operation::Max),
            (2, Operation::Sum),
            (3, Operation::Sum),
        ];
        let runs = plan_reduction_runs(&mut mixed);
        assert_eq!(runs.len(), 3);
        assert_eq!(runs[0].axes, [3, 2]);
        assert!(matches!(runs[0].operation, Operation::Sum));
        assert_eq!(runs[1].axes, [1]);
        assert!(matches!(runs[1].operation, Operation::Max));
        assert_eq!(runs[2].axes, [0]);
        assert!(matches!(runs[2].operation, Operation::Sum));

        let mut excluded = [
            (0, Operation::Min),
            (1, Operation::Min),
            (2, Operation::Prod),
            (3, Operation::Prod),
        ];
        let runs = plan_reduction_runs(&mut excluded);
        assert_eq!(
            runs.len(),
            3,
            "adjacent min may collapse; prod remains sequential"
        );
        assert_eq!(runs[2].axes, [1, 0]);
        assert!(matches!(runs[2].operation, Operation::Min));
    }

    #[test]
    fn homogeneous_runs_issue_one_backend_reduction_call() -> Result<()> {
        let input = Tensor::arange(0f32, 2. * 3. * 4., &Device::Cpu)?.reshape(&[2, 3, 4])?;

        reset_backend_reduction_call_count();
        (&input).reduce_axes(&mut [
            (0, Operation::Sum),
            (1, Operation::Sum),
            (2, Operation::Sum),
        ])?;
        assert_eq!(backend_reduction_call_count(), 1);

        reset_backend_reduction_call_count();
        (&input).reduce_axes(&mut [
            (0, Operation::Mean),
            (1, Operation::Mean),
            (2, Operation::Mean),
        ])?;
        assert_eq!(backend_reduction_call_count(), 1);

        reset_backend_reduction_call_count();
        (&input).reduce_axes(&mut [
            (0, Operation::Sum),
            (1, Operation::Max),
            (2, Operation::Sum),
        ])?;
        assert_eq!(backend_reduction_call_count(), 3);

        reset_backend_reduction_call_count();
        (&input).reduce_axes(&mut [(1, Operation::Min), (2, Operation::Min)])?;
        assert_eq!(backend_reduction_call_count(), 1);

        let strided = input.permute([0, 2, 1])?;
        reset_backend_reduction_call_count();
        let selected = (&strided).reduce_axes(&mut [(1, Operation::Max), (2, Operation::Max)])?;
        assert_eq!(backend_reduction_call_count(), 2);
        assert_eq!(
            selected.to_vec1::<f32>()?,
            strided.max(2)?.max(1)?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn reduce() -> Result<()> {
        let tests = vec![
            (
                Tensor::new(
                    &[
                        0.66984287f32,
                        0.52894678,
                        0.85415958,
                        0.17721198,
                        0.81804799,
                        0.80991797,
                        0.64868822,
                        0.96697902,
                        0.08047191,
                        0.46024353,
                        0.21955009,
                        0.31731976,
                        0.05446258,
                        0.39454557,
                        0.40949016,
                        0.21366165,
                        0.2357463,
                        0.93699481,
                        0.64522596,
                        0.4383618,
                        0.54871827,
                        0.87823442,
                        0.01261184,
                        0.90636503,
                    ],
                    &Device::Cpu,
                )?
                .reshape(&[4, 2, 3])?,
                [(0, Operation::Min)],
                Tensor::new(
                    &[
                        [0.05446258f32, 0.39454557, 0.08047191],
                        [0.17721198, 0.01261184, 0.31731976],
                    ],
                    &Device::Cpu,
                )?,
            ),
            (
                Tensor::new(
                    &[
                        0.66984287f32,
                        0.52894678,
                        0.85415958,
                        0.17721198,
                        0.81804799,
                        0.80991797,
                        0.64868822,
                        0.96697902,
                        0.08047191,
                        0.46024353,
                        0.21955009,
                        0.31731976,
                        0.05446258,
                        0.39454557,
                        0.40949016,
                        0.21366165,
                        0.2357463,
                        0.93699481,
                        0.64522596,
                        0.4383618,
                        0.54871827,
                        0.87823442,
                        0.01261184,
                        0.90636503,
                    ],
                    &Device::Cpu,
                )?
                .reshape(&[4, 2, 3])?,
                [(0, Operation::Max)],
                Tensor::new(
                    &[
                        [0.6698429f32, 0.966979, 0.8541596],
                        [0.87823445, 0.818048, 0.9369948],
                    ],
                    &Device::Cpu,
                )?,
            ),
        ];

        for (tensor, mut axes_operations, expected) in tests {
            assert_eq!(
                tensor.reduce_axes(&mut axes_operations)?.to_vec2::<f32>()?,
                expected.to_vec2::<f32>()?
            );
        }

        Ok(())
    }

    #[test]
    fn candle_transpose() -> Result<()> {
        let tests = vec![(
            Tensor::arange(0f32, (2 * 3 * 4) as f32, &Device::Cpu)?.reshape(&[2, 3, 4])?,
            &[2, 0, 1],
            Tensor::new(
                &[
                    [[0.0f32, 4.0, 8.0], [12.0, 16.0, 20.0]],
                    [[1.0, 5.0, 9.0], [13.0, 17.0, 21.0]],
                    [[2.0, 6.0, 10.0], [14.0, 18.0, 22.0]],
                    [[3.0, 7.0, 11.0], [15.0, 19.0, 23.0]],
                ],
                &Device::Cpu,
            )?,
        )];

        for (tensor, axes, expected) in tests {
            assert_eq!(
                Backend::transpose(&tensor, axes)?.to_vec3::<f32>()?,
                expected.to_vec3::<f32>()?
            );
        }

        Ok(())
    }

    #[test]
    fn tch_add_axes() -> Result<()> {
        let tests = vec![(
            Tensor::arange(0u8, 1 * 2 * 3, &Device::Cpu)?.reshape(&[1, 2, 3])?,
            5,
            &[(0, 5), (3, 3)],
            Tensor::new(
                vec![
                    0u8, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1,
                    2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3,
                    4, 5, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 0, 1, 2, 0, 1, 2,
                    0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5,
                ],
                &Device::Cpu,
            )?
            .reshape(&[5, 1, 2, 3, 3])?,
        )];

        for (tensor, naxes, pos2len, expected) in tests {
            assert_eq!(
                tensor
                    .add_axes(naxes, pos2len)?
                    .flatten_all()?
                    .to_vec1::<u8>()?,
                expected.flatten_all()?.to_vec1::<u8>()?
            );
        }

        Ok(())
    }
}
