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
}
