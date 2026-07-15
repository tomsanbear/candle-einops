use candle_core::{Result, Shape, Tensor};

use crate::Operation;

/// Tensor operations used by [`crate::einops!`].
///
/// Transformations return Candle [`Result`] values so backend failures retain
/// their original error and context.
pub trait Backend {
    type Output;
    fn shape(self) -> Vec<usize>;
    fn reshape(self, shape: &[usize]) -> Result<Self::Output>;
    fn transpose(self, axes: &[usize]) -> Result<Self::Output>;
    fn reduce_axes(self, axes_operations: &mut [(usize, Operation)]) -> Result<Self::Output>;
    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Result<Self::Output>;
}

impl<T: AsRef<Tensor>> Backend for T {
    type Output = Tensor;

    fn shape(self) -> Vec<usize> {
        self.as_ref().dims().to_vec()
    }

    fn reshape(self, shape: &[usize]) -> Result<Self::Output> {
        let shape = Shape::from_dims(shape);
        self.as_ref().reshape(shape)
    }

    fn transpose(self, axes: &[usize]) -> Result<Self::Output> {
        self.as_ref().permute(axes)
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

        axes_operations.sort_by_key(|(axis, _)| *axis);

        for (axis, operation) in axes_operations.iter().rev() {
            output = match operation {
                Operation::Min => output.min(*axis)?,
                Operation::Max => output.max(*axis)?,
                Operation::Sum => output.sum(&[*axis][..])?,
                Operation::Mean => output.mean(&[*axis][..])?,
                Operation::Prod => {
                    let axis_len = output.dim(*axis)?;
                    if axis_len == 0 {
                        let mut shape = output.dims().to_vec();
                        shape.remove(*axis);
                        Tensor::ones(Shape::from_dims(&shape), output.dtype(), output.device())?
                    } else {
                        let mut product = output.narrow(*axis, 0, 1)?.squeeze(*axis)?;
                        for index in 1..axis_len {
                            let factor = output.narrow(*axis, index, 1)?.squeeze(*axis)?;
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
        let mut output = self.as_ref().clone();

        let expected_naxes = output.rank() + pos2len.len();
        if naxes != expected_naxes {
            candle_core::bail!("add_axes: expected final rank {expected_naxes}, got {naxes}")
        }

        let mut repeats = vec![1; naxes];
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
            output = output.unsqueeze(axis_pos)?;
            repeats[axis_pos] = axis_len;
        }

        let shape = Shape::from_dims(&repeats[..]);
        output.repeat(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result};

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
