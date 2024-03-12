use candle_core::{Shape, Tensor};

use crate::backend::{Backend, Operation};

impl<T: AsRef<Tensor>> Backend for T {
    type Output = Tensor;

    fn shape(self) -> Vec<usize> {
        self.as_ref()
            .dims()
            .iter()
            .map(|&x| x as usize)
            .collect::<Vec<_>>()
    }

    fn reshape(self, shape: &[usize]) -> Self::Output {
        let shape = Shape::from_dims(shape);
        self.as_ref().reshape(shape).unwrap()
    }

    fn transpose(self, axes: &[usize]) -> Self::Output {
        self.as_ref().permute(axes).unwrap()
    }

    fn reduce_axes(self, axes_operations: &mut [(usize, Operation)]) -> Self::Output {
        let mut output = self.as_ref().clone();

        axes_operations.sort_by_key(|(axis, _)| *axis);

        for (axis, operation) in axes_operations.iter().rev() {
            output = match operation {
                Operation::Min => output.min(*axis).unwrap(),
                Operation::Max => output.max(*axis).unwrap(),
                Operation::Sum => output.sum(&[*axis][..]).unwrap(),
                Operation::Mean => output.mean(&[*axis][..]).unwrap(),
                // TODO: implement prod
            };
        }

        output
    }

    fn add_axes(self, naxes: usize, pos2len: &[(usize, usize)]) -> Self::Output {
        let mut output = self.as_ref().clone();

        let mut repeats = vec![1; naxes];

        for &(axis_pos, axis_len) in pos2len {
            output = output.unsqueeze(axis_pos).unwrap();
            repeats[axis_pos] = axis_len;
        }

        let shape = Shape::from_dims(&repeats[..]);
        output.repeat(shape).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Result};

    #[test]
    fn tch_reduce() -> Result<()> {
        let tests = vec![(
            Tensor::new(
                &[
                    0.66984287, 0.52894678, 0.85415958, 0.17721198, 0.81804799, 0.80991797,
                    0.64868822, 0.96697902, 0.08047191, 0.46024353, 0.21955009, 0.31731976,
                    0.05446258, 0.39454557, 0.40949016, 0.21366165, 0.2357463, 0.93699481,
                    0.64522596, 0.4383618, 0.54871827, 0.87823442, 0.01261184, 0.90636503,
                ],
                &Device::Cpu,
            )?
            .reshape(&[4, 2, 3]),
            [(0, Operation::Min)],
            Tensor::new(
                &[
                    0.05446258, 0.39454557, 0.08047191, 0.17721198, 0.01261184, 0.31731976,
                ],
                &Device::Cpu,
            )?
            .reshape(&[2, 3]),
        )];

        for (tensor, mut axes_operations, expected) in tests {
            assert_eq!(
                tensor.reduce_axes(&mut axes_operations).shape(),
                expected.shape()
            );
        }

        Ok(())
    }

    #[test]
    fn candle_transpose() -> Result<()> {
        let tests = vec![(
            Tensor::arange(0u8, 2 * 3 * 4 * 5, &Device::Cpu)?.reshape(&[2, 3, 4, 5]),
            &[3, 0, 2, 1],
            Tensor::new(
                vec![
                    0u8, 20, 40, 5, 25, 45, 10, 30, 50, 15, 35, 55, 60, 80, 100, 65, 85, 105, 70,
                    90, 110, 75, 95, 115, 1, 21, 41, 6, 26, 46, 11, 31, 51, 16, 36, 56, 61, 81,
                    101, 66, 86, 106, 71, 91, 111, 76, 96, 116, 2, 22, 42, 7, 27, 47, 12, 32, 52,
                    17, 37, 57, 62, 82, 102, 67, 87, 107, 72, 92, 112, 77, 97, 117, 3, 23, 43, 8,
                    28, 48, 13, 33, 53, 18, 38, 58, 63, 83, 103, 68, 88, 108, 73, 93, 113, 78, 98,
                    118, 4, 24, 44, 9, 29, 49, 14, 34, 54, 19, 39, 59, 64, 84, 104, 69, 89, 109,
                    74, 94, 114, 79, 99, 119,
                ],
                &Device::Cpu,
            )?
            .reshape(&[5, 2, 4, 3]),
        )];

        for (tensor, axes, expected) in tests {
            assert_eq!(Backend::transpose(&tensor, axes).shape(), expected.shape());
        }

        Ok(())
    }

    #[test]
    fn tch_add_axes() -> Result<()> {
        let tests = vec![(
            Tensor::arange(0f32, 1.0 * 2.0 * 3.0, &Device::Cpu)?.reshape(&[1, 2, 3]),
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
            .reshape(&[5, 1, 2, 3, 3]),
        )];

        for (tensor, naxes, pos2len, expected) in tests {
            assert_eq!(tensor.add_axes(naxes, pos2len).shape(), expected.shape());
        }

        Ok(())
    }
}
