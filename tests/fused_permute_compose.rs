use std::cell::RefCell;
use std::rc::Rc;

use candle_core::{Device, Result, Storage, Tensor, Var};
use candle_einops::{Backend, Operation, einops};

#[derive(Clone, Debug)]
struct RecordingBackend {
    shape: Vec<usize>,
    calls: Rc<RefCell<Vec<&'static str>>>,
}

impl RecordingBackend {
    fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            calls: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl Backend for RecordingBackend {
    type Output = Self;

    fn shape(self) -> Vec<usize> {
        self.shape
    }

    fn reshape(mut self, shape: &[usize]) -> Result<Self::Output> {
        self.calls.borrow_mut().push("reshape");
        self.shape = shape.to_vec();
        Ok(self)
    }

    fn transpose(mut self, axes: &[usize]) -> Result<Self::Output> {
        self.calls.borrow_mut().push("transpose");
        self.shape = axes.iter().map(|&axis| self.shape[axis]).collect();
        Ok(self)
    }

    fn permute_and_compose(
        mut self,
        _permutation: &[usize],
        output_shape: &[usize],
        _group_lengths: &[usize],
    ) -> Result<Self::Output>
    where
        Self::Output: Backend<Output = Self::Output>,
    {
        self.calls.borrow_mut().push("fused");
        self.shape = output_shape.to_vec();
        Ok(self)
    }

    fn reduce_axes(mut self, axes: &mut [(usize, Operation)]) -> Result<Self::Output> {
        self.calls.borrow_mut().push("reduce");
        axes.sort_by_key(|(axis, _)| *axis);
        for &(axis, _) in axes.iter().rev() {
            self.shape.remove(axis);
        }
        Ok(self)
    }

    fn add_axes(mut self, naxes: usize, positions: &[(usize, usize)]) -> Result<Self::Output> {
        self.calls.borrow_mut().push("repeat");
        let mut output = Vec::with_capacity(naxes);
        let mut input_axis = 0;
        for axis in 0..naxes {
            if let Some((_, extent)) = positions.iter().find(|(position, _)| *position == axis) {
                output.push(*extent);
            } else {
                output.push(self.shape[input_axis]);
                input_axis += 1;
            }
        }
        self.shape = output;
        Ok(self)
    }
}

impl Backend for &RecordingBackend {
    type Output = RecordingBackend;
    fn shape(self) -> Vec<usize> {
        self.shape.clone()
    }
    fn reshape(self, shape: &[usize]) -> Result<Self::Output> {
        self.clone().reshape(shape)
    }
    fn transpose(self, axes: &[usize]) -> Result<Self::Output> {
        self.clone().transpose(axes)
    }
    fn reduce_axes(self, axes: &mut [(usize, Operation)]) -> Result<Self::Output> {
        self.clone().reduce_axes(axes)
    }
    fn add_axes(self, naxes: usize, positions: &[(usize, usize)]) -> Result<Self::Output> {
        self.clone().add_axes(naxes, positions)
    }
}

#[derive(Clone, Debug)]
struct DefaultBackend(RecordingBackend);

impl Backend for DefaultBackend {
    type Output = Self;

    fn shape(self) -> Vec<usize> {
        self.0.shape
    }

    fn reshape(mut self, shape: &[usize]) -> Result<Self::Output> {
        self.0.calls.borrow_mut().push("reshape");
        self.0.shape = shape.to_vec();
        Ok(self)
    }

    fn transpose(mut self, axes: &[usize]) -> Result<Self::Output> {
        self.0.calls.borrow_mut().push("transpose");
        self.0.shape = axes.iter().map(|&axis| self.0.shape[axis]).collect();
        Ok(self)
    }

    fn reduce_axes(self, _axes: &mut [(usize, Operation)]) -> Result<Self::Output> {
        unreachable!()
    }

    fn add_axes(self, _naxes: usize, _positions: &[(usize, usize)]) -> Result<Self::Output> {
        unreachable!()
    }
}

impl Backend for &DefaultBackend {
    type Output = DefaultBackend;
    fn shape(self) -> Vec<usize> {
        self.0.shape.clone()
    }
    fn reshape(self, shape: &[usize]) -> Result<Self::Output> {
        self.clone().reshape(shape)
    }
    fn transpose(self, axes: &[usize]) -> Result<Self::Output> {
        self.clone().transpose(axes)
    }
    fn reduce_axes(self, axes: &mut [(usize, Operation)]) -> Result<Self::Output> {
        self.clone().reduce_axes(axes)
    }
    fn add_axes(self, naxes: usize, positions: &[(usize, usize)]) -> Result<Self::Output> {
        self.clone().add_axes(naxes, positions)
    }
}

#[test]
fn fused_codegen_is_narrow_and_default_backend_remains_compatible() -> Result<()> {
    let recording = RecordingBackend::new(&[2, 3, 4]);
    let calls = recording.calls.clone();
    let output = einops!("a b c -> c (a b)", recording)?;
    assert_eq!(output.shape, [4, 6]);
    assert_eq!(&*calls.borrow(), &["fused"]);

    let default = DefaultBackend(RecordingBackend::new(&[2, 3, 4]));
    let calls = default.0.calls.clone();
    let output =
        <DefaultBackend as Backend>::permute_and_compose(default, &[2, 0, 1], &[4, 6], &[1, 2])?;
    assert_eq!(output.0.shape, [4, 6]);
    assert_eq!(&*calls.borrow(), &["transpose", "reshape"]);

    let reduction = RecordingBackend::new(&[2, 3, 4, 5]);
    let calls = reduction.calls.clone();
    let _ = einops!("a b sum(reduced) c -> c (a b)", reduction)?;
    assert_eq!(&*calls.borrow(), &["reduce", "transpose", "reshape"]);

    let repeat = RecordingBackend::new(&[2, 3]);
    let calls = repeat.calls.clone();
    let _ = einops!("a b -> b (a repeated:2)", repeat)?;
    assert_eq!(&*calls.borrow(), &["transpose", "repeat", "reshape"]);
    Ok(())
}

fn storage_address(tensor: &Tensor) -> *const Storage {
    let (storage, _) = tensor.storage_and_layout();
    std::ptr::from_ref(&*storage)
}

fn assert_values_equal(left: &Tensor, right: &Tensor) -> Result<()> {
    assert_eq!(left.dims(), right.dims());
    assert_eq!(
        left.flatten_all()?.to_vec1::<f32>()?,
        right.flatten_all()?.to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn tensor_selected_views_preserve_values_storage_offsets_and_gradients() -> Result<()> {
    let device = Device::Cpu;
    let input = Var::from_vec((0..24).map(|v| v as f32).collect(), (2, 3, 4), &device)?;
    let old_permuted = input.permute([2, 0, 1])?;
    let old = Tensor::reshape(&old_permuted, (4, 6))?;
    let selected = einops!("a b c -> c (a b)", input.as_tensor())?;
    assert_values_equal(&selected, &old)?;
    assert_eq!(selected.stride(), [1, 4]);
    assert_eq!(
        storage_address(&selected),
        storage_address(input.as_tensor())
    );
    assert_ne!(storage_address(&old), storage_address(input.as_tensor()));

    let weights = Tensor::reshape(&Tensor::arange(1f32, 25., &device)?, (4, 6))?;
    let selected_gradients = selected.mul(&weights)?.sum_all()?.backward()?;
    let old_gradients = old.mul(&weights)?.sum_all()?.backward()?;
    assert_values_equal(
        selected_gradients.get(input.as_tensor()).unwrap(),
        old_gradients.get(input.as_tensor()).unwrap(),
    )?;

    let storage = Tensor::reshape(&Tensor::arange(0f32, 48., &device)?, (4, 3, 4))?;
    let offset = storage.narrow(0, 1, 2)?;
    let offset_output = einops!("a b c -> c (a b)", &offset)?;
    assert_eq!(storage_address(&offset_output), storage_address(&offset));
    assert_eq!(
        offset_output.layout().start_offset(),
        offset.layout().start_offset()
    );

    let nchw = Tensor::reshape(&Tensor::arange(0f32, 210., &device)?, (2, 3, 5, 7))?;
    let nhwc = einops!("n c h w -> n (h w) c", &nchw)?;
    let nhwc_permuted = nchw.permute([0, 2, 3, 1])?;
    let nhwc_old = Tensor::reshape(&nhwc_permuted, (2, 35, 3))?;
    assert_values_equal(&nhwc, &nhwc_old)?;
    assert_eq!(nhwc.stride(), [105, 1, 35]);
    assert_eq!(storage_address(&nhwc), storage_address(&nchw));
    Ok(())
}

#[test]
fn tensor_noncontiguous_zero_singleton_identity_and_fallback_boundaries() -> Result<()> {
    let device = Device::Cpu;
    let base = Tensor::reshape(&Tensor::arange(0f32, 24., &device)?, (2, 3, 4))?;
    let source = base.permute([2, 0, 1])?;
    let eligible =
        <&Tensor as Backend>::permute_and_compose(&source, &[0, 1, 2], &[4, 6], &[1, 2])?;
    let eligible_old = Tensor::reshape(&source, (4, 6))?;
    assert_values_equal(&eligible, &eligible_old)?;
    assert_eq!(storage_address(&eligible), storage_address(&source));
    assert_ne!(storage_address(&eligible_old), storage_address(&source));

    let fallback =
        <&Tensor as Backend>::permute_and_compose(&source, &[0, 1, 2], &[8, 3], &[2, 1])?;
    let fallback_old = Tensor::reshape(&source, (8, 3))?;
    assert_values_equal(&fallback, &fallback_old)?;
    assert_ne!(storage_address(&fallback), storage_address(&source));

    let singleton = Tensor::reshape(&Tensor::arange(0f32, 6., &device)?, (2, 1, 3))?;
    let singleton_output =
        <&Tensor as Backend>::permute_and_compose(&singleton, &[1, 0, 2], &[1, 6], &[1, 2])?;
    let singleton_permuted = singleton.permute([1, 0, 2])?;
    let singleton_old = Tensor::reshape(&singleton_permuted, (1, 6))?;
    assert_eq!(singleton_output.layout(), singleton_old.layout());

    let zero = Tensor::zeros((2, 0, 3), candle_core::DType::F32, &device)?;
    let zero_output =
        <&Tensor as Backend>::permute_and_compose(&zero, &[2, 0, 1], &[3, 0], &[1, 2])?;
    assert_eq!(zero_output.dims(), [3, 0]);
    assert_eq!(storage_address(&zero_output), storage_address(&zero));

    let identity =
        <&Tensor as Backend>::permute_and_compose(&base, &[0, 1, 2], &[2, 3, 4], &[1, 1, 1])?;
    assert_eq!(identity.layout(), base.layout());
    assert_eq!(storage_address(&identity), storage_address(&base));
    Ok(())
}

#[test]
fn fused_metadata_errors_are_deterministic_and_selected_errors_do_not_retry() -> Result<()> {
    let input = Tensor::reshape(&Tensor::arange(0f32, 24., &Device::Cpu)?, (2, 3, 4))?;
    assert!(
        <&Tensor as Backend>::permute_and_compose(&input, &[0, 0, 2], &[2, 12], &[1, 2]).is_err()
    );
    assert!(<&Tensor as Backend>::permute_and_compose(&input, &[0, 1, 2], &[24], &[2]).is_err());
    assert!(<&Tensor as Backend>::permute_and_compose(&input, &[0, 1, 2], &[24], &[4]).is_err());
    Ok(())
}
