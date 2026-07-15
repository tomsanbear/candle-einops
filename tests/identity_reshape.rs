use std::cell::RefCell;
use std::rc::Rc;

use candle_core::{DType, Device, Result, Storage, Tensor, Var};
use candle_einops::{Backend, Operation, einops};

fn storage_address(tensor: &Tensor) -> *const Storage {
    let (storage, _) = tensor.storage_and_layout();
    std::ptr::from_ref(&*storage)
}

fn assert_identity_metadata(input: &Tensor, output: &Tensor, context: &str) {
    assert_eq!(output.dims(), input.dims(), "{context}: shape");
    assert_eq!(output.id(), input.id(), "{context}: tensor identity");
    assert_eq!(
        storage_address(output),
        storage_address(input),
        "{context}: storage"
    );
    assert_eq!(output.layout(), input.layout(), "{context}: layout");
}

#[test]
fn non_contiguous_identity_reshape_preserves_storage_layout_values_and_autograd() -> Result<()> {
    let device = Device::Cpu;
    let input = Var::from_vec(
        (0..24).map(|value| value as f32).collect(),
        (2, 3, 4),
        &device,
    )?;
    let permuted = input.permute((1, 0, 2))?;
    assert!(!permuted.is_contiguous());

    let output = Backend::reshape(&permuted, permuted.dims())?;
    assert_identity_metadata(&permuted, &output, "non-contiguous");
    assert_eq!(
        output.flatten_all()?.to_vec1::<f32>()?,
        permuted.flatten_all()?.to_vec1::<f32>()?
    );

    let weights = Tensor::reshape(&Tensor::arange(1f32, 25., &device)?, (3, 2, 4))?;
    let gradients = output.mul(&weights)?.sum_all()?.backward()?;
    let gradient = gradients
        .get(input.as_tensor())
        .expect("identity reshape must retain the input autograd edge");
    assert_eq!(gradient.dims(), [2, 3, 4]);
    assert_eq!(
        gradient.flatten_all()?.to_vec1::<f32>()?,
        weights
            .permute((1, 0, 2))?
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    Ok(())
}

#[test]
fn contiguous_scalar_zero_singleton_and_offset_identities_are_shallow_clones() -> Result<()> {
    let device = Device::Cpu;
    let contiguous = Tensor::reshape(&Tensor::arange(0u32, 6, &device)?, (2, 3))?;
    let scalar = Tensor::new(7f32, &device)?;
    let zero = Tensor::zeros((2, 0, 3), DType::F32, &device)?;
    let singleton = Tensor::reshape(&Tensor::arange(0u32, 6, &device)?, (1, 6, 1))?;
    let storage = Tensor::reshape(&Tensor::arange(0u32, 20, &device)?, (5, 4))?;
    let offset = storage.narrow(0, 1, 3)?;
    assert!(offset.layout().start_offset() > 0);

    for (name, input) in [
        ("contiguous", contiguous),
        ("scalar", scalar),
        ("zero", zero),
        ("singleton", singleton),
        ("offset", offset),
    ] {
        let output = Backend::reshape(&input, input.dims())?;
        assert_identity_metadata(&input, &output, name);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct RecordingBackend {
    shape: Vec<usize>,
    reshapes: Rc<RefCell<Vec<Vec<usize>>>>,
}

impl RecordingBackend {
    fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            reshapes: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn unsupported<T>() -> Result<T> {
        candle_core::bail!("recording backend operation is outside this test")
    }
}

impl Backend for RecordingBackend {
    type Output = Self;

    fn shape(self) -> Vec<usize> {
        self.shape
    }

    fn reshape(mut self, shape: &[usize]) -> Result<Self::Output> {
        self.reshapes.borrow_mut().push(shape.to_vec());
        self.shape = shape.to_vec();
        Ok(self)
    }

    fn transpose(self, _axes: &[usize]) -> Result<Self::Output> {
        Self::unsupported()
    }

    fn reduce_axes(self, _axes_operations: &mut [(usize, Operation)]) -> Result<Self::Output> {
        Self::unsupported()
    }

    fn add_axes(self, _naxes: usize, _pos2len: &[(usize, usize)]) -> Result<Self::Output> {
        Self::unsupported()
    }
}

impl Backend for &RecordingBackend {
    type Output = RecordingBackend;

    fn shape(self) -> Vec<usize> {
        self.shape.clone()
    }

    fn reshape(self, shape: &[usize]) -> Result<Self::Output> {
        <RecordingBackend as Backend>::reshape(self.clone(), shape)
    }

    fn transpose(self, _axes: &[usize]) -> Result<Self::Output> {
        RecordingBackend::unsupported()
    }

    fn reduce_axes(self, _axes_operations: &mut [(usize, Operation)]) -> Result<Self::Output> {
        RecordingBackend::unsupported()
    }

    fn add_axes(self, _naxes: usize, _pos2len: &[(usize, usize)]) -> Result<Self::Output> {
        RecordingBackend::unsupported()
    }
}

#[test]
fn singleton_grouping_boundaries_remain_public_backend_operations() -> Result<()> {
    let device = Device::Cpu;
    let input = Tensor::reshape(&Tensor::arange(0u32, 6, &device)?, (2, 1, 3))?;
    let composed = einops!("row singleton column -> (row singleton) column", &input)?;
    assert_eq!(composed.dims(), [2, 3]);
    assert_eq!(
        composed.flatten_all()?.to_vec1::<u32>()?,
        [0, 1, 2, 3, 4, 5]
    );
    let decomposed = einops!(
        "(row singleton:1) column -> row singleton column",
        &composed
    )?;
    assert_eq!(decomposed.dims(), [2, 1, 3]);

    let recording = RecordingBackend::new(&[2, 1, 3]);
    let calls = recording.reshapes.clone();
    let output = einops!("row singleton column -> (row singleton) column", recording)?;
    assert_eq!(output.shape, [2, 3]);
    assert_eq!(&*calls.borrow(), &[vec![2, 3]]);
    Ok(())
}

#[test]
fn shape_changes_and_invalid_metadata_keep_existing_boundaries() -> Result<()> {
    let device = Device::Cpu;
    let contiguous = Tensor::reshape(&Tensor::arange(0u32, 6, &device)?, (2, 3))?;
    let non_contiguous = Tensor::transpose(&contiguous, 0, 1)?;
    let flattened = Backend::reshape(&non_contiguous, &[6])?;
    assert_eq!(flattened.dims(), [6]);
    assert_eq!(flattened.to_vec1::<u32>()?, [0, 3, 1, 4, 2, 5]);
    assert!(flattened.is_contiguous());
    assert_ne!(
        storage_address(&flattened),
        storage_address(&non_contiguous)
    );

    let error = Backend::reshape(&contiguous, &[4, 2])
        .expect_err("mismatched element counts must still fail");
    assert!(error.to_string().contains("shape mismatch in reshape"));

    let recording = RecordingBackend::new(&[5]);
    let calls = recording.reshapes.clone();
    let error = einops!("(row:2 column) -> row column", recording)
        .expect_err("invalid decomposition metadata must fail before reshape");
    assert!(error.to_string().contains("not divisible"));
    assert!(calls.borrow().is_empty());
    Ok(())
}
