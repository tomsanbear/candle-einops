//! Benchmark-only probes and candidates for repeated-label diagonal lowering.

use std::hint::black_box;

use candle_core::{CpuStorage, Device, Result, Storage, Tensor};
use candle_einops::einsum;
use criterion::Criterion;
use serde::Serialize;

use crate::{
    Clock, DeviceSynchronizer, Estimate, Fingerprint, Operation, RESULT_SCHEMA_VERSION, Scenario,
    ScenarioId, Synchronizer, WorkUnits, criterion_operation, prepare, summarize,
};

#[derive(Debug)]
pub struct CurrentLoweringProbe {
    pub adjacent: Tensor,
    pub flattened: Tensor,
    pub copied_elements: usize,
}

pub fn cpu_storage_id(tensor: &Tensor) -> Result<usize> {
    let (storage, _) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cpu(CpuStorage::F32(values)) => Ok(values.as_ptr() as usize),
        Storage::Cpu(CpuStorage::U32(values)) => Ok(values.as_ptr() as usize),
        storage => candle_core::bail!("expected CPU f32/u32 storage, received {storage:?}"),
    }
}

pub fn probe_current_interleaved_lowering(
    input: &Tensor,
    first_extent: usize,
    second_extent: usize,
) -> Result<CurrentLoweringProbe> {
    if input.dims() != [first_extent, second_extent, first_extent, second_extent] {
        candle_core::bail!("interleaved probe shape does not match i j i j extents")
    }
    let adjacent = input.permute((0, 2, 1, 3))?;
    let flattened = adjacent.reshape((
        first_extent.checked_mul(first_extent).ok_or_else(|| {
            candle_core::Error::msg("interleaved repeated extent product overflows usize")
        })?,
        second_extent,
        second_extent,
    ))?;
    let copied_elements = if cpu_storage_id(input)? == cpu_storage_id(&flattened)? {
        0
    } else {
        input.elem_count()
    };
    Ok(CurrentLoweringProbe {
        adjacent,
        flattened,
        copied_elements,
    })
}

pub fn build_repeated_indices(
    extent: usize,
    multiplicity: usize,
    device: &Device,
) -> Result<Tensor> {
    if multiplicity < 2 {
        candle_core::bail!("diagonal index multiplicity must be at least two")
    }
    let mut stride = 0_usize;
    let mut power = 1_usize;
    for _ in 0..multiplicity {
        stride = stride
            .checked_add(power)
            .ok_or_else(|| candle_core::Error::msg("diagonal stride overflows usize"))?;
        power = power
            .checked_mul(extent)
            .ok_or_else(|| candle_core::Error::msg("diagonal extent product overflows usize"))?;
    }
    let indices = (0..extent)
        .map(|coordinate| checked_u32_index(coordinate, stride))
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_vec(indices, extent, device)
}

pub fn build_interleaved_indices(
    first_extent: usize,
    second_extent: usize,
    device: &Device,
) -> Result<Tensor> {
    let dims = [first_extent, second_extent, first_extent, second_extent];
    let strides = contiguous_strides(&dims)?;
    let first_stride = strides[0]
        .checked_add(strides[2])
        .ok_or_else(|| candle_core::Error::msg("first interleaved stride overflows usize"))?;
    let second_stride = strides[1]
        .checked_add(strides[3])
        .ok_or_else(|| candle_core::Error::msg("second interleaved stride overflows usize"))?;
    let mut indices = Vec::with_capacity(first_extent.saturating_mul(second_extent));
    for first in 0..first_extent {
        for second in 0..second_extent {
            let first_offset = first
                .checked_mul(first_stride)
                .ok_or_else(|| candle_core::Error::msg("first diagonal offset overflows usize"))?;
            let second_offset = second
                .checked_mul(second_stride)
                .ok_or_else(|| candle_core::Error::msg("second diagonal offset overflows usize"))?;
            indices.push(u32::try_from(
                first_offset.checked_add(second_offset).ok_or_else(|| {
                    candle_core::Error::msg("interleaved diagonal offset overflows usize")
                })?,
            )?);
        }
    }
    Tensor::from_vec(indices, (first_extent, second_extent), device)?.flatten_all()
}

fn checked_u32_index(coordinate: usize, stride: usize) -> Result<u32> {
    let index = coordinate
        .checked_mul(stride)
        .ok_or_else(|| candle_core::Error::msg("diagonal index overflows usize"))?;
    u32::try_from(index).map_err(candle_core::Error::wrap)
}

fn contiguous_strides(dims: &[usize]) -> Result<Vec<usize>> {
    let mut stride = 1_usize;
    let mut strides = vec![0; dims.len()];
    for (index, &dim) in dims.iter().enumerate().rev() {
        strides[index] = stride;
        stride = stride
            .checked_mul(dim)
            .ok_or_else(|| candle_core::Error::msg("contiguous stride overflows usize"))?;
    }
    Ok(strides)
}

pub fn cached_flat_gather(
    input: &Tensor,
    indices: &Tensor,
    output_shape: &[usize],
) -> Result<Tensor> {
    if !input.is_contiguous() {
        candle_core::bail!("cached flat diagonal gather requires contiguous input")
    }
    let expected = output_shape.iter().try_fold(1_usize, |product, &extent| {
        product
            .checked_mul(extent)
            .ok_or_else(|| candle_core::Error::msg("diagonal output shape overflows usize"))
    })?;
    if indices.elem_count() != expected {
        candle_core::bail!(
            "diagonal index count {} does not match output elements {expected}",
            indices.elem_count()
        )
    }
    input
        .flatten_all()?
        .index_select(&indices.flatten_all()?, 0)?
        .reshape(output_shape)
}

#[derive(Clone, Copy, Debug)]
enum Pattern {
    Simple,
    Interleaved,
}

#[derive(Clone, Copy, Debug)]
pub struct DiagonalScenario {
    id: ScenarioId,
    pattern: Pattern,
    first_extent: usize,
    second_extent: usize,
}

impl DiagonalScenario {
    #[must_use]
    pub fn input_elements(self) -> usize {
        match self.pattern {
            Pattern::Simple => self.first_extent * self.first_extent,
            Pattern::Interleaved => {
                self.first_extent * self.second_extent * self.first_extent * self.second_extent
            }
        }
    }

    #[must_use]
    pub fn output_elements(self) -> usize {
        match self.pattern {
            Pattern::Simple => self.first_extent,
            Pattern::Interleaved => self.first_extent * self.second_extent,
        }
    }

    #[must_use]
    pub fn index_elements(self) -> usize {
        self.output_elements()
    }

    #[must_use]
    pub fn current_copy_elements(self) -> usize {
        match self.pattern {
            Pattern::Simple => 0,
            Pattern::Interleaved => self.input_elements(),
        }
    }

    pub fn build_indices(self, device: &Device) -> Result<Tensor> {
        match self.pattern {
            Pattern::Simple => build_repeated_indices(self.first_extent, 2, device),
            Pattern::Interleaved => {
                build_interleaved_indices(self.first_extent, self.second_extent, device)
            }
        }
    }

    fn input_shape(self) -> Vec<usize> {
        match self.pattern {
            Pattern::Simple => vec![self.first_extent, self.first_extent],
            Pattern::Interleaved => vec![
                self.first_extent,
                self.second_extent,
                self.first_extent,
                self.second_extent,
            ],
        }
    }

    fn output_shape(self) -> Vec<usize> {
        match self.pattern {
            Pattern::Simple => vec![self.first_extent],
            Pattern::Interleaved => vec![self.first_extent, self.second_extent],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct IndexPreparationRecord {
    pub schema_version: u32,
    pub scenario_id: ScenarioId,
    pub sample_count: usize,
    pub input_elements: usize,
    pub current_copy_elements: usize,
    pub output_elements: usize,
    pub index_elements: usize,
    pub index_preparation: Estimate,
    pub fingerprint: Fingerprint,
}

pub fn measure_index_preparation(
    scenario: DiagonalScenario,
    device: &Device,
    synchronizer: &dyn Synchronizer,
    clock: &dyn Clock,
    sample_count: usize,
    fingerprint: Fingerprint,
) -> Result<IndexPreparationRecord> {
    if sample_count == 0 {
        candle_core::bail!("index preparation sample count must be non-zero")
    }
    let mut samples = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        synchronizer.synchronize()?;
        let started = clock.now_ns();
        let indices = scenario.build_indices(device)?;
        black_box(&indices);
        synchronizer.synchronize()?;
        samples.push(clock.now_ns().saturating_sub(started));
        black_box(indices);
    }
    Ok(IndexPreparationRecord {
        schema_version: RESULT_SCHEMA_VERSION,
        scenario_id: scenario.id(),
        sample_count,
        input_elements: scenario.input_elements(),
        current_copy_elements: scenario.current_copy_elements(),
        output_elements: scenario.output_elements(),
        index_elements: scenario.index_elements(),
        index_preparation: summarize(&samples),
        fingerprint,
    })
}

impl Scenario for DiagonalScenario {
    fn id(&self) -> ScenarioId {
        self.id
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        WorkUnits::new(
            self.input_elements() as u64,
            (self.input_elements() * size_of::<f32>()) as u64,
            None,
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let input = Tensor::arange(0f32, self.input_elements() as f32, device)?
            .reshape(self.input_shape())?;
        Ok(vec![input, self.build_indices(device)?])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        match self.pattern {
            Pattern::Simple => einsum!("i i -> i", &inputs[0]),
            Pattern::Interleaved => einsum!("i j i j -> i j", &inputs[0]),
        }
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        cached_flat_gather(&inputs[0], &inputs[1], &self.output_shape())
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims()
            || library.flatten_all()?.to_vec1::<f32>()?
                != reference.flatten_all()?.to_vec1::<f32>()?
        {
            candle_core::bail!("diagonal spike candidate differs from current lowering")
        }
        Ok(())
    }
}

static SCENARIOS: [DiagonalScenario; 6] = [
    DiagonalScenario {
        id: ScenarioId::new("spike/diagonal/simple/n16"),
        pattern: Pattern::Simple,
        first_extent: 16,
        second_extent: 1,
    },
    DiagonalScenario {
        id: ScenarioId::new("spike/diagonal/simple/n64"),
        pattern: Pattern::Simple,
        first_extent: 64,
        second_extent: 1,
    },
    DiagonalScenario {
        id: ScenarioId::new("spike/diagonal/simple/n256"),
        pattern: Pattern::Simple,
        first_extent: 256,
        second_extent: 1,
    },
    DiagonalScenario {
        id: ScenarioId::new("spike/diagonal/interleaved/n4"),
        pattern: Pattern::Interleaved,
        first_extent: 4,
        second_extent: 4,
    },
    DiagonalScenario {
        id: ScenarioId::new("spike/diagonal/interleaved/n8"),
        pattern: Pattern::Interleaved,
        first_extent: 8,
        second_extent: 8,
    },
    DiagonalScenario {
        id: ScenarioId::new("spike/diagonal/interleaved/n16"),
        pattern: Pattern::Interleaved,
        first_extent: 16,
        second_extent: 16,
    },
];

#[must_use]
pub fn scenarios() -> &'static [DiagonalScenario] {
    &SCENARIOS
}

pub fn criterion_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in scenarios() {
        let prepared = prepare(scenario, &device).expect("diagonal setup");
        let id = scenario.id().as_str();
        criterion_operation(
            criterion,
            &format!("{id}/current"),
            &prepared,
            Operation::Library,
            &synchronizer,
            "current diagonal sample must succeed",
        );
        criterion_operation(
            criterion,
            &format!("{id}/cached-flat-gather"),
            &prepared,
            Operation::Reference,
            &synchronizer,
            "diagonal candidate sample must succeed",
        );
    }
}
