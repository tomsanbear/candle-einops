//! Benchmark-only broadcast-aware GEMM strategies and structural probes.

use candle_core::{Device, Result, Tensor};
use criterion::Criterion;

use crate::diagonal_spike::shares_storage;
use crate::{
    DeviceSynchronizer, Operation, Scenario, ScenarioId, WorkUnits, criterion_operation,
    deterministic_f32_values, prepare,
};

#[derive(Debug)]
pub struct EagerExpansionProbe {
    pub output: Tensor,
    pub left_copy_elements: usize,
    pub right_copy_elements: usize,
    pub peak_temporary_elements: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateStrategy {
    Direct,
    Sliced,
}

#[derive(Clone, Copy, Debug)]
pub struct BroadcastScenario {
    id: ScenarioId,
    case: BroadcastCase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StructuralMetrics {
    pub eager_copy_bytes: usize,
    pub eager_peak_temporary_elements: usize,
    pub eager_gemm_submissions: usize,
    pub selected_copy_bytes: usize,
    pub selected_peak_temporary_elements: usize,
    pub selected_gemm_submissions: usize,
}

#[derive(Clone, Copy, Debug)]
enum BroadcastCase {
    Left,
    Right,
    Both,
    LayoutHostile,
}

struct BroadcastShapes {
    batch: Vec<usize>,
    left: Vec<usize>,
    right: Vec<usize>,
    output: Vec<usize>,
    matrices: usize,
    m: usize,
    k: usize,
    n: usize,
}

fn broadcast_shapes(left: &Tensor, right: &Tensor) -> Result<BroadcastShapes> {
    if left.rank() < 2 || right.rank() < 2 {
        candle_core::bail!("broadcast GEMM operands must have rank at least two")
    }
    if left.dtype() != right.dtype() {
        candle_core::bail!("broadcast GEMM operands must have the same dtype")
    }
    if !left.device().same_device(right.device()) {
        candle_core::bail!("broadcast GEMM operands must be on the same device")
    }
    let left_dims = left.dims();
    let right_dims = right.dims();
    let (m, k) = (left_dims[left.rank() - 2], left_dims[left.rank() - 1]);
    let (right_k, n) = (right_dims[right.rank() - 2], right_dims[right.rank() - 1]);
    if k != right_k {
        candle_core::bail!("broadcast GEMM contracted dimensions differ: {k} and {right_k}")
    }

    let batch_rank = (left.rank() - 2).max(right.rank() - 2);
    let mut batch = vec![1; batch_rank];
    for (operand, dims) in [("left", left_dims), ("right", right_dims)] {
        let operand_batch = dims.len() - 2;
        let offset = batch_rank - operand_batch;
        for (index, &extent) in dims[..operand_batch].iter().enumerate() {
            let resolved = &mut batch[offset + index];
            if *resolved == 1 {
                *resolved = extent;
            } else if extent != 1 && extent != *resolved {
                candle_core::bail!(
                    "{operand} batch extent {extent} cannot broadcast with {}",
                    *resolved
                )
            }
        }
    }
    let matrices = batch.iter().try_fold(1_usize, |product, &extent| {
        product
            .checked_mul(extent)
            .ok_or_else(|| candle_core::Error::msg("broadcast GEMM batch product overflows usize"))
    })?;
    let mut left_shape = batch.clone();
    left_shape.extend_from_slice(&[m, k]);
    let mut right_shape = batch.clone();
    right_shape.extend_from_slice(&[k, n]);
    let mut output = batch.clone();
    output.extend_from_slice(&[m, n]);
    Ok(BroadcastShapes {
        batch,
        left: left_shape,
        right: right_shape,
        output,
        matrices,
        m,
        k,
        n,
    })
}

pub fn eager_expansion_probe(left: &Tensor, right: &Tensor) -> Result<EagerExpansionProbe> {
    let shapes = broadcast_shapes(left, right)?;
    let left_expanded = left.broadcast_as(shapes.left.as_slice())?;
    let right_expanded = right.broadcast_as(shapes.right.as_slice())?;
    let left_materialized = left_expanded.reshape((shapes.matrices, shapes.m, shapes.k))?;
    let right_materialized = right_expanded.reshape((shapes.matrices, shapes.k, shapes.n))?;
    let left_copy_elements = if shares_storage(left, &left_materialized) {
        0
    } else {
        left_materialized.elem_count()
    };
    let right_copy_elements = if shares_storage(right, &right_materialized) {
        0
    } else {
        right_materialized.elem_count()
    };
    let output = left_materialized
        .matmul(&right_materialized)?
        .reshape(shapes.output)?;
    Ok(EagerExpansionProbe {
        output,
        left_copy_elements,
        right_copy_elements,
        peak_temporary_elements: left_copy_elements.saturating_add(right_copy_elements),
    })
}

pub fn direct_broadcast_gemm(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let shapes = broadcast_shapes(left, right)?;
    if left.dims() != shapes.left || right.dims() != shapes.right {
        candle_core::bail!("direct GEMM requires operands that need no batch expansion")
    }
    left.matmul(right)
}

fn aligned_batch_matrix(tensor: &Tensor, batch: &[usize], coordinates: &[usize]) -> Result<Tensor> {
    let matrix_dims = &tensor.dims()[tensor.rank() - 2..];
    let mut aligned_shape = vec![1; batch.len().saturating_sub(tensor.rank() - 2)];
    aligned_shape.extend_from_slice(tensor.dims());
    let mut matrix = tensor.reshape(aligned_shape)?;
    for (axis, (&target, &coordinate)) in batch.iter().zip(coordinates).enumerate() {
        let extent = matrix.dims()[axis];
        if extent != 1 && extent != target {
            candle_core::bail!("slice candidate received an incompatible batch extent")
        }
        matrix = matrix.narrow(axis, if extent == 1 { 0 } else { coordinate }, 1)?;
    }
    matrix.reshape(matrix_dims)
}

fn batch_coordinates(mut flat: usize, batch: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; batch.len()];
    for (axis, &extent) in batch.iter().enumerate().rev() {
        if extent != 0 {
            coordinates[axis] = flat % extent;
            flat /= extent;
        }
    }
    coordinates
}

pub fn sliced_broadcast_gemm(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let shapes = broadcast_shapes(left, right)?;
    if shapes.matrices == 0 || shapes.m == 0 || shapes.n == 0 || shapes.k == 0 {
        return Tensor::zeros(shapes.output, left.dtype(), left.device());
    }
    let mut outputs = Vec::with_capacity(shapes.matrices);
    for matrix_index in 0..shapes.matrices {
        let coordinates = batch_coordinates(matrix_index, &shapes.batch);
        let left_matrix = aligned_batch_matrix(left, &shapes.batch, &coordinates)?;
        let right_matrix = aligned_batch_matrix(right, &shapes.batch, &coordinates)?;
        outputs.push(left_matrix.matmul(&right_matrix)?);
    }
    Tensor::stack(&outputs, 0)?.reshape(shapes.output)
}

pub fn selected_broadcast_gemm(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let shapes = broadcast_shapes(left, right)?;
    if left.dims() == shapes.left && right.dims() == shapes.right {
        direct_broadcast_gemm(left, right)
    } else {
        sliced_broadcast_gemm(left, right)
    }
}

static SCENARIOS: [BroadcastScenario; 4] = [
    BroadcastScenario {
        id: ScenarioId::new("spike/broadcast-gemm/left-broadcast"),
        case: BroadcastCase::Left,
    },
    BroadcastScenario {
        id: ScenarioId::new("spike/broadcast-gemm/right-broadcast"),
        case: BroadcastCase::Right,
    },
    BroadcastScenario {
        id: ScenarioId::new("spike/broadcast-gemm/both-broadcast"),
        case: BroadcastCase::Both,
    },
    BroadcastScenario {
        id: ScenarioId::new("spike/broadcast-gemm/layout-hostile"),
        case: BroadcastCase::LayoutHostile,
    },
];

#[must_use]
pub fn broadcast_scenarios() -> &'static [BroadcastScenario] {
    &SCENARIOS
}

impl BroadcastScenario {
    #[must_use]
    pub fn selected_strategy(self) -> CandidateStrategy {
        match self.case {
            BroadcastCase::Left | BroadcastCase::Right | BroadcastCase::Both => {
                CandidateStrategy::Sliced
            }
            BroadcastCase::LayoutHostile => CandidateStrategy::Direct,
        }
    }

    #[must_use]
    pub fn modeled_submissions(self) -> usize {
        match self.selected_strategy() {
            CandidateStrategy::Direct => 1,
            CandidateStrategy::Sliced => match self.case {
                BroadcastCase::Left | BroadcastCase::Right => 32,
                BroadcastCase::Both => 64,
                BroadcastCase::LayoutHostile => unreachable!("layout-hostile uses direct GEMM"),
            },
        }
    }

    pub fn structural_metrics(self, device: &Device) -> Result<StructuralMetrics> {
        let inputs = self.setup(device)?;
        let eager = eager_expansion_probe(&inputs[0], &inputs[1])?;
        Ok(StructuralMetrics {
            eager_copy_bytes: eager
                .left_copy_elements
                .saturating_add(eager.right_copy_elements)
                .saturating_mul(size_of::<f32>()),
            eager_peak_temporary_elements: eager.peak_temporary_elements,
            eager_gemm_submissions: 1,
            selected_copy_bytes: 0,
            selected_peak_temporary_elements: match self.selected_strategy() {
                CandidateStrategy::Direct => 0,
                CandidateStrategy::Sliced => self.output_elements(),
            },
            selected_gemm_submissions: self.modeled_submissions(),
        })
    }

    fn input_elements(self) -> usize {
        match self.case {
            BroadcastCase::Left | BroadcastCase::Right => 33 * 32 * 32,
            BroadcastCase::Both => 16 * 32 * 32,
            BroadcastCase::LayoutHostile => 32 * 32 * 32,
        }
    }

    pub fn output_elements(self) -> usize {
        match self.case {
            BroadcastCase::Left | BroadcastCase::Right => 32 * 32 * 32,
            BroadcastCase::Both => 64 * 32 * 32,
            BroadcastCase::LayoutHostile => 16 * 32 * 32,
        }
    }
}

impl Scenario for BroadcastScenario {
    fn id(&self) -> ScenarioId {
        self.id
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        WorkUnits::new(
            self.output_elements() as u64,
            ((self.input_elements() + self.output_elements()) * size_of::<f32>()) as u64,
            None,
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let tensor = |shape: &[usize], stream| {
            let elements = shape.iter().product();
            Tensor::from_vec(deterministic_f32_values(elements, stream), shape, device)
        };
        match self.case {
            BroadcastCase::Left => Ok(vec![tensor(&[1, 32, 32], 11)?, tensor(&[32, 32, 32], 12)?]),
            BroadcastCase::Right => Ok(vec![tensor(&[32, 32, 32], 13)?, tensor(&[1, 32, 32], 14)?]),
            BroadcastCase::Both => Ok(vec![
                tensor(&[8, 1, 32, 32], 15)?,
                tensor(&[1, 8, 32, 32], 16)?,
            ]),
            BroadcastCase::LayoutHostile => Ok(vec![
                tensor(&[16, 32, 32], 17)?.transpose(1, 2)?,
                tensor(&[16, 32, 32], 18)?,
            ]),
        }
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        Ok(eager_expansion_probe(&inputs[0], &inputs[1])?.output)
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        match self.selected_strategy() {
            CandidateStrategy::Direct => direct_broadcast_gemm(&inputs[0], &inputs[1]),
            CandidateStrategy::Sliced => sliced_broadcast_gemm(&inputs[0], &inputs[1]),
        }
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!(
                "broadcast GEMM strategies produced different shapes for {}",
                self.id.as_str()
            )
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        if let Some((index, (library, reference))) = library
            .iter()
            .zip(&reference)
            .enumerate()
            .find(|(_, (library, reference))| library != reference)
        {
            candle_core::bail!(
                "broadcast GEMM strategies differ for {} at {index}: {library} vs {reference}",
                self.id.as_str()
            )
        }
        Ok(())
    }
}

pub fn criterion_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in broadcast_scenarios() {
        let prepared = prepare(scenario, &device).expect("broadcast GEMM setup");
        let id = scenario.id().as_str();
        criterion_operation(
            criterion,
            &format!("{id}/eager"),
            &prepared,
            Operation::Library,
            &synchronizer,
            "eager broadcast GEMM sample must succeed",
        );
        criterion_operation(
            criterion,
            &format!("{id}/selected"),
            &prepared,
            Operation::Reference,
            &synchronizer,
            "selected broadcast GEMM sample must succeed",
        );
    }
}
