//! Focused evidence for layout-aware binary einsum operand packing.

use candle_core::{Device, Result, Tensor};
use candle_einops::__private::benchmark_pack_canonical_operand;
use criterion::Criterion;

use crate::{
    Backend, DeviceSynchronizer, Operation, Scenario, ScenarioId, ScenarioSupport, WorkUnits,
    criterion_operation, deterministic_f32_values, prepare,
};

const A: usize = 32;
const B: usize = 24;
const K: usize = 64;
const N: usize = 72;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Construct,
    Consume,
}

#[derive(Clone, Copy, Debug)]
pub struct BinaryOperandPackingScenario {
    id: ScenarioId,
    mode: Mode,
}

static SCENARIOS: [BinaryOperandPackingScenario; 2] = [
    BinaryOperandPackingScenario {
        id: ScenarioId::new("einsum/binary-packing/recovered-view/construct"),
        mode: Mode::Construct,
    },
    BinaryOperandPackingScenario {
        id: ScenarioId::new("einsum/binary-packing/recovered-view/consume"),
        mode: Mode::Consume,
    },
];

#[must_use]
pub fn scenarios() -> &'static [BinaryOperandPackingScenario] {
    &SCENARIOS
}

impl BinaryOperandPackingScenario {
    fn pack_library(input: &Tensor) -> Result<Tensor> {
        benchmark_pack_canonical_operand(input, &[1, A * B, K], &[0, 2, 1])
    }

    fn pack_reference(input: &Tensor) -> Result<Tensor> {
        input.reshape((1, A * B, K))
    }

    fn maybe_consume(&self, packed: Tensor, right: &Tensor) -> Result<Tensor> {
        match self.mode {
            Mode::Construct => Ok(packed),
            Mode::Consume => packed.matmul(&right.unsqueeze(0)?),
        }
    }
}

impl Scenario for BinaryOperandPackingScenario {
    fn id(&self) -> ScenarioId {
        self.id
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let elements = match self.mode {
            Mode::Construct => A * B * K,
            Mode::Consume => A * B * N,
        };
        let flops = (self.mode == Mode::Consume).then_some(2 * A * B * K * N);
        WorkUnits::new(
            elements as u64,
            ((A * B * K + K * N + elements) * size_of::<f32>()) as u64,
            flops.map(|value| value as u64),
        )
    }

    fn support(&self, backend: Backend) -> ScenarioSupport {
        if backend != Backend::Cpu && self.mode == Mode::Construct {
            ScenarioSupport::Unsupported(
                "library path is view-only and enqueues no accelerator work",
            )
        } else {
            ScenarioSupport::Supported
        }
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let source = Tensor::from_vec(deterministic_f32_values(K * A * B, 51), (K, A, B), device)?;
        let canonical = source.permute((1, 2, 0))?;
        let right = Tensor::from_vec(deterministic_f32_values(K * N, 52), (K, N), device)?;
        Ok(vec![canonical, right])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.maybe_consume(Self::pack_library(&inputs[0])?, &inputs[1])
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.maybe_consume(Self::pack_reference(&inputs[0])?, &inputs[1])
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("binary operand packing outputs have different shapes")
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        if library != reference {
            candle_core::bail!("binary operand packing outputs have different values")
        }
        Ok(())
    }
}

pub fn criterion_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in scenarios() {
        let prepared = prepare(scenario, &device).expect("binary operand packing setup");
        for operation in [Operation::Library, Operation::Reference] {
            let name = format!(
                "{}/{}",
                scenario.id().as_str(),
                match operation {
                    Operation::Library => "library",
                    Operation::Reference => "reference",
                }
            );
            criterion_operation(
                criterion,
                &name,
                &prepared,
                operation,
                &synchronizer,
                "binary operand packing sample must succeed",
            );
        }
    }
}
