//! Focused evidence for composition views selected outside the original static path.

use std::mem::size_of;

use candle_core::{Device, Result, Tensor};
use candle_einops::einops;
use criterion::Criterion;

use crate::{
    Backend, DeviceSynchronizer, Operation, Scenario, ScenarioId, ScenarioSupport, WorkUnits,
    criterion_operation, deterministic_f32_values, prepare,
};

const A: usize = 32;
const B: usize = 24;
const C: usize = 16;
const D: usize = 8;
const REDUCED: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Pattern {
    RuntimeEllipsis,
    PostReduction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Construct,
    Consume,
}

#[derive(Clone, Copy, Debug)]
pub struct ExtendedComposeScenario {
    id: ScenarioId,
    pattern: Pattern,
    mode: Mode,
}

static SCENARIOS: [ExtendedComposeScenario; 4] = [
    ExtendedComposeScenario {
        id: ScenarioId::new("layout/extended-compose/runtime-ellipsis/construct"),
        pattern: Pattern::RuntimeEllipsis,
        mode: Mode::Construct,
    },
    ExtendedComposeScenario {
        id: ScenarioId::new("layout/extended-compose/runtime-ellipsis/consume"),
        pattern: Pattern::RuntimeEllipsis,
        mode: Mode::Consume,
    },
    ExtendedComposeScenario {
        id: ScenarioId::new("layout/extended-compose/post-reduction/construct"),
        pattern: Pattern::PostReduction,
        mode: Mode::Construct,
    },
    ExtendedComposeScenario {
        id: ScenarioId::new("layout/extended-compose/post-reduction/consume"),
        pattern: Pattern::PostReduction,
        mode: Mode::Consume,
    },
];

#[must_use]
pub fn scenarios() -> &'static [ExtendedComposeScenario] {
    &SCENARIOS
}

impl ExtendedComposeScenario {
    fn maybe_consume(&self, tensor: Tensor) -> Result<Tensor> {
        match self.mode {
            Mode::Construct => Ok(tensor),
            Mode::Consume => tensor.contiguous(),
        }
    }

    fn input_elements(&self) -> usize {
        match self.pattern {
            Pattern::RuntimeEllipsis => A * B * C * D,
            Pattern::PostReduction => REDUCED * A * B * C,
        }
    }

    fn output_elements(&self) -> usize {
        match self.pattern {
            Pattern::RuntimeEllipsis => A * B * C * D,
            Pattern::PostReduction => A * B * C,
        }
    }
}

impl Scenario for ExtendedComposeScenario {
    fn id(&self) -> ScenarioId {
        self.id
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let input = self.input_elements();
        let output = self.output_elements();
        let traversals = match self.mode {
            Mode::Construct => 1,
            Mode::Consume => 2,
        };
        let flops = (self.pattern == Pattern::PostReduction).then_some(input);
        WorkUnits::new(
            output as u64,
            ((input + output * traversals) * size_of::<f32>()) as u64,
            flops.map(|value| value as u64),
        )
    }

    fn support(&self, backend: Backend) -> ScenarioSupport {
        if backend != Backend::Cpu
            && self.pattern == Pattern::RuntimeEllipsis
            && self.mode == Mode::Construct
        {
            ScenarioSupport::Unsupported(
                "library path is view-only and enqueues no accelerator work",
            )
        } else {
            ScenarioSupport::Supported
        }
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let elements = self.input_elements();
        let shape: &[usize] = match self.pattern {
            Pattern::RuntimeEllipsis => &[A, B, C, D],
            Pattern::PostReduction => &[REDUCED, A, B, C],
        };
        Ok(vec![Tensor::from_vec(
            deterministic_f32_values(elements, 61),
            shape,
            device,
        )?])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let output = match self.pattern {
            Pattern::RuntimeEllipsis => einops!("a b .. -> b (..) a", &inputs[0])?,
            Pattern::PostReduction => einops!("sum(reduced) a b c -> c (a b)", &inputs[0])?,
        };
        self.maybe_consume(output)
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let output = match self.pattern {
            Pattern::RuntimeEllipsis => inputs[0].permute([1, 2, 3, 0])?.reshape((B, C * D, A))?,
            Pattern::PostReduction => inputs[0].sum(0)?.permute([2, 0, 1])?.reshape((C, A * B))?,
        };
        self.maybe_consume(output)
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("extended composition outputs have different shapes")
        }
        if self.mode == Mode::Construct {
            let eager_cpu_rank_two =
                library.device().is_cpu() && self.pattern == Pattern::PostReduction;
            if eager_cpu_rank_two && (!library.is_contiguous() || !reference.is_contiguous()) {
                candle_core::bail!("CPU rank-two construction must use the eager layout")
            }
            if !eager_cpu_rank_two && (library.is_contiguous() || !reference.is_contiguous()) {
                candle_core::bail!(
                    "construction mode did not discriminate view from historical copy"
                )
            }
        }
        if self.mode == Mode::Consume && (!library.is_contiguous() || !reference.is_contiguous()) {
            candle_core::bail!("consumption mode outputs must both be contiguous")
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        for (index, (&library, &reference)) in library.iter().zip(&reference).enumerate() {
            let allowed = 1e-5 * reference.abs().max(1.);
            if (library - reference).abs() > allowed {
                candle_core::bail!("extended composition outputs differ at {index}")
            }
        }
        Ok(())
    }
}

pub fn criterion_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in scenarios() {
        let prepared = prepare(scenario, &device).expect("extended composition setup");
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
                "extended composition sample must succeed",
            );
        }
    }
}
