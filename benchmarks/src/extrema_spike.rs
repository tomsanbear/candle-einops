//! Multi-axis extrema lowering spike.

use std::mem::size_of;

use candle_core::{Device, Result, Tensor};
use criterion::Criterion;

use crate::{
    DeviceSynchronizer, Operation, Scenario, ScenarioId, WorkUnits, criterion_operation, prepare,
};

const A: usize = 32;
const B: usize = 24;
const C: usize = 16;
const D: usize = 8;
const INPUT_ELEMENTS: usize = A * B * C * D;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExtremaKind {
    Min,
    Max,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExtremaLayout {
    ContiguousTrailing,
    ContiguousLeading,
    StridedTrailing,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExtremaStructuralMetrics {
    pub current_submissions: usize,
    pub candidate_submissions: usize,
    pub candidate_materialized_elements: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ExtremaScenario {
    id: ScenarioId,
    layout: ExtremaLayout,
    kind: ExtremaKind,
}

static SCENARIOS: [ExtremaScenario; 6] = [
    ExtremaScenario {
        id: ScenarioId::new("spike/extrema/contiguous-trailing/min"),
        layout: ExtremaLayout::ContiguousTrailing,
        kind: ExtremaKind::Min,
    },
    ExtremaScenario {
        id: ScenarioId::new("spike/extrema/contiguous-trailing/max"),
        layout: ExtremaLayout::ContiguousTrailing,
        kind: ExtremaKind::Max,
    },
    ExtremaScenario {
        id: ScenarioId::new("spike/extrema/contiguous-leading/min"),
        layout: ExtremaLayout::ContiguousLeading,
        kind: ExtremaKind::Min,
    },
    ExtremaScenario {
        id: ScenarioId::new("spike/extrema/contiguous-leading/max"),
        layout: ExtremaLayout::ContiguousLeading,
        kind: ExtremaKind::Max,
    },
    ExtremaScenario {
        id: ScenarioId::new("spike/extrema/strided-trailing/min"),
        layout: ExtremaLayout::StridedTrailing,
        kind: ExtremaKind::Min,
    },
    ExtremaScenario {
        id: ScenarioId::new("spike/extrema/strided-trailing/max"),
        layout: ExtremaLayout::StridedTrailing,
        kind: ExtremaKind::Max,
    },
];

#[must_use]
pub fn scenarios() -> &'static [ExtremaScenario] {
    &SCENARIOS
}

pub fn structural_metrics(layout: ExtremaLayout) -> ExtremaStructuralMetrics {
    ExtremaStructuralMetrics {
        current_submissions: 2,
        candidate_submissions: 1,
        candidate_materialized_elements: usize::from(layout == ExtremaLayout::StridedTrailing)
            * INPUT_ELEMENTS,
    }
}

pub fn fixture(layout: ExtremaLayout, device: &Device) -> Result<Tensor> {
    let values = (0..INPUT_ELEMENTS)
        .map(|index| ((index * 17 + 11) % (INPUT_ELEMENTS + 1)) as f32)
        .collect::<Vec<_>>();
    match layout {
        ExtremaLayout::ContiguousTrailing | ExtremaLayout::ContiguousLeading => {
            Tensor::from_vec(values, (A, B, C, D), device)
        }
        ExtremaLayout::StridedTrailing => {
            Tensor::from_vec(values, (A, C, B, D), device)?.permute([0, 2, 1, 3])
        }
    }
}

fn reduce_once(input: &Tensor, axis: usize, kind: ExtremaKind) -> Result<Tensor> {
    match kind {
        ExtremaKind::Min => input.min(axis),
        ExtremaKind::Max => input.max(axis),
    }
}

pub fn sequential(input: &Tensor, layout: ExtremaLayout, kind: ExtremaKind) -> Result<Tensor> {
    match layout {
        ExtremaLayout::ContiguousLeading => {
            reduce_once(&reduce_once(input, 1, kind)?, 0, kind)
        }
        ExtremaLayout::ContiguousTrailing | ExtremaLayout::StridedTrailing => {
            reduce_once(&reduce_once(input, 3, kind)?, 2, kind)
        }
    }
}

pub fn collapsed_candidate(
    input: &Tensor,
    layout: ExtremaLayout,
    kind: ExtremaKind,
) -> Result<Tensor> {
    match layout {
        ExtremaLayout::ContiguousLeading => reduce_once(&input.reshape((A * B, C, D))?, 0, kind),
        ExtremaLayout::ContiguousTrailing | ExtremaLayout::StridedTrailing => {
            reduce_once(&input.reshape((A, B, C * D))?, 2, kind)
        }
    }
}

impl Scenario for ExtremaScenario {
    fn id(&self) -> ScenarioId {
        self.id
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let output_elements = match self.layout {
            ExtremaLayout::ContiguousLeading => C * D,
            ExtremaLayout::ContiguousTrailing | ExtremaLayout::StridedTrailing => A * B,
        };
        WorkUnits::new(
            INPUT_ELEMENTS as u64,
            ((INPUT_ELEMENTS + output_elements) * size_of::<f32>()) as u64,
            None,
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        Ok(vec![fixture(self.layout, device)?])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        sequential(&inputs[0], self.layout, self.kind)
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        collapsed_candidate(&inputs[0], self.layout, self.kind)
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims()
            || library.flatten_all()?.to_vec1::<f32>()?
                != reference.flatten_all()?.to_vec1::<f32>()?
        {
            candle_core::bail!("extrema spike outputs differ")
        }
        Ok(())
    }
}

pub fn criterion_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in scenarios() {
        let prepared = prepare(scenario, &device).expect("extrema spike setup");
        for operation in [Operation::Library, Operation::Reference] {
            let name = format!(
                "{}/{}",
                scenario.id().as_str(),
                match operation {
                    Operation::Library => "sequential",
                    Operation::Reference => "collapsed",
                }
            );
            criterion_operation(
                criterion,
                &name,
                &prepared,
                operation,
                &synchronizer,
                "extrema spike sample must succeed",
            );
        }
    }
}
