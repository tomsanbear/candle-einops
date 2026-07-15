//! Multi-axis extrema lowering spike.

use candle_core::{Device, Result, Tensor};

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

pub fn structural_metrics(layout: ExtremaLayout) -> ExtremaStructuralMetrics {
    ExtremaStructuralMetrics {
        current_submissions: 2,
        candidate_submissions: 2,
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
    sequential(input, layout, kind)
}
