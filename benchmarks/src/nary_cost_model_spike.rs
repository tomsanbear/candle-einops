//! Benchmark-only bounded n-ary contraction cost-model spike.

use std::hint::black_box;
use std::time::Instant;

use candle_core::{Device, Result, Tensor};
use candle_einops::__private::{
    EinsumAxisPattern, EllipsisEinsumSpec, benchmark_binary_graph_estimate,
    benchmark_nary_planner_selects_exact, execute_nary_einsum,
};
use criterion::Criterion;
use serde::Serialize;

use crate::{
    DeviceSynchronizer, Estimate, Fingerprint, Operation, RESULT_SCHEMA_VERSION, Scenario,
    ScenarioId, WorkUnits, criterion_operation, prepare, summarize,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AxisExtent {
    pub label: &'static str,
    pub extent: usize,
}

impl AxisExtent {
    #[must_use]
    pub const fn new(label: &'static str, extent: usize) -> Self {
        Self { label, extent }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutClass {
    Contiguous,
    Transposed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelOperand {
    stable_ordinal: usize,
    axes: Vec<AxisExtent>,
    layout: LayoutClass,
    members: u64,
}

impl ModelOperand {
    #[must_use]
    pub fn new(stable_ordinal: usize, axes: &[AxisExtent], layout: LayoutClass) -> Self {
        Self {
            stable_ordinal,
            axes: axes.to_vec(),
            layout,
            members: 1_u64.checked_shl(stable_ordinal as u32).unwrap_or(0),
        }
    }

    fn elements(&self) -> Result<u128> {
        checked_product(
            self.axes.iter().map(|axis| axis.extent as u128),
            "operand elements",
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkModel {
    operands: Vec<ModelOperand>,
    final_output: Vec<&'static str>,
    global_axis_order: Vec<&'static str>,
}

impl NetworkModel {
    pub fn new(operands: Vec<ModelOperand>, final_output: &[&'static str]) -> Result<Self> {
        if operands.is_empty() {
            candle_core::bail!("cost model requires at least one operand")
        }
        let mut seen_members = 0_u64;
        let mut global_axis_order = Vec::new();
        for operand in &operands {
            if operand.members == 0 || seen_members & operand.members != 0 {
                candle_core::bail!("cost model operand ordinals must be unique and below 64")
            }
            seen_members |= operand.members;
            let mut local = Vec::new();
            for axis in &operand.axes {
                if axis.label.is_empty() || local.contains(&axis.label) {
                    candle_core::bail!("cost model operand axes must be non-empty and unique")
                }
                local.push(axis.label);
                if !global_axis_order.contains(&axis.label) {
                    global_axis_order.push(axis.label);
                }
            }
        }
        for &label in final_output {
            if !global_axis_order.contains(&label) {
                candle_core::bail!("cost model output label `{label}` is absent from inputs")
            }
        }
        Ok(Self {
            operands,
            final_output: final_output.to_vec(),
            global_axis_order,
        })
    }

    #[must_use]
    pub fn operands(&self) -> &[ModelOperand] {
        &self.operands
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CostWeights {
    pub flop: u128,
    pub copy_byte: u128,
    pub intermediate_element: u128,
    pub peak_live_element: u128,
    pub submission: u128,
}

impl CostWeights {
    pub const CPU: Self = Self {
        flop: 1,
        copy_byte: 1,
        intermediate_element: 2,
        peak_live_element: 2,
        submission: 1_024,
    };
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct PairEstimate {
    pub flops: u128,
    pub output_elements: u128,
    pub copy_bytes: u128,
    pub submissions: u128,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PlanStep {
    pub members: (u64, u64),
    pub output_axes: Vec<&'static str>,
    pub estimate: PairEstimate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PlanMetrics {
    pub flops: u128,
    pub intermediate_elements: u128,
    pub output_elements: u128,
    pub copy_bytes: u128,
    pub peak_live_elements: u128,
    pub submissions: u128,
    pub score: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractionPlan {
    pub steps: Vec<PlanStep>,
    pub metrics: PlanMetrics,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FixtureKind {
    LinearChain,
    BalancedTree,
    BroadcastHeavy,
    LayoutHostile,
}

#[derive(Clone, Debug)]
pub struct NetworkFixture {
    pub id: &'static str,
    pub kind: FixtureKind,
    pub model: NetworkModel,
}

#[derive(Clone, Debug, Serialize)]
pub struct PlannerProbeRecord {
    pub schema_version: u32,
    pub scenario_id: &'static str,
    pub arity: usize,
    pub greedy_metrics: PlanMetrics,
    pub exact_metrics: PlanMetrics,
    pub greedy_members: Vec<(u64, u64)>,
    pub exact_members: Vec<(u64, u64)>,
    pub greedy_planner: Estimate,
    pub exact_planner: Estimate,
    pub selected_planner_p95_ns: u64,
    pub budget_us: u64,
    pub budget_met: bool,
    pub fingerprint: Fingerprint,
}

pub fn measure_fixture_planners(
    fixture: &NetworkFixture,
    samples: usize,
    fingerprint: Fingerprint,
) -> Result<PlannerProbeRecord> {
    if samples == 0 {
        candle_core::bail!("planner probe sample count must be non-zero")
    }
    let greedy = plan_output_greedy(&fixture.model, CostWeights::CPU)?;
    let exact = plan_bounded_exact(&fixture.model, CostWeights::CPU)?;
    let runtime_inputs = fixture_inputs(fixture, &Device::Cpu)?;
    let runtime_refs = runtime_inputs.iter().collect::<Vec<_>>();
    let input_axes = fixture
        .model
        .operands
        .iter()
        .map(|operand| {
            operand
                .axes
                .iter()
                .map(|axis| axis.label)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let patterns = input_axes
        .iter()
        .map(|axes| EinsumAxisPattern::new(axes, None))
        .collect::<Vec<_>>();
    let runtime_spec = EllipsisEinsumSpec::new(
        &patterns,
        EinsumAxisPattern::new(&fixture.model.final_output, None),
    );
    let production_selected_exact =
        benchmark_nary_planner_selects_exact(&runtime_refs, runtime_spec)?;
    if production_selected_exact != selected_uses_exact(&fixture.model)? {
        candle_core::bail!(
            "production and frozen selectors disagree for {}",
            fixture.id
        )
    }
    let mut greedy_samples = Vec::with_capacity(samples);
    let mut exact_samples = Vec::with_capacity(samples);
    let mut selected_samples = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        black_box(plan_output_greedy(&fixture.model, CostWeights::CPU)?);
        greedy_samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        let started = Instant::now();
        black_box(plan_bounded_exact(&fixture.model, CostWeights::CPU)?);
        exact_samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
        let started = Instant::now();
        let selected = benchmark_nary_planner_selects_exact(&runtime_refs, runtime_spec)?;
        black_box(selected);
        selected_samples.push(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    }
    selected_samples.sort_unstable();
    let p95_index = (samples - 1) * 95 / 100;
    let selected_planner_p95_ns = selected_samples[p95_index];
    let budget_us = 175;
    Ok(PlannerProbeRecord {
        schema_version: RESULT_SCHEMA_VERSION,
        scenario_id: fixture.id,
        arity: fixture.model.operands.len(),
        greedy_metrics: greedy.metrics,
        exact_metrics: exact.metrics,
        greedy_members: greedy.steps.iter().map(|step| step.members).collect(),
        exact_members: exact.steps.iter().map(|step| step.members).collect(),
        greedy_planner: summarize(&greedy_samples),
        exact_planner: summarize(&exact_samples),
        selected_planner_p95_ns,
        budget_us,
        budget_met: selected_planner_p95_ns <= budget_us * 1_000,
        fingerprint,
    })
}

fn checked_product(values: impl IntoIterator<Item = u128>, context: &'static str) -> Result<u128> {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.contains(&0) {
        return Ok(0);
    }
    values.into_iter().try_fold(1_u128, |product, value| {
        product
            .checked_mul(value)
            .ok_or_else(|| candle_core::Error::msg(format!("{context} overflows u128")))
    })
}

fn checked_add(left: u128, right: u128, context: &'static str) -> Result<u128> {
    left.checked_add(right)
        .ok_or_else(|| candle_core::Error::msg(format!("{context} overflows u128")))
}

fn axis_extent(operand: &ModelOperand, label: &str) -> Option<usize> {
    operand
        .axes
        .iter()
        .find(|axis| axis.label == label)
        .map(|axis| axis.extent)
}

fn resolve_extent(left: Option<usize>, right: Option<usize>, label: &str) -> Result<usize> {
    match (left, right) {
        (Some(left), Some(right)) if left == right => Ok(left),
        (Some(1), Some(right)) => Ok(right),
        (Some(left), Some(1)) => Ok(left),
        (Some(left), None) | (None, Some(left)) => Ok(left),
        (Some(left), Some(right)) => {
            candle_core::bail!("axis `{label}` cannot broadcast extents {left} and {right}")
        }
        (None, None) => candle_core::bail!("axis `{label}` is absent from both operands"),
    }
}

fn pair_details(
    state: &[ModelOperand],
    left: usize,
    right: usize,
    model: &NetworkModel,
) -> Result<(PairEstimate, ModelOperand)> {
    let left_operand = state
        .get(left)
        .ok_or_else(|| candle_core::Error::msg("left pair index is out of range"))?;
    let right_operand = state
        .get(right)
        .ok_or_else(|| candle_core::Error::msg("right pair index is out of range"))?;
    if left >= right {
        candle_core::bail!("pair indices must be ordered")
    }
    let mut union = Vec::new();
    for &label in &model.global_axis_order {
        if axis_extent(left_operand, label).is_some() || axis_extent(right_operand, label).is_some()
        {
            union.push(label);
        }
    }
    let mut output_labels = Vec::new();
    for &label in &union {
        let live_elsewhere = state.iter().enumerate().any(|(index, operand)| {
            index != left && index != right && axis_extent(operand, label).is_some()
        });
        if model.final_output.contains(&label) || live_elsewhere {
            output_labels.push(label);
        }
    }
    let output_axes = output_labels
        .iter()
        .map(|label| {
            resolve_extent(
                axis_extent(left_operand, label),
                axis_extent(right_operand, label),
                label,
            )
            .map(|extent| AxisExtent::new(label, extent))
        })
        .collect::<Result<Vec<_>>>()?;
    let labels = |operand: &ModelOperand| {
        operand
            .axes
            .iter()
            .map(|axis| axis.label)
            .collect::<Vec<_>>()
    };
    let dims = |operand: &ModelOperand| {
        operand
            .axes
            .iter()
            .map(|axis| axis.extent)
            .collect::<Vec<_>>()
    };
    let strides = |operand: &ModelOperand, dims: &[usize]| -> Result<Vec<usize>> {
        let mut storage_dims = dims.to_vec();
        if operand.layout == LayoutClass::Transposed && storage_dims.len() >= 2 {
            let last = storage_dims.len() - 1;
            storage_dims.swap(last - 1, last);
        }
        let mut storage_strides = vec![0; storage_dims.len()];
        let mut stride = 1usize;
        for (axis, &extent) in storage_dims.iter().enumerate().rev() {
            storage_strides[axis] = stride;
            stride = stride.saturating_mul(extent);
        }
        if operand.layout == LayoutClass::Transposed && storage_strides.len() >= 2 {
            let last = storage_strides.len() - 1;
            storage_strides.swap(last - 1, last);
        }
        Ok(storage_strides)
    };
    let left_labels = labels(left_operand);
    let left_dims = dims(left_operand);
    let left_strides = strides(left_operand, &left_dims)?;
    let right_labels = labels(right_operand);
    let right_dims = dims(right_operand);
    let right_strides = strides(right_operand, &right_dims)?;
    let graph = benchmark_binary_graph_estimate(
        &left_labels,
        &left_dims,
        &left_strides,
        &right_labels,
        &right_dims,
        &right_strides,
        &output_labels,
    )?;
    let members = left_operand.members | right_operand.members;
    Ok((
        PairEstimate {
            flops: graph.work,
            output_elements: graph.output_elements,
            copy_bytes: graph.copy_bytes,
            submissions: graph.submissions,
        },
        ModelOperand {
            stable_ordinal: left_operand
                .stable_ordinal
                .min(right_operand.stable_ordinal),
            axes: output_axes,
            layout: LayoutClass::Contiguous,
            members,
        },
    ))
}

pub fn estimate_pair(model: &NetworkModel, left: usize, right: usize) -> Result<PairEstimate> {
    Ok(pair_details(&model.operands, left, right, model)?.0)
}

fn initial_metrics(state: &[ModelOperand]) -> Result<PlanMetrics> {
    let peak_live_elements = state.iter().try_fold(0_u128, |sum, operand| {
        checked_add(sum, operand.elements()?, "initial live elements")
    })?;
    Ok(PlanMetrics {
        flops: 0,
        intermediate_elements: 0,
        output_elements: 0,
        copy_bytes: 0,
        peak_live_elements,
        submissions: 0,
        score: 0,
    })
}

fn accumulate_step(
    metrics: &mut PlanMetrics,
    state: &[ModelOperand],
    estimate: &PairEstimate,
) -> Result<()> {
    let live = state.iter().try_fold(0_u128, |sum, operand| {
        checked_add(sum, operand.elements()?, "live elements")
    })?;
    metrics.peak_live_elements = metrics.peak_live_elements.max(checked_add(
        live,
        estimate.output_elements,
        "peak live elements",
    )?);
    metrics.flops = checked_add(metrics.flops, estimate.flops, "plan FLOPs")?;
    metrics.intermediate_elements = checked_add(
        metrics.intermediate_elements,
        estimate.output_elements,
        "plan intermediate elements",
    )?;
    metrics.copy_bytes = checked_add(metrics.copy_bytes, estimate.copy_bytes, "plan copy bytes")?;
    metrics.submissions = checked_add(
        metrics.submissions,
        estimate.submissions,
        "plan submissions",
    )?;
    metrics.output_elements = estimate.output_elements;
    Ok(())
}

fn score(metrics: &PlanMetrics, weights: CostWeights) -> Result<u128> {
    [
        (metrics.flops, weights.flop),
        (metrics.copy_bytes, weights.copy_byte),
        (metrics.intermediate_elements, weights.intermediate_element),
        (metrics.peak_live_elements, weights.peak_live_element),
        (metrics.submissions, weights.submission),
    ]
    .into_iter()
    .try_fold(0_u128, |total, (value, weight)| {
        let term = value
            .checked_mul(weight)
            .ok_or_else(|| candle_core::Error::msg("cost-model weighted term overflows u128"))?;
        checked_add(total, term, "cost-model score")
    })
}

fn apply_pair(state: &mut Vec<ModelOperand>, left: usize, right: usize, output: ModelOperand) {
    state.remove(right);
    state.remove(left);
    state.insert(left, output);
}

pub fn plan_output_greedy(model: &NetworkModel, weights: CostWeights) -> Result<ContractionPlan> {
    let mut state = model.operands.clone();
    let mut metrics = initial_metrics(&state)?;
    let mut steps = Vec::new();
    while state.len() > 1 {
        let mut best: Option<(usize, usize, PairEstimate, ModelOperand)> = None;
        for left in 0..state.len() - 1 {
            for right in left + 1..state.len() {
                let (estimate, output) = pair_details(&state, left, right, model)?;
                let key = (
                    estimate.output_elements,
                    estimate.flops,
                    state[left].stable_ordinal,
                    state[right].stable_ordinal,
                    left,
                    right,
                );
                if best
                    .as_ref()
                    .is_none_or(|(best_left, best_right, best_estimate, _)| {
                        key < (
                            best_estimate.output_elements,
                            best_estimate.flops,
                            state[*best_left].stable_ordinal,
                            state[*best_right].stable_ordinal,
                            *best_left,
                            *best_right,
                        )
                    })
                {
                    best = Some((left, right, estimate, output));
                }
            }
        }
        let (left, right, estimate, output) =
            best.ok_or_else(|| candle_core::Error::msg("greedy planner found no pair"))?;
        accumulate_step(&mut metrics, &state, &estimate)?;
        steps.push(PlanStep {
            members: (state[left].members, state[right].members),
            output_axes: output.axes.iter().map(|axis| axis.label).collect(),
            estimate,
        });
        apply_pair(&mut state, left, right, output);
    }
    metrics.score = score(&metrics, weights)?;
    Ok(ContractionPlan { steps, metrics })
}

fn exact_search(
    model: &NetworkModel,
    weights: CostWeights,
    state: Vec<ModelOperand>,
    steps: Vec<PlanStep>,
    metrics: PlanMetrics,
    best: &mut Option<ContractionPlan>,
) -> Result<()> {
    if state.len() == 1 {
        let mut metrics = metrics;
        metrics.score = score(&metrics, weights)?;
        let candidate = ContractionPlan { steps, metrics };
        if best.as_ref().is_none_or(|current| {
            (candidate.metrics.score, &candidate.steps) < (current.metrics.score, &current.steps)
        }) {
            *best = Some(candidate);
        }
        return Ok(());
    }
    for left in 0..state.len() - 1 {
        for right in left + 1..state.len() {
            let (estimate, output) = pair_details(&state, left, right, model)?;
            let mut next_state = state.clone();
            let mut next_metrics = metrics.clone();
            accumulate_step(&mut next_metrics, &state, &estimate)?;
            let mut next_steps = steps.clone();
            next_steps.push(PlanStep {
                members: (state[left].members, state[right].members),
                output_axes: output.axes.iter().map(|axis| axis.label).collect(),
                estimate,
            });
            apply_pair(&mut next_state, left, right, output);
            exact_search(model, weights, next_state, next_steps, next_metrics, best)?;
        }
    }
    Ok(())
}

pub fn plan_bounded_exact(model: &NetworkModel, weights: CostWeights) -> Result<ContractionPlan> {
    if !(3..=6).contains(&model.operands.len()) {
        candle_core::bail!("bounded exact planner supports arity 3 through 6")
    }
    let metrics = initial_metrics(&model.operands)?;
    let mut best = None;
    exact_search(
        model,
        weights,
        model.operands.clone(),
        Vec::new(),
        metrics,
        &mut best,
    )?;
    best.ok_or_else(|| candle_core::Error::msg("bounded exact planner found no plan"))
}

pub fn selected_uses_exact(model: &NetworkModel) -> Result<bool> {
    plan_output_greedy(model, CostWeights::CPU)?;
    Ok(false)
}

fn matrix_chain(
    id: &'static str,
    kind: FixtureKind,
    dimensions: [usize; 5],
    first_layout: LayoutClass,
    batch: Option<usize>,
) -> Result<NetworkFixture> {
    let [a, b, c, d, e] = dimensions;
    let operand = |ordinal, labels: [(&'static str, usize); 2], layout| {
        let mut axes = batch
            .map(|extent| {
                vec![AxisExtent::new(
                    "batch",
                    if ordinal == 0 { 1 } else { extent },
                )]
            })
            .unwrap_or_default();
        axes.extend(labels.map(|(label, extent)| AxisExtent::new(label, extent)));
        ModelOperand::new(ordinal, &axes, layout)
    };
    let operands = vec![
        operand(0, [("a", a), ("b", b)], first_layout),
        operand(1, [("b", b), ("c", c)], LayoutClass::Contiguous),
        operand(2, [("c", c), ("d", d)], LayoutClass::Contiguous),
        operand(3, [("d", d), ("e", e)], LayoutClass::Contiguous),
    ];
    let final_output = if batch.is_some() {
        vec!["batch", "a", "e"]
    } else {
        vec!["a", "e"]
    };
    Ok(NetworkFixture {
        id,
        kind,
        model: NetworkModel::new(operands, &final_output)?,
    })
}

pub fn network_fixtures() -> Vec<NetworkFixture> {
    vec![
        matrix_chain(
            "spike/nary-cost/linear-chain",
            FixtureKind::LinearChain,
            [30, 35, 15, 5, 10],
            LayoutClass::Contiguous,
            None,
        )
        .expect("bounded linear fixture"),
        matrix_chain(
            "spike/nary-cost/balanced-tree",
            FixtureKind::BalancedTree,
            [128, 8, 1, 8, 128],
            LayoutClass::Contiguous,
            None,
        )
        .expect("bounded balanced fixture"),
        matrix_chain(
            "spike/nary-cost/broadcast-heavy",
            FixtureKind::BroadcastHeavy,
            [32, 32, 15, 5, 10],
            LayoutClass::Contiguous,
            Some(32),
        )
        .expect("bounded broadcast fixture"),
        matrix_chain(
            "spike/nary-cost/layout-hostile",
            FixtureKind::LayoutHostile,
            [30, 35, 15, 5, 10],
            LayoutClass::Transposed,
            None,
        )
        .expect("bounded layout fixture"),
    ]
}

#[derive(Clone, Debug)]
pub struct NetworkScenario {
    fixture: NetworkFixture,
    current_plan: ContractionPlan,
    selected_plan: ContractionPlan,
}

pub fn network_scenarios() -> Vec<NetworkScenario> {
    network_fixtures()
        .into_iter()
        .map(|fixture| {
            let current_plan = plan_output_greedy(&fixture.model, CostWeights::CPU)
                .expect("bounded fixture has a greedy plan");
            let selected_plan = if selected_uses_exact(&fixture.model)
                .expect("bounded fixture has a planner selection")
            {
                plan_bounded_exact(&fixture.model, CostWeights::CPU)
                    .expect("selected bounded fixture has an exact plan")
            } else {
                current_plan.clone()
            };
            NetworkScenario {
                fixture,
                current_plan,
                selected_plan,
            }
        })
        .collect()
}

fn fixture_inputs(fixture: &NetworkFixture, device: &Device) -> Result<Vec<Tensor>> {
    fixture
        .model
        .operands
        .iter()
        .map(|operand| {
            let shape = operand
                .axes
                .iter()
                .map(|axis| axis.extent)
                .collect::<Vec<_>>();
            let elements = usize::try_from(operand.elements()?)?;
            let values = (0..elements)
                .map(|index| 0.5 + (index % 13) as f32 * 0.001)
                .collect::<Vec<_>>();
            if operand.layout == LayoutClass::Transposed {
                let mut storage_shape = shape.clone();
                let last = storage_shape.len() - 1;
                storage_shape.swap(last - 1, last);
                Tensor::from_vec(values, storage_shape, device)?.transpose(last - 1, last)
            } else {
                Tensor::from_vec(values, shape, device)
            }
        })
        .collect()
}

fn execute_spec(
    tensors: &[&Tensor],
    inputs: &[Vec<&'static str>],
    output: &[&'static str],
) -> Result<Tensor> {
    let patterns = inputs
        .iter()
        .map(|axes| EinsumAxisPattern::new(axes, None))
        .collect::<Vec<_>>();
    execute_nary_einsum(
        tensors,
        EllipsisEinsumSpec::new(&patterns, EinsumAxisPattern::new(output, None)),
    )
}

fn execute_plan(
    inputs: &[Tensor],
    fixture: &NetworkFixture,
    plan: &ContractionPlan,
) -> Result<Tensor> {
    let mut operands = inputs
        .iter()
        .zip(&fixture.model.operands)
        .map(|(tensor, operand)| {
            (
                tensor.clone(),
                operand
                    .axes
                    .iter()
                    .map(|axis| axis.label)
                    .collect::<Vec<_>>(),
                operand.members,
            )
        })
        .collect::<Vec<_>>();
    for step in &plan.steps {
        let left = operands
            .iter()
            .position(|operand| operand.2 == step.members.0)
            .expect("plan left membership remains live");
        let right = operands
            .iter()
            .position(|operand| operand.2 == step.members.1)
            .expect("plan right membership remains live");
        let (left, right) = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        let right_operand = operands.remove(right);
        let left_operand = operands.remove(left);
        let output = execute_spec(
            &[&left_operand.0, &right_operand.0],
            &[left_operand.1, right_operand.1],
            &step.output_axes,
        )?;
        operands.insert(
            left,
            (
                output,
                step.output_axes.clone(),
                step.members.0 | step.members.1,
            ),
        );
    }
    let final_operand = operands.pop().expect("plan retains one operand");
    execute_spec(
        &[&final_operand.0],
        &[final_operand.1],
        &fixture.model.final_output,
    )
}

impl Scenario for NetworkScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new(self.fixture.id)
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let input_elements = self
            .fixture
            .model
            .operands
            .iter()
            .map(|operand| operand.elements().expect("bounded input elements"))
            .sum::<u128>();
        WorkUnits::new(
            u64::try_from(self.selected_plan.metrics.output_elements)
                .expect("bounded output elements"),
            u64::try_from((input_elements + self.selected_plan.metrics.output_elements) * 4)
                .expect("bounded workload bytes"),
            Some(u64::try_from(self.selected_plan.metrics.flops).expect("bounded FLOPs")),
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        fixture_inputs(&self.fixture, device)
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let tensors = inputs.iter().collect::<Vec<_>>();
        let input_axes = self
            .fixture
            .model
            .operands
            .iter()
            .map(|operand| operand.axes.iter().map(|axis| axis.label).collect())
            .collect::<Vec<Vec<_>>>();
        execute_spec(&tensors, &input_axes, &self.fixture.model.final_output)
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        execute_plan(inputs, &self.fixture, &self.current_plan)
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("n-ary cost-model paths produced different shapes")
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        for (index, (&library, &reference)) in library.iter().zip(&reference).enumerate() {
            let tolerance = 0.002 * reference.abs().max(1.);
            if (library - reference).abs() > tolerance {
                candle_core::bail!(
                    "n-ary cost-model paths differ at {index}: {library} vs {reference}"
                )
            }
        }
        Ok(())
    }
}

pub fn criterion_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in network_scenarios() {
        let prepared = prepare(&scenario, &device).expect("n-ary network setup");
        let id = scenario.id().as_str();
        criterion_operation(
            criterion,
            &format!("{id}/selected"),
            &prepared,
            Operation::Library,
            &synchronizer,
            "selected n-ary sample must succeed",
        );
        criterion_operation(
            criterion,
            &format!("{id}/current"),
            &prepared,
            Operation::Reference,
            &synchronizer,
            "current n-ary sample must succeed",
        );
    }
}
