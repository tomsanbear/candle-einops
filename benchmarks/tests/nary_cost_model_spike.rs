use candle_core::{Device, Result};
use candle_einops_benchmarks::Scenario;
use candle_einops_benchmarks::nary_cost_model_spike::{
    AxisExtent, CostWeights, FixtureKind, LayoutClass, ModelOperand, NetworkModel, estimate_pair,
    network_fixtures, network_scenarios, plan_bounded_exact, plan_output_greedy,
    selected_uses_exact,
};

#[test]
fn bounded_exact_planner_beats_output_greedy_on_frozen_counterexamples() -> Result<()> {
    let fixtures = network_fixtures();
    assert_eq!(fixtures.len(), 4);
    for fixture in fixtures {
        let greedy = plan_output_greedy(&fixture.model, CostWeights::CPU)?;
        let exact = plan_bounded_exact(&fixture.model, CostWeights::CPU)?;
        assert!(
            exact.metrics.score < greedy.metrics.score,
            "{} must remain a greedy counterexample: greedy {:?}, exact {:?}",
            fixture.id,
            greedy.metrics,
            exact.metrics
        );
        assert!(exact.metrics.flops <= greedy.metrics.flops);
        assert!(exact.metrics.peak_live_elements <= greedy.metrics.peak_live_elements);
    }
    Ok(())
}

#[test]
fn selected_hybrid_obeys_the_frozen_arity_and_work_budget() -> Result<()> {
    let fixtures = network_fixtures();
    assert!(!selected_uses_exact(&fixtures[0].model)?);
    assert!(selected_uses_exact(&fixtures[1].model)?);
    assert!(selected_uses_exact(&fixtures[2].model)?);
    assert!(!selected_uses_exact(&fixtures[3].model)?);

    for arity in 3..=6 {
        let model = NetworkModel::new(
            (0..arity)
                .map(|ordinal| {
                    ModelOperand::new(ordinal, &[AxisExtent::new("i", 4)], LayoutClass::Contiguous)
                })
                .collect(),
            &["i"],
        )?;
        assert_eq!(
            plan_bounded_exact(&model, CostWeights::CPU)?.steps.len(),
            arity - 1
        );
        if arity > 4 {
            assert!(!selected_uses_exact(&model)?);
        }
    }
    Ok(())
}

#[test]
fn exact_planner_is_bounded_and_breaks_ties_by_stable_membership() -> Result<()> {
    let equal = NetworkModel::new(
        vec![
            ModelOperand::new(0, &[AxisExtent::new("i", 4)], LayoutClass::Contiguous),
            ModelOperand::new(1, &[AxisExtent::new("i", 4)], LayoutClass::Contiguous),
            ModelOperand::new(2, &[AxisExtent::new("i", 4)], LayoutClass::Contiguous),
        ],
        &["i"],
    )?;
    let first = plan_bounded_exact(&equal, CostWeights::CPU)?;
    let second = plan_bounded_exact(&equal, CostWeights::CPU)?;
    assert_eq!(first, second);
    assert_eq!(first.steps[0].members, (1, 2));

    let too_small = NetworkModel::new(equal.operands()[..2].to_vec(), &["i"])?;
    assert!(plan_bounded_exact(&too_small, CostWeights::CPU).is_err());
    let too_large = NetworkModel::new(
        (0..7)
            .map(|ordinal| {
                ModelOperand::new(ordinal, &[AxisExtent::new("i", 4)], LayoutClass::Contiguous)
            })
            .collect(),
        &["i"],
    )?;
    assert!(plan_bounded_exact(&too_large, CostWeights::CPU).is_err());
    Ok(())
}

#[test]
fn model_checks_overflow_zero_k_and_broadcast_materialization() -> Result<()> {
    let overflow = NetworkModel::new(
        vec![
            ModelOperand::new(
                0,
                &[
                    AxisExtent::new("m", usize::MAX),
                    AxisExtent::new("k", usize::MAX),
                ],
                LayoutClass::Contiguous,
            ),
            ModelOperand::new(
                1,
                &[
                    AxisExtent::new("k", usize::MAX),
                    AxisExtent::new("n", usize::MAX),
                ],
                LayoutClass::Contiguous,
            ),
            ModelOperand::new(2, &[AxisExtent::new("n", 1)], LayoutClass::Contiguous),
        ],
        &["m"],
    )?;
    assert!(plan_bounded_exact(&overflow, CostWeights::CPU).is_err());

    let zero_k = NetworkModel::new(
        vec![
            ModelOperand::new(
                0,
                &[AxisExtent::new("m", 8), AxisExtent::new("k", 0)],
                LayoutClass::Contiguous,
            ),
            ModelOperand::new(
                1,
                &[AxisExtent::new("k", 0), AxisExtent::new("n", 8)],
                LayoutClass::Contiguous,
            ),
            ModelOperand::new(2, &[AxisExtent::new("n", 8)], LayoutClass::Contiguous),
        ],
        &["m"],
    )?;
    assert_eq!(estimate_pair(&zero_k, 0, 1)?.flops, 0);

    let broadcast = network_fixtures()
        .into_iter()
        .find(|fixture| fixture.kind == FixtureKind::BroadcastHeavy)
        .unwrap();
    let estimate = estimate_pair(&broadcast.model, 0, 1)?;
    assert_eq!(estimate.copy_bytes, 128 * 1024);
    assert_eq!(estimate.submissions, 1);
    Ok(())
}

#[test]
fn whole_network_scenarios_are_exactly_bounded_and_correct() -> Result<()> {
    let scenarios = network_scenarios();
    assert_eq!(
        scenarios
            .iter()
            .map(|scenario| scenario.id().as_str())
            .collect::<Vec<_>>(),
        vec![
            "spike/nary-cost/linear-chain",
            "spike/nary-cost/balanced-tree",
            "spike/nary-cost/broadcast-heavy",
            "spike/nary-cost/layout-hostile",
        ]
    );
    for scenario in scenarios {
        let inputs = scenario.setup(&Device::Cpu)?;
        let current = scenario.run_library(&inputs)?;
        let selected = scenario.run_reference(&inputs)?;
        scenario.check(&current, &selected)?;
    }
    Ok(())
}
