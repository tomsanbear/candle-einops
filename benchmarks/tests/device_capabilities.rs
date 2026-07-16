use candle_einops_benchmarks::{
    Backend, Scenario, binary_operand_packing, extended_compose, identity_reshape_scenarios,
    partition_scenarios, permute_compose_layout_spike, repeat_broadcast_scenarios,
};

#[test]
fn accelerator_partition_reports_every_view_only_scenario() {
    let binary = binary_operand_packing::scenarios();
    let repeats = repeat_broadcast_scenarios();
    let identities = identity_reshape_scenarios();
    let permutations = permute_compose_layout_spike::scenarios();
    let extended = extended_compose::scenarios();
    let scenarios = binary
        .iter()
        .map(|scenario| scenario as &dyn Scenario)
        .chain(repeats.iter().map(|scenario| scenario as &dyn Scenario))
        .chain(identities.iter().map(|scenario| scenario as &dyn Scenario))
        .chain(permutations.iter().map(|scenario| scenario as &dyn Scenario))
        .chain(extended.iter().map(|scenario| scenario as &dyn Scenario));

    let (supported, skipped) = partition_scenarios(scenarios, Backend::Metal);
    assert!(supported.iter().all(|scenario| {
        !scenario.id().as_str().ends_with("/construct")
            || scenario.id().as_str().contains("post-reduction")
    }));
    assert_eq!(
        skipped
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect::<Vec<_>>(),
        [
            "einsum/binary-packing/recovered-view/construct",
            "repeat/broadcast/single-axis/construct",
            "repeat/broadcast/two-axis/construct",
            "reshape/identity/contiguous/construct",
            "reshape/identity/non-contiguous/construct",
            "layout/permute-compose/c-ab/construct",
            "layout/permute-compose/n-hw-c/construct",
            "layout/extended-compose/runtime-ellipsis/construct",
        ]
    );
    assert!(skipped.iter().all(|scenario| !scenario.reason.is_empty()));
}

#[test]
fn cpu_partition_keeps_view_and_materializing_scenarios() {
    let identities = identity_reshape_scenarios();
    let scenarios = identities
        .iter()
        .map(|scenario| scenario as &dyn Scenario);
    let (supported, skipped) = partition_scenarios(scenarios, Backend::Cpu);
    assert_eq!(supported.len(), 4);
    assert!(skipped.is_empty());
}
