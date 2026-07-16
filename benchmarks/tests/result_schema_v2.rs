use candle_einops_benchmarks::{
    Availability, Backend, BenchmarkDocument, CompiledFeatures, CpuImplementation,
    ExecutionProfile, RunMetadata, SkippedScenario,
};

#[test]
fn availability_requires_exactly_one_honest_state() {
    assert!(Availability::available(42_u64, "public API").validate().is_ok());
    assert!(
        Availability::<u64>::unavailable("external profiler required")
            .validate()
            .is_ok()
    );
    assert!(
        Availability {
            value: Some(42_u64),
            source: None,
            reason: Some("contradictory".to_owned()),
        }
        .validate()
        .is_err()
    );
}

#[test]
fn document_v2_owns_run_identity_records_and_explicit_skips() {
    let profile = ExecutionProfile::new(Backend::Cpu, CpuImplementation::Baseline, 0);
    let run = RunMetadata::collect(profile, CompiledFeatures::NONE).expect("run metadata");
    let document = BenchmarkDocument::new(
        run,
        Vec::new(),
        vec![SkippedScenario::new(
            "layout/view-only",
            "view-only scenarios do not enqueue accelerator work",
        )],
    )
    .expect("valid document");
    let value = serde_json::to_value(document).expect("serialize document");
    assert_eq!(value["schema_version"], 2);
    assert_eq!(value["run"]["backend"], "cpu");
    assert_eq!(value["run"]["cpu_implementation"], "baseline");
    assert_eq!(value["run"]["synchronization"], "candle_device_synchronize");
    assert!(value["records"].is_array());
    assert_eq!(value["skipped"][0]["scenario_id"], "layout/view-only");
}
