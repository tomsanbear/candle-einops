use candle_core::Device;
use candle_einops_benchmarks::{
    Availability, Backend, CompiledFeatures, CpuImplementation, DeviceDiagnostics,
    DeviceMemorySnapshot, ExecutionProfile, RunMetadata,
};

#[test]
fn cpu_diagnostics_are_explicitly_unavailable() {
    let before = DeviceMemorySnapshot::collect(&Device::Cpu, Backend::Cpu);
    let after = DeviceMemorySnapshot::collect(&Device::Cpu, Backend::Cpu);
    let diagnostics = DeviceDiagnostics::from_snapshots(before, after);

    diagnostics.validate().unwrap();
    for metric in [
        &diagnostics.allocated_bytes_before,
        &diagnostics.allocated_bytes_after,
        &diagnostics.free_bytes_before,
        &diagnostics.free_bytes_after,
        &diagnostics.total_bytes,
        &diagnostics.device_elapsed_ns,
        &diagnostics.kernel_count,
        &diagnostics.command_buffer_count,
        &diagnostics.enqueue_count,
    ] {
        assert!(matches!(metric, Availability { value: None, source: None, reason: Some(reason) } if !reason.is_empty()));
    }
}

#[test]
fn run_metadata_is_collected_from_the_created_device() {
    let profile = ExecutionProfile::new(Backend::Cpu, CpuImplementation::Baseline, 0);
    let metadata = RunMetadata::collect_for_device(
        profile,
        CompiledFeatures::NONE,
        &Device::Cpu,
    )
    .unwrap();

    assert_eq!(metadata.device_name, "CPU (baseline)");
    assert!(metadata.device_identity.value.is_none());
    metadata.validate().unwrap();
}

#[cfg(feature = "metal")]
#[test]
fn metal_reports_physical_identity_and_allocated_bytes() {
    let profile = ExecutionProfile::new(Backend::Metal, CpuImplementation::Baseline, 0);
    let device = profile.create_device(CompiledFeatures::CURRENT).unwrap();
    let metadata = RunMetadata::collect_for_device(profile, CompiledFeatures::CURRENT, &device)
        .unwrap();
    let snapshot = DeviceMemorySnapshot::collect(&device, Backend::Metal);

    assert!(!metadata.device_name.starts_with("Metal device"));
    assert!(metadata.device_identity.value.is_some());
    assert!(snapshot.allocated_bytes.value.is_some());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_reports_physical_identity_versions_and_memory() {
    let profile = ExecutionProfile::new(Backend::Cuda, CpuImplementation::Baseline, 0);
    let device = profile.create_device(CompiledFeatures::CURRENT).unwrap();
    let metadata = RunMetadata::collect_for_device(profile, CompiledFeatures::CURRENT, &device)
        .unwrap();
    let snapshot = DeviceMemorySnapshot::collect(&device, Backend::Cuda);

    assert!(!metadata.device_name.starts_with("CUDA device"));
    assert!(metadata.device_identity.value.is_some());
    assert!(metadata.driver_version.value.is_some());
    assert!(metadata.runtime_version.value.is_some());
    assert!(snapshot.free_bytes.value.is_some());
    assert!(snapshot.total_bytes.value.is_some());
}
