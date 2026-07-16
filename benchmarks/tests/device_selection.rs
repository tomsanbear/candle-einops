use candle_core::DeviceLocation;
use candle_einops_benchmarks::{
    Backend, CompiledFeatures, CpuImplementation, ExecutionProfile,
};

#[test]
fn execution_profiles_reject_feature_and_device_mismatches() {
    let baseline = ExecutionProfile::new(Backend::Cpu, CpuImplementation::Baseline, 0);
    assert!(baseline.validate(CompiledFeatures::NONE).is_ok());

    let indexed_cpu = ExecutionProfile::new(Backend::Cpu, CpuImplementation::Baseline, 1);
    assert!(indexed_cpu.validate(CompiledFeatures::NONE).is_err());

    let metal = ExecutionProfile::new(Backend::Metal, CpuImplementation::Baseline, 0);
    assert!(metal.validate(CompiledFeatures::NONE).is_err());

    let accelerated_cpu =
        ExecutionProfile::new(Backend::Cpu, CpuImplementation::Accelerate, 0);
    assert!(accelerated_cpu.validate(CompiledFeatures::NONE).is_err());

    let conflicting = CompiledFeatures {
        accelerate: true,
        mkl: false,
        metal: true,
        cuda: false,
    };
    assert!(metal.validate(conflicting).is_err());
}

#[test]
fn baseline_profile_constructs_the_requested_cpu_device() {
    let profile = ExecutionProfile::new(Backend::Cpu, CpuImplementation::Baseline, 0);
    let device = profile
        .create_device(CompiledFeatures::NONE)
        .expect("baseline CPU must be available");
    assert_eq!(device.location(), DeviceLocation::Cpu);
}
