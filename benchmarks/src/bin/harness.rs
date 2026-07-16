use std::error::Error;
use std::fs;
use std::path::PathBuf;

use candle_einops_benchmarks::{
    Backend, BenchmarkRecord, CompiledFeatures, CpuImplementation, DeviceSynchronizer,
    ExecutionProfile, Fingerprint, MonotonicClock, PlumbingScenario, Scenario,
    binary_fast_path_scenarios, binary_operand_packing, broadcast_gemm_spike, diagonal_spike,
    extended_compose, extrema_spike, identity_reshape_scenarios, measure_pair,
    nary_cost_model_spike, permute_compose_layout_spike, prepare, product_scenarios,
    reduction_fusion_scenarios, repeat_broadcast_scenarios, zero_k_scenarios,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut filter: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut samples = 5;
    let mut include_plumbing = false;
    let mut backend = Backend::Cpu;
    let mut cpu_implementation = CpuImplementation::Baseline;
    let mut device_index = 0;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--filter" => filter = Some(arguments.next().ok_or("--filter requires a value")?),
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output requires a value")?,
                ));
            }
            "--samples" => {
                samples = arguments
                    .next()
                    .ok_or("--samples requires a value")?
                    .parse()?;
            }
            "--include-plumbing" => include_plumbing = true,
            "--backend" => {
                backend = match arguments.next().as_deref() {
                    Some("cpu") => Backend::Cpu,
                    Some("metal") => Backend::Metal,
                    Some("cuda") => Backend::Cuda,
                    _ => return Err("--backend requires cpu, metal, or cuda".into()),
                };
            }
            "--cpu-implementation" => {
                cpu_implementation = match arguments.next().as_deref() {
                    Some("baseline") => CpuImplementation::Baseline,
                    Some("accelerate") => CpuImplementation::Accelerate,
                    Some("mkl") => CpuImplementation::Mkl,
                    _ => {
                        return Err(
                            "--cpu-implementation requires baseline, accelerate, or mkl".into(),
                        );
                    }
                };
            }
            "--device-index" => {
                device_index = arguments
                    .next()
                    .ok_or("--device-index requires a value")?
                    .parse()?;
            }
            _ => return Err(format!("unknown benchmark argument: {argument}").into()),
        }
    }

    let plumbing = PlumbingScenario;
    let products = product_scenarios();
    let binary = binary_fast_path_scenarios();
    let binary_packing = binary_operand_packing::scenarios();
    let zero_k = zero_k_scenarios();
    let reductions = reduction_fusion_scenarios();
    let repeats = repeat_broadcast_scenarios();
    let identity_reshapes = identity_reshape_scenarios();
    let permute_compositions = permute_compose_layout_spike::scenarios();
    let extended_compositions = extended_compose::scenarios();
    let extrema = extrema_spike::scenarios();
    let nary_costs = nary_cost_model_spike::network_scenarios();
    let mut scenarios: Vec<&dyn Scenario> = if include_plumbing {
        vec![&plumbing]
    } else {
        products
            .iter()
            .map(|scenario| scenario as &dyn Scenario)
            .collect()
    };
    if !include_plumbing {
        scenarios.extend(
            diagonal_spike::scenarios()
                .iter()
                .map(|scenario| scenario as &dyn Scenario),
        );
        scenarios.extend(binary.iter().map(|scenario| scenario as &dyn Scenario));
        scenarios.extend(
            binary_packing
                .iter()
                .map(|scenario| scenario as &dyn Scenario),
        );
        scenarios.extend(zero_k.iter().map(|scenario| scenario as &dyn Scenario));
        scenarios.extend(reductions.iter().map(|scenario| scenario as &dyn Scenario));
        scenarios.extend(repeats.iter().map(|scenario| scenario as &dyn Scenario));
        scenarios.extend(
            identity_reshapes
                .iter()
                .map(|scenario| scenario as &dyn Scenario),
        );
        scenarios.extend(
            permute_compositions
                .iter()
                .map(|scenario| scenario as &dyn Scenario),
        );
        scenarios.extend(
            extended_compositions
                .iter()
                .map(|scenario| scenario as &dyn Scenario),
        );
        scenarios.extend(extrema.iter().map(|scenario| scenario as &dyn Scenario));
        scenarios.extend(
            broadcast_gemm_spike::broadcast_scenarios()
                .iter()
                .map(|scenario| scenario as &dyn Scenario),
        );
        scenarios.extend(nary_costs.iter().map(|scenario| scenario as &dyn Scenario));
    }
    let selected = scenarios
        .into_iter()
        .filter(|scenario| {
            filter
                .as_deref()
                .is_none_or(|value| scenario.id().as_str().contains(value))
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err("no benchmark scenarios matched".into());
    }

    let profile = ExecutionProfile::new(backend, cpu_implementation, device_index);
    let device = profile.create_device(CompiledFeatures::CURRENT)?;
    let synchronizer = DeviceSynchronizer(&device);
    let clock = MonotonicClock;
    let fingerprint = Fingerprint::collect_for(profile)?;
    let records = selected
        .into_iter()
        .map(|scenario| {
            let prepared = prepare(scenario, &device)?;
            let measurement = measure_pair(&prepared, &synchronizer, &clock, samples)?;
            Ok(BenchmarkRecord::from_measurement(
                &prepared,
                &measurement,
                fingerprint.clone(),
            )?)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let document = serde_json::to_string_pretty(&records)?;
    println!("{document}");
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, format!("{document}\n"))?;
    }
    Ok(())
}
