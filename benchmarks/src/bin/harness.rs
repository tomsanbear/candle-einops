use std::error::Error;
use std::fs;
use std::path::PathBuf;

use candle_einops_benchmarks::{
    Backend, BenchmarkDocument, BenchmarkRecord, CompiledFeatures, CpuImplementation,
    DeviceDiagnostics, DeviceMemorySnapshot, DeviceSynchronizer, ExecutionProfile, MonotonicClock,
    PlumbingScenario, RunMetadata, Scenario,
    binary_fast_path_scenarios, binary_operand_packing, broadcast_gemm_spike,
    capture_operation as capture_one_operation, diagonal_spike, extended_compose, extrema_spike,
    identity_reshape_scenarios, measure_pair,
    nary_cost_model_spike, partition_scenarios, permute_compose_layout_spike, prepare,
    product_scenarios, reduction_fusion_scenarios, repeat_broadcast_scenarios, zero_k_scenarios,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut filter: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut samples = 5;
    let mut include_plumbing = false;
    let mut backend = Backend::Cpu;
    let mut cpu_implementation = CpuImplementation::Baseline;
    let mut device_index = 0;
    let mut capture_operation = None;
    let mut capture_output = None;
    let mut capture_warmups = 3;
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
            "--capture-operation" => {
                capture_operation = match arguments.next().as_deref() {
                    Some("library") => Some(candle_einops_benchmarks::Operation::Library),
                    Some("reference") => Some(candle_einops_benchmarks::Operation::Reference),
                    _ => return Err("--capture-operation requires library or reference".into()),
                };
            }
            "--capture-output" => {
                capture_output = Some(PathBuf::from(
                    arguments.next().ok_or("--capture-output requires a value")?,
                ));
            }
            "--capture-warmups" => {
                capture_warmups = arguments
                    .next()
                    .ok_or("--capture-warmups requires a value")?
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
    let (selected, skipped) = partition_scenarios(selected, profile.backend);
    if selected.is_empty() {
        let reasons = skipped
            .iter()
            .map(|scenario| format!("{}: {}", scenario.scenario_id, scenario.reason))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!("no supported benchmark scenarios matched ({reasons})").into());
    }
    let device = profile.create_device(CompiledFeatures::CURRENT)?;
    if let Some(operation) = capture_operation {
        if selected.len() != 1 {
            return Err(format!(
                "capture filter must match exactly one supported scenario, matched {}",
                selected.len()
            )
            .into());
        }
        let prepared = prepare(selected[0], &device)?;
        capture_one_operation(
            &prepared,
            operation,
            &device,
            profile.backend,
            capture_output.as_deref(),
            capture_warmups,
        )?;
        println!(
            "captured {} operation for {}",
            match operation {
                candle_einops_benchmarks::Operation::Library => "library",
                candle_einops_benchmarks::Operation::Reference => "reference",
            },
            selected[0].id().as_str()
        );
        return Ok(());
    }
    if capture_output.is_some() {
        return Err("--capture-output requires --capture-operation".into());
    }
    let synchronizer = DeviceSynchronizer(&device);
    let clock = MonotonicClock;
    let run = RunMetadata::collect_for_device(profile, CompiledFeatures::CURRENT, &device)?;
    let records = selected
        .into_iter()
        .map(|scenario| {
            let prepared = prepare(scenario, &device)?;
            let before = DeviceMemorySnapshot::collect(&device, profile.backend);
            let measurement = measure_pair(&prepared, &synchronizer, &clock, samples)?;
            let after = DeviceMemorySnapshot::collect(&device, profile.backend);
            let diagnostics = DeviceDiagnostics::from_snapshots(before, after);
            Ok(BenchmarkRecord::from_measurement_with_diagnostics(
                &prepared,
                &measurement,
                diagnostics,
            )?)
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let document = BenchmarkDocument::new(run, records, skipped)?;
    let document = serde_json::to_string_pretty(&document)?;
    println!("{document}");
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, format!("{document}\n"))?;
    }
    Ok(())
}
