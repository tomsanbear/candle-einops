use std::error::Error;
use std::fs;
use std::path::PathBuf;

use candle_core::Device;
use candle_einops_benchmarks::{
    BenchmarkRecord, DeviceSynchronizer, Fingerprint, MonotonicClock, PlumbingScenario, Scenario,
    diagonal_spike, measure_pair, prepare, product_scenarios,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut filter: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut samples = 5;
    let mut include_plumbing = false;
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
            _ => return Err(format!("unknown benchmark argument: {argument}").into()),
        }
    }

    let plumbing = PlumbingScenario;
    let products = product_scenarios();
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

    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    let clock = MonotonicClock;
    let fingerprint = Fingerprint::collect_cpu()?;
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
