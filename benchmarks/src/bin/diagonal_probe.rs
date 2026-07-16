use std::error::Error;
use std::fs;
use std::path::PathBuf;

use candle_core::Device;
use candle_einops_benchmarks::{
    DeviceSynchronizer, Fingerprint, MonotonicClock, Scenario, diagonal_spike,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut filter: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut samples = 101;
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
            _ => return Err(format!("unknown probe argument: {argument}").into()),
        }
    }

    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    let clock = MonotonicClock;
    let fingerprint = Fingerprint::collect_cpu()?;
    let records = diagonal_spike::scenarios()
        .iter()
        .filter(|scenario| {
            filter
                .as_deref()
                .is_none_or(|value| scenario.id().as_str().contains(value))
        })
        .map(|scenario| {
            diagonal_spike::measure_index_preparation(
                scenario,
                &device,
                &synchronizer,
                &clock,
                samples,
                fingerprint.clone(),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    if records.is_empty() {
        return Err("no diagonal probe scenarios matched".into());
    }
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
