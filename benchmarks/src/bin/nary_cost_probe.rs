use std::error::Error;
use std::fs;
use std::path::PathBuf;

use candle_einops_benchmarks::Fingerprint;
use candle_einops_benchmarks::nary_cost_model_spike::{measure_fixture_planners, network_fixtures};

fn main() -> Result<(), Box<dyn Error>> {
    let mut output: Option<PathBuf> = None;
    let mut samples = 1_001;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
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
            _ => return Err(format!("unknown planner probe argument: {argument}").into()),
        }
    }

    let fingerprint = Fingerprint::collect_cpu()?;
    let records = network_fixtures()
        .iter()
        .map(|fixture| measure_fixture_planners(fixture, samples, fingerprint.clone()))
        .collect::<Result<Vec<_>, _>>()?;
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
