//! Shared measurement contract for the repository's isolated benchmark suite.

use std::fmt;
use std::hint::black_box;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::{Device, Result, Tensor};
use criterion::Criterion;
use serde::{Deserialize, Serialize};

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("benchmark backends `cuda` and `metal` are mutually exclusive");

pub const RESULT_SCHEMA_VERSION: u32 = 1;
pub const CANDLE_VERSION: &str = "0.11.0";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ScenarioId(&'static str);

impl ScenarioId {
    #[must_use]
    pub const fn new(value: &'static str) -> Self {
        assert!(!value.is_empty(), "scenario ids must not be empty");
        Self(value)
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkUnits {
    pub elements: u64,
    pub bytes: u64,
    pub flops: Option<u64>,
}

impl WorkUnits {
    #[must_use]
    pub const fn new(elements: u64, bytes: u64, flops: Option<u64>) -> Self {
        Self {
            elements,
            bytes,
            flops,
        }
    }

    fn validate(self) -> std::result::Result<(), ValidationError> {
        if self.elements == 0 || self.bytes == 0 {
            return Err(ValidationError::new(
                "work units require non-zero elements and bytes",
            ));
        }
        Ok(())
    }
}

pub trait Scenario {
    fn id(&self) -> ScenarioId;
    fn tracked(&self) -> bool;
    fn work(&self) -> WorkUnits;
    fn setup(&self, device: &Device) -> Result<Vec<Tensor>>;
    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor>;
    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor>;
    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()>;
}

pub struct PreparedScenario<'a> {
    scenario: &'a dyn Scenario,
    inputs: Vec<Tensor>,
}

impl PreparedScenario<'_> {
    #[must_use]
    pub fn id(&self) -> ScenarioId {
        self.scenario.id()
    }

    #[must_use]
    pub fn tracked(&self) -> bool {
        self.scenario.tracked()
    }

    #[must_use]
    pub fn work(&self) -> WorkUnits {
        self.scenario.work()
    }
}

pub fn prepare<'a>(scenario: &'a dyn Scenario, device: &Device) -> Result<PreparedScenario<'a>> {
    let inputs = scenario.setup(device)?;
    let library = scenario.run_library(&inputs)?;
    let reference = scenario.run_reference(&inputs)?;
    scenario.check(&library, &reference)?;
    Ok(PreparedScenario { scenario, inputs })
}

pub trait Synchronizer {
    fn synchronize(&self) -> Result<()>;
}

pub struct DeviceSynchronizer<'a>(pub &'a Device);

impl Synchronizer for DeviceSynchronizer<'_> {
    fn synchronize(&self) -> Result<()> {
        self.0.synchronize()
    }
}

pub trait Clock {
    fn now_ns(&self) -> u64;
}

pub struct MonotonicClock;

impl Clock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        let nanos = ORIGIN.get_or_init(Instant::now).elapsed().as_nanos();
        u64::try_from(nanos).unwrap_or(u64::MAX)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Library,
    Reference,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub confidence_level: f64,
    pub lower_bound_ns: f64,
    pub upper_bound_ns: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    pub median_ns: f64,
    pub confidence_interval: ConfidenceInterval,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SampleSet {
    pub samples_ns: Vec<u64>,
    pub estimate: Estimate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairMeasurement {
    pub library: SampleSet,
    pub reference: SampleSet,
    pub library_to_reference_ratio: f64,
    pub order: [Operation; 2],
}

pub fn measure_pair(
    prepared: &PreparedScenario<'_>,
    synchronizer: &dyn Synchronizer,
    clock: &dyn Clock,
    sample_count: usize,
) -> Result<PairMeasurement> {
    if sample_count == 0 {
        candle_core::bail!("benchmark sample count must be non-zero")
    }
    let mut library = Vec::with_capacity(sample_count);
    let mut reference = Vec::with_capacity(sample_count);
    for _ in 0..sample_count {
        library.push(measure_operation(
            prepared,
            Operation::Library,
            synchronizer,
            clock,
        )?);
        reference.push(measure_operation(
            prepared,
            Operation::Reference,
            synchronizer,
            clock,
        )?);
    }
    let library_estimate = summarize(&library);
    let reference_estimate = summarize(&reference);
    let ratio = library_estimate.median_ns / reference_estimate.median_ns;
    Ok(PairMeasurement {
        library: SampleSet {
            samples_ns: library,
            estimate: library_estimate,
        },
        reference: SampleSet {
            samples_ns: reference,
            estimate: reference_estimate,
        },
        library_to_reference_ratio: ratio,
        order: [Operation::Library, Operation::Reference],
    })
}

fn measure_operation(
    prepared: &PreparedScenario<'_>,
    operation: Operation,
    synchronizer: &dyn Synchronizer,
    clock: &dyn Clock,
) -> Result<u64> {
    synchronizer.synchronize()?;
    let started = clock.now_ns();
    let output = match operation {
        Operation::Library => prepared.scenario.run_library(&prepared.inputs)?,
        Operation::Reference => prepared.scenario.run_reference(&prepared.inputs)?,
    };
    black_box(&output);
    synchronizer.synchronize()?;
    let finished = clock.now_ns();
    black_box(output);
    Ok(finished.saturating_sub(started))
}

fn summarize(samples: &[u64]) -> Estimate {
    let median_ns = median(samples);
    let mut bootstrap = Vec::with_capacity(1_000);
    let mut state = 0x6840_1587_6003_2202_u64;
    for _ in 0..1_000 {
        let mut resample = Vec::with_capacity(samples.len());
        for _ in samples {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            resample.push(samples[(state as usize) % samples.len()]);
        }
        bootstrap.push(median(&resample));
    }
    bootstrap.sort_by(f64::total_cmp);
    Estimate {
        median_ns,
        confidence_interval: ConfidenceInterval {
            confidence_level: 0.95,
            lower_bound_ns: bootstrap[24],
            upper_bound_ns: bootstrap[974],
        },
    }
}

fn median(samples: &[u64]) -> f64 {
    let mut values = samples.to_vec();
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] as f64 + values[middle] as f64) / 2.0
    } else {
        values[middle] as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Fingerprint {
    pub git_sha: String,
    pub rust_version: String,
    pub candle_version: String,
    pub os: String,
    pub architecture: String,
    pub backend: Backend,
    pub device: String,
    pub driver: Option<String>,
}

impl Fingerprint {
    pub fn collect_cpu() -> std::result::Result<Self, ValidationError> {
        let fingerprint = Self {
            git_sha: command_output("git", &["rev-parse", "HEAD"])?,
            rust_version: command_output(
                std::env::var("RUSTC").as_deref().unwrap_or("rustc"),
                &["--version"],
            )?,
            candle_version: CANDLE_VERSION.to_owned(),
            os: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            backend: Backend::Cpu,
            device: "cpu".to_owned(),
            driver: None,
        };
        fingerprint.validate()?;
        Ok(fingerprint)
    }

    pub fn validate(&self) -> std::result::Result<(), ValidationError> {
        if self.git_sha.len() != 40 || !self.git_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(ValidationError::new(
                "git_sha must be a full 40-character SHA",
            ));
        }
        for (name, value) in [
            ("rust_version", self.rust_version.as_str()),
            ("candle_version", self.candle_version.as_str()),
            ("os", self.os.as_str()),
            ("architecture", self.architecture.as_str()),
            ("device", self.device.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ValidationError::new(format!("{name} must not be empty")));
            }
        }
        if self.backend != Backend::Cpu && self.driver.as_deref().is_none_or(str::is_empty) {
            return Err(ValidationError::new(
                "accelerator fingerprints require a driver identity",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    pub schema_version: u32,
    pub scenario_id: String,
    pub tracked: bool,
    pub workload: WorkUnits,
    pub sample_count: usize,
    pub library: Estimate,
    pub reference: Estimate,
    pub library_to_reference_ratio: f64,
    pub fingerprint: Fingerprint,
}

impl BenchmarkRecord {
    pub fn from_measurement(
        prepared: &PreparedScenario<'_>,
        measurement: &PairMeasurement,
        fingerprint: Fingerprint,
    ) -> std::result::Result<Self, ValidationError> {
        let record = Self {
            schema_version: RESULT_SCHEMA_VERSION,
            scenario_id: prepared.id().as_str().to_owned(),
            tracked: prepared.tracked(),
            workload: prepared.work(),
            sample_count: measurement.library.samples_ns.len(),
            library: measurement.library.estimate.clone(),
            reference: measurement.reference.estimate.clone(),
            library_to_reference_ratio: measurement.library_to_reference_ratio,
            fingerprint,
        };
        record.validate()?;
        Ok(record)
    }

    pub fn validate(&self) -> std::result::Result<(), ValidationError> {
        if self.schema_version != RESULT_SCHEMA_VERSION {
            return Err(ValidationError::new("unsupported benchmark result schema"));
        }
        if self.scenario_id.trim().is_empty() || self.sample_count == 0 {
            return Err(ValidationError::new(
                "scenario_id and sample_count must be non-empty",
            ));
        }
        self.workload.validate()?;
        self.fingerprint.validate()?;
        validate_estimate("library", &self.library)?;
        validate_estimate("reference", &self.reference)?;
        if !self.library_to_reference_ratio.is_finite() || self.library_to_reference_ratio <= 0.0 {
            return Err(ValidationError::new(
                "library_to_reference_ratio must be finite and positive",
            ));
        }
        Ok(())
    }
}

fn validate_estimate(name: &str, estimate: &Estimate) -> std::result::Result<(), ValidationError> {
    let interval = &estimate.confidence_interval;
    if !estimate.median_ns.is_finite()
        || estimate.median_ns <= 0.0
        || interval.confidence_level != 0.95
        || !interval.lower_bound_ns.is_finite()
        || !interval.upper_bound_ns.is_finite()
        || interval.lower_bound_ns > estimate.median_ns
        || estimate.median_ns > interval.upper_bound_ns
    {
        return Err(ValidationError::new(format!(
            "{name} estimate has invalid median or confidence interval"
        )));
    }
    Ok(())
}

fn command_output(program: &str, args: &[&str]) -> std::result::Result<String, ValidationError> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| ValidationError::new(format!("could not execute {program}: {error}")))?;
    if !output.status.success() {
        return Err(ValidationError::new(format!(
            "{program} failed while collecting benchmark metadata"
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|error| ValidationError::new(format!("invalid {program} output: {error}")))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError(String);

impl ValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ValidationError {}

pub struct PlumbingScenario;

impl Scenario for PlumbingScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new("plumbing/smoke")
    }

    fn tracked(&self) -> bool {
        false
    }

    fn work(&self) -> WorkUnits {
        WorkUnits::new(4, 16, None)
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        Ok(vec![Tensor::new(&[1f32, 2., 3., 4.], device)?])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].reshape((2, 2))
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        inputs[0].reshape((2, 2))
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("plumbing outputs have different shapes")
        }
        Ok(())
    }
}

pub fn criterion_plumbing_benchmark(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let scenario = PlumbingScenario;
    let prepared = prepare(&scenario, &device).expect("plumbing setup must succeed");
    let synchronizer = DeviceSynchronizer(&device);
    let clock = MonotonicClock;
    criterion.bench_function("plumbing/smoke-untracked", |bencher| {
        bencher.iter(|| {
            measure_operation(&prepared, Operation::Library, &synchronizer, &clock)
                .expect("plumbing sample must succeed")
        });
    });
}
