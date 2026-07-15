//! Shared measurement contract for the repository's isolated benchmark suite.

use std::fmt;
use std::hint::black_box;
use std::mem::size_of;
use std::process::Command;
use std::sync::OnceLock;
use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};
use candle_einops::{einops, einsum};
use criterion::Criterion;
use serde::{Deserialize, Serialize};

pub mod binary_operand_packing;
pub mod broadcast_gemm_spike;
pub mod diagonal_spike;
pub mod extended_compose;
pub mod extrema_spike;
pub mod nary_cost_model_spike;
pub mod permute_compose_layout_spike;

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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingOrderPolicy {
    #[default]
    FixedLibraryThenReference,
    AlternatingLibraryThenReference,
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
    pub order_policy: SamplingOrderPolicy,
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
    for sample in 0..sample_count {
        let order = if sample.is_multiple_of(2) {
            [Operation::Library, Operation::Reference]
        } else {
            [Operation::Reference, Operation::Library]
        };
        for operation in order {
            let duration = measure_operation(prepared, operation, synchronizer, clock)?;
            match operation {
                Operation::Library => library.push(duration),
                Operation::Reference => reference.push(duration),
            }
        }
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
        order_policy: SamplingOrderPolicy::AlternatingLibraryThenReference,
    })
}

fn run_operation(prepared: &PreparedScenario<'_>, operation: Operation) -> Result<Tensor> {
    match operation {
        Operation::Library => prepared.scenario.run_library(&prepared.inputs),
        Operation::Reference => prepared.scenario.run_reference(&prepared.inputs),
    }
}

/// Executes one Criterion operation and waits for asynchronous device work while
/// keeping the result alive. Criterion owns the surrounding timing interval.
pub fn run_synchronized_operation(
    prepared: &PreparedScenario<'_>,
    operation: Operation,
    synchronizer: &dyn Synchronizer,
) -> Result<Tensor> {
    let output = run_operation(prepared, operation)?;
    black_box(&output);
    synchronizer.synchronize()?;
    Ok(black_box(output))
}

fn measure_operation(
    prepared: &PreparedScenario<'_>,
    operation: Operation,
    synchronizer: &dyn Synchronizer,
    clock: &dyn Clock,
) -> Result<u64> {
    synchronizer.synchronize()?;
    let started = clock.now_ns();
    let output = run_operation(prepared, operation)?;
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
    #[serde(default)]
    pub sampling_order_policy: SamplingOrderPolicy,
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
            sampling_order_policy: measurement.order_policy,
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

/// Portable spike candidate which reduces one axis with a balanced tree of
/// Candle multiplication operations. This is benchmark-only and deliberately
/// does not claim to be a native reduction.
pub fn balanced_product_axis(input: &Tensor, axis: usize) -> Result<Tensor> {
    let axis_len = input.dim(axis)?;
    if axis_len == 0 {
        let mut shape = input.dims().to_vec();
        shape.remove(axis);
        return Tensor::ones(shape, input.dtype(), input.device());
    }
    let mut factors = (0..axis_len)
        .map(|index| input.narrow(axis, index, 1)?.squeeze(axis))
        .collect::<Result<Vec<_>>>()?;
    while factors.len() > 1 {
        let mut products = Vec::with_capacity(factors.len().div_ceil(2));
        let mut factor_iter = factors.into_iter();
        while let Some(left) = factor_iter.next() {
            products.push(match factor_iter.next() {
                Some(right) => left.mul(&right)?,
                None => left,
            });
        }
        factors = products;
    }
    Ok(factors.pop().expect("non-empty product factors"))
}

#[derive(Clone, Copy, Debug)]
pub struct ProductScenario {
    id: ScenarioId,
    factors: usize,
    two_axes: bool,
}

impl ProductScenario {
    const fn one_axis(id: &'static str, factors: usize) -> Self {
        Self {
            id: ScenarioId::new(id),
            factors,
            two_axes: false,
        }
    }

    const fn two_axes(id: &'static str, factors: usize) -> Self {
        Self {
            id: ScenarioId::new(id),
            factors,
            two_axes: true,
        }
    }
}

pub(crate) fn deterministic_f32_values(elements: usize, stream: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            let value = index
                .wrapping_mul(37)
                .wrapping_add(stream.wrapping_mul(101))
                % 251;
            (value as f32 - 125.) / 64.
        })
        .collect()
}

pub fn product_scenarios() -> [ProductScenario; 4] {
    [
        ProductScenario::one_axis("product/sequential-vs-balanced/k-8", 8),
        ProductScenario::one_axis("product/sequential-vs-balanced/k-64", 64),
        ProductScenario::one_axis("product/sequential-vs-balanced/k-512", 512),
        ProductScenario::two_axes("product/sequential-vs-balanced/two-axis-8x8", 64),
    ]
}

impl ProductScenario {
    fn candidate(&self, input: &Tensor) -> Result<Tensor> {
        if self.two_axes {
            balanced_product_axis(&input.reshape((256, self.factors))?, 1)
        } else {
            balanced_product_axis(input, 1)
        }
    }
}

impl Scenario for ProductScenario {
    fn id(&self) -> ScenarioId {
        self.id
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let elements = 256 * self.factors;
        WorkUnits::new(
            elements as u64,
            (elements * std::mem::size_of::<f32>()) as u64,
            Some((256 * self.factors.saturating_sub(1)) as u64),
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let values = (0..256 * self.factors)
            .map(|index| 1. + ((index % 7) as f32 - 3.) * 0.0001)
            .collect::<Vec<_>>();
        let input = if self.two_axes {
            Tensor::from_vec(values, (256, 8, 8), device)?
        } else {
            Tensor::from_vec(values, (256, self.factors), device)?
        };
        Ok(vec![input])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        if self.two_axes {
            einops!("rows prod(left right) -> rows", &inputs[0])
        } else {
            einops!("rows prod(columns) -> rows", &inputs[0])
        }
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.candidate(&inputs[0])
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!(
                "product outputs have different shapes: {:?} and {:?}",
                library.dims(),
                reference.dims()
            )
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        let tolerance = f32::EPSILON * self.factors as f32 * 8.;
        for (index, (&library, &reference)) in library.iter().zip(&reference).enumerate() {
            let allowed = tolerance * reference.abs().max(1.);
            if (library - reference).abs() > allowed {
                candle_core::bail!(
                    "product output {index} differs: {library} vs {reference}, tolerance {allowed}"
                )
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum BinaryMechanism {
    Hadamard,
    Outer,
    Rank2Matmul,
    Rank3Matmul,
}

pub struct BinaryFastPathScenario {
    id: &'static str,
    mechanism: BinaryMechanism,
    dimensions: [usize; 4],
}

#[must_use]
pub fn binary_fast_path_scenarios() -> Vec<BinaryFastPathScenario> {
    use BinaryMechanism::{Hadamard, Outer, Rank2Matmul, Rank3Matmul};
    vec![
        BinaryFastPathScenario {
            id: "einsum/binary/hadamard-overhead",
            mechanism: Hadamard,
            dimensions: [1, 32, 1, 1],
        },
        BinaryFastPathScenario {
            id: "einsum/binary/hadamard-throughput",
            mechanism: Hadamard,
            dimensions: [1, 262_144, 1, 1],
        },
        BinaryFastPathScenario {
            id: "einsum/binary/outer-overhead",
            mechanism: Outer,
            dimensions: [1, 7, 1, 9],
        },
        BinaryFastPathScenario {
            id: "einsum/binary/outer-throughput",
            mechanism: Outer,
            dimensions: [1, 480, 1, 544],
        },
        BinaryFastPathScenario {
            id: "einsum/binary/rank2-matmul-overhead",
            mechanism: Rank2Matmul,
            dimensions: [1, 7, 8, 9],
        },
        BinaryFastPathScenario {
            id: "einsum/binary/rank2-matmul-throughput",
            mechanism: Rank2Matmul,
            dimensions: [1, 120, 128, 136],
        },
        BinaryFastPathScenario {
            id: "einsum/binary/rank3-matmul-overhead",
            mechanism: Rank3Matmul,
            dimensions: [2, 7, 8, 9],
        },
        BinaryFastPathScenario {
            id: "einsum/binary/rank3-matmul-throughput",
            mechanism: Rank3Matmul,
            dimensions: [8, 56, 64, 72],
        },
    ]
}

impl Scenario for BinaryFastPathScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new(self.id)
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let [b, m, k, n] = self.dimensions;
        let (inputs, output, flops) = match self.mechanism {
            BinaryMechanism::Hadamard => (m * 2, m, m),
            BinaryMechanism::Outer => (m + n, m * n, m * n),
            BinaryMechanism::Rank2Matmul => (m * k + k * n, m * n, 2 * m * k * n),
            BinaryMechanism::Rank3Matmul => (b * (m * k + k * n), b * m * n, 2 * b * m * k * n),
        };
        WorkUnits::new(
            u64::try_from(output).expect("bounded benchmark elements"),
            u64::try_from((inputs + output) * size_of::<f32>()).expect("bounded benchmark bytes"),
            Some(u64::try_from(flops).expect("bounded benchmark FLOPs")),
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let [b, m, k, n] = self.dimensions;
        let tensor = |shape: &[usize], stream| {
            let elements = shape.iter().product();
            Tensor::from_vec(deterministic_f32_values(elements, stream), shape, device)
        };
        match self.mechanism {
            BinaryMechanism::Hadamard => Ok(vec![tensor(&[m], 1)?, tensor(&[m], 2)?]),
            BinaryMechanism::Outer => Ok(vec![tensor(&[m], 3)?, tensor(&[n], 4)?]),
            BinaryMechanism::Rank2Matmul => Ok(vec![tensor(&[m, k], 5)?, tensor(&[k, n], 6)?]),
            BinaryMechanism::Rank3Matmul => {
                Ok(vec![tensor(&[b, m, k], 7)?, tensor(&[b, k, n], 8)?])
            }
        }
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        match self.mechanism {
            BinaryMechanism::Hadamard => {
                einsum!("feature, feature -> feature", &inputs[0], &inputs[1])
            }
            BinaryMechanism::Outer => einsum!("row, column -> row column", &inputs[0], &inputs[1]),
            BinaryMechanism::Rank2Matmul => einsum!(
                "row inner, inner column -> row column",
                &inputs[0],
                &inputs[1]
            ),
            BinaryMechanism::Rank3Matmul => einsum!(
                "batch row inner, batch inner column -> batch row column",
                &inputs[0],
                &inputs[1]
            ),
        }
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        match self.mechanism {
            BinaryMechanism::Hadamard => inputs[0].mul(&inputs[1]),
            BinaryMechanism::Outer => inputs[0]
                .unsqueeze(1)?
                .broadcast_mul(&inputs[1].unsqueeze(0)?),
            BinaryMechanism::Rank2Matmul | BinaryMechanism::Rank3Matmul => {
                inputs[0].matmul(&inputs[1])
            }
        }
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("binary fast-path outputs have different shapes")
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        if library != reference {
            candle_core::bail!("binary fast-path outputs have different values")
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ZeroKStructuralMetrics {
    pub output_elements: u64,
    pub hypothetical_contraction_flops: u64,
    pub gemm_submissions: u64,
    pub current_public_operations: u64,
    pub candidate_public_operations: u64,
    pub current_temporary_elements: u64,
    pub candidate_temporary_elements: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct ZeroKScenario {
    id: &'static str,
    rows: usize,
    columns: usize,
}

#[must_use]
pub fn zero_k_scenarios() -> [ZeroKScenario; 3] {
    [
        ZeroKScenario {
            id: "einsum/zero-k/output-1x1",
            rows: 1,
            columns: 1,
        },
        ZeroKScenario {
            id: "einsum/zero-k/output-64x64",
            rows: 64,
            columns: 64,
        },
        ZeroKScenario {
            id: "einsum/zero-k/output-512x512",
            rows: 512,
            columns: 512,
        },
    ]
}

impl ZeroKScenario {
    #[must_use]
    pub fn structural_metrics(&self) -> ZeroKStructuralMetrics {
        ZeroKStructuralMetrics {
            output_elements: u64::try_from(self.rows * self.columns)
                .expect("bounded zero-K output elements"),
            hypothetical_contraction_flops: 0,
            gemm_submissions: 0,
            current_public_operations: 3,
            candidate_public_operations: 3,
            current_temporary_elements: 2,
            candidate_temporary_elements: 0,
        }
    }

    fn graph_preserving_reference(&self, left: &Tensor, right: &Tensor) -> Result<Tensor> {
        let anchor = |operand: &Tensor| operand.unsqueeze(0)?.narrow(0, 0, 0)?.sum_all();
        anchor(left)?
            .add(&anchor(right)?)?
            .broadcast_as((self.rows, self.columns))
    }
}

pub fn zero_k_cat_candidate(left: &Tensor, right: &Tensor, shape: &[usize]) -> Result<Tensor> {
    let anchor = |operand: &Tensor| operand.unsqueeze(0)?.narrow(0, 0, 0)?.sum_all();
    anchor(left)?.add(&anchor(right)?)?.broadcast_as(shape)
}

impl Scenario for ZeroKScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new(self.id)
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let output_elements = self.structural_metrics().output_elements;
        WorkUnits::new(
            output_elements,
            output_elements * size_of::<f32>() as u64,
            Some(0),
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        Ok(vec![
            Tensor::zeros((self.rows, 0), DType::F32, device)?,
            Tensor::zeros((0, self.columns), DType::F32, device)?,
        ])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        einsum!(
            "row inner, inner column -> row column",
            &inputs[0],
            &inputs[1]
        )
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.graph_preserving_reference(&inputs[0], &inputs[1])
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != [self.rows, self.columns] || library.dims() != reference.dims() {
            candle_core::bail!("zero-K outputs have different shapes")
        }
        if library
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .any(|&value| value != 0.)
        {
            candle_core::bail!("zero-K library output is not exactly zero")
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum ReductionLayout {
    ContiguousTrailing,
    StridedNonAdjacent,
}

#[derive(Clone, Copy)]
enum ReductionKind {
    Sum,
    Mean,
}

pub struct ReductionFusionScenario {
    id: &'static str,
    layout: ReductionLayout,
    kind: ReductionKind,
}

#[must_use]
pub fn reduction_fusion_scenarios() -> [ReductionFusionScenario; 4] {
    use ReductionKind::{Mean, Sum};
    use ReductionLayout::{ContiguousTrailing, StridedNonAdjacent};
    [
        ReductionFusionScenario {
            id: "reduce/fusion/contiguous-trailing/sum",
            layout: ContiguousTrailing,
            kind: Sum,
        },
        ReductionFusionScenario {
            id: "reduce/fusion/contiguous-trailing/mean",
            layout: ContiguousTrailing,
            kind: Mean,
        },
        ReductionFusionScenario {
            id: "reduce/fusion/strided-non-adjacent/sum",
            layout: StridedNonAdjacent,
            kind: Sum,
        },
        ReductionFusionScenario {
            id: "reduce/fusion/strided-non-adjacent/mean",
            layout: StridedNonAdjacent,
            kind: Mean,
        },
    ]
}

impl ReductionFusionScenario {
    const INPUT_ELEMENTS: usize = 16 * 16 * 32 * 32;

    fn output_elements(&self) -> usize {
        match self.layout {
            ReductionLayout::ContiguousTrailing => 16 * 16,
            ReductionLayout::StridedNonAdjacent => 16 * 32,
        }
    }
}

impl Scenario for ReductionFusionScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new(self.id)
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let output_elements = self.output_elements();
        WorkUnits::new(
            Self::INPUT_ELEMENTS as u64,
            ((Self::INPUT_ELEMENTS + output_elements) * size_of::<f32>()) as u64,
            Some(Self::INPUT_ELEMENTS as u64),
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let values = (0..Self::INPUT_ELEMENTS)
            .map(|index| (index % 257) as f32 / 257.)
            .collect::<Vec<_>>();
        let input = match self.layout {
            ReductionLayout::ContiguousTrailing => {
                Tensor::from_vec(values, (16, 16, 32, 32), device)?
            }
            ReductionLayout::StridedNonAdjacent => {
                Tensor::from_vec(values, (16, 32, 16, 32), device)?.permute([0, 2, 1, 3])?
            }
        };
        Ok(vec![input])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        match (self.layout, self.kind) {
            (ReductionLayout::ContiguousTrailing, ReductionKind::Sum) => {
                einops!("batch channel sum(row column) -> batch channel", &inputs[0])
            }
            (ReductionLayout::ContiguousTrailing, ReductionKind::Mean) => einops!(
                "batch channel mean(row column) -> batch channel",
                &inputs[0]
            ),
            (ReductionLayout::StridedNonAdjacent, ReductionKind::Sum) => einops!(
                "batch sum(row) channel sum(column) -> batch channel",
                &inputs[0]
            ),
            (ReductionLayout::StridedNonAdjacent, ReductionKind::Mean) => einops!(
                "batch mean(row) channel mean(column) -> batch channel",
                &inputs[0]
            ),
        }
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let axes: &[usize] = match self.layout {
            ReductionLayout::ContiguousTrailing => &[2, 3],
            ReductionLayout::StridedNonAdjacent => &[1, 3],
        };
        match self.kind {
            ReductionKind::Sum => inputs[0].sum(axes),
            ReductionKind::Mean => inputs[0].mean(axes),
        }
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("reduction fusion outputs have different shapes")
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        let tolerance = 1e-5f32;
        for (index, (&library, &reference)) in library.iter().zip(&reference).enumerate() {
            let allowed = tolerance * reference.abs().max(1.);
            if (library - reference).abs() > allowed {
                candle_core::bail!(
                    "reduction fusion output {index} differs: {library} vs {reference}, tolerance {allowed}"
                )
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum RepeatFamily {
    SingleAxis,
    TwoAxis,
}

#[derive(Clone, Copy)]
enum RepeatMode {
    Construct,
    Consume,
}

pub struct RepeatBroadcastScenario {
    id: &'static str,
    family: RepeatFamily,
    mode: RepeatMode,
}

#[must_use]
pub fn repeat_broadcast_scenarios() -> [RepeatBroadcastScenario; 4] {
    use RepeatFamily::{SingleAxis, TwoAxis};
    use RepeatMode::{Construct, Consume};
    [
        RepeatBroadcastScenario {
            id: "repeat/broadcast/single-axis/construct",
            family: SingleAxis,
            mode: Construct,
        },
        RepeatBroadcastScenario {
            id: "repeat/broadcast/single-axis/consume",
            family: SingleAxis,
            mode: Consume,
        },
        RepeatBroadcastScenario {
            id: "repeat/broadcast/two-axis/construct",
            family: TwoAxis,
            mode: Construct,
        },
        RepeatBroadcastScenario {
            id: "repeat/broadcast/two-axis/consume",
            family: TwoAxis,
            mode: Consume,
        },
    ]
}

impl RepeatBroadcastScenario {
    fn input_side(&self) -> usize {
        match self.family {
            RepeatFamily::SingleAxis => 256,
            RepeatFamily::TwoAxis => 128,
        }
    }

    fn output_elements(&self) -> usize {
        match self.family {
            RepeatFamily::SingleAxis => 256 * 32 * 256,
            RepeatFamily::TwoAxis => 16 * 128 * 128 * 16,
        }
    }

    fn maybe_consume(&self, tensor: Tensor) -> Result<Tensor> {
        match self.mode {
            RepeatMode::Construct => Ok(tensor),
            RepeatMode::Consume => tensor.contiguous(),
        }
    }
}

impl Scenario for RepeatBroadcastScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new(self.id)
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let input_elements = self.input_side() * self.input_side();
        let output_elements = self.output_elements();
        WorkUnits::new(
            output_elements as u64,
            ((input_elements + output_elements) * size_of::<f32>()) as u64,
            None,
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let side = self.input_side();
        let values = (0..side * side)
            .map(|index| (index % 251) as f32 / 251.)
            .collect::<Vec<_>>();
        Ok(vec![
            Tensor::from_vec(values, (side, side), device)?.permute([1, 0])?,
        ])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let repeated = match self.family {
            RepeatFamily::SingleAxis => einops!("row column -> copies:32 row column", &inputs[0])?,
            RepeatFamily::TwoAxis => {
                einops!("row column -> row outer:16 column inner:16", &inputs[0])?
            }
        };
        self.maybe_consume(repeated)
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let repeated = match self.family {
            RepeatFamily::SingleAxis => inputs[0].unsqueeze(0)?.repeat((32, 1, 1))?,
            RepeatFamily::TwoAxis => inputs[0]
                .unsqueeze(1)?
                .unsqueeze(3)?
                .repeat((1, 16, 1, 16))?,
        };
        self.maybe_consume(repeated)
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("repeat broadcast outputs have different shapes")
        }
        if matches!(self.mode, RepeatMode::Construct) && library.is_contiguous() {
            candle_core::bail!("repeat construct output unexpectedly materialized")
        }
        let library = library.flatten_all()?.to_vec1::<f32>()?;
        let reference = reference.flatten_all()?.to_vec1::<f32>()?;
        for (index, (&library, &reference)) in library.iter().zip(&reference).enumerate() {
            let tolerance = if matches!(self.mode, RepeatMode::Consume) {
                1e-4 * reference.abs().max(1.)
            } else {
                0.
            };
            if (library - reference).abs() > tolerance {
                candle_core::bail!(
                    "repeat broadcast output {index} differs: {library} vs {reference}, tolerance {tolerance}"
                )
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IdentityReshapeLayout {
    Contiguous,
    NonContiguous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IdentityReshapeMode {
    Construct,
    Consume,
}

pub struct IdentityReshapeScenario {
    id: &'static str,
    layout: IdentityReshapeLayout,
    mode: IdentityReshapeMode,
}

#[must_use]
pub fn identity_reshape_scenarios() -> [IdentityReshapeScenario; 4] {
    use IdentityReshapeLayout::{Contiguous, NonContiguous};
    use IdentityReshapeMode::{Construct, Consume};
    [
        IdentityReshapeScenario {
            id: "reshape/identity/contiguous/construct",
            layout: Contiguous,
            mode: Construct,
        },
        IdentityReshapeScenario {
            id: "reshape/identity/contiguous/consume",
            layout: Contiguous,
            mode: Consume,
        },
        IdentityReshapeScenario {
            id: "reshape/identity/non-contiguous/construct",
            layout: NonContiguous,
            mode: Construct,
        },
        IdentityReshapeScenario {
            id: "reshape/identity/non-contiguous/consume",
            layout: NonContiguous,
            mode: Consume,
        },
    ]
}

impl IdentityReshapeScenario {
    const SIDE: usize = 512;

    fn maybe_consume(&self, tensor: Tensor) -> Result<Tensor> {
        match self.mode {
            IdentityReshapeMode::Construct => Ok(tensor),
            IdentityReshapeMode::Consume => tensor.contiguous(),
        }
    }
}

impl Scenario for IdentityReshapeScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new(self.id)
    }

    fn tracked(&self) -> bool {
        true
    }

    fn work(&self) -> WorkUnits {
        let elements = Self::SIDE * Self::SIDE;
        let traversals = match self.mode {
            IdentityReshapeMode::Construct => 1,
            IdentityReshapeMode::Consume => 2,
        };
        WorkUnits::new(
            elements as u64,
            (elements * traversals * size_of::<f32>()) as u64,
            None,
        )
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        let elements = Self::SIDE * Self::SIDE;
        let input = Tensor::from_vec(
            deterministic_f32_values(elements, 31),
            (Self::SIDE, Self::SIDE),
            device,
        )?;
        let input = match self.layout {
            IdentityReshapeLayout::Contiguous => input,
            IdentityReshapeLayout::NonContiguous => input.permute([1, 0])?,
        };
        Ok(vec![input])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        let reshaped = candle_einops::Backend::reshape(&inputs[0], inputs[0].dims())?;
        self.maybe_consume(reshaped)
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.maybe_consume(inputs[0].clone())
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        if library.dims() != reference.dims() {
            candle_core::bail!("identity reshape outputs have different shapes")
        }
        let expected_contiguous = self.mode == IdentityReshapeMode::Consume
            || self.layout == IdentityReshapeLayout::Contiguous;
        if library.is_contiguous() != expected_contiguous {
            candle_core::bail!("identity reshape output has an unexpected layout")
        }
        if library.flatten_all()?.to_vec1::<f32>()? != reference.flatten_all()?.to_vec1::<f32>()? {
            candle_core::bail!("identity reshape outputs have different values")
        }
        Ok(())
    }
}

pub(crate) fn criterion_operation(
    criterion: &mut Criterion,
    name: &str,
    prepared: &PreparedScenario<'_>,
    operation: Operation,
    synchronizer: &dyn Synchronizer,
    failure: &'static str,
) {
    synchronizer
        .synchronize()
        .expect("Criterion pre-synchronization must succeed");
    criterion.bench_function(name, |bencher| {
        bencher
            .iter(|| run_synchronized_operation(prepared, operation, synchronizer).expect(failure));
    });
}

pub fn criterion_plumbing_benchmark(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let scenario = PlumbingScenario;
    let prepared = prepare(&scenario, &device).expect("plumbing setup must succeed");
    let synchronizer = DeviceSynchronizer(&device);
    criterion_operation(
        criterion,
        "plumbing/smoke-untracked",
        &prepared,
        Operation::Library,
        &synchronizer,
        "plumbing sample must succeed",
    );
}

pub fn criterion_product_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in product_scenarios() {
        let prepared = prepare(&scenario, &device).expect("product setup must succeed");
        criterion_operation(
            criterion,
            scenario.id().as_str(),
            &prepared,
            Operation::Library,
            &synchronizer,
            "product sample must succeed",
        );
    }
}

pub fn criterion_binary_fast_paths(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in binary_fast_path_scenarios() {
        let prepared = prepare(&scenario, &device).expect("binary fast-path setup must succeed");
        for operation in [Operation::Library, Operation::Reference] {
            let name = format!(
                "{}/{}",
                scenario.id().as_str(),
                match operation {
                    Operation::Library => "library",
                    Operation::Reference => "reference",
                }
            );
            criterion_operation(
                criterion,
                &name,
                &prepared,
                operation,
                &synchronizer,
                "binary fast-path sample must succeed",
            );
        }
    }
}

pub fn criterion_zero_k(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in zero_k_scenarios() {
        let prepared = prepare(&scenario, &device).expect("zero-K setup must succeed");
        for operation in [Operation::Library, Operation::Reference] {
            let name = format!(
                "{}/{}",
                scenario.id().as_str(),
                match operation {
                    Operation::Library => "library",
                    Operation::Reference => "reference",
                }
            );
            criterion_operation(
                criterion,
                &name,
                &prepared,
                operation,
                &synchronizer,
                "zero-K sample must succeed",
            );
        }
    }
}

pub fn criterion_reduction_fusion(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in reduction_fusion_scenarios() {
        let prepared = prepare(&scenario, &device).expect("reduction fusion setup must succeed");
        for operation in [Operation::Library, Operation::Reference] {
            let name = format!(
                "{}/{}",
                scenario.id().as_str(),
                match operation {
                    Operation::Library => "library",
                    Operation::Reference => "reference",
                }
            );
            criterion_operation(
                criterion,
                &name,
                &prepared,
                operation,
                &synchronizer,
                "reduction fusion sample must succeed",
            );
        }
    }
}

pub fn criterion_repeat_broadcast(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in repeat_broadcast_scenarios() {
        let prepared = prepare(&scenario, &device).expect("repeat broadcast setup must succeed");
        for operation in [Operation::Library, Operation::Reference] {
            let name = format!(
                "{}/{}",
                scenario.id().as_str(),
                match operation {
                    Operation::Library => "library",
                    Operation::Reference => "reference",
                }
            );
            criterion_operation(
                criterion,
                &name,
                &prepared,
                operation,
                &synchronizer,
                "repeat broadcast sample must succeed",
            );
        }
    }
}

pub fn criterion_identity_reshape(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let synchronizer = DeviceSynchronizer(&device);
    for scenario in identity_reshape_scenarios() {
        let prepared = prepare(&scenario, &device).expect("identity reshape setup must succeed");
        for operation in [Operation::Library, Operation::Reference] {
            let name = format!(
                "{}/{}",
                scenario.id().as_str(),
                match operation {
                    Operation::Library => "library",
                    Operation::Reference => "reference",
                }
            );
            criterion_operation(
                criterion,
                &name,
                &prepared,
                operation,
                &synchronizer,
                "identity reshape sample must succeed",
            );
        }
    }
}
