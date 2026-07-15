use std::sync::{Arc, Mutex};

use candle_core::{Device, Result, Tensor};
use candle_einops_benchmarks::broadcast_gemm_spike::broadcast_scenarios;
use candle_einops_benchmarks::{
    Backend, BenchmarkRecord, Clock, Fingerprint, Operation, SamplingOrderPolicy, Scenario,
    ScenarioId, Synchronizer, WorkUnits, binary_fast_path_scenarios, measure_pair, prepare,
    reduction_fusion_scenarios, repeat_broadcast_scenarios, run_synchronized_operation,
};

#[derive(Clone)]
struct EventLog(Arc<Mutex<Vec<&'static str>>>);

impl EventLog {
    fn push(&self, event: &'static str) {
        self.0.lock().expect("event log lock").push(event);
    }

    fn take(&self) -> Vec<&'static str> {
        std::mem::take(&mut *self.0.lock().expect("event log lock"))
    }
}

fn flat_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.flatten_all()?.to_vec1::<f32>()
}

fn assert_discriminating_inputs(scenario: &dyn Scenario) -> Result<Vec<Tensor>> {
    let first = scenario.setup(&Device::Cpu)?;
    let second = scenario.setup(&Device::Cpu)?;
    assert_eq!(first.len(), second.len(), "{}", scenario.id().as_str());
    for (operand, (first, second)) in first.iter().zip(&second).enumerate() {
        let first_values = flat_f32(first)?;
        let second_values = flat_f32(second)?;
        assert_eq!(first.dims(), second.dims());
        assert_eq!(
            first_values,
            second_values,
            "{} operand {operand}",
            scenario.id().as_str()
        );
        assert!(
            first_values.windows(2).any(|pair| pair[0] != pair[1]),
            "{} operand {operand} is constant",
            scenario.id().as_str()
        );
    }
    if first.len() == 2 && first[0].elem_count() == first[1].elem_count() {
        assert_ne!(
            flat_f32(&first[0])?,
            flat_f32(&first[1])?,
            "{} operands are indistinguishable",
            scenario.id().as_str()
        );
    }
    Ok(first)
}

#[test]
fn binary_and_broadcast_inputs_are_deterministic_and_discriminating() -> Result<()> {
    for scenario in binary_fast_path_scenarios() {
        let inputs = assert_discriminating_inputs(&scenario)?;
        let id = scenario.id().as_str();
        if id.contains("outer") {
            assert_ne!(inputs[0].dim(0)?, inputs[1].dim(0)?, "{id}");
        } else if id.contains("rank2-matmul") {
            let [m, k] = <[usize; 2]>::try_from(inputs[0].dims()).unwrap();
            let [right_k, n] = <[usize; 2]>::try_from(inputs[1].dims()).unwrap();
            assert_eq!(k, right_k, "{id}");
            assert_eq!(
                3,
                [m, k, n]
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                "{id}"
            );
        } else if id.contains("rank3-matmul") {
            let [batch, m, k] = <[usize; 3]>::try_from(inputs[0].dims()).unwrap();
            let [right_batch, right_k, n] = <[usize; 3]>::try_from(inputs[1].dims()).unwrap();
            assert_eq!((batch, k), (right_batch, right_k), "{id}");
            assert_eq!(
                3,
                [m, k, n]
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                "{id}"
            );
        }
    }
    for scenario in broadcast_scenarios() {
        assert_discriminating_inputs(scenario)?;
    }
    Ok(())
}

#[test]
fn repeat_broadcast_scenarios_are_exactly_the_ticket_owned_matrix() -> Result<()> {
    let scenarios = repeat_broadcast_scenarios();
    let ids = scenarios
        .iter()
        .map(|scenario| scenario.id().as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [
            "repeat/broadcast/single-axis/construct",
            "repeat/broadcast/single-axis/consume",
            "repeat/broadcast/two-axis/construct",
            "repeat/broadcast/two-axis/consume",
        ]
    );
    for scenario in &scenarios {
        assert!(scenario.tracked());
        prepare(scenario, &Device::Cpu)?;
    }
    Ok(())
}

#[test]
fn reduction_fusion_scenarios_are_exactly_the_ticket_owned_matrix() -> Result<()> {
    let scenarios = reduction_fusion_scenarios();
    assert_eq!(scenarios.len(), 4);
    let ids = scenarios
        .iter()
        .map(|scenario| scenario.id().as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        [
            "reduce/fusion/contiguous-trailing/sum",
            "reduce/fusion/contiguous-trailing/mean",
            "reduce/fusion/strided-non-adjacent/sum",
            "reduce/fusion/strided-non-adjacent/mean",
        ]
    );
    for scenario in &scenarios {
        assert!(scenario.tracked());
        prepare(scenario, &Device::Cpu)?;
    }
    Ok(())
}

#[test]
fn binary_fast_path_scenarios_are_tracked_unique_and_correct() -> Result<()> {
    let scenarios = binary_fast_path_scenarios();
    assert_eq!(scenarios.len(), 8);
    let mut ids = scenarios
        .iter()
        .map(|scenario| scenario.id().as_str())
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 8);
    for scenario in &scenarios {
        assert!(scenario.tracked());
        prepare(scenario, &Device::Cpu)?;
    }
    Ok(())
}

struct ContractScenario {
    events: EventLog,
}

impl Scenario for ContractScenario {
    fn id(&self) -> ScenarioId {
        ScenarioId::new("plumbing/contract")
    }

    fn tracked(&self) -> bool {
        false
    }

    fn work(&self) -> WorkUnits {
        WorkUnits::new(4, 16, None)
    }

    fn setup(&self, device: &Device) -> Result<Vec<Tensor>> {
        self.events.push("setup");
        Ok(vec![Tensor::new(&[1f32, 2., 3., 4.], device)?])
    }

    fn run_library(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.events.push("library");
        Ok(inputs[0].clone())
    }

    fn run_reference(&self, inputs: &[Tensor]) -> Result<Tensor> {
        self.events.push("reference");
        Ok(inputs[0].clone())
    }

    fn check(&self, library: &Tensor, reference: &Tensor) -> Result<()> {
        self.events.push("correctness");
        assert_eq!(library.dims(), reference.dims());
        Ok(())
    }
}

struct ContractSynchronizer {
    events: EventLog,
}

impl Synchronizer for ContractSynchronizer {
    fn synchronize(&self) -> Result<()> {
        self.events.push("sync");
        Ok(())
    }
}

struct ContractClock {
    events: EventLog,
    ticks: Mutex<u64>,
}

impl Clock for ContractClock {
    fn now_ns(&self) -> u64 {
        self.events.push("clock");
        let mut ticks = self.ticks.lock().expect("clock lock");
        *ticks += 10;
        *ticks
    }
}

#[test]
fn setup_and_correctness_are_outside_the_timed_sample() -> Result<()> {
    let events = EventLog(Arc::new(Mutex::new(Vec::new())));
    let scenario = ContractScenario {
        events: events.clone(),
    };
    let prepared = prepare(&scenario, &Device::Cpu)?;
    assert_eq!(
        events.take(),
        ["setup", "library", "reference", "correctness"]
    );

    let synchronizer = ContractSynchronizer {
        events: events.clone(),
    };
    let clock = ContractClock {
        events: events.clone(),
        ticks: Mutex::new(0),
    };
    let result = measure_pair(&prepared, &synchronizer, &clock, 3)?;
    assert_eq!(result.library.samples_ns, [10, 10, 10]);
    assert_eq!(result.reference.samples_ns, [10, 10, 10]);
    assert_eq!(
        events.take(),
        [
            "sync",
            "clock",
            "library",
            "sync",
            "clock",
            "sync",
            "clock",
            "reference",
            "sync",
            "clock",
            "sync",
            "clock",
            "reference",
            "sync",
            "clock",
            "sync",
            "clock",
            "library",
            "sync",
            "clock",
            "sync",
            "clock",
            "library",
            "sync",
            "clock",
            "sync",
            "clock",
            "reference",
            "sync",
            "clock",
        ]
    );
    assert_eq!(
        result.order_policy,
        SamplingOrderPolicy::AlternatingLibraryThenReference
    );

    let fingerprint = Fingerprint {
        git_sha: "a".repeat(40),
        rust_version: "rustc test".to_owned(),
        candle_version: "0.11.0".to_owned(),
        os: "test-os".to_owned(),
        architecture: "test-arch".to_owned(),
        backend: Backend::Cpu,
        device: "test-cpu".to_owned(),
        driver: None,
    };
    let record = BenchmarkRecord::from_measurement(&prepared, &result, fingerprint)
        .expect("valid benchmark record");
    let mut serialized = serde_json::to_value(&record).expect("serialize benchmark record");
    assert_eq!(
        serialized["sampling_order_policy"],
        "alternating_library_then_reference"
    );
    serialized
        .as_object_mut()
        .unwrap()
        .remove("sampling_order_policy");
    let legacy: BenchmarkRecord =
        serde_json::from_value(serialized).expect("deserialize legacy v1 record");
    assert_eq!(legacy.schema_version, 1);
    assert_eq!(
        legacy.sampling_order_policy,
        SamplingOrderPolicy::FixedLibraryThenReference
    );
    Ok(())
}

#[test]
fn criterion_sample_uses_no_json_clock_and_synchronizes_after_execution() -> Result<()> {
    let events = EventLog(Arc::new(Mutex::new(Vec::new())));
    let scenario = ContractScenario {
        events: events.clone(),
    };
    let prepared = prepare(&scenario, &Device::Cpu)?;
    events.take();
    let synchronizer = ContractSynchronizer {
        events: events.clone(),
    };
    let output = run_synchronized_operation(&prepared, Operation::Library, &synchronizer)?;
    assert_eq!(output.dims(), &[4]);
    assert_eq!(events.take(), ["library", "sync"]);
    Ok(())
}

#[test]
fn incomplete_or_invalid_metadata_is_rejected() {
    let incomplete = serde_json::from_str::<BenchmarkRecord>(r#"{"schema_version":1}"#);
    assert!(incomplete.is_err());

    let invalid = Fingerprint {
        git_sha: "not-a-sha".to_owned(),
        rust_version: String::new(),
        candle_version: "0.11.0".to_owned(),
        os: String::new(),
        architecture: String::new(),
        backend: Backend::Cpu,
        device: String::new(),
        driver: None,
    };
    assert!(invalid.validate().is_err());
}
