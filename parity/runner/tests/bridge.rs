use std::collections::BTreeMap;
use std::process::Command;

use candle_einops_parity_runner::{
    NormalizedResponse, Operation, OracleClient, OracleRequest, OracleValue, PROTOCOL_VERSION,
    ParityConfig, PatternId, persist_replay,
};

fn transpose_request(case_id: &str) -> OracleRequest {
    OracleRequest {
        case_id: case_id.to_string(),
        pattern_id: PatternId::new("rearrange/transpose-v1").expect("stable pattern id"),
        operation: Operation::Rearrange,
        pattern: "rows columns -> columns rows".to_string(),
        reduction: None,
        dtype: "float64".to_string(),
        shape: vec![2, 3],
        values: [0., 1., 2., 3., 4., 5.]
            .into_iter()
            .map(OracleValue::from)
            .collect(),
        axes_lengths: BTreeMap::new(),
    }
}

#[test]
fn live_client_validates_hello_reuses_one_child_and_preserves_order() {
    let mut client = OracleClient::spawn_uv().expect("locked Python oracle starts");
    assert_eq!(client.protocol_version(), PROTOCOL_VERSION);
    let child_id = client.child_id();

    let responses = client
        .evaluate(&[
            transpose_request("ordered-0"),
            transpose_request("ordered-1"),
        ])
        .expect("ordered batch succeeds");
    assert_eq!(client.child_id(), child_id, "the child must be persistent");
    assert_eq!(responses[0].case_id(), Some("ordered-0"));
    assert_eq!(responses[1].case_id(), Some("ordered-1"));
    assert!(matches!(responses[0], NormalizedResponse::Success(_)));

    let mut invalid = transpose_request("invalid-shape");
    invalid.values.truncate(1);
    let responses = client
        .evaluate(&[transpose_request("success"), invalid])
        .expect("oracle errors are protocol data");
    let NormalizedResponse::Success(success) = &responses[0] else {
        panic!("expected success: {:?}", responses[0]);
    };
    assert_eq!(success.shape, [3, 2]);
    assert_eq!(success.values.len(), 6);
    let NormalizedResponse::Error(error) = &responses[1] else {
        panic!("expected normalized error: {:?}", responses[1]);
    };
    assert_eq!(error.case_id.as_deref(), Some("invalid-shape"));
    assert_eq!(error.error.kind, "ValueError");
    assert!(!error.error.message.contains("Traceback"));

    let request = transpose_request("replay-7");
    let json = serde_json::to_string(&request).expect("request serializes");
    let directory = tempfile::tempdir().expect("temporary replay directory");
    let path = directory.path().join("failure.json");
    persist_replay(&path, &request).expect("failure is persisted");
    assert_eq!(
        std::fs::read_to_string(&path).expect("replay is readable"),
        json
    );
    let from_json = client.replay_json(&json).expect("JSON replay succeeds");
    let from_file = client.replay_file(&path).expect("file replay succeeds");
    assert_eq!(from_json, from_file);
    assert_eq!(from_json.case_id(), Some("replay-7"));

    assert_eq!(client.child_id(), child_id, "all calls reuse one child");
    let status = client.shutdown().expect("child exits after stdin closes");
    assert!(status.success());
}

#[test]
fn incompatible_hello_version_is_rejected() {
    let mut command = Command::new("sh");
    command.args([
        "-c",
        "printf '%s\\n' '{\"kind\":\"hello\",\"protocol_version\":999,\"service\":\"fake\"}'",
    ]);
    let error = OracleClient::spawn(command).expect_err("version mismatch must fail");
    assert!(error.to_string().contains("protocol version 999"));
}

#[test]
fn out_of_order_case_identity_is_rejected() {
    let mut command = Command::new("sh");
    command.args([
        "-c",
        "printf '%s\\n' '{\"kind\":\"hello\",\"protocol_version\":1,\"service\":\"candle-einops-python-oracle\"}'; read ignored; printf '%s\\n' '{\"case_id\":\"wrong\",\"ok\":true,\"shape\":[],\"values\":[]}'",
    ]);
    let mut client = OracleClient::spawn(command).expect("valid fake hello");
    let error = client
        .evaluate(&[transpose_request("expected")])
        .expect_err("mismatched response id must fail");
    assert!(
        error
            .to_string()
            .contains("expected `expected`, received `wrong`")
    );
}

#[test]
fn property_configuration_is_bounded_and_accepts_deterministic_overrides() {
    let defaults = ParityConfig::from_overrides(None, None, None).expect("defaults");
    assert!(defaults.cases > 0 && defaults.cases <= 256);
    assert!(defaults.max_elements > 0 && defaults.max_elements <= 4096);

    let configured = ParityConfig::from_overrides(Some("17"), Some("42"), Some("128"))
        .expect("valid deterministic overrides");
    assert_eq!(configured.cases, 17);
    assert_eq!(configured.seed, 42);
    assert_eq!(configured.max_elements, 128);
    assert_eq!(configured.proptest_config().cases, 17);
    assert!(ParityConfig::from_overrides(Some("0"), None, None).is_err());
    assert!(ParityConfig::from_overrides(None, None, Some("5000")).is_err());
}
