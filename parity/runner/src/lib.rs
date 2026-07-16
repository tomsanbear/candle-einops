//! Typed, persistent bridge to the locked Python einops oracle.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

use proptest::test_runner::{Config as ProptestConfig, RngSeed};
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;
pub const SERVICE_NAME: &str = "candle-einops-python-oracle";

const DEFAULT_CASES: u32 = 64;
const DEFAULT_SEED: u64 = 0x5eed_2026_cafe_f00d;
const DEFAULT_MAX_ELEMENTS: usize = 512;
const MAX_CASES: u32 = 256;
const MAX_ELEMENTS: usize = 4096;

pub type BridgeResult<T> = Result<T, BridgeError>;

#[derive(Debug)]
pub struct BridgeError {
    message: String,
}

impl BridgeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    fn context(context: &str, error: impl fmt::Display) -> Self {
        Self::new(format!("{context}: {error}"))
    }

    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(message)
    }
}

impl fmt::Display for BridgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for BridgeError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PatternId(String);

impl PatternId {
    pub fn new(value: impl Into<String>) -> BridgeResult<Self> {
        let value = value.into();
        if value.is_empty()
            || !value.chars().all(|character| {
                character.is_ascii_alphanumeric()
                    || matches!(character, '-' | '_' | '/' | '.' | ':')
            })
        {
            return Err(BridgeError::new(format!(
                "invalid stable pattern id `{value}`"
            )));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    fn is_valid(&self) -> bool {
        Self::new(self.0.clone()).is_ok()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operation {
    Rearrange,
    Repeat,
    Reduce,
    Einsum,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OracleValue {
    Number(f64),
    Symbol(String),
}

impl From<f64> for OracleValue {
    fn from(value: f64) -> Self {
        Self::Number(value)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OracleRequest {
    pub case_id: String,
    pub pattern_id: PatternId,
    pub operation: Operation,
    pub pattern: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduction: Option<String>,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub values: Vec<OracleValue>,
    #[serde(default)]
    pub axes_lengths: BTreeMap<String, usize>,
}

impl OracleRequest {
    fn validate_identity(&self) -> BridgeResult<()> {
        if self.case_id.is_empty() {
            return Err(BridgeError::new("oracle case id must not be empty"));
        }
        if !self.pattern_id.is_valid() {
            return Err(BridgeError::new(format!(
                "invalid stable pattern id `{}`",
                self.pattern_id.as_str()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EinsumOperand {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub values: Vec<OracleValue>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EinsumRequest {
    pub case_id: String,
    pub pattern_id: PatternId,
    pub operation: Operation,
    pub pattern: String,
    pub operands: Vec<EinsumOperand>,
}

impl EinsumRequest {
    fn validate_identity(&self) -> BridgeResult<()> {
        validate_identity(&self.case_id, &self.pattern_id)
    }
}

fn validate_identity(case_id: &str, pattern_id: &PatternId) -> BridgeResult<()> {
    if case_id.is_empty() {
        return Err(BridgeError::new("oracle case id must not be empty"));
    }
    if !pattern_id.is_valid() {
        return Err(BridgeError::new(format!(
            "invalid stable pattern id `{}`",
            pattern_id.as_str()
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuccessResponse {
    pub case_id: String,
    pub shape: Vec<usize>,
    pub values: Vec<OracleValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedError {
    pub kind: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub case_id: Option<String>,
    pub error: NormalizedError,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum NormalizedResponse {
    Success(SuccessResponse),
    Error(ErrorResponse),
}

impl NormalizedResponse {
    pub fn case_id(&self) -> Option<&str> {
        match self {
            Self::Success(response) => Some(&response.case_id),
            Self::Error(response) => response.case_id.as_deref(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct Hello {
    kind: String,
    protocol_version: u32,
    service: String,
}

#[derive(Debug, Deserialize)]
struct WireResponse {
    case_id: Option<String>,
    ok: bool,
    #[serde(default)]
    shape: Option<Vec<usize>>,
    #[serde(default)]
    values: Option<Vec<OracleValue>>,
    #[serde(default)]
    error: Option<NormalizedError>,
}

impl WireResponse {
    fn normalize(self) -> BridgeResult<NormalizedResponse> {
        if self.ok {
            let case_id = self
                .case_id
                .ok_or_else(|| BridgeError::new("successful oracle response has no case id"))?;
            let shape = self.shape.ok_or_else(|| {
                BridgeError::new(format!(
                    "successful oracle response `{case_id}` has no shape"
                ))
            })?;
            let values = self.values.ok_or_else(|| {
                BridgeError::new(format!(
                    "successful oracle response `{case_id}` has no values"
                ))
            })?;
            if self.error.is_some() {
                return Err(BridgeError::new(format!(
                    "successful oracle response `{case_id}` also contains an error"
                )));
            }
            Ok(NormalizedResponse::Success(SuccessResponse {
                case_id,
                shape,
                values,
            }))
        } else {
            if self.shape.is_some() || self.values.is_some() {
                return Err(BridgeError::new(
                    "failed oracle response also contains success fields",
                ));
            }
            let error = self
                .error
                .ok_or_else(|| BridgeError::new("failed oracle response has no error"))?;
            Ok(NormalizedResponse::Error(ErrorResponse {
                case_id: self.case_id,
                error,
            }))
        }
    }
}

#[derive(Debug)]
pub struct OracleClient {
    child: Child,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: BufReader<ChildStdout>,
    protocol_version: u32,
    finished: bool,
}

impl OracleClient {
    pub fn spawn_uv() -> BridgeResult<Self> {
        let mut command = Command::new("uv");
        command.args([
            "run",
            "--project",
            "parity",
            "--frozen",
            "--no-sync",
            "--managed-python",
            "--no-build",
            "python",
            "parity/oracle.py",
        ]);
        command.current_dir(repository_root());
        Self::spawn(command)
    }

    pub fn spawn(mut command: Command) -> BridgeResult<Self> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = command
            .spawn()
            .map_err(|error| BridgeError::context("spawn Python oracle", error))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| BridgeError::new("Python oracle stdin was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BridgeError::new("Python oracle stdout was not piped"))?;
        let mut client = Self {
            child,
            stdin: Some(BufWriter::new(stdin)),
            stdout: BufReader::new(stdout),
            protocol_version: 0,
            finished: false,
        };
        let hello: Hello = client.read_message("oracle hello")?;
        if hello.kind != "hello" {
            return Err(BridgeError::new(format!(
                "expected oracle hello, received kind `{}`",
                hello.kind
            )));
        }
        if hello.protocol_version != PROTOCOL_VERSION {
            return Err(BridgeError::new(format!(
                "unsupported oracle protocol version {}, expected {PROTOCOL_VERSION}",
                hello.protocol_version
            )));
        }
        if hello.service != SERVICE_NAME {
            return Err(BridgeError::new(format!(
                "unexpected oracle service `{}`, expected `{SERVICE_NAME}`",
                hello.service
            )));
        }
        client.protocol_version = hello.protocol_version;
        Ok(client)
    }

    pub fn protocol_version(&self) -> u32 {
        self.protocol_version
    }

    pub fn child_id(&self) -> u32 {
        self.child.id()
    }

    pub fn evaluate(
        &mut self,
        requests: &[OracleRequest],
    ) -> BridgeResult<Vec<NormalizedResponse>> {
        let mut identities = HashSet::with_capacity(requests.len());
        for request in requests {
            request.validate_identity()?;
            if !identities.insert(request.case_id.as_str()) {
                return Err(BridgeError::new(format!(
                    "duplicate oracle case id `{}` in one batch",
                    request.case_id
                )));
            }
        }

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| BridgeError::new("Python oracle stdin is closed"))?;
        for request in requests {
            serde_json::to_writer(&mut *stdin, request)
                .map_err(|error| BridgeError::context("serialize oracle request", error))?;
            stdin
                .write_all(b"\n")
                .map_err(|error| BridgeError::context("write oracle request delimiter", error))?;
        }
        stdin
            .flush()
            .map_err(|error| BridgeError::context("flush oracle request batch", error))?;

        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            let wire: WireResponse = self.read_message("oracle response")?;
            let response = wire.normalize()?;
            let received = response.case_id().unwrap_or("<none>");
            if received != request.case_id {
                return Err(BridgeError::new(format!(
                    "oracle response order mismatch: expected `{}`, received `{received}`",
                    request.case_id
                )));
            }
            responses.push(response);
        }
        Ok(responses)
    }

    pub fn evaluate_einsum(
        &mut self,
        requests: &[EinsumRequest],
    ) -> BridgeResult<Vec<NormalizedResponse>> {
        let mut identities = HashSet::with_capacity(requests.len());
        for request in requests {
            request.validate_identity()?;
            if request.operation != Operation::Einsum {
                return Err(BridgeError::new(format!(
                    "einsum request `{}` has non-einsum operation",
                    request.case_id
                )));
            }
            if !identities.insert(request.case_id.as_str()) {
                return Err(BridgeError::new(format!(
                    "duplicate oracle case id `{}` in one batch",
                    request.case_id
                )));
            }
        }

        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| BridgeError::new("Python oracle stdin is closed"))?;
        for request in requests {
            serde_json::to_writer(&mut *stdin, request)
                .map_err(|error| BridgeError::context("serialize einsum oracle request", error))?;
            stdin
                .write_all(b"\n")
                .map_err(|error| BridgeError::context("write oracle request delimiter", error))?;
        }
        stdin
            .flush()
            .map_err(|error| BridgeError::context("flush oracle request batch", error))?;

        let mut responses = Vec::with_capacity(requests.len());
        for request in requests {
            let wire: WireResponse = self.read_message("oracle response")?;
            let response = wire.normalize()?;
            let received = response.case_id().unwrap_or("<none>");
            if received != request.case_id {
                return Err(BridgeError::new(format!(
                    "oracle response order mismatch: expected `{}`, received `{received}`",
                    request.case_id
                )));
            }
            responses.push(response);
        }
        Ok(responses)
    }

    pub fn replay_json(&mut self, json: &str) -> BridgeResult<NormalizedResponse> {
        let envelope = serde_json::from_str::<serde_json::Value>(json)
            .map_err(|error| BridgeError::context("parse replay JSON", error))?;
        let operation = envelope
            .get("operation")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| BridgeError::new("replay JSON has no string operation"))?;
        let response = if operation == "einsum" {
            let request = serde_json::from_value::<EinsumRequest>(envelope)
                .map_err(|error| BridgeError::context("parse einsum replay JSON", error))?;
            self.evaluate_einsum(&[request])?
        } else {
            let request = serde_json::from_value::<OracleRequest>(envelope)
                .map_err(|error| BridgeError::context("parse replay JSON", error))?;
            self.evaluate(&[request])?
        };
        response
            .into_iter()
            .next()
            .ok_or_else(|| BridgeError::new("replay produced no response"))
    }

    pub fn replay_file(&mut self, path: impl AsRef<Path>) -> BridgeResult<NormalizedResponse> {
        let path = path.as_ref();
        let json = fs::read_to_string(path).map_err(|error| {
            BridgeError::context(&format!("read replay file `{}`", path.display()), error)
        })?;
        self.replay_json(&json)
    }

    pub fn shutdown(mut self) -> BridgeResult<ExitStatus> {
        self.stdin.take();
        let status = self
            .child
            .wait()
            .map_err(|error| BridgeError::context("wait for Python oracle", error))?;
        self.finished = true;
        Ok(status)
    }

    fn read_message<T>(&mut self, context: &str) -> BridgeResult<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut line = String::new();
        self.stdout
            .read_line(&mut line)
            .map_err(|error| BridgeError::context(&format!("read {context}"), error))?;
        if line.is_empty() {
            return Err(BridgeError::new(format!(
                "{context} ended before a JSONL message"
            )));
        }
        serde_json::from_str(&line)
            .map_err(|error| BridgeError::context(&format!("decode {context}"), error))
    }
}

impl Drop for OracleClient {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        self.stdin.take();
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            Ok(None) | Err(_) => {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
        self.finished = true;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParityConfig {
    pub cases: u32,
    pub seed: u64,
    pub max_elements: usize,
}

impl ParityConfig {
    pub fn from_env() -> BridgeResult<Self> {
        let cases = std::env::var("CANDLE_EINOPS_PARITY_CASES").ok();
        let seed = std::env::var("CANDLE_EINOPS_PARITY_SEED").ok();
        let max_elements = std::env::var("CANDLE_EINOPS_PARITY_MAX_ELEMENTS").ok();
        Self::from_overrides(cases.as_deref(), seed.as_deref(), max_elements.as_deref())
    }

    pub fn from_overrides(
        cases: Option<&str>,
        seed: Option<&str>,
        max_elements: Option<&str>,
    ) -> BridgeResult<Self> {
        let cases = parse_override(cases, DEFAULT_CASES, "case count")?;
        let seed = parse_override(seed, DEFAULT_SEED, "seed")?;
        let max_elements = parse_override(max_elements, DEFAULT_MAX_ELEMENTS, "element bound")?;
        if cases == 0 || cases > MAX_CASES {
            return Err(BridgeError::new(format!(
                "parity case count must be in 1..={MAX_CASES}, got {cases}"
            )));
        }
        if max_elements == 0 || max_elements > MAX_ELEMENTS {
            return Err(BridgeError::new(format!(
                "parity element bound must be in 1..={MAX_ELEMENTS}, got {max_elements}"
            )));
        }
        Ok(Self {
            cases,
            seed,
            max_elements,
        })
    }

    pub fn proptest_config(self) -> ProptestConfig {
        ProptestConfig {
            cases: self.cases,
            rng_seed: RngSeed::Fixed(self.seed),
            ..ProptestConfig::default()
        }
    }
}

fn parse_override<T>(value: Option<&str>, default: T, name: &str) -> BridgeResult<T>
where
    T: std::str::FromStr,
    T::Err: fmt::Display,
{
    match value {
        Some(value) => value.parse().map_err(|error| {
            BridgeError::context(&format!("invalid parity {name} `{value}`"), error)
        }),
        None => Ok(default),
    }
}

pub fn persist_replay<T>(path: impl AsRef<Path>, request: &T) -> BridgeResult<()>
where
    T: Serialize,
{
    let displayed_path = path.as_ref();
    let path = if displayed_path.is_absolute() {
        displayed_path.to_path_buf()
    } else {
        repository_root().join(displayed_path)
    };
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| {
            BridgeError::context(
                &format!("create replay directory `{}`", parent.display()),
                error,
            )
        })?;
    }
    let json = serde_json::to_string(request)
        .map_err(|error| BridgeError::context("serialize replay request", error))?;
    fs::write(&path, &json).map_err(|error| {
        BridgeError::context(&format!("write replay file `{}`", path.display()), error)
    })?;
    eprintln!(
        "parity replay saved to {}; run python3 .github/scripts/test_python_parity.py --replay-file {}",
        displayed_path.display(),
        displayed_path.display()
    );
    Ok(())
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("parity/runner is two levels below the repository root")
        .to_path_buf()
}
