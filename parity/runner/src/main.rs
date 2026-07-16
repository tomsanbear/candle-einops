use std::process::ExitCode;

use candle_einops_parity_runner::{BridgeResult, OracleClient};

fn run() -> BridgeResult<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let mut client = OracleClient::spawn_uv()?;
    let response = match arguments.as_slice() {
        [flag, json] if flag == "--replay-json" => client.replay_json(json)?,
        [flag, path] if flag == "--replay-file" => client.replay_file(path)?,
        _ => {
            return Err(candle_einops_parity_runner::BridgeError::usage(
                "usage: candle-einops-parity-runner (--replay-json JSON | --replay-file PATH)",
            ));
        }
    };
    println!(
        "{}",
        serde_json::to_string(&response).expect("normalized response is serializable")
    );
    let status = client.shutdown()?;
    if !status.success() {
        return Err(candle_einops_parity_runner::BridgeError::usage(format!(
            "Python oracle exited with {status}"
        )));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
