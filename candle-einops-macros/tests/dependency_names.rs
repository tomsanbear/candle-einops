use std::{path::Path, process::Command};

#[test]
fn downstream_dependency_and_hygiene_fixtures_execute() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .expect("macro crate has a workspace root");
    let target_dir = workspace_dir.join("target/dependency-name-fixtures");

    for fixture in [
        "normal",
        "renamed",
        "hygiene-ignored-len",
        "hygiene",
        "keyword-alias",
    ] {
        let manifest = manifest_dir
            .join("tests/fixtures")
            .join(fixture)
            .join("Cargo.toml");
        let output = Command::new(env!("CARGO"))
            .args(["run", "--quiet", "--manifest-path"])
            .arg(&manifest)
            .env("CARGO_TARGET_DIR", &target_dir)
            .output()
            .expect("fixture cargo run should start");

        assert!(
            output.status.success(),
            "{fixture} dependency fixture failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
