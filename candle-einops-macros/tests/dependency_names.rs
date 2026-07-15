use std::{path::Path, process::Command};

#[test]
fn normal_and_renamed_runtime_dependencies_compile() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .expect("macro crate has a workspace root");
    let target_dir = workspace_dir.join("target/dependency-name-fixtures");

    for fixture in ["normal", "renamed"] {
        let manifest = manifest_dir
            .join("tests/fixtures")
            .join(fixture)
            .join("Cargo.toml");
        let output = Command::new(env!("CARGO"))
            .args(["check", "--manifest-path"])
            .arg(&manifest)
            .env("CARGO_TARGET_DIR", &target_dir)
            .output()
            .expect("fixture cargo check should start");

        assert!(
            output.status.success(),
            "{fixture} dependency fixture failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
