#!/usr/bin/env python3
"""Package, safely unpack, and test both workspace publication artifacts."""

from pathlib import Path
import json
import os
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CARGO = os.environ.get("CARGO", "cargo")


def run(arguments: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    command = [CARGO, *arguments]
    print(f"+ {' '.join(command)} (in {cwd})", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def package_versions() -> dict[str, str]:
    output = subprocess.run(
        [CARGO, "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    metadata = json.loads(output.stdout)
    return {package["name"]: package["version"] for package in metadata["packages"]}


def safe_unpack(archive: Path, destination: Path) -> Path:
    destination = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as package:
        members = package.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"artifact contains an unsupported link: {member.name}")
        package.extractall(destination)

    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"expected one package root in {archive}, found {roots}")
    return roots[0]


def test_artifact(package: Path, env: dict[str, str]) -> None:
    manifest = package / "Cargo.toml"
    run(
        ["test", "--manifest-path", str(manifest), "--all-targets", "--all-features"],
        cwd=package,
        env=env,
    )
    run(
        ["test", "--manifest-path", str(manifest), "--doc", "--all-features"],
        cwd=package,
        env=env,
    )


def test_downstream_consumers(runtime: Path, temporary: Path, env: dict[str, str]) -> None:
    runtime_path = json.dumps(str(runtime))
    consumers = [
        (
            "normal-consumer",
            f'candle-einops = {{ path = {runtime_path} }}',
            "use candle_core::{Device, Result, Tensor};\nuse candle_einops::einsum;",
        ),
        (
            "renamed-consumer",
            f'tensor-ops = {{ package = "candle-einops", path = {runtime_path} }}',
            "use candle_core::{Device, Result, Tensor};\nuse tensor_ops::einsum;",
        ),
        (
            "keyword-consumer",
            f'type = {{ package = "candle-einops", path = {runtime_path} }}',
            "use r#match::{Device, Result, Tensor};\nuse r#type::einsum;",
        ),
    ]
    source_template = """{imports}

fn main() -> Result<()> {{
    let matrix = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
    let transposed = einsum!("rows columns -> columns rows", &matrix)?;
    assert_eq!(transposed.to_vec2::<f32>()?, [[0., 3.], [1., 4.], [2., 5.]]);
    let product = einsum!("row inner, inner column -> row column", &matrix, &transposed)?;
    assert_eq!(product.to_vec2::<f32>()?, [[5., 14.], [14., 50.]]);
    let reduced = einsum!(".. column -> column", &matrix)?;
    assert_eq!(reduced.to_vec1::<f32>()?, [3., 5., 7.]);
    let square = Tensor::arange(0f32, 9f32, &Device::Cpu)?.reshape((3, 3))?;
    let diagonal = einsum!("index index -> index", &square)?;
    assert_eq!(diagonal.to_vec1::<f32>()?, [0., 4., 8.]);
    let weights = Tensor::new(&[1f32, 1.], &Device::Cpu)?;
    let nary = einsum!(
        "row inner, inner column, column -> row",
        &matrix,
        &transposed,
        &weights,
    )?;
    assert_eq!(nary.to_vec1::<f32>()?, [19., 64.]);
    Ok(())
}}
"""

    for name, runtime_dependency, imports in consumers:
        consumer = temporary / name
        (consumer / "src").mkdir(parents=True)
        candle_dependency = (
            'match = { package = "candle-core", version = "0.11" }'
            if name == "keyword-consumer"
            else 'candle-core = "0.11"'
        )
        (consumer / "Cargo.toml").write_text(
            f"""[package]
name = "{name}"
version = "0.0.0"
edition = "2024"
publish = false

[dependencies]
{candle_dependency}
{runtime_dependency}

[workspace]
""",
            encoding="utf-8",
        )
        (consumer / "src/main.rs").write_text(
            source_template.format(imports=imports), encoding="utf-8"
        )
        run(["run", "--manifest-path", str(consumer / "Cargo.toml")], cwd=consumer, env=env)


def main() -> None:
    versions = package_versions()
    run(["package", "--workspace"], cwd=ROOT)

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")).resolve()
    archives = {
        name: target_dir / "package" / f"{name}-{versions[name]}.crate"
        for name in ["candle-einops-macros", "candle-einops"]
    }

    with tempfile.TemporaryDirectory(prefix="candle-einops-artifacts-") as temporary:
        temporary_dir = Path(temporary)
        macro_dir = safe_unpack(archives["candle-einops-macros"], temporary_dir / "macro")
        runtime_dir = safe_unpack(archives["candle-einops"], temporary_dir / "runtime")

        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(temporary_dir / "target")
        env["RUSTDOCFLAGS"] = "-D warnings"

        print("Testing macro artifact first", flush=True)
        test_artifact(macro_dir, env)

        cargo_config = temporary_dir / ".cargo"
        cargo_config.mkdir()
        macro_path = json.dumps(str(macro_dir))
        (cargo_config / "config.toml").write_text(
            "[patch.crates-io]\n"
            f"candle-einops-macros = {{ path = {macro_path} }}\n",
            encoding="utf-8",
        )

        print("Testing runtime artifact against the unpacked macro artifact", flush=True)
        test_artifact(runtime_dir, env)
        print("Testing normal, renamed, and keyword downstream consumers", flush=True)
        test_downstream_consumers(runtime_dir, temporary_dir, env)


if __name__ == "__main__":
    main()
