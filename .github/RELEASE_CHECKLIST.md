# 0.2.0 release checklist

Run these commands from a clean checkout before publishing:

- `cargo +1.94.0 test --workspace --all-targets --all-features`
- `cargo +stable test --workspace --all-targets --all-features`
- `cargo +stable fmt --all -- --check`
- `cargo +stable clippy --workspace --all-targets --all-features -- -D warnings -A dead-code -A clippy::excessive-precision -A clippy::identity-op -A clippy::map-flatten`
- `RUSTDOCFLAGS="-D warnings" cargo +stable doc --workspace --all-features --no-deps`
- `RUSTDOCFLAGS="-D warnings" cargo +stable test --doc --workspace --all-features`
- `python3 .github/scripts/validate_ci_policy.py`
- `python3 .github/scripts/validate_artifact_policy.py`
- `python3 .github/scripts/validate_einsum_release.py`
- `python3 .github/scripts/validate_python_parity_policy.py`
- `python3 .github/scripts/validate_performance_harness_policy.py`
- `python3 .github/scripts/test_validate_performance_harness_policy.py`
- `uv sync --frozen --project benchmarks/reporting`
- `uv run --frozen --project benchmarks/reporting python .github/scripts/test_generate_performance_report.py`
- `uv run --frozen --project benchmarks/reporting python .github/scripts/generate_performance_report.py --check`
- `actionlint .github/workflows/ci.yml`
- `cargo deny --all-features check`
- `python3 .github/scripts/test_python_parity.py`
- `python3 .github/scripts/test_published_artifacts.py`
- `cargo +stable test -p candle-einops --lib --test behavior --test errors --test einsum_semantic_matrix`

Confirm that both artifacts contain their declared dual-license texts and that
the artifact gate executes normal, renamed, and keyword-alias downstream
consumers. Publish `candle-einops-macros` 0.2.0 first, wait for its index entry,
then publish `candle-einops` 0.2.0. Its exact macro dependency protects the
private generated-code ABI.

Tagging, publishing, and creating a GitHub release are deliberate manual steps.
