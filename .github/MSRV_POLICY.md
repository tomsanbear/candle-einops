# MSRV and dependency resolution policy

Both workspace crates declare Rust 1.94 as their minimum supported Rust version
(MSRV). CI runs the complete workspace test suite with Rust 1.94 and stable.
The workspace uses resolver 3, so Cargo selects versions compatible with the
active toolchain when dependency `rust-version` metadata is available.

`Cargo.lock` is intentionally untracked because both workspace members are
libraries. CI performs unlocked resolution so the Rust 1.94 job exercises a
consumer-compatible MSRV graph while stable exercises the newest compatible
graph. The dependency-policy job checks the same current, unlocked graph for
advisories, approved licenses, wildcard requirements, and unexpected sources.

This policy favors consumer compatibility and early ecosystem-drift detection
over byte-for-byte CI reproducibility. A dependency update that breaks either
toolchain or policy gate must be constrained or upgraded explicitly; it must
not be hidden with a broad exception. Advisory exceptions require a concrete
dependency path, rationale, and review date in `deny.toml`.

Release packaging also resolves unlocked, matching normal library consumers.
The generated package archives contain a resolution lockfile for verification,
but that generated `Cargo.lock` does not constrain downstream users.
