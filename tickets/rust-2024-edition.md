---
id: rust-2024-edition
title: Migrate both crates to Rust 2024
status: done
priority: p0
dependencies: [candle-011-baseline]
related: []
scopes: [tooling, macros, tests]
shared_scopes: [ticketing]
paths: []
tags: [candle-0.11-modernization]
---
# Migrate both crates to Rust 2024

## Goal

Build both published crates with the Rust 2024 edition and an explicit, verified MSRV.

## Gap

Both crates still declare the Rust 2021 edition, and the repository previously had no declared minimum supported Rust version.

## Work

Set both manifests to edition 2024, retain the verified Rust 1.94 MSRV established by the Candle 0.11 baseline, and run the complete workspace gates on both 1.94 and stable.

## Acceptance

Cargo metadata reports edition 2024 and rust-version 1.94 for both crates; formatting, tests, Clippy, documentation, and packaging pass without edition regressions.

## Refs

Rust 1.94 is the lowest version verified with Candle 0.11 on aarch64 macOS. Raising the floor further would not unlock additional language features needed by this library.
