# Performance harness

This standalone, unpublished crate owns measurement methodology. It deliberately
contains only an untracked plumbing scenario; mechanism benchmarks are added by
separate tickets.

Use the repository wrapper for every supported operation:

```console
python3 .github/scripts/run_benchmarks.py compile
python3 .github/scripts/run_benchmarks.py smoke
python3 .github/scripts/run_benchmarks.py run --filter rearrange/view-permute --output target/benchmarks/result.json
```

The wrapper pins Rust 1.94, uses this crate's committed lockfile, and shares the
root ignored `target/benchmarks` directory. `run` selects tracked scenarios by
substring. The foundation has none, so `smoke` opts into the untracked plumbing
fixture; downstream scenario tickets populate normal runs.

Each scenario has an immutable id, deterministic setup, a library operation, a
direct Candle reference, an untimed correctness check, and elements/bytes with
optional FLOPs. A sample synchronizes immediately before the clock starts and
after its black-boxed output is produced. The output remains alive through the
second synchronization.

The versioned JSON record is the automation contract. It contains paired
library/reference medians, a deterministic 95% bootstrap interval, their ratio,
work units, and a git/Rust/Candle/platform/device fingerprint. Criterion output
is useful for local inspection but is secondary and must not be parsed by
automation or committed.

CPU is the default backend. `--backend metal` and `--backend cuda` establish
mutually exclusive feature builds for later device-specific instrumentation;
this foundation does not claim GPU timing, allocation, kernel, or enqueue
metrics. Profilers should wrap the supported command rather than bypassing its
locked build. There is no universal timing baseline or CI regression gate.
