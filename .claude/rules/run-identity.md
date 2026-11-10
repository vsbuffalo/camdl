---
paths:
  - "rust/crates/runid/**"
  - "rust/crates/cli/src/resolve.rs"
  - "rust/crates/cli/src/fit/cas.rs"
description: CAS / run-identity — what re-keys a run, what is presentation, and the required reading before touching either
---

# CAS and run identity

Applies to anything that feeds a `run_id`: a new `SimConfig` / `FitConfigV2`
field, a new identity level, an output-affecting CLI flag.

## Required reading

- The `runid` crate doc (`rust/crates/runid/src/lib.rs`) — the two hashing paths
  and the version stack.
- `rust/crates/cli/src/resolve.rs` — the factored model / config / params /
  scenario / seed levels; `normalize_for_hash`, which they share, is in
  `rust/crates/runid/src/inputs.rs`.
- `rust/crates/cli/src/fit/cas.rs` — the fit canonical-JSON hash.

## The rule

A field that **changes stored bytes is identity** — it must re-key. A
**re-encoding of the same values is presentation** — strip it.

Re-keys are deliberate and version-bumped, never collateral. A silent re-key
invalidates the cache and breaks the reproducibility claim the store exists to
make; a missed re-key silently serves a stale result for different inputs, which
is worse.
