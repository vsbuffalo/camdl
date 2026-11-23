---
paths:
  - "rust/crates/store/**"
  - "rust/crates/store-derive/**"
  - "rust/crates/ir/src/model_hash.rs"
  - "rust/crates/ir/src/model_hash/**"
  - "rust/crates/cli/src/camdl_store.rs"
  - "rust/crates/cli/src/resolve.rs"
description: CAS / run-identity — what re-keys a run, what is presentation, and the required reading before touching either
---

# CAS and run identity

Applies to anything that feeds a run id: a new field of an identity type
(`sim_store::SimConfig`, a fit problem's or run's identity), an output-affecting
CLI flag.

## Required reading

- The run-store proposal's "API 1: inputs and identity" and its "Keys" section
  (`docs/dev/proposals/2026-11-20-run-store-api.md`): the input classes, the
  `Identity` derive and the encoding.
- `rust/crates/store/src/hash.rs` and `identity.rs`: the one encoder and the
  identity types it hashes.
- `rust/crates/cli/src/camdl_store.rs`: camdl's ambient identity and the
  compiled model's stored part.
- `rust/crates/ir/src/model_hash.rs`: the model's structural hash, which camdl
  prints and compares models by and which no identity holds.

## The rule

A field that **changes stored bytes is identity** — it must re-key. A
**re-encoding of the same values is presentation** — strip it.

Re-keys are deliberate and version-bumped, never collateral. A silent re-key
invalidates the cache and breaks the reproducibility claim the store exists to
make; a missed re-key silently serves a stale result for different inputs, which
is worse.
