# Model identity is gradient-independent: strip derived-gradient maps from `normalize_for_hash`

Status: proposed\
Area: CAS / run-identity (`runid`, `cli/src/resolve.rs`)

## Problem

A run's model identity — the `ir` level of `run_id`, computed as
`model.content_hash()` over `normalize_for_hash(model)`
(`runid/src/inputs.rs:192`, `cli/src/resolve.rs`) — folds in the
compiler-emitted gradient maps: `Transition::hash_into` hashes `rate_grad` and
`rate_state_grad` (`runid/src/ir_hash.rs:325-330`), and the observation/σ²
analogues hash `projection_state_grad`, `ic_grad`, `sigma_sq_grad`.
`normalize_for_hash` strips only `output.format` and
`simulation.time_semantics`.

These maps are **redundant** in an identity hash: each is deterministic autodiff
of the transition rates / observation arguments, over the model's parameters,
tables, and forcings — all of which are already hashed. Two models with
identical dynamics therefore always have identical gradients; including the maps
makes identity strictly finer-grained than the semantics warrant, sensitive to a
_compile detail_ that cannot affect any output.

This is latent until gradient emission becomes optional. It does with
`camdlc --no-state-grad` (gh#439): the same model compiled lean (for `simulate`
/ `mh`, which never read the state-Jacobian) vs full (for `nuts` on `ode`, which
does) now hashes to two different model identities. The consequence is that the
runtime cannot dispatch lean compilation by method (gh#439 fix A2) without
silently re-keying every `simulate` / `batch` / `predict` / non-NUTS-fit leaf in
the store — a collateral re-key the `runid` crate forbids.

## Decision

Treat the compiler-derived gradient maps as **presentation, not identity** — the
same class as `output.format`. Drop them from the IR content hash itself
(`runid::ir_hash`, the `ContentAddressed for Model` tree) by not folding them
into `hash_into`: transition `rate_grad` / `rate_state_grad`; each likelihood
`Diffable`'s `grad` / `proj_grad`; overdispersion `sigma_sq_grad`;
`projection_state_grad`; model `ic_grad`. Bump the hashing-schema version `SV`
(the runid-stack version governing _what is hashed_, distinct from
`ir/VERSION`).

This lives at the `content_hash` layer rather than
`resolve.rs::normalize_for_hash` because the batch identity paths
(`pfilter_cas`, `survey_cas`, `sim_ensemble_cas`) compute their model level via
`ModelDigest::from_model` on the _raw_ model, bypassing `normalize_for_hash`
entirely — a normalize-only strip would leave batch/pfilter/survey
gradient-dependent, a silent gap. `content_hash` is the shared substrate every
identity path routes through.

The IR _content_ on disk is unchanged (goldens keep their maps — this touches
the hash computation only, not serialization), so **no golden regeneration** and
no `ir/VERSION` bump. Model identity becomes a function of the semantic model
(rates, params, structure, observation forms) and is invariant to which
gradients were emitted.

## Why this is safe (information-preserving)

The stripped fields carry no information beyond what remains hashed: gradients
are a pure function of `(rates, params, tables, forcings)`. There is no model
with identical rates/params/tables and different gradient maps. `DEUnsupported`
markers (a gradient the compiler could not emit) are likewise derived — they
depend on the rate structure and the estimated-parameter set, both hashed — and
matter to the fit-time capability gate, not to output identity. So stripping
loses nothing and removes the false distinction.

## Re-key

Blanking hashed fields changes every model hash → a run_id re-key. This is a
**deliberate, version-bumped** re-key, not collateral: bump the `runid` version
stack (the run_id version, not `ir/VERSION`); flip the existing
gradient-emission distinctness test (it currently asserts lean vs full IR
produce _different_ model hashes; it must now assert they produce the _same_
model hash); add a run_id-stability test pinning the new key on a reference
model.

Back-compat: a non-goal at alpha. Existing store leaves under old keys are not
deleted but will not be found under new keys; runs recompute once. After this,
run_id is stable across all future gradient-emission changes (A2, the low-rank
Jacobian, an adjoint path) rather than re-keying on each.

## Unblocks

With model identity gradient-independent, gh#439 fix A2 (runtime dispatches
`--no-state-grad` for every method except `nuts`+`ode`, with the flag folded
into `ir_cache_key`) is run_id-neutral and drops in cleanly.
