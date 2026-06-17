# CAS/run-input type consolidation: resolver choke points, domain backends, and typed identity

- Date: 2026-06-17
- Status: Active implementation RFC. **Supersedes the design/sequencing of**
  `2026-06-16-input-surface-and-cas-unification.md`; that doc is retained as the
  input-surface map, guarantees ledger, and adversarial-review record.
- Scope: input identity, CAS write path, batch/design migration, backend/method
  typing
- Non-goal: timeline/schedule runtime refactor (`gh#233`)
- Already landed in PR #244 (do not re-do): all of Layer 0 — the wall-clock
  watchdog removed for a deterministic `pf_max_substeps` budget (C0.2);
  batch/compare `deny_unknown_fields` + `[stages.*]` typo rejection; the
  anti-drift pinned-encoding golden; AND **C0.1a** (batch `[design.*]` atomic
  writes + a `run.json` completion-marker hit authority — closes the
  partial-write-as-hit hole, no re-key). The remaining design work is **C0.1b**
  (route the design store through `runid`'s CasStore), folded into PR D. The
  next _new_ PR is **PR B** (backend domain types).

## Summary

The `runid` core is the right center of gravity: resolved values derive
`RunInput`, floats enter identity only through `FiniteF64`, maps are canonical,
hashes are domain-separated, and CAS writes are atomic. The remaining risk is
the transition layer around it. Several user surfaces and legacy paths can still
compute identity, paths, or cache hits adjacent to `runid` instead of through
one canonical resolver/store seam.

The target invariant is:

```text
user surface(s)
    -> one resolved artifact input
    -> one typed RunInput value / factored levels
    -> one run_id
    -> one resolved CAS write (atomic commit OR streaming claim/finalize)
```

This proposal does **not** collapse every input shape into one struct. That was
the earlier overreach. Raw CLI/TOML structs are presentation surfaces; resolved
identity structs are content-addressed values. The consolidation point is the
resolver boundary, not serde/clap itself.

The highest priority fixes are:

1. Batch `[design.*]`: stop trusting `traj.tsv.exists()` as a cache-hit
   authority. SHIPPED as C0.1a (atomic write + `run.json` completion marker, no
   re-key); the full route-through-`runid`-`CasStore` migration (C0.1b) is
   deferred to PR D.
2. Quarantine legacy `hashing.rs`; no new run identity construction outside
   `RunInput`/`ContentAddressed`.
3. Introduce one resolver per artifact kind, returning a common
   `ResolvedArtifact` shape consumed by the only legal writer.
4. Replace ambiguous backend naming with domain backend types: `ForwardBackend`
   and `InferenceBackend`.
5. Make the fit method registry typed (`FitAlgorithm`, `InferenceBackend`) and
   keep capability validation as a separate layer, not as a replacement for
   impossible-state types.
6. Add an input-surface differential harness: semantic changes re-key;
   provenance/execution-only changes do not.

## Verification snapshot

Claims below were checked against the `worktree-cas-cleanup` tree on 2026-06-17.

```text
$ rg -n "enum Backend|FitBackend|StageBackend|backend" \
    rust/crates/cli/src/args rust/crates/cli/src/fit \
    rust/crates/runid/src/inputs.rs rust/crates/cli/src/run_meta.rs \
    rust/crates/cli/src/batch.rs -S

Confirmed:
- args::types::Backend is the 3-variant forward backend.
- run_meta::Backend is the 2-variant fit-stage backend.
- runid::inputs::Backend is another 3-variant identity backend.
- fit.toml currently uses both args::types::Backend for [config].backend and
  run_meta::Backend for [stages.*].backend.
```

```text
$ rg -n "design|cache hit|atomic|CasSink|batch|design" \
    docs/dev/proposals/2026-06-16-input-surface-and-cas-unification.md \
    rust/crates/cli/src/batch.rs rust/crates/cli/src/*cas*.rs \
    rust/crates/cli/src/resolve.rs -S

Confirmed:
- the proposal identifies legacy hashing.rs for batch [design.*];
- batch.rs has comments at the design cache-hit/write path stating the full fix
  is the CasSink/runid atomic migration;
- CasSink already exists for normal batch cells.
```

```text
$ rg -n "struct Capabilities|bitflags|Capabilities|required_capabilities|\
check_model_capabilities|validate_combo|METHODS" rust/crates -S

Confirmed:
- sim::Capabilities is a model-feature x backend system;
- fit::methods::METHODS is a separate algorithm x backend registry;
- check_model_capabilities has a separate inference capability table.
```

These facts are the reason this proposal chooses resolver/store consolidation
and domain backend types instead of a single backend enum or a broad
capabilities-only rewrite.

## Problem

The current architecture has a strong intended invariant, but the invariant is
not yet expressed as a single choke point. New code can still:

- parse a user surface locally and forget to feed a semantic field into
  identity;
- add a display field to `run.json.inputs` and assume it is hashed;
- compute or reuse a legacy path/hash instead of going through `runid`;
- decide a cache hit by checking for a data file rather than a completed CAS
  record;
- append optional artifacts to an already committed CAS leaf;
- use a "backend" type whose domain is unclear.

These are agent-hostile surfaces: a local change can look plausible while
breaking the global identity invariant.

### Current backend smell

There are at least three backend value types:

- `crate::args::types::Backend`: `Gillespie | ChainBinomial | Ode`. This is a
  forward simulation backend.
- `crate::run_meta::Backend`: `ChainBinomial | Ode`. This is a fit/inference
  backend.
- `runid::inputs::Backend`: `Gillespie | ChainBinomial | Ode`. This is a
  resolved identity backend for trajectory simulation.

The two sets are not all redundant. The two-variant fit backend is doing real
work: it prevents `gillespie` from being represented as a fit-stage backend. But
the shared name `Backend` hides the domain boundary, and the current proposal
sentence "collapse the two `Backend` enums into one" is too broad.

The correct consolidation is:

```rust
pub enum ForwardBackend {
    Gillespie,
    ChainBinomial,
    Ode,
}

pub enum InferenceBackend {
    ChainBinomial,
    Ode,
}
```

Use `ForwardBackend` on the CLI/config forward-simulation surfaces; the resolver
maps it into the existing `runid::inputs::Backend` _identity_ type (which is
**not** renamed — see Layer 3). Use `InferenceBackend` on the fit-stage,
pfilter, survey, and profile surfaces. Conversions from `ForwardBackend` to
`InferenceBackend` are explicit and fallible (`Gillespie` -> error).

### Current capability smell

The existing `sim::Capabilities` system is valuable, but it is not a drop-in
replacement for typed backend domains. It currently answers "does this model
require features this backend can provide?" It does not by itself answer "is
this fitting algorithm valid on this backend?"

The existing docs already separate the axes:

```text
model-feature x backend       -> sim::Capabilities
algorithm x backend           -> METHODS registry
model-feature x algorithm     -> scattered/ad-hoc gates
```

Therefore the right next step is not "replace the restricted inference backend
enum with capabilities." The right next step is:

- keep impossible states unrepresentable at parse/identity boundaries;
- make the algorithm/backend registry typed;
- then express richer algorithm requirements as typed requirements/provisions
  inside that registry.

## Design

### Layer 0: fix the active correctness holes first

#### C0.1 Batch design CAS hit/write authority

`batch [design.*]` must stop using bare `traj.tsv` existence as a cache-hit
authority. A partial or aborted write can leave a file that the next run treats
as valid. This violates the CAS invariant.

Rule:

```text
File existence is never a cache-hit authority. A hit requires a completed,
parseable record; writes are atomic (a crash leaves the old state or nothing,
never a truncated artifact read back as valid).
```

This splits into a shipped correctness fix and a deferred full migration.

**C0.1a — shipped in PR #244 (correctness fix, no re-key).** Closes the
partial-write-as-hit silent-wrong-answer within the existing legacy design-store
layout:

- `traj.tsv` and the `run.json` marker are written atomically (temp + rename),
  marker LAST — the final `traj.tsv` is always complete-or-absent;
- a design hit requires a parseable `run.json` completion marker
  (`design_cell_complete`), never bare `traj.tsv` existence; a partial cell
  re-runs. `plan_runs`' `traj_exists` stays as a necessary-but-not-sufficient
  prefilter the design path re-validates against the marker;
- old complete caches still hit (their marker parses); only partial ones re-run.

Acceptance (shipped): a `traj.tsv`-only cell and a truncated/unparseable marker
are NOT hits; a valid marker is (`design_cache_hit_requires_completion_marker`).

**C0.1b — deferred to PR D (full migration, re-keys the design store).** The
stronger end state: route design cells through `runid`'s `CasStore` so they are
first-class `runid` leaves (same `ResolvedArtifact`/`TrajectoryInput` identity,
atomic checksummed commit + `lookup`), retiring the legacy `designs/.../sims/…`
layout and the `traj.tsv`-existence planner entirely. This lands with the
resolver/store choke point (PR D); it changes the design-store _layout_ (a
deliberate, documented turnover — NOT a `runid` identity re-key), and only then
does the static gate below apply:

```text
$ rg -n "traj.tsv.*exists|exists\\(\\).*traj.tsv|metadata\\(.*traj" rust/crates/cli/src/batch.rs
```

must find no design cache-hit decision (the `plan_runs` prefilter is gone once
design routes through `CasStore::lookup`). Until C0.1b lands, C0.1a's
marker-authority is the correctness guarantee.

#### C0.2 Delete or demote wall-clock output influence

The wall-clock watchdog must not affect whether a stored scientific artifact is
produced. It is machine-dependent and outside typed input identity. The
deterministic substep budget is the right compute-blowup guard.

End state:

- no output-producing CAS path consults `CAMDL_PF_WALLCLOCK_TIMEOUT_S`;
- if an interactive wall-clock guard remains, it is UI-only and cannot commit or
  suppress a CAS result;
- `pf_max_substeps` stays a typed execution budget. It need not be part of the
  completed artifact identity if the policy is: a stricter budget may abort
  before producing a result, but an existing completed result is still valid for
  the same scientific inputs.

The proposal and docs must state that cache policy explicitly.

### Layer 1: one resolved artifact seam per artifact kind

Introduce a common resolved shape consumed by the store writer:

```rust
pub struct ResolvedArtifact {
    pub kind: ArtifactKind,
    pub levels: Vec<LevelId>,
    pub run_id: ContentHash,
    pub display_inputs: serde_json::Value,
}
```

Each artifact kind has exactly one resolver:

```rust
resolve_trajectory(...)
resolve_synthetic_obs(...)
resolve_pfilter_eval(...)
resolve_survey(...)
resolve_profile_point(...)
resolve_fit_stage(...)
resolve_projection(...)
resolve_event_log_child(...)
```

The resolver owns the transformation from user-facing surface values to resolved
identity values:

- raw `f64` -> `FiniteF64`;
- path -> `DataDigest` / `ContentHash`;
- base seed -> process seed;
- requested obs alignment -> resolved obs alignment;
- stage config -> canonical digest;
- parent artifact path -> `ArtifactRef`.

The only legal writer consumes `ResolvedArtifact`, and it must admit **both**
store write modes — the store has two today: `commit_atomic` (the caller hands
over a finished artifact set) and `claim_streaming(...).finalize(...)` (runners
like `fit` claim a leaf and stream output into it as they go). A single
`commit_resolved(store, resolved, artifacts)` is therefore too narrow as
written; widen it to a resolved-writer API:

```rust
enum WriteMode {
    Atomic(Artifacts),  // hand over a finished artifact set
    Streaming,          // claim the leaf, stream into claim.dir(), then finalize
}

fn begin_resolved_write(store, resolved: &ResolvedArtifact, mode: WriteMode)
    -> Result<ResolvedWrite, CasError>;
```

The load-bearing invariant is **"all writes are resolved first"** — every write
goes through `ResolvedArtifact`, so identity is computed once (by the resolver)
before any bytes land — NOT "all writes use one atomic artifact list." A
`commit_resolved_atomic(store, &resolved, artifacts)` convenience may wrap the
`Atomic` mode, but it must not be the _only_ door, or it excludes `fit`'s
streaming path (the failure mode of the earlier single-driver sketch).

This does not require CLI and TOML to share one parse type. It requires every
surface to converge through the same resolver before writing.

Acceptance:

- no command computes a run path or `run_id` directly;
- new artifact kinds cannot call `CasStore::commit_atomic` without first
  constructing the resolved artifact shape;
- `run.json.inputs` is built from `ResolvedArtifact::display_inputs`, never used
  as identity.

### Layer 2: quarantine legacy identity construction

The `runid` crate is the identity mechanism. Legacy `hashing.rs` is a migration
island, not a second identity API.

Rename or module-document it as legacy:

```rust
mod legacy_identity_do_not_extend;
```

or keep the file name but add allowlist tests that prevent new call sites.

Forbidden for run identity outside the allowlist:

```rust
Sha256::new()
serde_json::to_string(...) as a hash payload
canonical_params(...)
model_hash(...)
sim_hash(...)
scen_hash(...)
fit_content_hash(...)
```

Artifact byte digests are allowed. Run identity must go through
`RunInput`/`ContentAddressed`.

Acceptance:

```text
$ rg -n "model_hash\\(|sim_hash\\(|scen_hash\\(|canonical_params\\(|fit_content_hash\\(" \
    rust/crates/cli/src rust/crates/runid/src
```

matches only the explicit legacy allowlist and tests for the legacy island.

### Layer 3: backend domains, not one enum

Rename and consolidate by domain.

```rust
// CLI/config-facing forward simulation backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum ForwardBackend {
    Gillespie,
    ChainBinomial,
    Ode,
}

// Fit/inference backend. No Gillespie until an inference method actually
// supports a Gillespie process interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferenceBackend {
    ChainBinomial,
    Ode,
}
```

**`runid::inputs::Backend` is hash schema — do NOT casually rename it.** The
`RunInput` derive's domain-separation tag is `module_path!() :: TypeName`
(`runid-derive/src/lib.rs:55`), so renaming the _identity_ type changes its tag,
which changes its `content_hash`, which **re-keys every sim/pfilter/survey/
profile artifact** — a store-wide invalidation (the anti-drift golden in PR #244
pins `Backend::ChainBinomial` precisely to catch this). For a clarity-only
refactor that is not worth it. Options, in order of preference: (a) keep the
`runid` identity type named `Backend` and introduce `ForwardBackend` only as the
CLI/config-facing type that the resolver maps _into_ `runid::inputs::Backend`;
(b) add an explicit, stable `#[run_input(type_tag = "...")]` to the derive and
rename behind it; or (c) do it as a deliberate `HASH_VERSION` migration. Rule:
treat `runid` identity type _names_ as part of the hash schema, not as
free-to-rename Rust identifiers.

`run_meta::Backend` becomes `InferenceBackend` (a CLI/config value type — _not_
a `runid` identity type). It is serialized into the fit blob as a snake_case
string, so preserving its serde wire spelling keeps the blob byte-identical:
**this rename does not re-key.**

Fit surfaces that mean "stage backend" use `InferenceBackend`. Surfaces that
mean "generate a forward trajectory" use `ForwardBackend`.

Explicit conversions:

```rust
impl TryFrom<ForwardBackend> for InferenceBackend {
    type Error = BackendDomainError;

    fn try_from(b: ForwardBackend) -> Result<Self, Self::Error> {
        match b {
            ForwardBackend::ChainBinomial => Ok(Self::ChainBinomial),
            ForwardBackend::Ode => Ok(Self::Ode),
            ForwardBackend::Gillespie => Err(BackendDomainError::NotInferenceBackend),
        }
    }
}
```

This is the important type-system property: a fit-stage identity cannot contain
`Gillespie`. A future Gillespie inference method must add an explicit variant or
a new process interface; it cannot arrive by accident because someone reused the
forward enum.

#### Global fit `[config].backend`

The current global fit config uses the 3-variant forward backend. That is a
domain leak unless the field is strictly a forward-simulation default for
synthetic/initialization paths.

Choose one of these end states:

1. If it is an inference default, change it to `InferenceBackend`.
2. If it is only a forward-simulation default, rename it to make that explicit.
3. If stage backends are authoritative, remove/deprecate the global backend and
   require each stage to declare its backend.

Do not keep a field named simply `backend` in fit config if it can parse
`gillespie` but is later interpreted as an inference backend.

Acceptance:

```text
$ rg -n "pub backend: .*args::types::Backend|pub backend: .*ForwardBackend" \
    rust/crates/cli/src/fit
```

must have only intentionally forward-simulation fields, each documented as such.

### Layer 4: typed fit method registry

The method registry is currently stringly. Keep the registry, but type it:

```rust
pub enum FitAlgorithm {
    If2,
    Pgas,
    Pmmh,
    Mh,
    Pfilter,
    NlSbplx,
    NlBobyqa,
}

pub struct InferenceMethod {
    pub algorithm: FitAlgorithm,
    pub backend: InferenceBackend,
    pub category: MethodCategory,
    pub status: MethodStatus,
    pub one_liner: &'static str,
    pub use_for: &'static str,
    pub status_note: &'static str,
    pub requirements: MethodRequirements,
}
```

`validate_combo` becomes typed internally:

```rust
pub fn validate_combo(
    algorithm: FitAlgorithm,
    backend: InferenceBackend,
) -> Result<&'static InferenceMethod, MethodError>;
```

String parsing remains at the surface only. Once parsed, no code should compare
algorithm/backend strings to decide behavior.

Acceptance:

```text
$ rg -n "algorithm ==|backend ==|match \\(algorithm, backend\\)|backend: \"|algorithm: \"" \
    rust/crates/cli/src/fit
```

should only find parser/display tests and the registry literal. Dispatch logic
uses typed enums.

### Layer 5: capabilities as requirements/provisions, not enum collapse

Capabilities are still useful. They should evolve inside the typed registry as
requirements, not replace the restricted `InferenceBackend` type.

Keep `sim::Capabilities` for model-feature requirements:

```rust
pub struct MethodRequirements {
    pub model_caps_allowed: sim::Capabilities,
    pub process_caps_required: InferenceProcessCaps,
}
```

Add a separate process capability vocabulary only if/when needed:

```rust
bitflags::bitflags! {
    pub struct InferenceProcessCaps: u32 {
        const FIXED_DT_TRANSITION_KERNEL = 1 << 0;
        const OBSERVATION_LOG_LIKELIHOOD = 1 << 1;
        const COMPLETE_DATA_LOG_DENSITY  = 1 << 2;
        const PATH_GRADIENTS             = 1 << 3;
        const DETERMINISTIC_TRAJECTORY   = 1 << 4;
    }
}
```

Do not overload `sim::Capabilities` with algorithm machinery concepts. Existing
flags like `BALANCE`, `OVERDISPERSION`, and `REAL_COMPARTMENTS` describe model
features, not inference algorithm interfaces.

This gives the future Gillespie story a clean path:

- adding Gillespie to inference is not "let the forward backend parse";
- it means adding an inference process implementation with typed provisions,
  adding registry rows for algorithms that can consume those provisions, and
  adding tests for the resulting likelihood/objective.

### Layer 6: run.json display inputs are not identity

`RunRecord` is not hashed. That is correct. It is also a trap.

Rule:

```text
No semantic field may be added to run.json.inputs unless it is either:
1. derived from an existing hashed RunInput field, or
2. simultaneously added to the relevant resolved RunInput type.
```

Display/provenance-only fields are allowed, but the differential harness must
pin that changing them does not change `run_id`.

Examples:

- label, argv, color, progress mode: provenance/execution, no re-key;
- model bytes, params, data bytes, seed, backend, dt, algorithm knobs, obs
  alignment, output schedule: semantic/artifact bytes, re-key;
- event log request: not an append to trajectory; either part of trajectory
  identity up front or a child artifact keyed by parent + event-log inputs.

### Layer 7: child artifacts for optional outputs

A committed CAS leaf is immutable. Optional artifacts must not be appended to an
existing leaf on a cache hit.

Every optional artifact is one of:

1. part of the original leaf identity and manifest;
2. a child artifact keyed on parent `run_id` plus its own inputs;
3. a non-CAS mirror outside the store.

Preferred end state:

- event log: child artifact;
- synthetic observations: child artifact;
- lineage projections/realizations: child artifacts;
- posterior predictive summaries and plots: child artifacts or non-CAS mirrors,
  depending on whether they are intended as reusable scientific artifacts.

Acceptance:

No command writes a new file into an already completed leaf directory without a
new CAS commit/record path.

### Layer 8: input-surface differential harness

Add tests that enumerate semantic/provenance/execution fields per artifact kind.

For each field:

```text
baseline input A
mutated input B

if field is semantic or artifact-byte-affecting:
    run_id(A) != run_id(B)

if field is provenance or execution-only:
    run_id(A) == run_id(B)
```

This is the guard that catches "field was parsed but not hashed" and "field was
hashed but should have been presentation-only."

Start with:

- trajectory;
- batch cell;
- pfilter eval;
- fit stage;
- profile point;
- synthetic obs child;
- event-log child when introduced.

The test should exercise the resolver, not hand-construct the `RunInput` value
directly; otherwise it misses the surface-to-identity seam where the real bugs
occur.

## Sequencing

### PR A: cheap correctness completion, no re-key where possible — SHIPPED (PR #244)

1. Batch `[design.*]` **C0.1a** atomic write + completion-marker hit authority
   (the full C0.1b runid routing is deferred to PR D).
2. Wall-clock output influence removed (C0.2).
3. Anti-drift encoding golden.
4. Unknown-key rejection + stage-key typo rejection.

All landed in PR #244.

### PR B: backend/domain type cleanup

1. Rename `args::types::Backend` -> `ForwardBackend` (CLI/config surface; not a
   `runid` identity type, so no re-key).
2. **Do NOT rename `runid::inputs::Backend`** — it is hash schema (Layer 3).
   Keep the identity type as-is; `ForwardBackend` is the CLI/config type the
   resolver maps _into_ it. A future rename, if wanted, rides a stable
   `type_tag` or a deliberate `HASH_VERSION` migration — never this PR.
3. Rename `run_meta::Backend` -> `InferenceBackend`, **preserving its snake_case
   serde spelling** (the fit blob stays byte-identical — no re-key).
4. Fix the fit global `[config].backend` domain leak by one of the Layer-3
   choices.
5. Add the explicit `TryFrom<ForwardBackend> for InferenceBackend` (fallible:
   `Gillespie` -> `BackendDomainError`).
6. Preserve all serde wire spelling.

Goal: **zero re-key.** This PR changes Rust type names and the CLI/config
surface, not the structural identity encoding. If any change would move a pinned
run-id golden (PR #244's anti-drift golden), stop — that means a `runid`
identity type was touched; back it out or escalate it to an explicit, documented
`HASH_VERSION` migration.

### PR C: typed method registry

1. Add `FitAlgorithm`.
2. Convert `METHODS` to typed entries.
3. Convert `validate_combo`, `status_note`, rendering, and dispatch to typed
   enums internally.
4. Keep parser/display string functions at the boundary.

### PR D: resolver/store choke point

1. Introduce `ResolvedArtifact`.
2. Move each command to its artifact resolver.
3. Make the store writer consume the resolved artifact in **both** modes
   (`begin_resolved_write` with `WriteMode::Atomic` / `WriteMode::Streaming`),
   so `fit`'s streaming path is covered, not just atomic commits.
4. Remove direct run path/hash construction from command bodies.

### PR E: differential harness and legacy quarantine

1. Add the input-surface differential harness.
2. Add the legacy hashing allowlist test.
3. Remove or rename remaining legacy identity helpers once no production
   call-sites remain.

## Non-goals

- Do not refactor scheduling/time-spine code here. That is `gh#233`.
- Do not merge raw CLI/TOML parse structs with resolved identity structs when
  fields require transformation (`f64`, paths, maps, seeds, obs alignment).
- Do not add `Gillespie` to inference merely because a capability table could
  reject it later. Unsupported states should stay unrepresentable at the fit
  backend type.
- Do not rewrite inference math or particle filters as part of CAS identity
  cleanup.

## Acceptance gates

### Static search gates

```text
rg -n "traj.tsv.*exists|exists\\(\\).*traj.tsv|metadata\\(.*traj" rust/crates/cli/src/batch.rs
```

This gate applies to **C0.1b** (PR D): once the design store routes through
`CasStore::lookup`, no design cache-hit decision reads file existence. It does
NOT apply to the shipped C0.1a, which deliberately keeps `plan_runs`'
`traj_exists` as a necessary-but-not-sufficient prefilter re-validated against a
`run.json` completion marker; C0.1a's gate is the behavioral one below ("A
partial `batch [design.*]` output file is not a cache hit").

```text
rg -n "model_hash\\(|sim_hash\\(|scen_hash\\(|canonical_params\\(|fit_content_hash\\(" \
    rust/crates/cli/src rust/crates/runid/src
```

Only legacy allowlist/test call-sites.

```text
rg -n "pub enum Backend|struct Backend|type Backend" rust/crates
```

No ambiguous `Backend` type remains **outside `runid::inputs::Backend`** (the
identity/hash-schema type, deliberately kept — Layer 3). Every other occurrence
is `ForwardBackend`/`InferenceBackend` or a more specific domain name.

```text
rg -n "algorithm ==|backend ==|match \\(algorithm, backend\\)|backend: \"|algorithm: \"" \
    rust/crates/cli/src/fit
```

No behavioral dispatch through raw strings outside parser/display tests and the
typed registry literal.

### Behavioral gates

- A partial `batch [design.*]` output file is not a cache hit.
- Unknown keys in every user-authored TOML config fail loudly.
- Mutating every semantic trajectory input changes `run_id`.
- Mutating provenance-only fields does not change `run_id`.
- A fit stage cannot represent `InferenceBackend::Gillespie`.
- A config with `backend = "gillespie"` in a fit-stage backend slot fails at
  parse/validation with a domain-specific error.
- Existing completed CAS leaves are never modified to append optional artifacts.

### Test gates

At minimum:

```text
cd rust && cargo test -p runid
cd rust && cargo test -p camdl-cli batch
cd rust && cargo test -p camdl-cli fit
cd rust && cargo test
```

If any step intentionally re-keys artifacts, include the before/after rationale
in the PR description and update pinned run-id goldens deliberately.

## Design rationale

### Why not one backend enum?

Because it makes an unsupported fit state representable. A forward backend and
an inference backend are different domains today. `Gillespie` is a valid forward
simulator; it is not a valid fit-stage backend in the current inference stack. A
single enum would force every fit path to remember to reject it later.

The type system can do better: fit-stage identity cannot contain the variant.

### Why not capabilities-only?

Capabilities are useful for validation, but they are not a substitute for domain
types. The existing capability system is model-feature oriented. A
capabilities-only fit backend would still parse and store unsupported
combinations, then rely on later validation to reject them. That is weaker than
making impossible combinations unrepresentable in resolved identity.

Use capabilities for extensibility and model-dependent gates. Use types for
domain boundaries.

### Why not one CLI/TOML/identity struct?

Because raw surfaces and identity values have different invariants.

- CLI/TOML wants `f64`, paths, optional defaults, aliases, and helpful parse
  errors.
- Identity wants `FiniteF64`, content digests, resolved seeds, canonical maps,
  and include-by-default hashing.

For the small subset that is genuinely parseable and content-addressable without
transformation, shared structs are fine. For transforming fields, the resolver
is the seam.

### Why `ResolvedArtifact`?

It forces the ordering:

```text
resolve first, hash second, commit third
```

Commands should not be able to write a stored result while bypassing identity.
The struct is intentionally boring: kind, levels, run_id, display inputs. Its
value is architectural, not algorithmic.

## Final target

After this work, the common path for stored artifacts is — **resolve first, then
write in whichever mode the artifact needs** (`commit_resolved_atomic` is the
convenience wrapper for the atomic mode; runners that stream use the
`begin_resolved_write(..., WriteMode::Streaming)` claim/finalize path):

```rust
// Atomic: a fully-computed artifact set (e.g. a forward trajectory).
let resolved = resolve_trajectory(surface, context)?;
let staged   = run_engine(...)?;
commit_resolved_atomic(&store, &resolved, staged)?;

// Streaming: a runner that writes into the claimed leaf as it goes (e.g. fit).
let resolved = resolve_fit_stage(surface, context)?;
let write    = begin_resolved_write(&store, &resolved, WriteMode::Streaming)?;
run_fit_into(write.dir(), ...)?;
write.finalize()?;

// Child artifact (keyed on parent run_id + its own inputs), atomic:
let resolved = resolve_event_log_child(parent_ref, event_log_surface)?;
commit_resolved_atomic(&store, &resolved, staged)?;
```

No command computes identity ad hoc. No CAS path trusts file existence. No
semantic input is display-only by accident. No unsupported fit backend is
representable in resolved identity. That is the level of type pressure needed
before starting the separate `gh#233` schedule/runtime consolidation.
