# Run-identity and store-contract hardening

Date: 2026-08-23 Status: draft for maintainer review Audit ref:
`docs/dev/reviews/2026-08-23-run-identity-and-store-design-review.md`

## Problem

One defect class has now produced four incident waves: a knob that changes what
a run computes or stores is absent from its `run_id`, so a cache hit serves a
result computed under different settings — the silent-wrong-answer mode this
store exists to prevent. gh#514 (5 flags), gh#540 (13), 2026-08-23 (6 fit flags,
fixed in `1759cbf2`..`95df25ba`), and the 2026-08-23 audit (~10 more across
simulate/profile/pfilter). Four waves of one class is a shape problem, not a
vigilance problem.

Two independent deep reviews (see Audit ref) reached the same verdict: the
hashing core and the store's durability/reclaim machinery are **well engineered
and stay**. The defects concentrate in two missing contracts:

1. **Identity is assembled per command, exclude-by-default, from values that can
   diverge from what the run consumes.** The CLI never adopted
   `#[derive(RunInput)]` (zero uses outside `runid`); four `Stage` identity arms
   and the pfilter/survey/ensemble/profile blobs hand-enumerate included fields;
   and `simulate` resolves its model twice, mirroring only a subset of overrides
   into the hashed copy (`--integrator`/`--param-vec` missed — the third
   recurrence of the divergent-resolution shape after gh#616).
2. **The store's commit contract has no doors for divergence, overwrite, or
   augmentation.** `AlreadyCompleted` silently discards freshly staged bytes
   without comparing them to the incumbent; `--force` has four different
   behaviors at five call sites, none of them "overwrite"; a completed leaf
   cannot gain an artifact, so obs/event_log grew workarounds; skip/force policy
   lives caller-side in five parallel implementations (one correct).

## What this proposal does NOT do

No rewrite of `hash.rs`, `float.rs`, the derive, `kind.rs`, the reclaim/lock
machinery, or the pinned goldens. No change to the canonical encoding. Regime 2
(canonical-JSON digest with named subtractions, skip-if-default stability for
`FitConfigV2`) remains sanctioned — it expresses config evolution the derive
should not.

## Design

### S1. Divergence check at commit (store)

At the `AlreadyCompleted` discard site, compare the staged manifest (name →
SHA-256, already computed) against the incumbent's `record.artifacts` (already
parsed): equal → dedup as today; any shared name with a differing digest → new
`CasError::DivergentRecompute { file, ours, theirs }`, staging dir
**quarantined** (evidence), loud failure; staged strict superset with equal
shared digests → route to S4's augment (until S4 lands: report and dedup). Runs
are seeded-deterministic, so a divergence is by definition an identity bug or
nondeterminism — both must be loud. ~50 lines, zero extra I/O, no schema change,
no re-key. **This lands first and alone if nothing else does**: it converts the
entire missed-re-key class from silent to loud at first occurrence, with the
file named.

### S2. `WritePolicy` — skip/force policy moves inside the seam (store)

```rust
enum WritePolicy { Reuse, Force }
enum WriteVerdict { CacheHit(Box<RunRecord>, PathBuf), MustRun(WriteTicket) }
fn begin_resolved_write(store, root, resolved, meta, mode, policy)
    -> Result<WriteVerdict, CasError>;
```

`Force` becomes the store's one overwrite door: quarantine-then-reclaim a
`Completed` same-identity leaf under the existing collision-aware machinery
(never across identities). All six call-site policies collapse into it:
`CasSink::should_run` delegates; fit's force/resume hard-error, survey's bare
`exists()` check (an identity-unaware cache that can serve the wrong survey),
profile's parallel scan, simulate's dead force flag, and pfilter's
claim-failure-as-warning all become matches on `WriteVerdict`. A new command
cannot get skip/force wrong because the only way to write is to receive a
verdict. No re-key.

### S3. RAII claim guard + the promised sweep (store)

`ResolvedClaim`/`StreamClaim` gain `Drop`: if not finalized, write
`status: Failed` (the state exists and is currently written nowhere) and remove
`.lock` — the IR-cache lock in `util.rs` already models this pattern. PID
reclaim stays as the backstop for `kill -9` and the `process::exit` sites (which
bypass `Drop`; their migration to error returns is the separately-planned fit
orchestrator refactor). Add the store-open lifecycle sweep the module doc
already promises but which does not exist: clear dead-PID
`.lock`/`.reclaim`/`.lock.new`, orphaned `.staging/*`, and mark reclaim-eligible
`Running` leaves `Failed`. No re-key.

### S4. `augment` + real obs children (store)

Store-level `augment(leaf, id, name, bytes)` under the leaf's lock: fsync the
file, atomically rewrite `run.json` with the extended manifest. This is the
missing operation behind the event_log loss and the obs `Stale(OrphanFiles)`
flip. Obs becomes a real child leaf with its own `run.json`, written
declare-first (the API enforces the ordering). `children`/`artifacts` are
recorded-not-hashed, so no re-key. Largest store item; last.

### I1. Resolve-once seam per command (identity)

Per command, one resolution function produces one value that both the hash and
the run consume — ownership makes divergence unrepresentable:

```rust
pub struct ResolvedSimulate {
    model: ir::Model,        // anchors substituted, --integrator/--param-vec applied
    config: SimConfig,
    params: ResolvedParams,
    scenario: ResolvedScenario,
    seeds: SeedPlan,
    provenance: RunProvenance,   // display only, never hashed
}
pub fn resolve_simulate(args: SimulateArgs) -> Result<ResolvedSimulate, ResolveError>;
impl ResolvedSimulate {
    pub fn identity(&self) -> ResolvedTrajectory; // pure, reads only self
    pub fn into_job(self) -> SimulateJob;         // consumes the SAME value
}
```

Simulate first (it has the live divergences); the immediate fix for
`--integrator`/`--param-vec` (apply the overrides in `build_simulate_cas_sink`,
mirroring the gh#616 anchor fix) lands ahead of the full seam, with a
differential test asserting the run path's and identity path's resolved
model/params agree. The fit analogue finishes what `apply_cli_overrides`
started: `resolve_fit_stage` takes the post-override `Stage` by ownership and
`*StageOpts::from_stage` becomes the only constructor from that same value.
Default-flag runs re-key nothing; runs using the previously-missed flags
correctly split — that is the fix, not collateral.

### I2. `canonical_config_hash` helper + subtractive `Stage` arms (identity)

`runid::canonical_config_hash<T: Serialize>(value: &T, exclude: &[&str])
-> Result<ContentHash, CasError>`
— the struct-level finiteness gate runs **inside**, pre-serialization (fixing
the four call sites where `ensure_finite` is applied to an already-built `Value`
and is therefore vacuous), exclusions are named and subtractive. The four
enumerated `Stage::identity_payload` arms (PGAS/PMMH/Mh/Nuts) convert to
`to_value(self)` minus their extension dimensions (`sweeps`/`iterations`, plus
`n_trajectories` for PGAS) — the same pattern `fit_config_blob_hash` and the
IF2/PFilter/Nlopt arms already use. The pfilter/survey/ensemble/ profile blobs
migrate onto the helper. Exclude-by-default becomes unwritable by accident.
**Re-keys**: fit stages of the four converted variants (the subtractive field
set differs from the enumerations), plus profile leaves (which absorb the
audit's `--rw-sd`/`--init`/`--condition-from`/ `--pf-max-substeps` keying in the
same pass — one re-key, not two), plus gh#730's `--dt-check-strict` resolved
threshold. All in one flagged batch.

### I3. Exhaustive destructuring in `ir_hash.rs` (identity)

Every struct impl destructures exhaustively
(`let Model { name, …, quantities: _, contrasts: _ } = self;`) so a new IR field
is a compile error at the moment it is born, forcing the hash-or-exclude
decision. Converts the silent-omission failure mode (two same-version models
differing in a new field colliding) into a compiler stop. Byte-neutral, no
re-key.

### I4. Small hardening (identity)

`#[run_input(index = N)]` on the derive (or a full-variant pin test per derived
enum) to close the positional-variant-index reorder hazard; a stable explicit
tag string replacing `module_path!()` if any module reshuffle is planned. Delete
the never-constructed composed leaf-input structs (`PfilterEvalInput`,
`SurveyInput`, `FitStageInput`, `ProfilePointInput`, `SyntheticObsInput`,
`TrajectoryInput`) — wiring them would re-key those kinds for no behavioral
gain; the shipped level factorings are sound and I2's helper is the honest seam.
Per the delete-on-sight rule.

## Sequencing and re-key inventory

| Phase                  | Items                                                                                                                                                                                                  | Re-keys                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| 1 (campaign-safe, now) | S1 divergence check; simulate `--integrator`/`--param-vec` parity + differential test; pfilter `--condition-from` (hash-neutral for unconditioned runs); I3 destructuring; vacuous-`ensure_finite` fix | nothing for existing valid runs; flag-using runs split correctly                |
| 2                      | S2 WritePolicy + call-site collapse; S3 guard + sweep                                                                                                                                                  | none                                                                            |
| 3 (one flagged batch)  | I2 helper + subtractive Stage arms + profile/pfilter blob migration + profile gap keying + gh#730                                                                                                      | four Stage variants' fit stages; all profile leaves; conditioned pfilter leaves |
| 4                      | S4 augment + obs child leaves; I1 full seam (simulate, then fit); I4 hardening                                                                                                                         | none (I1: default runs unchanged)                                               |

Phase 3 is the batch the `StageCommon` reshape (from the 2026-08-23 fit-CLI
review) should ride in, so mh/nl-* re-key once more at most, not repeatedly.
Every re-key lands with a `run_id`-stability test pinning what does **not**
change, per `.claude/rules/run-identity.md`.

## Decisions

- **`--dates` on the `SimEnsemble` artifact: normalize out, not key.** The
  ensemble is rendered from a date-free buffer (matching the `Sim` leaves);
  `--dates` remains presentation for the `-o` mirror. Decided per the
  presentation rule.
- **`batch [obs] enabled`: keyed** (folded into the `config` level via
  `TrajectoryCtx`) rather than detect-and-rerun — an obs-bearing leaf is a
  distinct artifact; detection papers over identity. Decided; S1 backstops it
  meanwhile.
- **pfilter claim failure is an error, not a warning.** Decided; falls out of
  S2's verdict type.

## Named follow-ups (genuinely the maintainer's)

- **gh#731: cluster/shared-store locking.** The `.lock` records PID only; PID
  liveness is host-local. If fits run on shared storage across hosts, the lock
  needs host + process-start-time. Blocked on the maintainer confirming whether
  cross-host store sharing is a real deployment; recommendation if yes: extend
  the lock record (no re-key — locks are not identity).
- **Timing of Phase 3's re-key** relative to the active outbreak-fitting
  campaign: the batch invalidates profile and four-variant fit-stage caches.
  Recommendation: land Phases 1–2 now; hold Phase 3 until the current fits land
  or the maintainer waves it through.
