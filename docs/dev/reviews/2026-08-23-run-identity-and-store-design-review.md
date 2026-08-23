# Run-identity and store-protocol design review

Date: 2026-08-23 Scope: `rust/crates/runid` + `rust/crates/runid-derive`
(identity/hashing and store/staging halves), the five CLI identity surfaces
(`fit/cas.rs`, `profile_cas.rs`, `pfilter_cas.rs`, `survey_cas.rs`,
`sim_ensemble_cas.rs`, `Stage::identity_payload`), and every claim/commit call
site. Method: two independent deep-review agents (one per half), each given the
full incident record as established fact, plus a same-day identity-gap audit
across all CAS-writing commands. Line references are as of
`worktree-refactor-cli` at `95df25ba`. Verification status: the three most
load-bearing claims were independently re-verified before this document was
written — `RunStatus::Failed` is written nowhere in the workspace;
`commit_atomic`'s `AlreadyCompleted` arm deletes the staged directory with no
byte comparison (`store.rs:305-308`, its comment _assumes_ identicality);
`ensure_finite` applied to a built `serde_json::Value` cannot detect non-finites
(`json!`/`to_value` collapse NaN/Inf to `Null` before the gate runs). Other
line-level claims are the reviewing agents', spot-checked but not exhaustively
re-traced.

Prior context: the missing-knob incident record is gh#514 (5 chain-start flags),
gh#540 (13 sampler flags), 2026-08-23 (6 fit flags, fixed in
`1759cbf2`..`95df25ba`), and the same-day audit (~10 gaps across
simulate/profile/pfilter; gh#726, gh#730).

Proposal that acts on this review:
`docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md`.

---

## Part 1 — Identity and hashing

**Verdict: the core is well designed; the system is sound-but-incomplete at
exactly one seam, and the CLI has grown a third identity regime the design never
sanctioned.** Across four incident waves, zero defects were in the hashing core
and all were in the CLI layer above it — either a knob hashed from a
pre-override value (the resolution split) or a knob absent from a
hand-enumerated payload (exclude-by-default surfaces).

### The hashing core is sound

- Framing is rigorous: length-prefixing kills concatenation ambiguity
  (`hash.rs:26-29, 200-209`); enums write a fixed-width `u32` index before
  payload; `Option` is presence-tagged; maps sort keys; `usize` is
  width-normalized; per-type domain tags + per-type `schema_version` prevent
  cross-type coincidences. The `run_id` root obeys the same rules
  (`kind.rs:79-88`).
- The two float policies are correct and type-enforced: `f64` is deliberately
  not `ContentAddressed` (`hash.rs:316-317`); a field must choose `FiniteF64`
  (finite, `-0.0` normalized) or the raw-bits IR policy matching
  `ConstExpr::PartialEq`.
- Version layering is coherent: `HASH_VERSION` (whole store) > per-type
  `schema_version` > `ir_version` inside `ModelDigest`.
- Stability tests are real: `canonical_encoding_is_pinned` pins 12 literal
  digests including two full `run_id`s; `model_golden_hash` pins the IR
  encoding; `macro_eq.rs` pins macro output ≡ hand impl.

Genuine weaknesses in the core's margins:

1. **Derived enums hash positional variant indices** (`.enumerate()` over
   declaration order, `runid-derive/src/lib.rs:97`). A mid-list insertion or
   reorder silently renumbers — and renumbering is not just churn: a new run of
   variant-now-at-index-1 hashes identically to an old stored run of the variant
   that used to be at index 1, a cross-generation stale hit. The hand impls
   handle this with permanent explicit indices and append-only comments
   (`ir_hash.rs:503-529`); the derive has no `#[run_input(index = N)]`, and the
   pin tests cover only one variant per enum. Latent, not live.
2. **`module_path!()`-based type tags** mean moving a type between modules
   silently re-keys every leaf of that type. Safe direction, but silent —
   inconsistent with "re-keys are deliberate, never collateral".
3. **`ensure_finite` is vacuous at four call sites** because it is applied to an
   already-built `serde_json::Value`: `pfilter_cas.rs:73, 90`,
   `survey_cas.rs:79, 89`, `sim_ensemble_cas.rs` blobs, `profile_cas.rs:75-76`.
   `fit/cas.rs` does it correctly (struct-level, pre-serialization). Two pfilter
   evals at `beta=NaN` vs `beta=Inf` would collide today if upstream validation
   admits them. This is also the clearest exhibit that the hand-rolled regime
   drifts under copy-paste.

### Three regimes, one unsanctioned

- **Regime 1 — `#[derive(RunInput)]`** (include-by-default, provenance opt-out,
  compile error on unhashable fields). Used by `runid::inputs` types; the
  simulate/batch `config` level. Never produced a missing-knob incident —
  structurally it cannot.
- **Regime 2 — canonical-JSON digest of a whole serde struct minus named
  exclusions.** Sanctioned by the crate doc and _needed_: `skip_serializing_if`
  gives skip-if-default hash stability so the fit config can evolve without
  re-keying old runs — something the derive cannot and should not express.
  `fit_config_blob_hash` (`fit/cas.rs:304-315`) is this regime done right:
  serialize everything, then `remove("stages")/("fit_seeds")/("output_dir")` —
  include-by-default with named, documented subtractions.
- **Regime 3 — hand-enumerated `json!` subsets.** Outside the documented
  contract, exclude-by-default, and the home of the missing-knob half of every
  incident wave: `Stage::identity_payload`'s PGAS/PMMH/Mh/Nuts arms destructure
  with `..` and enumerate _included_ fields (the invariant lives in comments:
  "`burnin_dt` … MUST be listed here", twice); the
  pfilter/survey/sim_ensemble/profile-method blobs are enumerated subsets — the
  "hash-a-recipe antipattern" `runid`'s own doc names. The audit's
  profile/pfilter gaps are precisely fields these enumerations never listed.

The bypass is **mostly adoption debt with one small design gap**: regime 2 has
no owned helper in `runid`, so every call site hand-rolls canonicalize +
finiteness gate + subtract, and hand-rolls drift. The missing piece is modest —
a `runid::canonical_config_hash<T: Serialize>(value, exclude: &[&str])` with the
struct-level finiteness gate inside. The extension-dimension need
(`sweeps`/`iterations` excluded so resumable runs share a prefix identity) is
already representable as "excluded here, folded separately as `target_length`";
only the subtractive spelling was missed. Damning adoption detail: the composed
leaf-input types the original design specified (`PfilterEvalInput`,
`SurveyInput`, `FitStageInput`, `ProfilePointInput`, `SyntheticObsInput`,
`TrajectoryInput`) are constructed nowhere outside `runid`'s own tests — the
step that was to bind them grew regime 3 instead.

### The identity-path/run-path split is the top structural flaw

`build_simulate_cas_sink` loads and resolves the model a **second time**
(`main.rs:2428-2432`) and hand-mirrors a subset of the run path's overrides into
the hashed copy — obs anchors (post-gh#616) and params yes, `--integrator` and
`--param-vec` no. The fit side is the same disease: `apply_cli_overrides` exists
only to force override-writes before the claim, and `CliStageOverrides`' doc
admits the residual hazard ("a flag added to `FitRunArgs` and forgotten here
still bypasses"). Three recurrences (gh#616, `--integrator`, `--param-vec`) of
one shape — identity computed from a different value-set than the run consumes —
is a missing seam, not three bugs. The fix is a per-command resolved-input type
whose `identity()` and `into_job()` consume the same value (ownership makes
divergence unrepresentable); see the proposal.

### `ir_hash.rs`: sound today, wrong failure mode tomorrow

When the IR grows a field the answer is **silent omission**: every impl reads
`self.field` rather than exhaustively destructuring, so a new field compiles
clean, hashes nothing, and two same-version models differing only in that field
collide. The team knows the trap is live (the F23 comment in `main.rs:2441-2447`
warns two vintages would share a `run_id` over the excluded `quantities`). The
cheap fix: exhaustive destructuring
(`let Model { name, …, quantities: _, contrasts: _ } = self;`) in every struct
impl, so a new IR field is a compile error forcing the hash-or-exclude decision
at birth. Byte-neutral. `normalize_for_hash` itself is principled and narrow
(two fields, idempotence tripwire, gh#442 centralization); the per-impl
omissions scattered through the file become visible `field: _` entries under the
same fix. Enums are already safe (exhaustive `match`).

---

## Part 2 — Store write/staging protocol

**Verdict: sound core, incomplete contract.** The bottom half of `store.rs` —
durability ordering, rename atomicity, collision handling, PID-checked reclaim —
is genuinely well engineered, with the concurrency corner cases test-hooked. The
top half — the commit contract exposed to callers — is missing three doors
(divergence detection, overwrite, post-completion augmentation), and because
those doors don't exist, five callers each improvised their own policy around
the store.

### State machine and crash-safety

On-disk states derived by `lookup`/`resolve_claim_dir`: Miss; `Stale(Corrupt)`
(quarantined, never blind-cleared); orphan partial (quarantined);
`Stale(Incomplete)` (same identity, `Running`; reclaimed on next claim); `Hit`
(Completed + exact-set); `Stale(OrphanFiles|SchemaDrift)`; `Collision`
(different identity — disambiguate, never touch).

- **`RunStatus::Failed` is dead code** — declared, never written. Clean failure
  and `kill -9` are indistinguishable on disk (both `Running`).
- **Dead-claimant recovery exists and is good**: PID liveness via
  `kill(pid, 0)`, reclaim serialized through a second `.reclaim` O_EXCL lock
  with post-acquire re-confirmation, takeover renames the new lock over the old
  so `.lock` is never observably absent, and `ReclaimRaceCompleted` refuses to
  wipe a leaf a concurrent claimant finalized. Best-built part of the system.
- **But recovery is claim-time only**, and the "store-open lifecycle sweep" the
  module doc twice defers to **does not exist** (no sweep of `.staging/`,
  `.quarantine/`, or stranded `.reclaim` serializers anywhere in the workspace).
  A crashed fit's leaf sits `Running` until the same identity is re-run; a
  stranded `.reclaim` is a documented permanent wedge whose documented remover
  is nowhere.
- **PID liveness is host-local; the lock is filesystem-scoped.** The doc
  motivates reclaim with a cluster re-submit — but on shared storage a remote
  holder is probed against the wrong process table (reclaim-under a live holder,
  or wedge on a dead one whose PID is locally live). If cluster use is real, the
  lock needs host + process-start-time, not PID alone.
- Atomicity is real (per-file `sync_all`, staging-dir fsync, rename,
  destination-parent fsync; mode-B commit point is the `run.json.tmp` rename).
  Crash mid-commit leaves an unswept `.staging/*` orphan and no visible leaf.

### The `AlreadyCompleted` contract is the central flaw

`commit_atomic` → `AlreadyCompleted` → `remove_dir_all(staging)` → return the
incumbent (`store.rs:305-308`). No comparison between staged and incumbent bytes
— at the one moment the store holds both sides of its own "same key ⟺ same
bytes" claim, it checks nothing. Discard-identical-recompute is the right
default; discard-without-checking converts every identity hole in every caller
into a permanent silent wrong answer. This is the mechanism behind the
`--integrator`, `--dates`, and `--event-log` audit findings (and the
`batch.rs:1319-1323` comment claiming `--force` rescues event_log is false on
both halves — simulate never consults force, and batch's force-recompute still
ends at the same discard).

The cheap check exists: at the discard site the staged manifest (name → SHA-256,
already computed) and the incumbent's `record.artifacts` (already parsed) are
both in memory. Equal → benign dedup. Shared name with differing digest →
identity bug, fail loudly, quarantine the staging dir as evidence. Staged strict
superset → the "leaf gains an artifact" case. Zero extra I/O; runs are
seeded-deterministic, so a divergence _is_ a bug by definition.

### Force: one flag, four behaviors, none of them "overwrite"

Complete map: batch consults force in `should_run` (recompute — but commit still
discards, so force cannot change stored bytes); simulate threads force into a
sink that never reads it (dead flag); fit and survey skip the lookup and then
die on `claim_streaming`'s `AlreadyCompleted` (hard error, documented as known);
pfilter and profile have no force at all. The store has no overwrite door, and
five call sites each discovered that and coped differently. Force must become a
store-level `WritePolicy`.

### The sink abstraction is one seam in name, five in fact

`begin_resolved_write` genuinely unifies record construction and write
mechanics. But read/skip/force policy was left caller-side and five parallel
implementations grew: `CasSink::should_run` (the only correct one); `SimSink`
(no skip, leans on the discard semantics); fit's inline lookup; profile's own
cache scan; survey's bare `landscape.tsv.exists()` — no identity gate, no
`Completed` gate, no exact-set, so a short-hash collision at that path serves
the wrong survey as "cached", and a crash between `write_landscape` and
`finalize` leaves a file the next run calls cached while the store would call
the leaf `Stale(Incomplete)`. pfilter treats _any_ claim failure as a warning
and moves on (first run's `loglik.toml` stays under the shared key).

### Children and augmentation

The declared-children design (recorded-not-hashed; exact-set stops at child
boundaries) is right on paper. In practice: the obs "child" is not a child
artifact (raw unsynced writes, no `run.json`, a made-up child id pointing at no
record — nothing can ever validate it); adding `--obs` over a pre-existing leaf
discards the staged record that declared the child, then flips the leaf
`Stale(OrphanFiles)` via the orphan scan (safe direction, but a routine
operation invalidates a good artifact), and on a `Hit` nothing checks whether
the requested obs child exists (batch `--obs` over a non-obs store silently
produces no obs). `event_log` as an own-file is honest but exposes the missing
operation: there is no way for a completed leaf to gain an artifact. The general
fix is a store-level `augment` operation under the leaf's lock, plus promoting
obs to a real child leaf written declare-first.

### Residual concurrency notes

O_EXCL lock is authoritative; lookup→claim TOCTOU closed by the under-lock
re-check; reclaim double-entry and false-orphan races closed with dedicated test
hooks. Residuals: the cross-host PID problem; unsweepable stranded serializers;
a latent mixed-mode hole (`commit_atomic`'s `Reclaim` arm removes a
same-identity `Running` leaf without checking for a live `.lock` — harmless only
while no artifact kind uses both modes); pfilter/profile writing payload files
with bare `std::fs::write` and `let _ =` error discards, bypassing the fsync'd
`StreamClaim::write`, so a leaf can finalize `Completed` with a payload file
that was never written.
