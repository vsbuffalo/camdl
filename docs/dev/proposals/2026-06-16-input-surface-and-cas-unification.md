# Input surface mapping and input-addressed identity unification

Date: 2026-06-16 Status: Draft — REVISED after a 2026-06-16 adversarial review
(§11). Diagnosis (§1–§3) stands; the prescription is narrowed: D1's
"structural-on-the-surface-struct, zero transcription" is **infeasible** (the
identity layer deliberately excludes `f64`/`IndexMap` from `ContentAddressed`)
and is corrected to "dedup the surface, keep the transcribing resolve"; D3 (one
`build` driver) is demoted to the existing shared substrate; the cheap path
(C2/C3/C4 + the differential test, zero re-key) is the first PR. Builds on:
`2026-05-31-content-addressed-run-identity.md` (the foundational CAS design)
Issues: gh#241 (CLI↔TOML struct duplication), gh#156 (OutputView, the first
shared struct), gh#147 (legacy-store migration)

## Why this document exists

The original goal was that _every_ input that can change a computed artifact
flows through one typed system, so that content-addressed-storage (CAS)
invalidation is automatic and bug-free rather than discipline-dependent. The
foundational proposal
([2026-05-31](2026-05-31-content-addressed-run-identity.md)) specified that
system: a three-stage pipeline
`Raw CLI/TOML --Resolve--> Vec<RunInput> --hash--> run_id`, with the discipline
"enumerate the complete input set, default everything in; over-invalidation is
cheap, under-invalidation is a silent wrong answer."

The _identity_ half of that system — the `runid` crate, its
`#[derive(RunInput)]` macro, the factored per-level digests, content-not-path
digesting, the finiteness gate, presentation normalization — was built and is
sound. The _surface_ half was not. CLI argument structs (clap-only) and
config-file structs (serde-only) remain two disjoint hierarchies that model the
same concepts three and four times over (gh#241), the `Resolve` step exists as
scattered per-command functions rather than one enforced seam, and a legacy
hashing island (`hashing.rs`) still backs part of `batch`. None of these is
currently producing a wrong hash, but together they mean the safety property —
_every output-affecting input is hashed_ — holds today by author vigilance, not
by construction. That is the gap this document maps and proposes to close.

This document (1) maps the **complete** input surface with an exhaustiveness
argument; (2) classifies what is architecturally sound versus the actual gap;
and (3) states the design that closes the gap — include-by-default identity, a
shared resolve/identity substrate, and a two-test completeness gate — with the
load-bearing decisions recorded (§7, as revised by §11) and a short list of
items still open. §1.1 states up front, precisely, what the design guarantees
and what it does not, because over-claiming a caching guarantee is itself a
silent-wrong-answer risk.

## 0. Terminology: we are input-addressed, not content-addressed

We have loosely called this "CAS." The accurate term is **input-addressed**. A
leaf's address is `run_id = hash(resolved input set)` (`kind.rs:79`), and the
store _path_ is keyed by that input hash (`layout.rs:122`) — never by the output
bytes. We _do_ record output content hashes (`FileChecksum.digest` per output
file, `record.rs:51`; `ArtifactRef.digest` for consumed upstreams,
`inputs.rs:183`), but those are for verification and lineage, not addressing.
Conventional CAS (git, IPFS) is _output_-addressed: path = hash of contents. We
are not that. The foundational proposal's phrase "content-addressed run
identity" is only defensible read as "the content hash of the _input set_" —
which is exactly input-addressing. The docs should say input-addressed;
"content-addressed" should be reserved for the recorded output digests.

## 1. The three layers (and why they cannot be one)

The clean intuition — "one struct that is the CLI flag, the TOML key, _and_ the
cache key" — is right about the destination and wrong about the mechanism. It
conflates three genuinely distinct concerns:

1. **Surface** — how input is _expressed_. CLI flags (clap), TOML keys (serde).
   Wants ergonomics: defaults, `conflicts_with`, env fallbacks, short flags.
   This is presentation.
2. **Resolution** — surface → canonical semantic values. A real, _lossy_,
   non-identity function. The cases that prove it cannot be skipped:
   - `--seed 5` resolves to `process_seed = mix_cell_seed(5, point_idx, rep)`.
     Hashing the base seed would alias a lone run and a sweep-point — a
     documented silent-wrong-answer (`runid/src/inputs.rs:97-111`). The hashed
     value (`process_seed`) does not exist on the surface at all.
   - `--table foo=path.csv` resolves to a content digest. The path is
     provenance; the _bytes_ are identity (`runid/src/inputs.rs:84-88`, built in
     `fit/cas.rs:246`).
   - `--time-format auto` resolves to a concrete `numeric`/`date` choice that
     reinterprets the same data bytes into different observations (`2026-05-31`
     proposal §Resolve).
   - the requested `obs_alignment` resolves _per algorithm_ to the value that
     actually drives the posterior (gh#189, `fit/cas.rs:370`).
   - an omitted flag and an explicit-at-default flag must hash _equal_ (the
     `skip_serializing_if` machinery, `fit/config_v2.rs:304,315`).
3. **Identity** — the resolved values + content digests that determine the
   artifact, hashed. This is `runid::inputs` (`SimConfig`, `FitDigest`, …).

The monad the original intuition senses is real, but it is the **pipeline**, not
a struct: `Surface --resolve--> Identity --hash--> run_id`, where `resolve` is
the bind. Fields that pass through resolution unchanged (`backend`, `dt`,
`t_start`, `t_end`) _can_ live in one dual-derive struct shared by the surface
and a thin identity wrapper — that is exactly what gh#241/gh#156 propose. Fields
that resolution transforms (seed, tables, alignment, time-format) _structurally
cannot_. So the unification target is "one struct per _unchanged_ concept, one
total resolver for the rest," not "one struct for everything." Separating
surface from identity is the correct architecture, not the failure — the `Seed`
example alone vindicates it.

### 1.1 What the design guarantees — and what it does not

The goal is to move "every output-affecting input is hashed" from author
vigilance toward a machine-checked guarantee. It is worth being exact about
which parts the _types_ prove, which only _tests_ can catch, and which are out
of reach in principle — over-claiming here is itself a silent-wrong-answer risk.

**The types guarantee (compile-time, structural):**

- _Identity ≠ surface._ Only the resolved `Identity` is `ContentAddressed`;
  there is no hash instance for a raw surface struct, so `hash(surface)` does
  not typecheck. You can hash only the resolved value.
- _No forgotten field within a struct._ The `RunInput` derive folds every field
  include-by-default; a field whose type is not `ContentAddressed` is a compile
  error. You cannot silently omit a field that is _in_ the struct.
- _No unfilled identity field._ A resolved/level struct is built as a literal,
  so adding an identity field is a compile error until `resolve` produces it
  (the resolved→hash direction).
- _Sound composition._ A downstream leaf's identity embeds its upstreams'
  `run_id`s (a Merkle DAG); the transitive closure is captured by construction,
  with length-prefixed, type-tagged framing so distinct level sequences cannot
  alias, and `HASH_VERSION`/`schema_version` making any turnover an explicit,
  versioned act.

**Only tests can catch (detection, not prevention):**

- _Honest provenance._ That a `#[run_input(provenance)]` field truly does not
  affect output — the derive excludes it whether the claim is true or not.
  Caught only by the differential output test (§4.2): run the model with the
  field perturbed, assert the bytes are identical.
- _Resolve correctness._ That `resolve` actually threads a transforming surface
  input (`--seed → process_seed`) into a level — a `resolve` unit test, not a
  type.
- _No accidental re-key._ That a refactor or the dedup does not silently move a
  `run_id` — the run_id-stability golden (§4.2).
- _Output stability._ That a fixed identity still reproduces the same bytes —
  `camdl verify` (§10). Types cannot _prevent_ nondeterminism (RNG consumption
  order, parallel-reduction associativity); they only let us _detect_ divergence
  after the fact.

**Out of reach (needs runtime, or unprovable in principle):**

- _Hermeticity_ — that the declared surface is the _whole_ input. An undeclared
  channel (the `CAMDL_PF_WALLCLOCK_TIMEOUT_S` env leak, G2; an unsandboxed file
  read) is invisible to both the types and the completeness tests. It is closed
  by _removing_ undeclared channels: the model side is hermetic by construction
  (the IR is a term of camdl's pure DSL, and `read()`-pulled files are inlined
  into the hashed IR), and the env leak is a runtime fix (C4), not a type one.
- _Universal reproducibility_ — "rerunning anywhere, anytime yields the same
  bytes" is a claim about machines and future executions, not a proposition
  inside the type system. It can be _falsified_ (by `camdl verify`) but never
  _proved_ — Popper, not Curry–Howard.

The invariance axis follows directly: provenance values (labels, paths,
`output_dir`, `--format`, argv, thread count) may vary freely without changing
identity; semantic values must change it; and the one part of that split the
types cannot self-check — whether a provenance claim is honest — is exactly what
the differential test exists to verify.

## 2. The complete input surface (verified)

### 2.1 How we know this is exhaustive

The surface is **closed by construction**, and the closure has exactly three
doors:

- **Every CLI flag is a clap struct field.** clap is the only argument parser;
  there is no hand-rolled flag matching. Verified:
  `rg "derive\(.*(Parser|Subcommand|Args)" rust/crates/cli/src` matches in
  exactly two files (`main.rs`, `args/mod.rs`). The only argv that bypasses
  typed parsing is two deliberate `trailing_var_arg` escape hatches —
  `Passthrough` (forwards to `camdlc`, `main.rs:381`) and `If2Args._ignored`
  (deprecation catch-all, `args/mod.rs:1323`).
- **Every TOML key is a serde struct field.** `fit.toml`→`FitConfigV2`,
  `batch.toml`→`ExperimentToml`, `compare.toml`→`CompareToml`. Where
  `deny_unknown_fields` is present the set is _closed_ (an unknown key errors);
  where it is absent the _honored_ set is narrower than the _declared_ set (see
  §3, gap G3).
- **Environment variables** that are neither flag nor key. Enumerated in §2.4 —
  this is the one door the clap∪serde claim does not cover, and it has one live
  leak.

So: enumerate all clap derives + all serde config derives + all
`env`/`std::env::var` reads, and the union is the whole surface. That
enumeration is below.

### 2.2 CLI surface (clap)

All clap derives live in `main.rs` (the subcommand tree: `Cli`/`Command` + 5
group enums) and `args/mod.rs` + `args/types.rs` (33 `Args` structs + custom
value types). Subcommands: `simulate`, `batch run|status`,
`fit run|summary|diff|table|new|methods`, `pfilter`, `if2` (deprecated),
`profile`, `survey`, `eval`, `data split`, `list`, `show`, `cat`, `reindex`,
`compare`, `label`, `compile|check|doctest|inspect` (passthrough),
`lineage realize|tree|sojourn|cohort`, `docs`, `mre fit|simulate`.

Only **six** structs are shared today via `#[command(flatten)]`:
`ModelOverrides`, `InferenceModelOverrides`, `ScenarioArgs`, `SimBackend`,
`InferenceCore`, `FlowProjection`. Three high-traffic commands (`fit run`,
`survey`, `simulate`) do _not_ flatten the shared groups and reimplement
seed/parallel/scenario inline.

### 2.3 Config-file surface (serde)

| File           | Root struct      | file:line             | `deny_unknown_fields`       | Hashed via                                                          |
| -------------- | ---------------- | --------------------- | --------------------------- | ------------------------------------------------------------------- |
| `fit.toml`     | `FitConfigV2`    | `fit/config_v2.rs:24` | yes (top-level + 12 nested) | whole-blob serde (`fit/cas.rs:282`) → `FitDigest.fit_toml`          |
| `batch.toml`   | `ExperimentToml` | `batch.rs:71`         | **no (0/11 structs)**       | legacy `hashing.rs` for `[design.*]`; `runid` for plain `batch run` |
| `compare.toml` | `CompareToml`    | `compare.rs:24`       | **no**                      | not hashed (pure inspection input)                                  |

`FitConfigV2` is the single largest surface (5947 lines of `config_v2.rs`). Its
hash membership is governed by `skip_serializing_if` (the serde analogue of
"keep-out-of-hash-at-default"); the complete set is six fields: `simplex_groups`
(`:65`), `condition_from` (`:163`, gh#134), `obs_alignment` (`:304`, gh#189),
`allow_degenerate_rates` (`:315`, gh#189), `DataSpec.file` (`:353`), plus
`compiled_ir` (`#[serde(skip)]`, `:178`). Per-stage identity is governed
separately by `Stage::identity_payload()` (`config_v2.rs:1450`), which
deliberately excludes the "extension dimension" (PGAS `sweeps`, IF2/PMMH/Mh
`iterations`) so `--resume` can extend a chain without re-keying, folding it
into `cas_target_length` at the stage level instead.

### 2.4 Environment variables (the third door)

Most are correctly **provenance** (`CAMDL_OUTPUT`, `CAMDL_OUTPUT_DIR`,
`CAMDL_*_CACHE*`, color/log vars) or verified **numerically inert** optimization
toggles (`CAMDL_EVAL_UNRESOLVED`, `CAMDL_NO_BINDING_CACHE`, `CAMDL_EVAL_FLAT`,
`CAMDL_NO_CONSTANT_FOLD` — the last is structure-shaping but folded into the IR
cache key, `util.rs:435`). `CAMDL_SEED` is correctly **identity** (enters via
`process_seed`). The one live leak is **`CAMDL_PF_WALLCLOCK_TIMEOUT_S`**
(`degeneracy.rs:72`): a machine-speed-dependent watchdog that can _abort_ a run,
neither a flag nor a key, neutralized for `fit` (`pf_wallclock_disabled: true`,
`runner.rs:647`) but **live and un-hashed for `pfilter`, `survey`, `profile`**
(`pfilter.rs:436`, `survey.rs:995`, `profile.rs:1261`). See gap G2.

### 2.5 Duplication inventory (the gh#241 core, expanded)

| Concept                    | Occurrences                                                                                                                                                                                                                                        | Divergence                                                                                                                                                                                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| backend                    | `SimBackend` (`args/mod.rs:331`, `Option<Backend>`), `ProfileArgs` (`:1551`, **`Option<String>`**), `FitBackendConfig` (`config_v2.rs:295`), `ConfigSection` (`batch.rs:224`)                                                                      | type splits `Backend` vs `String`; **two different `Backend` enums coexist** — `args::types::Backend{Gillespie,ChainBinomial,Ode}` vs `run_meta::Backend{ChainBinomial,Ode}` — and a single `fit.toml` uses **both** (`[config].backend` is the former, `[stages.*].backend` the latter) |
| dt                         | `SimBackend` (`:335`, `Option<f64>`), `InferenceCore` (`:353`, `f64`=1.0), `FitBackendConfig` (`:297`), `ConfigSection` (`batch.rs:226`)                                                                                                           | `Option` vs concrete-default; `default_dt` duplicated in `config_v2.rs:321` and `batch.rs:239`                                                                                                                                                                                           |
| seed(s)                    | scalar in `InferenceCore`/`SimulateArgs`/`FitRunArgs`/`SurveyArgs` (defaults 1/1/None/42); `SeedSpec` clap (`types.rs:189`); `SeedsSpec` custom-deserialize (`config_v2.rs:509`); `SeedsSection` (`batch.rs:244`); `fit_seeds` (`config_v2.rs:46`) | four "produce a `Vec<u64>`" surfaces, divergent defaults, only `SimulateArgs` reads `CAMDL_SEED`                                                                                                                                                                                         |
| parallel                   | 5 structs, all `env=CAMDL_PARALLEL`                                                                                                                                                                                                                | `usize`+0 vs `Option<usize>`                                                                                                                                                                                                                                                             |
| output / output_dir / root | `--output` (5 commands, varying semantics), `--output-dir`/`root` (7 structs, `env=CAMDL_OUTPUT_DIR`)                                                                                                                                              | one regression test pins the defaults in lockstep (`args/mod.rs:2584`)                                                                                                                                                                                                                   |
| scenario                   | `ScenarioArgs` (shared), `SimulateArgs` inline (plural), `SurveyArgs` inline (no enable/disable)                                                                                                                                                   | three shapes                                                                                                                                                                                                                                                                             |

## 3. Architecture verdict: what is sound, what is the gap

**Sound (do not redesign).** The argument is a syllogism: _if the hash function
computes correct identities for the inputs it is given, the bugs can only live
upstream of it (the surface and the seam that feeds it), so the work is
feeding-it-correctly, not rebuilding it._ The premise is established by (i) the
passing fold-tests in `fit/cas.rs` (each asserts a field changes the run_id and
its at-default value does not re-key), (ii) the gap-hunt finding **zero**
silent-collision gaps in the _primary-artifact_ identity computation across
sim/pfilter/survey/profile (the one plausible exception, `--integrator`, folds
in via the IR digest) — though the review (§11) found the _sub-artifact_ layer
under-audited: `simulate --event-log` is a real G1 silent collision — and (iii)
all four gaps below being _upstream_ of the hash, never the `runid` hash
miscomputed. The identity layer's _computation_ is careful and correct (the bugs
are upstream, which is the point); the claim is "don't rebuild the hash," not
"there are no gaps":

- Factored per-level hashing for reuse (`resolve.rs:180`), include-by-default
  _within_ each level (the `RunInput` derive folds every non-provenance field; a
  non-`ContentAddressed` field is a _compile error_,
  `runid-derive/src/lib.rs:17-19`).
- Content-not-path digesting for every computation-feeding file (data, tables,
  holdout, resume state, survey landscape — all verified content-hashed,
  §appendix).
- The IR channel is clean: the whole canonical IR is content-hashed
  (`ModelDigest.ir`), `read()`-pulled coupling/contact files are inlined into
  the IR at compile time (`expander.ml:3682`), and the compiler hash,
  `ir_version`, and engine version all re-key.
- The `--integrator` choice (a plausible gap — `SimConfig` has no integrator
  field) in fact reaches the hash via the model IR digest
  (`ir_hash.rs:866-875`); the CLI override is applied pre-hash (`util.rs:2290`).
  **Not a gap.**
- The finiteness gate (`fit/cas.rs:240`) prevents NaN/Inf collapsing to `null`
  and colliding.

**gh#241 classified.** Per the discrepancy-classification rule: gh#241 is a
_surface DRY/maintainability_ issue (the same concept modeled 3–4× with
divergent types and defaults), **not** a CAS-correctness bug. The hashes are
correct today; the danger is _prospective_ (drift between the copies, and the
dedup refactor itself re-keying runs). The unification must be gated by a
run_id-stability test (gh#241 body).

**The actual gaps** — the places where the safety property holds by vigilance,
not construction:

- **G1 — No total, enforced resolver.** Each artifact kind hand-assembles its
  identity from a hand-filled context (`TrajectoryCtx` in `resolve.rs:41`,
  `FitStageCtx` in `fit/cas.rs:51`). The original `Resolve` trait
  (`Raw*Cli/Toml → Vec<RunInput>`) was never realized as one seam. Nothing at
  compile time guarantees a _new surface field that affects output_ reaches the
  identity struct. The completeness that exists is local (within a struct /
  within the fit blob); the surface→resolved transcription is unguarded. _This
  is the structural root of "the discipline gets skipped."_
- **G2 — `CAMDL_PF_WALLCLOCK_TIMEOUT_S` is live and un-hashed on
  `pfilter`/`survey`/`profile`** (§2.4). Output-affecting (abort vs complete),
  machine-dependent, outside clap∪serde.
- **G3 — Silent-drop config keys.** `batch.toml` and `compare.toml` have _zero_
  `deny_unknown_fields` coverage; a typo'd key is dropped, neither applied nor
  hashed. Worse, the `fit.toml` `Stage` enum (`config_v2.rs:982`) is
  `#[serde(tag = "algorithm")]`, and serde **cannot** apply
  `deny_unknown_fields` to an internally-tagged enum — so a typo inside
  `[stages.mle]` (`partalces = 2000`) is silently dropped and the default used.
  This is the highest-impact footgun (stage tuning keys are exactly where users
  iterate) and it is _not_ a one-line fix.
- **G4 — Three hashing mechanisms, one legacy island.** `runid` structural
  derive + fit canonical-JSON blob + legacy `hashing.rs`
  (`model_hash`/`sim_hash`/`scen_hash`). The legacy path still backs `batch`'s
  `[design.*]` store (a separate, non-atomic, un-checksummed layout) and is
  **integrator-blind** (`rg integrator hashing.rs` → no matches): two models
  differing only in declared integrator/tolerances collide on `sim_hash` there.
  The gh#147 CasSink migration that deletes this island is incomplete.

## 4. The design: include-by-default + structural enforcement

### 4.1 Why the old posture was fragile (the root cause)

The system did not fail from under-ambition; it failed from being _over-precise
in the wrong direction_. Two postures coexist in the code:

- **Enumerate-what-matters (fragile).** The sim path lists `SimConfig`'s fields
  by hand (`inputs.rs:135`); a new output-affecting field must be _remembered_
  and added, or it silently collides. The surface duplication (§2.5) multiplies
  the hazard — the same concept threaded through 3–4 structs is 3–4× the chance
  to miss one.
- **Include-everything-subtract-inert (robust).** The fit path hashes the
  _whole_ canonicalized config and explicitly removes three keys
  (`fit/cas.rs:282`). A forgotten field is automatically hashed (over-invalidate
  — safe); the only thing requiring thought is _exclusion_.

The robust posture is also the _simpler_ one. "Does `dt` affect the output,
which exact fields matter?" is hard reasoning and a one-way trap: wrong toward
inclusion costs a recompute, wrong toward exclusion serves a corrupted answer
silently. Defaulting to include **inverts the burden of proof onto exclusion**,
where rigor belongs and is cheap to demand (a justification plus a test). This
is "make illegal states unrepresentable" applied to caching — the illegal state,
_output-affecting input not hashed_, becomes structurally biased toward
over-invalidation (legal, slow) instead of collision (illegal, corrupting). The
design below makes this posture universal and turns the residual transcription
gap from a discipline into a type/test obligation.

### 4.2 Three layers of enforcement

**Layer 0 — shrink the surface (Lever A, gh#241/gh#156).** One dual-derive
struct per _resolution-invariant_ concept (`BackendConfig`, `SeedsConfig`,
`OutputView`), `#[derive(clap::Args, Serialize, Deserialize)]`,
`#[command(flatten)]` into each CLI subcommand and embedded as a TOML
`[section]`. Collapse the two `Backend` enums into one. Fewer transcription
points, fewer places to forget.

**Layer 1 — for the `ContentAddressed`-compatible subset, the surface struct
_is_ the identity field (zero transcription).** This is where the "one struct"
intuition is achievable — but the subset is narrower than the first draft
claimed (corrected, §11): it is the fields whose type is _already_
`ContentAddressed` **and** clap- **and** serde-parseable — the `Backend` enum,
integer counts, `bool`, `String`. For those the shared struct _also_ derives
`RunInput` and is embedded directly into the identity leaf; one definition
serves CLI, TOML, and hash at once. `dt` and other floats are **not** in this
subset (`f64` is not `ContentAddressed`; `FiniteF64` is not `Deserialize`), nor
are maps — they go through Layer 2's transcription. The original "one struct"
vision, scoped to exactly the subset where it is type-sound.

**Layer 2 — for transforming fields, a typed `resolve` seam + a default-include
resolved struct.** `seed`/`tables`/`time_format`/`obs_alignment` genuinely
transform (base seed → `process_seed`, path → content digest, request →
resolved). They get a real `resolve(Surface) -> Result<Resolved, Error>`.
`Resolved` is built as a struct literal, so adding a `Resolved` field forces the
resolver to fill it (compile error otherwise) — the resolved→hash direction is
closed by the type checker. Each level's identity is the **whole** content hash
of its resolved value, never an enumerated subset (enumerating a subset is the
"hash-a-recipe" antipattern the foundational proposal already forbids).

**Layer 3 — the completeness gate, on `RunInput` (no separate attribute
system).** The classification already exists: `#[run_input(provenance)]` _is_
the semantic/provenance distinction — a field is folded (semantic) by default or
carries the opt-out (provenance). A separate `#[cas(semantic|provenance)]`
attribute would be a second source of truth for the identical fact and is
rejected. Include-by-default also removes the need to _force_ a per-field
decision: forgetting to annotate a new field folds it (over-invalidate — safe),
never drops it (collision). So Layer 3 is two tests layered on `RunInput`, keyed
off the single existing annotation:

1. **run_id-stability golden** (the gh#241 gate). Pin representative `run_id`s;
   a refactor, the dedup, or a folded field added without thought fails loudly —
   forcing the author to accept the re-key or add `#[run_input(provenance)]`.
   Independent of `RunInput`; needed regardless.
2. **provenance-honesty differential test** (the real audit — an _output_ test,
   not a `run_id` test). For each `#[run_input(provenance)]` field, hold all
   else fixed, run a trivial fast model with the field set two ways, and assert
   the **output bytes are identical**. A diff means the annotation is a lie —
   the field is excluded from the `run_id` but changes the output, i.e. a silent
   collision. This is the one dangerous bug `RunInput` cannot catch on its own
   (the derive excludes a provenance field from the hash whether or not the
   claim is honest), and a `run_id`-level test cannot catch it either (the
   `run_id` is unchanged _by construction_). It requires running the computation
   — it is the same machinery as `camdl verify` (§10), just with a provenance
   field perturbed. Run it across _all_ provenance fields on one SIR-sized model
   so there is no "covered elsewhere" gap.

A `run_id`-perturbation test on _semantic_ fields is deliberately omitted:
flipping a folded field moves the hash by construction (already pinned by the
`macro_eq` golden, `runid/src/macro_eq.rs`), so it only re-tests the derive. The
lone non-tautological slice — does `resolve` actually thread a _transforming_
input (`--seed → process_seed`) into a level? — belongs in `resolve`'s own unit
tests.

To make test 2 generative without a second annotation, extend the `RunInput`
derive to expose the field list it already computes —
`fn run_input_fields() -> &'static [(&'static str, Membership)]`
(`Folded | Provenance`) — so the harness enumerates fields off the same single
source of truth that drives the hash. One annotation, one derive (emitting hash
_and_ audit metadata), two tests.

Defense in depth: Layer 1 erases the gap for the easy fields, Layer 2 makes the
resolved→hash direction a compile error, Layer 3's golden catches unintended
re-keys and its differential test catches dishonest provenance — the one thing
types cannot.

### 4.3 Closing the loud gaps (Lever C)

- **C1** — complete the gh#147 CasSink migration; route `batch [design.*]`
  through `runid`; delete
  `hashing.rs::{model_hash,sim_hash,scen_hash,canonical_params}` and the second
  store layout (removes G4).
- **C2** — `deny_unknown_fields` on the `batch.toml` + `compare.toml` structs
  (removes most of G3).
- **C3** — the `Stage` tagged-enum typo footgun: a post-parse key-validation
  pass per algorithm. (serde cannot apply `deny_unknown_fields` to an
  internally-tagged enum; the validation pass is lighter than restructuring
  `Stage` into tag + newtype-per-algorithm and causes no identity churn.)
- **C4** — thread `pf_wallclock_disabled: true` on the content-addressed path
  for `pfilter`/`survey`/`profile`, making the watchdog inert on all CAS paths
  as `fit` already does (removes G2; simpler than promoting it to a hashed
  field, and matches `fit`).

### 4.4 The uniform `Resolve`/`build` core (the target shape for G1)

Today each artifact kind hand-rolls its resolution (`resolve_trajectory`,
`resolve_fit_stage`, `pfilter_cas`, …) plus its own store dance. The target is
the foundational proposal's `Resolve` trait plus _one_ driver — generalizing
what already exists. A `Key`/input-edge in camdl is _one level_, and its digest
is the content hash of a resolved typed value, so the toy's flat `Vec<Key>` is
our `Vec<LevelId>` and `derivation_hash` is `run_id` generalized:

```rust
struct Level { name: &'static str, label: String, hash: ContentHash, schema_version: u16 } // = LevelId
struct FnId  { kind: ArtifactKind, fn_version: ContentHash }   // fn_version = the git hash (see below)
struct Derivation { func: FnId, levels: Vec<Level> }

fn derivation_hash(d: &Derivation) -> ContentHash {            // = run_id (kind.rs:79), generalized
    let mut h = CanonicalHasher::new();
    h.write_u16(HASH_VERSION);
    d.func.kind.hash_into(&mut h);
    d.func.fn_version.hash_into(&mut h);
    h.write_len(d.levels.len() as u64);
    for l in &d.levels { l.hash.hash_into(&mut h); }
    h.finalize()
}

trait Resolve {                                                // the bind; one impl per command — closes G1
    fn kind(&self) -> ArtifactKind;
    fn fn_version(&self) -> ContentHash;
    fn resolve(&self, ctx: &Ctx) -> Result<Vec<Level>, ResolveError>;   // effectful: reads files → digests
}

fn build<R: Resolve>(store: &Store, surface: &R, ctx: &Ctx,            // one driver for every kind
                     run: impl FnOnce(&Derivation) -> Artifacts) -> Result<Outcome> {
    let levels = surface.resolve(ctx)?;
    let d      = Derivation { func: FnId { kind: surface.kind(), fn_version: surface.fn_version() }, levels };
    let run_id = derivation_hash(&d);
    let path   = store_path(&store.root, d.func.kind, &d.levels);       // input-addressed → known before running
    match store.lookup(&path, run_id) {
        Hit(rec) => Ok(Outcome::Cached(rec)),
        _        => store.commit(path, run_id, &d.levels, run(&d)),
    }
}
```

**The shared surface struct (Layer 0) flows three ways from one definition**,
and composes with this core without transcription:

> **Corrected after review (§11):** the shared _surface_ struct derives only
> `clap::Args + serde`. It cannot _also_ derive `RunInput`, because `RunInput`
> requires every field to be `ContentAddressed`, and the identity layer
> deliberately excludes `f64` and `IndexMap`/`HashMap` (`hash.rs:316`) to force
> the resolved-type policy (`FiniteF64`, `BTreeMap`). `FiniteF64` has no
> `Deserialize`, so no single type for `dt` satisfies clap + serde + `RunInput`
> at once. The surface struct is shared; the _resolved_ identity type (with
> `FiniteF64`/`BTreeMap`) is distinct, and `resolve` transcribes — the existing
> `SimConfig` + `resolve.rs` pattern, which is correct and stays.

```rust
// Shared SURFACE struct — clap + serde only. (Layer 0 dedup.)
#[derive(Clone, clap::Args, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    #[arg(long)] #[serde(default = "default_backend")] pub backend: Backend,  // enum: ContentAddressed-OK
    #[arg(long)] #[serde(default = "default_dt")]      pub dt: f64,           // raw f64: NOT hashable as-is
}
// (1) CLI: #[command(flatten)] config: BackendConfig in FitRunArgs and SimulateArgs
// (2) TOML: the [config] section of fit.toml (replaces the bespoke FitBackendConfig)
// (3) hash, fit:  serializes INTO the fit_toml blob automatically (no extra wiring)
// (3) hash, sim:  resolve() transcribes → SimConfig { backend, dt: FiniteF64::new(dt)? }, then RunInput folds it
// Layer 1 ("surface struct IS the identity field, zero transcription") therefore holds only for the
// ContentAddressed-compatible subset — the `backend` enum and integer counts — NOT for `dt`/maps.
```

The `fit` stage `Resolve` then assembles all inputs — model (compiled IR), data
files (path → _content_ digest), the whole-blob config (with `BackendConfig`
riding inside it), the stage block, the upstream `deps`, the seed — into the
three levels, which is exactly `fit_level_digest` + `StageLevel` today
(`fit/cas.rs:317,401`):

```rust
impl Resolve for FitStageSurface<'_> {
    fn kind(&self) -> ArtifactKind { ArtifactKind::FitStage }
    fn fn_version(&self) -> ContentHash { engine_ir_commitment() }
    fn resolve(&self, ctx: &Ctx) -> Result<Vec<Level>, ResolveError> {
        let fit = FitDigest {
            model:        ModelDigest::from_model(self.model, ctx.ir_version, ctx.engine.clone()),
            data:         build_data_digests(&self.config.data.resolved_paths())?,  // path → bytes → digest
            holdout_data: build_holdout_digests(self.config)?,
            fit_toml:     fit_config_blob_hash(self.config)?,   // include-by-default; BackendConfig rides this
            engine:       ctx.engine.clone(),
        };
        let stage = StageLevel { config: stage_config_hash(&self.config.stages[self.stage_name])?,
                                 deps: Deps(self.deps.clone()) };               // the Merkle DAG edge
        let seed  = Seed { process_seed: self.seed, base_seed: self.seed };
        Ok(vec![
            Level::new("fit",   self.config.model.stem(),                           fit.content_hash()),
            Level::new("stage", format!("{:02}-{}", self.ordinal, self.stage_name), stage.content_hash()),
            Level::new("seed",  format!("seed_{}", self.seed),                      seed.content_hash()),
        ])
    }
}
```

**`fn_version` is the git hash, not the semver.** What folds today is
`VERSION_SHORT` = `semver +
"+" + GIT_HASH` (`version.rs:12`); the load-bearing
part is the **git hash** — a semver can be forgotten (two builds share `0.1.0`),
a git hash cannot (different commit ⇒ different hash, automatically). The semver
prefix is decorative and carries no identity weight. Precedent: `camdlc`'s
`GIT_HASH` already keys the IR cache (`util.rs:435`). One residual gap: a git
hash does not capture a _dirty working tree_ (uncommitted edits hash identically
to the clean commit — a silent stale hit in dev). The theoretical ideal is a
hash of the binary; the pragmatic fix is a `CAMDL_GIT_DIRTY` marker folded
alongside the hash. Released builds (clean tree) are unaffected.

**Structural on the _resolved_ types — the transcription stays (corrected,
§11).** The earlier draft proposed deriving `RunInput` directly on the surface
struct to delete the transcription. The review (§11) showed this is infeasible:
`RunInput` needs every field `ContentAddressed`, and `f64`/`IndexMap` are
deliberately not (`hash.rs:316`), so a struct cannot be both the serde surface
and the `RunInput` identity. The correct, and simpler, statement: the
**resolved** identity types (`SimConfig` with `FiniteF64`/`BTreeMap`, etc.)
derive `RunInput`; the **surface** structs derive clap+serde; `resolve`
transcribes between them. That transcription is the existing
`SimConfig`+`resolve.rs` pattern and is _correct_ — it is not the failure §1
indicts (the failure is the _duplicated, unguarded_ surface, not the existence
of a resolve step). Include-by-default is therefore fail-safe **within a derived
identity struct**, and the transcription hop is guarded by the Layer-3 tests
(§4.2), not by the type checker — §1.1 is corrected to say so. The blob route
stays for `fit` (the open-document include-by-default exemplar); migrating it to
a structural _resolved_ type is optional and re-keys once, for little gain.
During any such interim the shared `BackendConfig` rides the blob in `fit` and
is transcribed into `SimConfig` in `simulate` — safe because the `run_id` kind
tag separates the namespaces (a fit and a sim never share an id; the only
invariant is the local _within one identity, hash each level exactly one way_).

**Re-key caution.** Replacing `FitBackendConfig` with the shared `BackendConfig`
is blob-byte-identical only if the serde field set/order/skip-rules are
preserved. `FitBackendConfig` also carries `obs_alignment` +
`allow_degenerate_rates` (`config_v2.rs:304,315`); the shared core is
`{backend, dt}`. Keep those two as fit-specific siblings of `BackendConfig`
rather than widening the shared struct (which would force them onto `simulate`).
The run_id-stability golden (Layer 3) catches any accidental blob change.

## 5. The identity schema is ours — and the surface schema becomes ours too

We _define_ the identity schema: `runid::inputs` is a declarative statement of
exactly which resolved fields exist, in what order they hash, with
`#[run_input(schema_version = N)]` per type and the global `HASH_VERSION` as
migration levers — the direct analogue of `ir/schema.json` + `ir/VERSION` for
the IR (a deliberate, versioned, human-reviewed contract). The difference: the
IR schema is the cross-language OCaml↔Rust contract; the identity schema is
Rust-only (the OCaml side contributes only the IR, which enters as the
`ModelDigest` content hash).

What we do _not_ have today is an explicit _surface_ schema — the surface is
scattered across clap structs and serde structs (§2). Layer 0 creates one: the
shared dual-derive structs become the single declarative description of the
input surface, the missing sibling to the identity schema.

## 6. Input → file-path mapping (the Layout)

A leaf's identity is the _ordered tuple of per-level hashes along its path_; the
store path is a readable factoring of that tuple, one segment per level, each
`{label}-{hash8}` — the label is provenance (a rename → a cache miss, never a
wrong answer), the `hash8` is identity (`layout.rs:108`).
`store_path(root, kind, levels) = root/{kind_dir}/{seg_0}/…/{seg_n}`
(`layout.rs:122`). Navigation reads `run.json`, never the segments.

| Subcommand             | kind dir     | levels (in path order)                    | builder                   |
| ---------------------- | ------------ | ----------------------------------------- | ------------------------- |
| `simulate`             | `sims/`      | model · config · params · scenario · seed | `resolve.rs:180`          |
| `batch run` (ensemble) | `ensembles/` | model · config · params · grid            | `sim_ensemble_cas.rs:142` |
| `fit run`              | `fits/`      | fit · `NN`-stage · seed                   | `fit/cas.rs:417`          |
| `pfilter`              | `pfilters/`  | model · config · params · seed            | `pfilter_cas.rs:118`      |
| `survey`               | `surveys/`   | model · config · box · seed               | `survey_cas.rs:122`       |
| `profile`              | `profiles/`  | profile · point · stage · seed · start    | `profile_cas.rs:122`      |

What each level hashes: **model** = whole-IR `ModelDigest` (IR content hash +
`ir_version` + engine, `inputs.rs:154`); **config** = the `SimConfig` struct
(sim) or a whole-blob digest of the resolved config
(pfilter/survey/ensemble/fit); **params** = `ResolvedParams` (values + table
_content_ digests); **scenario** = `ResolvedScenario` delta; **seed** = the
resolved `process_seed`; **box** = LHS bounds + n_points (survey); **grid** =
the sorted ensemble cells (ensemble); **fit** = `FitDigest` (model + data
digests + whole canonical fit.toml blob); **NN-stage** = `StageLevel` (stage
config + `deps`); **point/stage/start** = the profile factoring. Invariant:
every level's hash is the **whole** content hash of a resolved value — never an
enumerated subset (Layer 2).

### 6.1 The type-level model (Rust)

The whole system is two type families and one arrow between them: a `Surface`
(presentation, _not_ `ContentAddressed`), an `Identity` (resolved,
content-addressed, factored into levels), and
`resolve : Surface -> Result<Identity>` — the effectful bind where presentation
collapses to semantics. Only the `Identity` can be hashed; the type checker
forbids hashing a `Surface`.

```rust
// ── SURFACE — clap + serde, presentation. NOT ContentAddressed. ──
struct Surface {
    model_path: PathBuf, backend: String, dt: f64,
    table_path: PathBuf, seed: u64, point_idx: u32,
}

// ── IDENTITY — resolved, content-addressed. Each level derives RunInput. ──
#[derive(RunInput)] enum Backend { Gillespie, ChainBinomial, Ode }
#[derive(RunInput)] struct Config { backend: Backend, dt: FiniteF64 }      // a LEVEL
#[derive(RunInput)] struct Params { table: ContentHash }                   // CONTENT, not path
#[derive(RunInput)] struct Seed {
    process_seed: u64,                          // mix(base, point_idx) — the hashed value
    #[run_input(provenance)] base_seed: u64,    // recorded in run.json, never hashed
}
#[derive(RunInput)] struct Model { ir: ContentHash, ir_version: String }
#[derive(RunInput)] struct SimIdentity {        // the leaf = a PRODUCT of levels
    model: Model, config: Config, params: Params, seed: Seed,
}

// ── RESOLVE — the bind: Surface -> Identity. Effectful, NOT identity-preserving. ──
fn resolve(s: &Surface) -> Result<SimIdentity, ResolveError> {
    Ok(SimIdentity {                            // struct literal: every level MUST be filled
        model:  compile_and_digest(&s.model_path)?,
        config: Config { backend: parse(&s.backend)?, dt: FiniteF64::new(s.dt)? },
        params: Params { table: digest_file(&s.table_path)? },   // path → content
        seed:   Seed { process_seed: mix(s.seed, s.point_idx), base_seed: s.seed },
    })
}
// store_path(root, Sim, levels(id)) = sims/{model-h8}/{config-h8}/{params-h8}/{seed-h8}
```

The real system matches this _structurally_, with two known differences:
`resolve` is split per artifact kind rather than one `Resolve` trait (G1 — and
the review, §11, keeps it that way: the distinct orchestrations differ at the
natural seam), and the fit `config` level uses the whole-blob form while sim
uses a structural-derive struct (both are include-by-default; the blob stays for
fit per §11, it is not "standardized away").

## 7. Sequencing, decisions, and remaining open questions

> These decisions were revised by the 2026-06-16 adversarial review — see §11
> for the verified findings and the corrected plan. The entries below are
> updated to match.

Recommended sequence: **cheap path first** (C2/C3/C4 + the differential test —
zero re-key, closes all present silent-wrong-answers, §11.2), then the
**run_id-stability + anti-drift goldens with an explicit coverage matrix**
(before any re-key), then **Layer 0 dedup** (surface-only), then C1's legacy
migration.

**Decisions made:**

- **D1 — Hashing route: structural on the _resolved_ types; dedup the _surface_
  (corrected).** The earlier "derive `RunInput` directly on the surface struct,
  zero transcription" is **infeasible** (`f64`/`IndexMap` are not
  `ContentAddressed` by design; `FiniteF64` has no `Deserialize`). Keep the
  transcribing `resolve` (the `SimConfig`+`resolve.rs` pattern is correct);
  share only the surface structs; Layer 1's zero-transcription applies to the
  `ContentAddressed`-compatible subset (enums, ints), not `dt`/maps.
  Classification stays the existing `#[run_input(provenance)]` opt-out — **no
  separate `CasAudit` attribute.** (§4.4, §11.1.)
- **D2 — `fn_version` = the git hash** (the `+<hash>` of `VERSION_SHORT`,
  `version.rs:12`), not the semver; add a `CAMDL_GIT_DIRTY` marker as a
  follow-up so a dirty working tree re-keys. (§4.4.)
- **D3 — DEMOTED to the existing shared substrate (corrected).** One `build`
  driver does not fit the real orchestration (fit's streaming-claim +
  multi-stage fold + sidecar; profile's record-less base; obs as a child). Keep
  the shared substrate that already exists
  (`level`/`derivation_hash`/`store_path`/`Deps`/ `digest_value`) and five
  distinct orchestrations — the "seam, not past it" rule. The `Resolve` trait
  may be a thin shared signature, not a god-driver. (§11.1.)

**Still open:**

1. Whether to migrate `FitConfigV2` blob → structural at all — the blob is the
   include-by-default exemplar and migrating re-keys every fit for little gain.
   **Revised lean (§11): keep the blob; do not bundle re-keys.** If migrated
   later, it is its own isolated, surfaced re-key with its own golden — _not_
   folded into the Layer-0 dedup (there is no key-migration tool, and the
   userbase runs multi-day fits, so two small revertible re-keys beat one
   combined one).
2. Whether to drop the semver prefix from `fn_version` entirely (redundant given
   the git hash) or keep it for human-readability in `run.json` (it is
   provenance regardless of which).
3. The exact `Stage` typo-validation shape for C3 (per-algorithm key allowlist
   vs restructuring the tagged enum) — serde cannot `deny_unknown_fields` an
   internally-tagged enum.

## 8. Rough size

Lever A (dedup) is net ≈ **−150 LoC** (≈ −350 duplicate definitions + custom
`Deserialize` impls, ≈ +200 shared), per gh#241 — the bulk being the seed
representations (four parse/resolve paths → one) and the backend/dt copies (four
→ one, plus the duplicated `default_backend`/`default_dt`). The enforcement
machinery (a small `RunInput`-derive extension to expose the field list + the
run_id-stability golden + the provenance-honesty differential harness) is net
**positive** — roughly +200–400 LoC, almost all test/harness, and lighter than a
separate `CasAudit` system would have been. So the change is **net-neutral to
slightly positive on lines**, and that is the right trade: the fragility was
_missing_ safety (omitted fields), not _present_ bloat, so removing it saves
little. The win is a structural guarantee bought with test code, not a smaller
diff — LOC was never the objective.

## 9. Views: a deferred read-side projection layer

A Nix-style alternative separates three concerns we currently entangle in the
storage path: a flat output-blob store (deduped truth), an input-addressed `drv`
cache table, and a _swappable view layer_ that projects human-navigable
hierarchies as symlinks — with labels excluded from every hash, so no choice of
hierarchy can perturb a key. The view layer carries a `ViewPolicy` (`Unique` = a
stable identity path, rebinding to a different output is a refused clash;
`Latest` = a mutable "current" pointer; `Collect` = accumulate, leaf named by
output hash) plus a pre-flight check that proves a batch clash-free before
writing.

This is genuinely cleaner as separation-of-concerns, and buys two things we
lack: **multiple simultaneous hierarchies** off the same results (by
`(model, scenario)`, by convergence, pooled replicates), and
**reorganize-without-recompute** (a label rename or re-factoring is zero cache
misses). It is **deferred, not adopted**, for concrete reasons:

- Our storage path is _input-addressed and known before the run_ (so `fit run`
  announces its output dir, `--dry-run` works); an output-addressed blob path is
  unknowable until after running.
- Symlinks are fragile on our deployment surface (HPC scratch, `rsync`, tar/zip,
  Windows); the single-tree-of-real-dirs travels cleanly, and the camdl-book /
  camdl-viewer consumers walk the tree + read `run.json` today.
- Output-byte dedup is near-worthless for stochastic simulation (a different
  seed → different bytes).
- Our fixed full-factoring _sidesteps_ the clash problem the view flexibility
  creates — we never drop a level, so we cannot clobber (at the cost of no
  pooled view).

Recommended future shape: a **read-side** `camdl view --by <schema>` that
materializes symlink (or copy/junction) projections from `run.json` on demand,
leaving the input-addressed tree as the source of truth — capturing the
`Latest`/`Collect` policy and the clash pre-flight without betting storage on
symlinks or migrating consumers.

## 10. The output-stability backstop: `camdl verify`

Input-addressing pins the _inputs_; identity-stability (Layer 3) checks the
address does not move spuriously. The complementary check is _output_ stability
— does a fixed identity still produce the same bytes? We already record the
hooks: `FileChecksum.digest` is each output file's SHA-256, "recorded for
integrity tooling (`camdl verify`); NOT checked on read today" (`record.rs:48`,
`hash.rs:95`). The missing piece is the falsification protocol — a
`camdl verify <run_id>` that re-executes a leaf cache-bypassed and compares the
new output digest to the recorded one. This is the backstop for the
silent-numerics-drift class types cannot prevent (RNG consumption order,
parallel-reduction associativity, a refactor that changes bytes without changing
inputs): on the model side determinism is structural over the IR evaluator, but
orchestration determinism is _detect-not-prevent_, and `verify` is the detector.
It pairs with the external-validation gates (he2010 loglik vs pomp), which are
the same protocol run in CI.

## 11. Adversarial review (2026-06-16): findings, corrections, revised plan

Four independent adversarial reviewers attacked this proposal against the code
(evidence, design feasibility, scope/ROI, the safety gate). The §1–§3 diagnosis
survived; the prescription is narrowed.

### 11.1 Verified findings

- **D1 is infeasible as originally written** (verified). `#[derive(RunInput)]`
  requires every field `ContentAddressed`; `f64` is deliberately excluded
  (`hash.rs:316`), there is no impl for `IndexMap`/`HashMap`, and `FiniteF64`
  has no `Deserialize`. So no struct can be both the serde surface and the
  `RunInput` identity for a float or map. **Correction:** keep the
  surface↔resolved split and the transcribing `resolve` (the existing
  `SimConfig`+`resolve.rs` is correct); dedup the _surface_ only; Layer 1's
  zero-transcription holds only for the `ContentAddressed`-compatible subset
  (enums, ints). §1.1, §4.2 Layer 1, and §4.4 corrected.
- **A live silent-collision bug, discovered (new G1 instance).**
  `simulate --event-log` writes `event_log.tsv` into the leaf
  (`args/mod.rs:588`) but is absent from `resolve.rs`/`inputs.rs` (verified) —
  unhashed, so a same-`run_id` hit silently drops it (and the obs sub-artifact
  is the same class: `kind.rs:29`, not yet a first-class leaf). camdl reports
  success and points at a non-existent file; `--force` does not recover it. The
  §3 claim "zero silent-collision gaps" is demoted to "zero in the
  _primary-artifact_ identity; _sub-artifacts_ were under-audited," and
  `--event-log`/obs are added to the §4.2 differential-test enumeration.
- **D3 (one `build` driver) does not fit the real orchestration.** Verified:
  `fit` uses the streaming claim API (`claim_streaming`/`finalize`), not
  `commit(Artifacts)`; it is a multi-stage topo-fold that reads each prior
  stage's on-disk `fit_state.toml` for `deps`, plus `--resume`/survey deps, plus
  a per-segment fit sidecar; `profile` has a record-less base segment; obs is a
  child, not a leaf. **Correction:** demote D3 to the _shared substrate that
  already exists_ (`level`/`derivation_hash`/
  `store_path`/`Deps`/`digest_value`); keep five distinct orchestrations (the
  "seam, not past it" rule).
- **G4 "delete `model_hash`" is a migration, not a delete** (verified ~10
  consumers in `fit`/`survey`/`profile`, plus the load-bearing `survey_top_k`
  cross-check at `init.rs:908`). And the gh#147-#3 _non-atomic-hit_ hole
  (`batch.rs:436`, a cache hit gated on bare `traj.tsv` existence) is inherited
  but was unnamed — it is the actual present silent-wrong-answer in the legacy
  path. C1 must (a) replace `model_hash`'s provenance/cross-check uses and (b)
  name the non-atomic-hit fix.
- **The safety gate has holes** (verified): the run_id-stability golden _does
  not exist yet_ (only relative fold-tests do); "representative" is undefined
  with no per-kind/per-field coverage matrix; the differential test scoped to
  "one SIR model" is a "covered-elsewhere" hand-wave the project's own rules
  forbid (`thread_count`, `ArtifactRef.kind`, resolved `obs_alignment`, swept
  `base_seed` are inert on a SIR sim but semantic on
  PGAS/multi-stream/parallel/deps cells); a `resolve` that threads the _wrong_
  surface field compiles, hashes, and moves the `run_id` — indistinguishable
  from a blessed re-key; and `macro_eq` pins derive≡hand (relative), not
  derive≡frozen-bytes (absolute), so a mirrored derive change re-keys the store
  while `macro_eq` stays green. Hand impls (`Deps`, the IR tree) sit outside
  both tests.
- **Re-key cost is under-weighted.** There is no key-migration tool (`reindex`
  only rebuilds the index, not addresses); the userbase runs multi-day national
  fits, so each re-key is a real compute-turnover.

### 11.2 Revised plan

- **First PR — the cheap path (zero re-key, closes all _present_
  silent-wrong-answers):** C4 (G2 wallclock — thread
  `pf_wallclock_disabled: true` on the `pfilter`/`survey`/`profile` CAS path;
  the seam exists), C2 (`deny_unknown_fields` on `batch.toml`/`compare.toml`),
  C3 (the `Stage` typo-validation pass), the **G4 correctness slice** (route
  `batch [design.*]` hits through the atomic `runid` store), and the
  **differential provenance-honesty test** (the one check types cannot do) — run
  **per cell** ({sim, PGAS, IF2, multi-stage-with-deps, survey, profile} ×
  {single/multi-thread} × {single/multi stream}), not on one SIR model.
- **Gate before any re-key:** build the run_id-stability golden with an explicit
  **coverage matrix** — one pinned-literal `run_id` per `ArtifactKind`, plus one
  invocation per output-affecting field set to a _non-default_ value — **plus**
  a pinned-literal `content_hash` anti-drift golden for the derived encoding
  (distinct from `macro_eq`), **plus** a resolve-correctness test (every surface
  field reaches _its_ resolved field). Extend coverage to the hand-written
  `ContentAddressed` impls.
- **Then the dedup (Lever A)**, surface-only, for the
  `ContentAddressed`-compatible subset; collapse the three `Backend` enums
  (`args::types`, `run_meta`, `runid::inputs`) deliberately, accepting the
  fit-level + stage-level (+ possibly sim) re-key as a single surfaced,
  versioned event.
- **Defer D3** (the uniform driver) and the `FitConfigV2` blob→structural
  migration. **Reverse §7.1:** do _not_ bundle re-keys "for a single turnover" —
  two isolated, independently-revertible re-keys with their own goldens are
  safer than one combined one, and there is no migration tool to soften either.

## Appendix — verified file:line index

- Identity types: `runid/src/inputs.rs` (leaf shapes + per-level digests);
  `runid-derive/src/lib.rs` (the derive, include-by-default,
  `#[run_input(provenance)]` opt-out, `schema_version`).
- Sim resolution: `resolve.rs:161` (`resolve_trajectory`), `TrajectoryCtx` at
  `:41`.
- Fit resolution: `fit/cas.rs:392` (`resolve_fit_stage`), blob hash `:282`,
  finiteness gate `:240`, per-stage identity `config_v2.rs:1450`.
- Other CAS builders: `pfilter_cas.rs:81`, `survey_cas.rs:80`,
  `profile_cas.rs:86`, `sim_ensemble_cas.rs:95`.
- Legacy island: `hashing.rs:{32,104,127,168}`, used `batch.rs:{416,513,540}`;
  layout `run_paths.rs:61`.
- Integrator-in-hash: `ir_hash.rs:866-875`; CLI override pre-hash
  `util.rs:2290`.
- Versions that re-key: IR cache key `util.rs:435`; `ModelDigest`
  `inputs.rs:154`; engine `version.rs:12`.
