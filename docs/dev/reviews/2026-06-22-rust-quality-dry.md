# Rust runtime quality & DRY review (2026-06-22)

**Scope:** the whole Rust workspace
(`rust/crates/{cli,sim,io,ir,runid,runid-derive,external-harness}`), ~150k LoC.
**Axis:** code quality, DRY/repetition, dead code, structural debt, un-named
tolerances, mechanical cleanups. **Not** correctness, not OCaml, not performance
architecture — correctness items found in passing are listed at the end for a
later pass.

**Method:** six read-only review agents, one per slice (cli/fit-drivers,
cli/fit-config+reporting, cli-ingest/util, cli-subcommands, sim/inference,
sim-core+small-crates), each verifying findings against code with pasted
snippets; the orchestrator then diffed the per-slice primitive catalogs to find
workspace-spanning forks no single slice could see. Every file:line below was
read; the cross-slice site lists were re-confirmed by direct grep (commands in
the "Verification" note per finding).

---

## Verdict

**This is not poor-quality code, and the LoC growth is mostly legitimate
surface, not sprawl.** The highest-risk area — the inference math — is the
_best_-disciplined part of the tree: the weight/likelihood/transform primitives
are genuinely single-sourced, and two consolidations the git history claims
(`gamma_multiplier_log_density` for gh#197, the `Schedule` boundary cursor for
gh#233) are real and intact. The debt is **concentrated, not diffuse**, and
falls into four buckets:

1. **A handful of genuine forks in inference math** (Tier 1) — small in count,
   high in stakes, because a one-sided edit silently moves a posterior.
2. **Confirmed forks in plumbing** (Tier 2) — date arithmetic, CAS helpers, the
   validation harness's probit table, seed derivation. Several are
   already-drifted or are documented-but-never-deduplicated ("same shape as X"
   comments that were never wired to X).
3. **A few god-modules / god-functions** (Tier 3) — `util.rs`, `config_v2.rs`,
   `FitRunConfig::build`, `CompiledModel::new`.
4. **Un-named tolerance sprawl** (Tier 4) — ~700 inline `1e-N` literals; most
   are test asserts, but a real subset are bare epsilons in control flow.

Two of the orchestrator's own seed leads were **wrong and are corrected below**
(the gh#233 boundary seam is already consolidated; the duplicated coefficient
table is the _probit_ table, not digamma) — recorded here because "verify, don't
describe" cuts both ways.

Totals: **~40 consolidation findings**; **~25 correctness items** kicked to the
next pass.

---

## What's clean (verified — answers "is this poor quality?")

- **Inference weight primitives are single-sourced with zero inline
  reimplementations.** `log_sum_exp` (`types.rs:350`), `normalize_log_weights`
  (`types.rs:399`), `ParticleSwarm::ess` (`types.rs:334`), `logw_variance`
  (`types.rs:367`). Every consumer in `particle_filter`/`correlated_pf`/`if2`/
  `prequential`/`ancestor_trace` delegates. (One exception: `if2.rs` inlines the
  ESS body — Tier 1-D.)
- **Observation log-PMFs are single-sourced** in `obs_loglik.rs` (`negbin`,
  `poisson`, `binom`, `beta_binomial`, `discretized_normal`); `obs_model.rs`
  (dmeasure dispatch), `pgas.rs`, `pgas_grad.rs` all call them — no second copy
  of any count log-pmf. `lgamma`/`digamma` are single-sourced
  (`obs_loglik.rs:10/43`), and `gamma_multiplier_log_density`
  (`obs_loglik.rs:87`) is shared by the value (`pgas.rs:975`) and gradient
  (`pgas_grad.rs`) paths — the gh#197 fix held.
- **CORRECTION — the gh#233 `Schedule` boundary seam is consolidated, not
  forked.** My seed lead (four incompatible accessors, an unused `next_stop`) is
  **stale**. `Schedule::next_stop` is live (`gillespie.rs:237,310`,
  `ode.rs:623`), `arrive` is the shared boundary-dispatch seam, `next_boundary`
  is private, and `schedule.rs:154-160` explicitly documents which same-valued
  epsilons (`RATE_EPSILON`, the probability clamp, `pgas::GRID_STEP_EPS`) are
  _deliberately_ kept distinct. This is the rubric applied correctly — a model
  to copy, not a finding.
- **`runid` is a clean single-hasher design** (SHA-256, two genuinely-distinct
  paths — structural-framed vs opaque-bytes — hashing disjoint things). The
  derive macro and hand-written IR impls share one encoding pinned by a
  byte-equivalence test (`macro_eq.rs`). No duplicate float canonicalizer.
- **Shared UI infra routes through one seam.** `progress.rs` (`Reporter`/`Task`/
  `count_style`/`fmt_rate`) and `style.rs` are single-source, and `batch`/
  `profile`/`pfilter`/`survey` all use them — no re-rolled `MultiProgress`/
  `ProgressStyle` anywhere. `evidence.rs` is the single home for
  decibans/evidence math; `compare.rs` calls into it rather than re-deriving.
- **Dead code is genuinely rare.** 2 `#[allow(dead_code)]` workspace-wide, both
  borderline-legit (a test helper; a deserialized-but-unread schema field). No
  `v1`-beside-`v2` production paths. No `&Vec<T>`/`&String` parameters of note.
- **The `methods.rs` capability-table "fork" is deliberate and test-pinned**
  (gh#191 withholds `REAL_COMPARTMENTS` from chain-binomial inference); not
  accidental drift.

---

## Master findings table (S1/S2, severity-ordered)

| #    | Sev | Class                       | Location(s)                                                                                                                                                                                             | One-line                                                                                                                                                             | Consolidate to                                                                  |
| ---- | --- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| T1-A | S1  | fork                        | `prior.rs:96` vs `hierarchical.rs:175`+                                                                                                                                                                 | 7-arm prior-density duplication (Normal/LogNormal/HalfNormal/Beta/Gamma/Exp/Uniform), byte-identical formulas                                                        | `Prior::log_density`                                                            |
| T1-B | S1  | magic-num (value/grad pair) | binom-prob `1e-15` ×4 (`pgas.rs:677,695`,`pgas_grad.rs:176,206`); IVP frac `1e-10` ×5 (`pgas.rs:853,1239`,`pgas_grad.rs:435`,`types.rs:73,146`); overdisp `1e-30` ×2 (`pgas.rs:967`,`pgas_grad.rs:350`) | un-named clamps duplicated across value & gradient; one-sided edit breaks NUTS energy conservation                                                                   | named consts (`BINOM_PROB_EPS`, `PROB_FRACTION_EPS`, `OVERDISP_SIGMA_SQ_FLOOR`) |
| T1-C | S1  | fork                        | `correlated_pf.rs:603` vs `resampling.rs:18`                                                                                                                                                            | `sorted_systematic_resample` byte-identical to `systematic_resample` modulo uniform source (and misnamed — nothing sorts)                                            | `systematic_resample_core(weights, u0)`                                         |
| T1-D | S1  | fork                        | `if2.rs:568-584` vs `types.rs:334`                                                                                                                                                                      | ESS recomputed inline, duplicating `ParticleSwarm::ess` degenerate-case branches                                                                                     | free `ess_from_log_weights(&[f64])`                                             |
| X-1  | S1  | fork (cross-slice ×6)       | `ir/caltime.rs` + `browse.rs:1612` + `cas/mod.rs:61` + `fit_table.rs:264` + `table_row.rs:413` + `sim/inference/diagnostic.rs:448`                                                                      | Hinnant civil-date arithmetic copied 6× across 4 crates                                                                                                              | `ir::caltime::{rata_die,date_from_rata_die}` (the documented SoT)               |
| X-2  | S1  | fork (already drifted)      | `obs_loglik.rs:269` vs `external-harness/compare.rs:225`                                                                                                                                                | probit (`normal_quantile`/`inv_norm`) table byte-identical; harness copy lacks the `p` clamp → ±inf where runtime is finite                                          | add `sim` dep, call `normal_quantile`                                           |
| X-3  | S1  | fork (cross-slice ×6/×3)    | `level()`: `fit/cas.rs:70`,`resolve.rs:197`,`sim_ensemble_cas.rs:75`,`pfilter_cas.rs:57`,`survey_cas.rs:58`,`profile_cas.rs:63`; `data_digests` ×3                                                      | `LevelId` ctor with hardcoded `schema_version:1` (identity-bearing) copied 6×; `data_digests` 3×                                                                     | `crate::fit::cas` + `const LEVEL_SCHEMA_VERSION`                                |
| X-4  | S1  | fork                        | `rng.rs:133` vs `lineage/mod.rs:143`                                                                                                                                                                    | `expand_u64_to_seed` (ChaCha8 32-byte expansion) duplicated; lineage copy admits it in a comment                                                                     | `pub(crate) rng::expand_u64_to_seed`                                            |
| T2-E | S1  | fork                        | `propensity.rs:630-653` vs `691-721`                                                                                                                                                                    | per-rate NaN/OOB/negative error classification duplicated verbatim ("replicated verbatim" comment)                                                                   | `classify_rate(p, model, tr, t)`                                                |
| T2-F | S1  | fork                        | `gillespie.rs:248-271` vs `317-340`                                                                                                                                                                     | `arrive`-dispatch closure pair byte-identical between both boundary branches (`diff`-empty)                                                                          | `arrive_at_boundary(...)`                                                       |
| T2-G | S1  | fork (×4 compute)           | `gating.rs:148`, `fit_summary.rs:614,1315`, `method_result.rs:376`                                                                                                                                      | compound IF2 convergence gate (`delta_db`/`se_floor_db`/`threshold_db`) computed 4× independently; `init.rs:1085` hardcodes `30.0` instead of `gate.decibans_thresh` | `gating::compound_gate_legs(state, gate) -> GateLegs`                           |
| T2-H | S1  | fork                        | `mod.rs:1876` (`cmd_fit_diff`) vs `config_diff.rs::compare`                                                                                                                                             | second full config-diff that bypasses `ConfigDiff`; bounds tol disagrees (`>1e-15` vs exact `>0.0`)                                                                  | render `ConfigDiff::compare`                                                    |
| T2-I | S1  | fork (×2/×4)                | `cas/typed.rs` (1 caller); `build_parallel_pool` ×4 (`batch.rs:37`,`profile.rs:1194`,`survey.rs:139`,`pfilter.rs:551`); `run_pooled` ×2                                                                 | cli-local `ContentHash` dup of `runid::ContentHash` (false doc-claim); rayon-pool helper forked 4× ("same shape as X" comment never wired)                           | delete `cas/typed`; hoist pool helpers to `util`                                |
| T2-J | S1  | fork (cross-slice ×5)       | `fit_table.rs:264`,`table_row.rs:413` (ISO parse) + the X-1 sites; SE-aware threshold `gating.rs:162,236`,`dt_check.rs:127`,`init.rs:1085`                                                              | ISO8601→unix parser forked; SE-floor `max(thresh, k·σ·NATS_TO_DB)` spelled 5 ways with k∈{4,8}                                                                       | `ir::caltime` parser; `se_aware_threshold(floor, σ, k)` + named k consts        |
| T2-K | S1  | fork                        | `pfilter_cas.rs:65`/`survey_cas.rs:63`/`profile_cas.rs:68` + the 4 `*_cas.rs` `resolve_*` scaffolds                                                                                                     | `data_digests` + the params-sort/finiteness-gate/`level`-list/`run_id` scaffold copied per subcommand                                                                | shared `crate::fit::cas` helpers + `Resolved` struct                            |
| T2-L | S2  | fork                        | `pgas.rs:1041` vs `pmmh.rs:1126`                                                                                                                                                                        | `struct Diagnostics` + `compute_diagnostics` byte-identical modulo one param-extract closure                                                                         | move to `runner.rs`; pass `extract` closure                                     |
| T2-M | S2  | fork (intra-file)           | `compare.rs:230` vs `367`                                                                                                                                                                               | Δelpd sort-order block duplicated between table and markdown renderers (silent row-order divergence)                                                                 | `render_order(rows, base_idx, base, t_mismatch)`                                |
| T2-N | S2  | fork (cross-slice ×4-5)     | `synthetic.rs:209,217`,`pfilter_cas.rs:130`,`sim_ensemble_cas.rs:82`,`runner.rs:1658`                                                                                                                   | integer-snap float formatter ("render int if within 1e-9") copied                                                                                                    | `util::fmt_int_clean(f64)` + named snap const                                   |
| T2-O | S2  | fork                        | `ir/observation.rs:109` vs `ir/model.rs:67`                                                                                                                                                             | `RegularSchedule` ≡ `RegularOutputSchedule` (same 3 `f64` fields)                                                                                                    | one inner geometry struct; two enums keep it                                    |
| T2-P | S2  | fork (×2 / ×3)              | `method_result.rs:462` vs `552` (summary-map); `:260` vs `:309` (θ̂ filter)                                                                                                                              | per-method PGAS/PMMH result parsing duplicated                                                                                                                       | `summary_scalar_map`, `estimated_theta_hat`                                     |
| T2-Q | S2  | fork                        | `ode_integrator.rs:43` vs `ode.rs:136`                                                                                                                                                                  | RK4 tableau weights spelled by hand in two steppers (justified-variant: different state shapes)                                                                      | shared `rk4_combine`/`rk4_stage` helpers                                        |
| T2-R | S2  | fork                        | `writer.rs:100` vs `tree.rs:709,775`                                                                                                                                                                    | line-list `parent_kind` tag encode/decode forked across module boundary; Parquet vs TSV disagree on unknown-tag strictness                                           | `ParentRef::from_tag(&str)`                                                     |
| T3   | S2  | god-module                  | `util.rs` (7 subsystems), `config_v2.rs` (5 resp.), `runner.rs:119-707` (`build`, 588 lines), `compiled_model.rs:730-1293` (`new`, 563 lines), `multi_stream_obs.rs` (mild)                             | SRP violations                                                                                                                                                       | named module/fn splits (see Tier 3)                                             |

---

## Tier 1 — inference-math forks (fix first; scientific stakes)

These are the findings where a silent divergence moves a posterior that informs
policy. Small in number, highest in priority.

### T1-A — `Prior::log_density` and the hierarchical hyperprior density are the same 7 distributions, written twice

`prior.rs:96` and the `match HierarchicalKind` at `hierarchical.rs:175`+ both
evaluate `log p(θ)` over the same `(natural, transformed)` signature for Normal,
LogNormal/TransformedNormal, HalfNormal, Beta, Gamma, Exponential, and
Uniform/LogUniform — with **byte-identical formulas**. Beta arm:

```rust
// prior.rs:133            (alpha-1)·ln natural + (beta-1)·ln(1-natural) - [lgamma(α)+lgamma(β)-lgamma(α+β)]
// hierarchical.rs:223     identical
```

A hyperprior and a prior _are_ the same density. Drift across these two = an
inconsistent hierarchical posterior, undetectably.

**Fix:** `hierarchical.rs` maps its `HierarchicalKind` + resolved
hyperparameters into a `Prior` value and calls `Prior::log_density`, deleting
the parallel match. If a hyperprior needs a density `Prior` lacks, add the arm
to `Prior` — the single density authority — not a second match.

### T1-B — un-named probability clamps duplicated across value & gradient paths

Three distinct probability/variance thresholds are bare literals, each appearing
in _both_ the density (`pgas.rs`) and its gradient (`pgas_grad.rs`):

- binomial split/exit interior clamp `1e-15` — `pgas.rs:677,695`,
  `pgas_grad.rs:176,206`
- IVP / logit fraction clamp `1e-10` — `pgas.rs:853,1239`, `pgas_grad.rs:435`,
  `types.rs:73,146`
- overdispersion variance floor `1e-30` (gates whether a gamma multiplier is
  counted) — `pgas.rs:967`, `pgas_grad.rs:350`

```rust
// pgas.rs:695        let p_split = (eff_rate/rate_remaining).clamp(1e-15, 1.0 - 1e-15);
// pgas_grad.rs:206   identical
```

These are **matched pairs that must stay equal**: if the value path and gradient
path clamp differently, NUTS sees a non-conservative Hamiltonian → energy drift
→ spurious divergences or biased sampling. The `1e-15` here also collides in
value (but not concept) with `RATE_EPSILON` and `MIN_STEP_EPS`.

**Fix:** three named consts with doc comments (`BINOM_PROB_EPS`,
`PROB_FRACTION_EPS`, `OVERDISP_SIGMA_SQ_FLOOR`), each referenced by both the
value and gradient site. Keep names distinct from the `1e-15` time/rate floors
per "distinct concepts keep distinct names even at equal value."
`correlated_pf.rs:551` reuses `1e-10` for a _different_ concept (base-uniform
clamp) — give it its own name, don't fold it in.

### T1-C — `sorted_systematic_resample` is a byte-identical fork (verified seed)

`correlated_pf.rs:603` and `resampling.rs:18` have identical selection loops;
only the uniform source differs (`rng.uniform()` vs passed `base_uniform`). The
name also lies — nothing is sorted.

**Fix:** extract
`systematic_resample_core(weights: &[f64], u0: f64) -> Vec<usize>` in
`resampling.rs` (taking already-normalized weights + base uniform).
`systematic_resample` draws `rng.uniform()` and delegates; `correlated_pf`
passes `base_uniform` and deletes its copy. Rename to drop "sorted".

### T1-D — `if2` ESS inline fork

`if2.rs:568-584` recomputes `ESS = (Σw)²/Σw²` on max-shifted log-weights with
the same `0.0`-on-non-finite contract as `ParticleSwarm::ess` (`types.rs:334`) —
the comment admits it ("same formula as `ParticleSwarm::ess()` … inlined here"
because IF2 holds `Vec<f64>`, not a `ParticleSwarm`).

**Fix:** extract `ess_from_log_weights(&[f64]) -> f64` in `types.rs` (the
existing home of `ess`/`log_sum_exp`/`logw_variance` as free fns over `&[f64]`);
both `ParticleSwarm::ess` and `if2` call it.

---

## Cross-slice forks (orchestrator synthesis — no single slice saw these whole)

### X-1 — civil-date arithmetic copied at 6 sites across 4 crates

Howard Hinnant's Gregorian era arithmetic (`146097`-day era, `719468` epoch
shift) is implemented at six production sites:

| Site                                                    | Direction                      |
| ------------------------------------------------------- | ------------------------------ |
| `ir/caltime.rs:59 rata_die` + `:160 date_from_rata_die` | **canonical (documented SoT)** |
| `cli/browse.rs:1612 days_from_civil`                    | civil→days                     |
| `cli/cas/mod.rs:61 civil_from_secs`                     | days→civil                     |
| `cli/fit/fit_table.rs:283` (in `parse_iso_to_unix`)     | civil→days                     |
| `cli/fit/table_row.rs:443 days_from_civil`              | civil→days                     |
| `sim/inference/diagnostic.rs:448-450`                   | days→civil                     |

CLAUDE.md explicitly names `ir/src/caltime.rs::rata_die` as the single source of
truth ("mirror only with an equivalence test"). `browse.rs` even admits its copy
is "the inverse of the one in cas.rs." Drift = wrong fit-listing ages/sorts and
wrong provenance timestamps.

**Fix:** add `pub fn parse_iso8601_utc_to_unix(&str) -> Option<i64>` to
`ir::caltime` (layered on `rata_die`; the existing `parse_iso_date` rejects the
`T…` time-of-day form, so the _parser_ is new but the day-count must call
`rata_die`), and route all five non-canonical sites through `ir::caltime`. Mind
the epoch-offset convention difference (`-719468` vs `rata_die`'s 1970 base)
when wiring — that mismatch is itself the argument for one home.
**Verification:**
`rg -n '146097|146_097|719468|719_468|days_from_civil|rata_die|civil_from_secs' rust/crates`.

### X-2 — probit table forked into the validation harness, already drifted

`obs_loglik.rs:269 normal_quantile` and
`external-harness/compare.rs:225 inv_norm` share the byte-identical
Beasley-Springer-Moro coefficient tables (`A`/`B`/`C`/`D`,
`C[0] = -7.784894002430293e-03`). But `normal_quantile` clamps its input
(`p.clamp(1e-300, 1.0 - 1e-16)`) and `inv_norm` does **not** — so the harness's
probit returns ±inf/NaN at `p∈{0,1}` where the runtime is finite. The thing that
_certifies the runtime correct_ validates against a drifted copy of the
runtime's own quantile. `external-harness/Cargo.toml` has no `sim` dependency —
which is why the copy exists.

**Fix:** `normal_quantile` is already `pub`. Add `sim = { path = "../sim" }` to
external-harness and call it (deletes ~30 lines, inherits the clamp); or hoist
the table to a shared `numerics` leaf crate. **Verification:**
`rg -n 'fn normal_quantile|fn inv_norm|7\.784894002430293' rust/crates`;
`rg -n 'sim' rust/crates/external-harness/Cargo.toml` → no match.

> **Note — corrected seed lead.** I originally labeled this a duplicated
> _digamma/lgamma_ table. It is the _probit/`normal_quantile`_ table. `lgamma`
> and `digamma` are in fact single-sourced (`obs_loglik.rs:10/43`, imported by
> `prior.rs`/`hierarchical.rs`). The duplication is real — and worse than I
> guessed (already drifted) — but the function was misnamed in the seed.

### X-3 — `LevelId` constructor + `data_digests` copied across the CAS surface

`fn level(name, label, hash) -> LevelId` with a hardcoded `schema_version: 1`
appears at **6 production sites** (`fit/cas.rs:70`, `resolve.rs:197`,
`sim_ensemble_cas.rs:75`, `pfilter_cas.rs:57`, `survey_cas.rs:58`,
`profile_cas.rs:63`); `data_digests` at 3 (`pfilter_cas`, `survey_cas`,
`profile_cas`). The schema version is identity-bearing (folds into every
`run_id`); a bump that lands in 5 copies but not the 6th silently keeps the old
key for one artifact kind. The four `*_cas.rs` `resolve_*` also share the
params-sort → `ensure_finite` → `level`-list → `run_id` scaffold.

**Fix:** hoist `level` (with `const LEVEL_SCHEMA_VERSION: u32 = 1`),
`data_digests`, and a `Resolved { levels, run_id }` helper into
`crate::fit::cas` (already the home of `digest_value`/`ensure_finite`); the
per-kind _level composition_ stays in each file (the genuine difference), the
scaffold consolidates. **Verification:**
`rg -n 'fn level\(|fn data_digests' rust/crates --type rust -g '!*/tests/*'`.

### X-4 — ChaCha8 seed expansion duplicated

`expand_u64_to_seed(u64) -> [u8;32]` (same three multipliers
`0x9e3779b97f4a7c15`/`0x6c62272e07bb0142`/`0xd800000000000000`) is
byte-identical at `rng.rs:133` and `lineage/mod.rs:143`; the lineage copy's
comment says "Duplicated rather than exported." The "byte layout consistent with
StatefulRng" guarantee is enforced by nothing — if the sim seeding changes, the
lineage RNG silently diverges.

**Fix:** make `rng::expand_u64_to_seed` `pub(crate)`;
`LineageRng::from_sim_seed` calls it.

### X-5 — the `~1e-9` schedule/time tolerance family

A `1e-9` tolerance for "two times are the same instant / include the grid
endpoint" is named three ways and inlined several more across sim **and** cli:
`schedule`-grid end-bounds (`output.rs:9`, `intervention.rs:93,99` inline;
`reactive.rs` `EMIT_EPS`/`WINDOW_EPS`; `intervention.rs` `GRID_TOL`), obs-time
matching (`util.rs:1404,1446`, `caltime_load.rs:289`), grid-endpoint inclusion
(`eval.rs:160`, `main.rs:1869`), and fit time-alignment (`runner.rs` ×9). `ir`
already has the model to copy: `validate.rs:296 INT_TOL = 1e-9`, named and
documented.

This is **not** "one `EPS` everywhere" — some are genuinely distinct concepts at
the same value (the maintainer's rule). The fix is: name each concept once in
its owning module, and _the maintainer decides_ which are the same concept (e.g.
should cli's obs-time-match epsilon BE sim's schedule-grid const, since obs
times are matched against the sim grid?). Captured here as one coordinated
cleanup so the decision is made once, not 15 times.

---

## Tier 3 — god-modules / god-functions (structural; do each as its own commit)

- **`util.rs` (3,944 lines) — 7 subsystems.** seeding · camdlc
  discovery/version-check · IR caching · model loading · scenario/summary ·
  simulation running · TSV output. The file is already sectioned with comment
  banners that map 1:1 to a split: `camdlc.rs`, `ir_cache.rs`, `model_load.rs`,
  `sim_run.rs`, `obs_window.rs`, `traj_output.rs`, leaving a small `util.rs` (or
  `seed.rs`).
- **`config_v2.rs` (6,074 lines) — 5 responsibilities.** types+serde+defaults ·
  resolution/expansion · validation · migration string-scanning · `impl Stage`.
  Cleanest first carve: `migrate.rs` (the three legacy-detection text scanners,
  zero coupling to the typed structs). NB: `FitConfigV2`/`EstimateSpecV2` carry
  a `V2` suffix with **no surviving V1** — drop the suffix (separate commit).
- **`FitRunConfig::build` (`runner.rs:119-707`, 588 lines).** load +
  resolve-params
  - compile + obs-stream disk I/O + condition-from resolution + infer-setup.
    Extract `load_and_compile_model`, `resolve_all_params`, `load_obs_streams`,
    `resolve_condition_from_holes`; `build` becomes a ~40-line assembler.
- **`CompiledModel::new` (`compiled_model.rs:730-1293`, 563 lines).** highest
  blast-radius function in `sim`. Split into `resolve_expressions`,
  `resolve_schedules_and_balance`, `derive_and_check_capabilities`,
  `build_forcings`; `new` orchestrates.
- **`multi_stream_obs.rs` (1,907 lines) — mild.** Separable: the bind-time
  validation/diagnostic types (`Finding`/`Severity`/`BindReport`/`render`) →
  `multi_stream_obs/bind.rs`, leaving scoring/projection/sampling in the parent.
  (No hot-loop clones found here — all 28 `.clone()` are setup or test.)

---

## Tier 4 — un-named tolerances in control flow (name once, in the owning module)

Beyond the X-5 `1e-9` family and the T1-B clamps:

- **fit drivers** (`runner.rs`): IF2 eval interval `10`, eval-particle cap
  `500`, multimodal ll-spread `50.0`, low-ESS fraction `0.05`, param-near-bound
  `0.01`, degenerate-W rel-scale `1e-6`, MAD floors
  `1e-15`/`3.0·mad`/`0.5·good_mad`, rw_sd heuristics `/20.0`,`/6.0`. R-hat
  verdict bands `1.1`/`1.5` and acceptance band `0.10`/`0.50` inlined in
  `pgas.rs`/`pmmh.rs`/`runner.rs` while `RHAT_THRESHOLD=1.1` exists named
  (`runner.rs:2279`) — reference it; give the IF2-optimizer Â its own
  `A_AGREEMENT_THRESHOLD` (distinct concept, same value).
- **gate/dt** (`gating.rs`,`dt_check.rs`): SE-floor multiplier `8.0` (and its
  `4.0` half) bare at every gate site → `SE_FLOOR_DB_MULT`; τ table `0.5`/`2.0`/
  `0.1`/`0.5` bare in `dt_check.rs:114` (note `0.5` means two different things).
  Â colour cutoffs `1.05`/`1.10` in `fit_summary.rs:683` should reference
  `gating::{A_SOFT,A_HARD}`.
- **cli ingest**: obs-time match `1e-9` (`util.rs`,`caltime_load.rs:289`),
  substep- distinct `1e-12` (`caltime_load.rs:268`) — distinct concept from the
  `1e-9`, keep distinct names; grid-endpoint `1e-9` (`eval.rs`,`main.rs`);
  progress-span div-guard `1e-9` (`util.rs:2723`, distinct).
- **subcommands**: int-vs-float TSV threshold `1e15` (`batch.rs:1659`) →
  `MAX_EXACT_INT_DRAW`; survey `DOUCET=1.7` is correctly named (model).
- **sim core**: calendar constants in `ir/caltime.rs` (`694025` epoch offset —
  the load-bearing OCaml-matched one — plus `719468`/`146097`/`1e6` frac-day
  grid) should be named consts; `BATCH_ROWS=8192` defined twice
  (`writer.rs:197`/`event_log_io.rs:384`).
- **TSV precision fork**: `io/trajectories.rs:289,299` writes real values at
  `{:.6}` while `cli/util.rs:2927,2931` writes `{:.4}` — for a format the `io`
  crate doc calls unified. Name `REAL_FMT_PRECISION` once; route `simulate`'s
  writer through `io::trajectories`. (Which precision is canonical → maintainer
  call; also kicked to correctness since mixed-precision files may have
  shipped.)

---

## Tier 5 — mechanical / clarity (cheap)

- Version-header `writeln!(f, "# {VERSION}")` ×6 in `runner.rs` →
  `write_version_header`.
- `derive_chain_seed` inlines `0x9e3779b97f4a7c15` next to the same-valued named
  `SEED_MIX_DRAW` (`util.rs:25`) → add distinct-named `SEED_MIX_CHAIN`.
- `Vec<&String>` sort-scratch in `batch.rs` (×7), `survey.rs:761` → `&str`.
- dead passthrough param `_landscape_path` in `survey.rs:1404` → delete wrapper.
- stale module comment `main.rs:6` (`cas_read` "transitional alongside
  run_meta") → reword (they read disjoint files).
- `PriorSource`→wire-string `match` duplicated (`profile.rs:914`,`mod.rs:1833`)
  → `impl PriorSource::as_wire_str`.
- `cas.rs` internal: `cas_dep_from_dir` ≡ `cas_survey_dep` → `cas_dep_for(...)`.
- `runid/hash.rs:256` `write_str_map` takes `&String` keys → `&str`.

---

## Kicked to the correctness pass (NOT investigated here — input for the next review)

Seed/RNG aliasing: `loglik_eval.rs:263` (`chain*10_000+k`), `runner.rs:1897`
(`seed+chain*1000+it`), `mod.rs:1542` (`seed ^ r*0x7f4a7c15`) — confirm offsets
can't alias paired-seed CRN. `cas.rs:412` collapses `process_seed`/`base_seed`
to one value. `sim_job.rs:166` invents seed `1` for an empty explicit-seed list.

Numerics: `pgas_grad.rs:182` binomial gradient uses `1-p` not the stable clamped
`q`. `obs_model.rs:222` inlines a Binomial gradient — verify it matches the pgas
form. `config_diff.rs:188` vs `mod.rs:1937` bounds-equality `>0.0` vs `>1e-15`
(the two `fit diff` surfaces can disagree). `io` vs `cli` real-value TSV
precision (`{:.6}` vs `{:.4}`).

Edge cases / robustness: `config_v2.rs:584` `parse_seed_range` unbounded range
size. `method_result.rs:430` `?` inside a per-line loop aborts the whole ESS
parse on one bad row. `synthetic.rs:163` NaN obs time survives dedup.
`tree.rs:778` Parquet reader maps unknown `parent_kind` → `None` silently while
TSV (`tree.rs:714`) hard-errors. `compiled_model.rs:69` `CubicSpline::new` uses
`assert!` (panic) for user-reachable forcing-data validation — against "never
panic for user-facing errors." `main.rs:2040` `snap_at` find-first-≤ snapshot
selection. `caltime_load.rs:265` substep-collision key cast under large dt.

---

## Recommended sequencing

1. **Tier 1 (A–D)** — inference-math forks. Each is small, each is its own TDD
   commit (extract core → assert both callers identical → delete copy). T1-B is
   the highest-leverage: it removes a class of value/gradient drift that NUTS
   cannot self-diagnose.
2. **X-2 (probit) and X-1 (civil-date)** — both confirmed forks, X-2 already
   drifted. X-2 is a ~30-line deletion + one Cargo dep; X-1 is a `caltime`
   extension + 5 call-site rewrites.
3. **X-3 / X-4 / T2-K (CAS + seed plumbing)** — identity-bearing; consolidate to
   `crate::fit::cas` and `rng`.
4. **Tier 3 god-module splits** — mechanical moves, one commit each, no behavior
   change; do `util.rs` and `config_v2.rs` first (highest reader cost).
5. **Tier 4 tolerance naming** — coordinate the X-5 `1e-9` decision once, then
   sweep. Tier 5 is fill-in.

Each fix is read-only-reviewed here; none applied. Per CLAUDE.md, dead-code
deletions and module splits land as their own commits before any substantive
change.
