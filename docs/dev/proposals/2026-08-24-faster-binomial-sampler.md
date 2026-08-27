# A faster binomial sampler for PGAS

Date: 2026-08-24 Status: proposed\
Note ref: `docs/dev/notes/2026-08-24-pgas-binomial-sampler-is-half-the-fit.md`\
Supersedes: the env-var-identity-class draft of this file (see "What changed and
why" below)

Re-keying is authorised by the maintainer (2026-08-24).

## Problem

Profiling `bvd_province_nu_carec.camdl` at production settings puts **82.4%** of
samples in `chain_binomial::step_one` and **38.9%** in `StatefulRng::binomial`
alone; a ceiling probe that made the draw free moved the whole fit **2.01×**.
The sampler plus the RNG bytes it consumes is about half the run. `rand_distr`
0.4.3's BTPE pays ~10 setup constants and two `Uniform` constructions per call,
then walks from the distribution's mode to the sampled value with one f64
division per step — which fires on almost every draw. Full evidence in the note.

**The sampler ratio is measured at 1.48×** by `binomial_ab`, over three stable
runs. **The whole-fit factor is not measured and cannot be yet** — BTRS is
unreachable from production, so ~1.15× is Amdahl on 1.48× at the profiled ≈39%,
spanning 1.11–1.19× across a 30–50% share. Two earlier figures in this family
were wrong: `1.25–1.35×` from flop counting (the mistake
[`2026-06-14-flat-bytecode-evaluator.md`](../notes/2026-06-14-flat-bytecode-evaluator.md)
records — synthetic 2.5×, real corpus 1.27×), and `1.58× → ~1.17×` from a bench
blend that weighted cells equally and omitted the BINV-routed draws (§4).

All three profile numbers above come from a run where NUTS did nothing —
`n_leapfrog = 1`, 0% acceptance, θ frozen at an infeasible start. The ≈39% share
is robust to that; the `(n, p)` cell set the 1.48× rests on is not. §4 and the
note carry the analysis.

## What changed and why

The first draft of this proposal made the sampler selectable by an **environment
variable**, and — having found that env vars do not enter the run address —
proposed a general env-var Identity/Provenance registry whose Identity members
would be folded into the engine-version string. Adversarial review killed all
three parts, and the maintainer confirmed env vars were never meant to be in
scope. Recording it so the reasoning is not repeated:

- **The split already exists, compile-enforced.** `runid-derive`'s doc:
  `#[derive(RunInput)]` is "include-by-default", `#[run_input(provenance)]`
  skips a field, "a field whose type is not `ContentAddressed` is a compile
  error — you cannot forget to make an input hashable", and "the macro
  **replaces** hand-written canonical hashing; **there is never a second
  implementation**." 18 live `#[run_input(provenance)]` uses. Folding a toggle
  into a version string is that second implementation.
- **The repo already decided this class the other way.** gh#241 removed
  `CAMDL_PF_WALLCLOCK_TIMEOUT_S` rather than hashing it —
  `sim/src/inference/degeneracy.rs`: "an un-typed input channel outside the
  CLI/TOML surface. **It is removed**" — and replaced it with a typed
  `--pf-max-substeps`. `2026-06-16-input-surface-and-cas-unification.md` §2.4
  ("Environment variables — the third door") already classifies this variable
  set. Neither was cited.
- **The string fold was defective on its own terms**: unvalidated values alias
  two configurations onto one address (`CAMDL_BINOMIAL="btrs+prng=x"` vs two
  separate toggles), the order of two active toggles was unspecified, and
  `scripts/gc_camdlc.sh`'s `sed -n 's/.*+\([0-9a-f][0-9a-f]*\).*/\1/p'` binds to
  the last `+` and would resolve the live compiler's hash to `b`, then delete
  it.

One claim from that draft survived scrutiny and is **partly refuted**: suffixing
`VERSION_SHORT` does **not** break the camdlc handshake or the IR cache — both
key on `version::GIT_HASH` (`cli/src/util.rs`). The third of that claim was
wrong, and in the direction that matters: **the CAS fit store does key on
`VERSION_SHORT`.** `cli/src/fit/cas.rs` puts `EngineVersion(engine_version)`
into `FitDigest` twice, and `cli/src/fit/mod.rs` sets `engine_version` to
`crate::version::VERSION_SHORT`. (The two are not independent: `VERSION_SHORT`
is the package version, a `+`, and `GIT_HASH`, so the hash is a substring of
it.) The conclusion the claim was drawn for holds, but as written it told a
reader that engine version is not in the fit address, when it is: suffixing it
would re-key the entire store. Moot now; it was the risk gh#746 was filed
against.

## What this does NOT do

- **No IR change, no `ir/VERSION` bump, no golden moves.** Runtime numerics
  only.
- **No environment variable.** The A/B seam that already landed is a
  thread-local override used by tests and benches; the production door is a
  typed config field (§1).
- **No env-var registry.** Dropped entirely. `CAMDL_EVAL_FLAT`'s absence from
  run identity is a real defect but a separate one — gh#746, to be re-scoped.
- **Does not touch BINV.** `StatefulRng::binomial`'s small-`np` branch carries
  the gh#510 unbounded-loop and gh#525 large-`n` fixes.
- **Does not change `particles`, `sweeps` or `chains`.** See "The larger lever".

## Design

### 1. The knob is a typed stage field, and identity is free

Add `binomial: BinomialAlgorithm` (default `Btpe`) to `Stage::PGAS` in
`cli/src/fit/config_v2.rs`, spelled `binomial = "btrs"` in the stage TOML.

**It enters run identity with no plumbing at all.** `Stage::identity_payload()`
is already subtractive — it serialises the whole variant and removes only two
named keys (`sweeps` and `n_trajectories`, both folded elsewhere) — and its own
comment states the invariant: "a new field is hashed unless it is deliberately
named below." That is the seam the 2026-08-23 run-identity work built precisely
so that a new stage field could not be forgotten, and it makes every worry in
the previous draft evaporate: distinct addresses for the two arms, four
coexisting cells for the 2×2, and a `run.json` that says which sampler produced
it.

Two existing tests are the human-loop check and both must be updated
deliberately, not mechanically:
`identity_payload_includes_every_field_but_the_named_exclusions` enumerates the
included keys, and `identity_payload_is_byte_stable_against_recompiles` pins
golden bytes (so adding the field re-keys — authorised).

**Resolve once, so the hash and the run cannot disagree — but NOT through the
thread-local.** An earlier revision of this section said "the chain worker calls
`rng::set_binomial_algorithm(stage.binomial)` once at entry". **That does not
work, and would have produced exactly the defect this proposal exists to
prevent.** Recorded rather than quietly replaced, because it reads as obviously
correct:

- The draws do not happen on the thread that reads the config. `csmc_as`
  propagates particles with `counts.par_iter_mut()…map(… step_one(…, rng, …) …)`
  (`sim/src/inference/pgas.rs`), nested inside the per-chain
  `(0..n_chains).into_par_iter()` in `cli/src/fit/pgas.rs`, both on the one
  process-global pool `cli/src/fit/mod.rs` builds — whose own comment says "a
  fit nests Rayon parallelism (chains × particle filter) on the global pool".
- So a thread-local set at chain entry is visible only to whichever particles
  happen to be stolen by that same worker. Measured on a standalone harness
  against the real crate: **0 of 4096** particle draws saw the selection.
- The obvious repair — set it inside the `par_iter` closure — is worse. Rayon
  reuses pool workers across jobs, so the value persists: in a two-stage fit
  where stage 1 selects `btrs` and stage 2 selects nothing, **1985 of 2048**
  draws in stage 2 came out non-BTPE. A stage addressed as `btpe`, drawn with
  BTRS.

Either way the stage leaf would record a sampler the run did not use, and the
mixed case is not even reproducible run-to-run because the split depends on
work-stealing. `gate_pgas_thread_invariance.rs` is the one existing test that
would catch it.

**The resolved value must therefore travel as a value**, threaded from the
hashed field to `step_one` — through `PGASConfig` and the propagation call
chain, the way `params` and `per_eval` already travel. The thread-local stays
what it is today: a test/bench affordance, asserted to have no production caller
by `rng::btrs_tests::nothing_outside_tests_and_benches_selects_a_sampler`, with
`rng::active_binomial_algorithm()` available so "the sampler that ran equals the
one that was hashed" can be stated as an assertion at all.

### 2. BTRS — landed

`rng.rs`, commit `2b54fa6e`. Hörmann (1993), transcribed from TensorFlow's
`random_binomial_op.cc` (Apache-2.0, same as camdl), deliberately the
**TensorFlow variant** — TFP notes its own deviation from the paper ("there is a
log missing"), and the variant here is the one the domination proof below
verifies.

**Correctness rests on deterministic sweeps, not on the transcription** — and on
**three** conditions, not the two an earlier revision of this section claimed.
That revision said BTRS "is exact iff its hat dominates the pmf and its squeeze
never accepts what the density would reject". Those are necessary, not
sufficient, and the missing one was the only condition nothing tested:

- The hat's Jacobian cancels the proposal density exactly, so BTRS returns `k`
  with probability `exp(log_bound(k))/α`. Exactness therefore **also** requires
  `exp(log_bound(k)) ∝ pmf(k)`.
- `hat_dominates_and_squeeze_is_valid` cannot see that, structurally: it forms
  `V` **from** `log_bound`, so a k-dependent error that lowers `log_bound` keeps
  `V ≤ 1` and leaves the sweep green while the distribution is wrong.
- Demonstrated, not theorised. Deleting `- stirling_approx_tail(k)` from
  `log_bound` — the exact shape of the deviation TFP documents in its own BTRS —
  left the **entire suite green**, as did scaling all ten `TAIL` entries by
  1.10, zeroing them, a one-digit typo in one of them, and four mutations of the
  asymptotic series. All 13 constants in `stirling_approx_tail` were untested.
- `log_bound_is_proportional_to_the_exact_pmf` now pins it, walking the
  reference log-pmf by its own recurrence rather than from `lgamma` (whose
  k-dependent terms are ~1e8 at `n ≈ 9e6` and would swamp the ~1e-9 property).
  Verified to catch all six mutations.

The domination sweep remains the right instrument for the other two conditions,
and it is still the only one that works for a constant typo:

> A one-digit error in `b` (`2.53 → 2.63`) distorts the tails symmetrically
> about the mode, leaving the mean bias at **exactly zero** while the
> distribution is wrong. A moment test is structurally blind to it and a χ²
> needs ~10⁸ draws. The sweep catches it in milliseconds.

It calls `BtrsHat`'s own methods, so it checks the **shipped** arithmetic rather
than a second copy of the formula. But its **domain matters as much as its
statistic**, and this too was overclaimed as "catches every other
single-constant typo": the model-derived cells missed three typos that genuinely
break BTRS — `m` losing the `+1` in `(n+1)p`, `v_r`'s `4.2 → 4.1`, and `alpha`'s
`5.1 → 5.0`. Three adversarial cells now cover them, each the sole witness to
one.

Findings pinned as tests:

- **`BINV_THRESHOLD` is a correctness boundary, not a speed knob.** Restated
  with the measured numbers, which were both wrong before: the thinnest
  domination margin in the routed domain is **0.22%** (at `(23, 0.4583)`,
  `np = 10.54`), not "a few percent", and domination first fails at
  **`n·p ≈ 9.64`**, not 7 — a gap of 3.6% in `np`, not 30%.
  `the_hat_stops_dominating_below_the_routing_threshold` asserts the failure so
  nobody lowers the threshold to buy draws.
- **`BTRS_MAX_N` is the same kind of boundary at the other end.** `log_bound`'s
  `(n+1)·ln(…)` term loses precision as `n·ε`, so the hat stops dominating above
  `n ≈ 1e12`: `sup V` is 0.975 at 1e13, 1.06 at 1e15, 2.7e46 at 1e18, and at
  `n = u64::MAX` the mean comes out 8.6% low with a 130σ outlier. BTRS
  de-selects itself above the bound.
- **A non-finite `p` is guarded in `binomial`, not in the sampler.** NaN passes
  both range guards (every NaN comparison is false) and then fails to route to
  BINV for the same reason; BTPE absorbs it, BTRS **span forever**, with only a
  `debug_assert!` in front — i.e. nothing in release. The gh#510 hang class.
- **The support check is hoisted above the squeeze** — a deliberate deviation
  from the reference. A no-op wherever the in-support guarantee holds; where it
  might not, it redraws instead of returning a `k > n` that the `p > 0.5`
  reflection turns into a u64 underflow of ~1.8e19.

The distributional suite (exact-PMF χ² with BTPE as positive control, moments at
the province regimes, support, threshold continuity, the gh#510/gh#525
pathological inputs) is **permanent** and outlives any toggle. It is calibrated
against a **location** shift: `chi_square_rejects_a_one_percent_bias` fails if
the suite loses the power to detect a 1% shift in `p`. Do not read that as
calibration against the failures BTRS actually risks — the same machinery misses
`b: 2.53 → 2.63` on all seven cells at 200k draws, needing ~10⁷–10⁸. Tail-shape
distortion is what the sweeps are for.

### 3. Scope the flip properly — the sampler is not confined to `chain_binomial`

`.binomial()` has five production call sites, and only two are the hot path:

```
sim/src/chain_binomial.rs:726, 743     the profiled path
sim/src/inference/obs_model.rs:547, 565  HIGH-RISK per CLAUDE.md
sim/src/compiled_model.rs:2075
```

So flipping the default changes **observation-model draws and synthetic-data
generation on every backend**, including ODE and Gillespie — cells the "82.4% in
`chain_binomial::step_one`" framing never mentions.
`.claude/rules/sim-and-inference.md` requires the backend × method matrix to be
dense, so the flip commit must name the cells, not hand-wave them. The
re-baseline set is nameable: `gate_trajectory_baseline.rs`,
`gate_inference_baseline.rs`, `gate_pgas_density_baseline.rs`,
`gate_corner_case_baseline.rs`, re-captured with `CAMDL_CAPTURE_BASELINE=1`.

**And there is a third binomial path.** `chain_binomial.rs:708–724`: when
`binomial_z_values` is populated (correlated PF / PMMH), the total-exit draw
bypasses `rng.binomial` entirely for a **normal approximation**, while the split
draws in the same loop still route through it. After a default flip that is the
only inexact binomial left in propagation. Not this proposal's job to fix, but
"one sampler" must not be claimed.

`binomial_matches_rand_distr_in_value_and_rng_words_consumed` (`rng.rs`) asserts
exact value **and** `get_word_pos()` against `rand_distr` on three BTPE cases.
It passes today because BTPE is the default; the flip commit must split it so
the BINV half keeps its oracle rather than deleting the rows, and adjust its
`assert_eq!(compared, 64)` non-vacuity guard deliberately.

### 4. Bench before flipping — landed

`benches/binomial_ab.rs`. Median-of-9, 400k draws/cell, arms interleaved in one
process with the order alternating by rep parity, `[min,max]` printed per cell.

**Blended sampler factor: 1.48×** (1.45× total-exit, 1.55× arm-routed splits,
1.00× at the BINV-routed splits), stable across three consecutive runs. Cells
are weighted by how many of the 24 draws per particle-substep they stand for.

An earlier revision reported **1.58×** and two things were wrong with it:

- It weighted the five total-exit cells equally. `E` is 6 of the 12 total-exit
  draws (2 provinces × 3 stages) and was getting 1/5 of the weight — and it is
  the cell with the _lowest_ ratio, so correct weighting moves total-exit from
  1.55× to 1.45×.
- It omitted the splits that route to BINV in **both** arms, thereby asserting
  24/24 draws reach the arm under test. The model's competing exits include
  hazards ~3 decades below their group's dominant rate (`export_e` ≈ 6e-4 on
  `n_exit ≈ 190`, so `np ≈ 0.1`), which BTRS never touches. Their absence was
  already contradicted by the note's own BINV-at-17%-of-binomial-time figure.

Under its own superseded accounting this machine reports 1.51–1.52×, so roughly
half the gap is the methodology fix and half is between-session variation. **Two
significant figures is the available precision.**

**The "easy half" claim was backwards and is withdrawn.** It read: "the splits
sit where BTRS's squeeze fires least often … benching `np ≈ 190` alone would
have measured the easy half." The mechanism is right — `v_r` is 0.30 at
`(20, 0.5)` against 0.76 at `(400, 0.476)`, and BTRS _is_ slower at the splits
in absolute terms — but BTPE degrades more there (its immediate-accept triangle
collapses from 63% to 21% of the `u` range), so the **ratio is larger** at the
splits. `np ≈ 190` is the total-exit regime; benching it alone would have
reported ~1.45× and **understated** the win.

The bench prints the implied whole-fit factor across assumed shares 30–50%
rather than one number, because ≈39% is a profile measurement with a point or
two of discretion and is a **lower bound** (`rand_distr` only partially inlines
— proven by `UniformFloat::<f64>::new` appearing as its own 0.59% leaf). At
1.48× that band is **1.11–1.19×, centre 1.15×**.

**Every figure here comes from a run in which NUTS did nothing** —
`n_leapfrog = 1`, `tree_depth = 1`, 0% acceptance, θ frozen at an infeasible
start. The share survives that (see the note: even a 1000× rise in leapfrog
count moves the implied factor only 1.15× → 1.14×), but the `(n, p)` **cell set
does not**: `np` scales with prevalence and `kappa` is log-uniform over seven
decades, so the split regimes at stationarity are unknown. Two split cells sit
exactly on the `n·min(p,1−p) = 10` routing boundary, where a hair less `p` sends
them to BINV in both arms at 1.00×. Re-harvest from a healthy chain before the
flip.

### 5. The PRNG is a separate question, and its measurement is contingent

ChaCha8 block generation is 8.6% of the profile — worth ~1.06–1.09×, inferred,
not measured. BTRS keeps two uniforms per attempt so Phase 2 does not subsume
it.

Cheaper than it looks in one respect: `ChainResumeState` stores params,
trajectory and NUTS adaptation but **not** RNG state, and PGAS re-derives
per-particle streams from `(seed, particle_index)` each sweep, so the resume
format is not coupled to the generator. The costs: `StatefulRng` is a newtype
over `ChaCha8Rng` whose `inner_mut()` leaks the concrete type to 8 call sites;
`set_stream` is the independence contract 9600 streams rest on; every
determinism gate re-baselines.

**The equivalence suite must be parameterised over the PRNG before this lands.**
BTRS consumes a variable number of uniforms _correlated with the returned k_ —
inert under ChaCha8, but a data-dependent consumption stride over ~10¹² draws is
exactly how a weak generator's equidistribution defects surface. A
wall-clock-only 2×2 would ship the riskiest cell unmeasured.

### 6. The 2×2, then one commit

With both knobs typed and hashed, run the full factorial end-to-end on fit-stage
leaves. **State the prediction first:** BTRS reaches an expensive test far less
often than BTPE, so the PRNG's win measured on top of BTPE should overstate its
value in the final configuration. If the gains simply multiply, that prediction
was wrong and gets written down.

Then one commit flips the defaults and deletes the losing arm. It needs **its
own issue** — Phase 3 currently shares gh#748 with Phase 2, so when that closes
as "PRNG swapped" the deletion has no surviving tracker, which is the orphan
`rust-conventions.md` warns about.

## Sequencing and re-key inventory

| step | lands                                                        | re-keys                                                         |
| ---- | ------------------------------------------------------------ | --------------------------------------------------------------- |
| 1    | gh#747 — BTRS + the three sweeps + bench (**landed**)        | nothing (not reachable from production)                         |
| 2    | typed `binomial` on `Stage::PGAS` **threaded to `step_one`** | the two `identity_payload` tests; `btrs` runs get own addresses |
| 3    | gh#748 — typed `prng` field, default `chacha8`               | same                                                            |
| 4    | flip defaults, delete the losing arm, name the cells         | **the whole CAS store; every pinned-number gate re-baselines**  |

Only step 4 invalidates anything.

Step 4 is **gh#761**. It previously shared gh#748 with step 3, so when that
closed as "PRNG swapped" the deletion of the losing arm would have lost its
tracker — the orphaned-dual-path case `rust-conventions.md` warns about. Filed
separately so the two-sampler seam has a terminal step on the board.

Two prerequisites for step 4 that are not in the table because they are not
re-keys:

- **Re-harvest the `(n, p)` regimes from a healthy chain.** The whole cell set
  comes from a frozen infeasible θ (§4).
- **Run the backend-level statistical suite.**
  `sim/tests/statistical_distribution.rs` carries
  `test_overdispersion_variance_chain_binomial`, `test_pure_death_variance`,
  `test_pure_death_distribution` and `test_two_state_equilibrium` behind
  `#[ignore = "statistical test: run with --ignored in nightly CI"]`, so the
  tests that would catch a bad binomial **through the chain-binomial backend**
  are not in `make test`. They pass today and take 24 s in total. At that cost
  the `#[ignore]` is hard to justify; at minimum the flip commit must run them.
  Same for `pgas_obs_overdisp_smoke.rs`'s two `#[ignore]`d PGAS smoke tests,
  which exercise `obs_model.rs` — one of the two high-risk call sites §3 names.

## Decisions

1. **The knob is a typed stage field, not an environment variable** — following
   gh#241's shipped precedent, and riding `Stage::identity_payload`'s
   include-by-default subtraction so identity needs no new mechanism.
2. **The resolved config value travels as a value to `step_one`**, not through
   the thread-local. A thread-local set per chain worker reaches ~none of the
   draws, because they happen on rayon workers in a nested `par_iter` (§1). The
   thread-local remains a test/bench affordance with no production caller, and
   that is now asserted rather than asserted-about.
3. **Correctness is three deterministic conditions, not two.** Domination and
   squeeze validity by sweep, plus `exp(log_bound) ∝ pmf` by a separate test the
   sweep is structurally blind to. A distributional suite is necessary and
   provably insufficient; so, on its own, is the sweep.
4. **BTRS is the TensorFlow variant**, not the paper's, because that is the one
   the sweeps verify.
5. **Re-keying is authorised** and concentrated in step 4.
6. **1.48× is measured on the sampler; ~1.15× on the fit is IMPLIED.** The
   distinction is the decision. An earlier revision of this list said "1.17× is
   measured" in the same sentence as the rule forbidding unmeasured factors as
   outcomes — the factor had never been observed end to end, and still has not
   been. Quote the sampler ratio, or say "implied".

## The larger lever, still unclaimed

`particles = 9600` is justified in the fit config by bootstrap-filter loglik
**estimator spread**. That is the PMMH criterion; PGAS forms no such estimator —
verified: there is no `log_z` anywhere in `pgas.rs`, `csmc_as` returns only
`(PGASTrajectory, CSMCDiagnostics)`, the θ-move is NUTS on the complete-data
posterior at a fixed trajectory, and `n_particles` is read at exactly two sites,
both `csmc_as` calls. `N` buys trajectory mixing.

If mixing holds at 2400, that is **4× — more than every engineering item in this
proposal multiplied together.** Three cautions from review, all worth heeding:
the right objective is **ESS(θ) per wall-second** computed from `draws.tsv`, not
`trajectory_renewal` (a slot-identity statistic that counts a free particle
inheriting the reference's prefix as "renewed"); the resample stream consumes
`N−1` uniforms per resample, so runs at different `N` are not comparable at a
fixed seed and the sweep needs several seeds per `N`; and the sweep must
instrument the degeneracy counters, because `normalize_log_weights` falls back
to uniform when every particle's weight is `−inf`, which is not `categorical(W)`
and whose trigger probability _rises as N falls_.

This is a statistical call for the maintainer, not an engineering one.

## Named follow-ups (independent defects found while doing this)

- **gh#746, re-scope.** `CAMDL_EVAL_FLAT` still does not enter run identity: two
  runs under different evaluators wrote the same CAS leaf, and the variable is
  read only in `sim` (`flat_eval.rs`), never in `resolve.rs`/`cas.rs`. The
  re-scope must also fix the justification, which was wrong: an earlier revision
  said "there is no CI gate proving that", checked by filename. There is one —
  `crates/sim/tests/flat_eval_byte_identity.rs`, in `make test`, asserting
  `to_bits()` equality of `eval_flat` vs `eval_resolved` for every rate of every
  golden at 5 times × 3 state variants behind a `checked_models >= 10` floor.
  What is missing is a **run-level** gate of the shape `gate_licm_ab.rs` has
  (same trajectory hash, variable on vs off) — and that gap is wider than the
  evaluator: `propensity.rs` documents the flat path as using its own
  `FlatCache` and **not** entering `CacheScope`, which no expression-level test
  can see.
- **`fit predict` writes sampler-dependent artifacts into a Completed CAS leaf,
  unaddressed** — `predict.rs:2106–2181`, "regenerated, overwritten in place",
  absent from the record manifest, so the divergence check cannot see them. A
  `btrs` predict would silently overwrite a `btpe` predict.
- **`correlated_pf.rs:120, 137` carries the unfixed gh#525 defect** —
  `q.powi(n as i32)` wraps for `n > i32::MAX`, making the initial pmf term
  garbage. Same reachability argument as gh#525 (data-column denominators).
- **`n_particles == 0` underflows** `j_ref = n_particles − 1` (`pgas.rs`), and
  `n_particles == 1` degenerates silently to θ-only Gibbs on a frozen
  trajectory. `particles: usize` has no validator.
- **`nlopt_stage.rs:279` stores `VERSION` where every sibling stores
  `VERSION_SHORT`**, so every nlopt stage reports permanently stale against the
  `VERSION_SHORT` comparators.
