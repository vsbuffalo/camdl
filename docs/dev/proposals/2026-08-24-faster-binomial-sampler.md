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

**BTRS is now measured, not inferred: 1.58× on the sampler, ~1.17× on the fit**
(`cargo bench -p sim --bench binomial_ab`). An earlier draft of this proposal
carried `1.25–1.35×` from flop counting; that was wrong, by about the margin
[`2026-06-14-flat-bytecode-evaluator.md`](../notes/2026-06-14-flat-bytecode-evaluator.md)
records for this exact mistake (synthetic 2.5×, real corpus 1.27×).

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

One claim from that draft survived scrutiny and is **refuted**: suffixing
`VERSION_SHORT` does **not** break the camdlc handshake, the IR cache, or the
CAS index — all three key on `version::GIT_HASH`, a separate const. It is moot
now, but it was the risk gh#746 was filed against.

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

**Resolve once, so the hash and the run cannot disagree.** The chain worker
calls `rng::set_binomial_algorithm(stage.binomial)` once at entry, from the same
field that was hashed. The thread-local override is therefore not a second input
channel — it is the transport for the one resolved value, which is the shape
2026-08-23's I1 asks for. Nothing in production reads it from anywhere else, and
a test should assert that.

### 2. BTRS — landed

`rng.rs`, commit `2b54fa6e`. Hörmann (1993), transcribed from TensorFlow's
`random_binomial_op.cc` (Apache-2.0, same as camdl), deliberately the
**TensorFlow variant** — TFP notes its own deviation from the paper ("there is a
log missing"), and the variant here is the one the domination proof below
verifies.

**Correctness rests on a deterministic proof, not on the transcription.** BTRS
is exact iff its hat dominates the pmf and its squeeze never accepts what the
density would reject. Both are properties of the eight constants, so
`hat_dominates_and_squeeze_is_valid` sweeps them over the routed domain with no
draws and no seed. This is not belt-and-braces on top of a χ² — it is the only
instrument that works here:

> A one-digit error in `b` (`2.53 → 2.63`) distorts the tails symmetrically
> about the mode, leaving the mean bias at **exactly zero** while the
> distribution is wrong. A moment test is structurally blind to it and a χ²
> needs ~10⁸ draws. The sweep catches it, and every other single-constant typo,
> in milliseconds.

The sweep calls `BtrsHat`'s own methods, so it checks the **shipped** arithmetic
rather than a second copy of the formula.

Two findings fell out and are pinned as tests:

- **`BINV_THRESHOLD` is a correctness boundary, not a speed knob.** The
  domination margin is a few percent at `n·p = 10` and goes **negative** by
  `n·p ≈ 7`. `the_hat_stops_dominating_below_the_routing_threshold` asserts the
  failure, so nobody lowers the threshold to buy draws.
- **The support check is hoisted above the squeeze** — a deliberate deviation
  from the reference. A no-op wherever the in-support guarantee holds; where it
  might not, it redraws instead of returning a `k > n` that the `p > 0.5`
  reflection turns into a u64 underflow of ~1.8e19.

The distributional suite (exact-PMF χ² with BTPE as positive control, moments at
the province regimes, support, threshold continuity, the gh#510/gh#525
pathological inputs) is **permanent** and outlives any toggle. It is calibrated,
not asserted: `chi_square_rejects_a_one_percent_bias` fails if the suite ever
loses the power to detect a 1% shift in `p`.

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

`benches/binomial_ab.rs`, commit `133ef6cc`. Median-of-9, 400k draws/cell, arms
interleaved in one process. **The split-draw regimes are half the cells**: the
model draws 12 total-exit binomials and 12 competing-risk splits per
particle-substep, and the splits sit at `n ≈ 20..200` where BTRS's squeeze fires
least often (1.61× there, 1.54× at total-exit; BTPE 77 vs 53 ns/draw). Benching
`np ≈ 190` alone would have measured the easy half.

The bench prints the implied whole-fit factor across assumed shares 30–50%
rather than one number, because ≈39% is a profile measurement with a point or
two of discretion and is a **lower bound** (`rand_distr` only partially inlines
— proven by `UniformFloat::<f64>::new` appearing as its own 0.59% leaf).

### 5. The PRNG is a separate question, and its measurement is contingent

ChaCha8 block generation is 8.6% of the profile — worth ~1.08×, inferred, not
measured. BTRS keeps two uniforms per attempt so Phase 2 does not subsume it.

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

| step | lands                                                   | re-keys                                                         |
| ---- | ------------------------------------------------------- | --------------------------------------------------------------- |
| 1    | gh#747 — BTRS + domination proof + bench (**landed**)   | nothing (not reachable from production)                         |
| 2    | typed `binomial` field on `Stage::PGAS`, default `btpe` | the two `identity_payload` tests; `btrs` runs get own addresses |
| 3    | gh#748 — typed `prng` field, default `chacha8`          | same                                                            |
| 4    | flip defaults, delete the losing arm, name the cells    | **the whole CAS store; every pinned-number gate re-baselines**  |

Only step 4 invalidates anything.

## Decisions

1. **The knob is a typed stage field, not an environment variable** — following
   gh#241's shipped precedent, and riding `Stage::identity_payload`'s
   include-by-default subtraction so identity needs no new mechanism.
2. **The thread-local override is transport for the resolved config value**, set
   once per chain worker, never a second input channel.
3. **Correctness is the deterministic domination sweep.** A distributional suite
   is necessary but provably insufficient on its own.
4. **BTRS is the TensorFlow variant**, not the paper's, because that is the one
   the sweep verifies.
5. **Re-keying is authorised** and concentrated in step 4.
6. **1.17× is measured.** No proposal in this family quotes an unmeasured factor
   as an outcome again.

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
  runs under different evaluators wrote the same CAS leaf. Harmless only while
  the two are byte-identical, and there is no CI gate proving that
  (`gate_licm_ab.rs`, `gate_constant_fold_ab.rs`, `gate_binding_cache_ab.rs`
  exist; no `gate_eval_flat_ab.rs` does).
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
