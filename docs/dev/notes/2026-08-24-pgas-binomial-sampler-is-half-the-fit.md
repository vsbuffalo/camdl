# Half of a PGAS fit is one binomial sampler: BTPE's mode-walk on the province model

Date: 2026-08-24\
Project: camdl\
Tags: profiling, pgas, inference, rng, binomial, chain-binomial, samply

Sibling to
[`2026-06-14-pgas-trajectory-io-bottleneck.md`](2026-06-14-pgas-trajectory-io-bottleneck.md)
(where PGAS time went at polio scale) and
[`2026-06-14-flat-bytecode-evaluator.md`](2026-06-14-flat-bytecode-evaluator.md)
(`CAMDL_EVAL_FLAT`). Those two found I/O and rate eval on a **large, sparse**
metapop. This note profiles the opposite regime — a **small, dense** model with
a very large particle count — and the answer is different.

## Question

`../ebola-bdbv-camdl/models/bvd_province_nu_carec.camdl` under
`fit_province_nu_carec_16c.toml`: 16 chains × 8000 sweeps × 9600 particles. How
long does it actually take, where does the time go, and what is the achievable
speedup?

## The workload (measured, not assumed)

Two provinces × 8 compartments (E in 3 stages) = 19 integer compartments, **24
transitions in 12 source groups**, dt = 1.0 d, window `first_obs - 1 week` →
`last_obs` = **92 substeps**, **14 observation times**, 7 observation streams.

Per particle-substep the chain-binomial step draws **24 binomials**: one
total-exit `Binom(n_src, 1-exp(-Σr·dt))` per source group, plus `k-1`
competing-risk split draws for a group with `k` exits (12 + 12).

## Finding 1 — the fit is ~5.6 h and it is 82% one function

```
16 chains configured, 8 RAN (8 skipped via BadInit), 6 sweeps, --parallel 14:
                                       15.15 s wall, 156.4 s user, 6.9 GB RSS
  → 8-chain fit at 8000 sweeps  ≈ 5.6 h wall, ≈ 58 CPU-h     (M4 Max, 14 cores)
  → if all 16 survived          ≈ 9-10 h wall, ≈ 116 CPU-h
```

**Read that first line carefully: 8 of the 16 chains were skipped via
`BadInit`.** Half were refused at initialisation at seed 7, so every wall figure
below is an 8-chain figure. An earlier draft of this note paired the 8-chain
wall (5.6 h) with a 16-chain CPU total (87 CPU-h) — inconsistent by 1.53×, which
is how the refusals were found.

Both CPU-hour figures are `user × 8000/6` from the run above
(`156.4 s × 1333 = 57.9 CPU-h`), i.e. they include the ~31% rayon overhead the
measurement contains. A previous revision printed 46 CPU-h, which was
`8 × per-chain serial compute` — a different basis, silently substituted, and
one that also made the 1.53× reconciliation in the paragraph above unrecoverable
(`87/57.9 = 1.50`; `87/46 = 1.89`). Keep one basis.

The refusal rate is a **finding in its own right**: the config's own header puts
this family's baseline at 12.5%, and 16 chains was chosen precisely because "8
chains has repeatedly left too few survivors to read R-hat". At seed 7 it
delivers exactly 8.

samply leaf self-time, serial run, 14,331 samples on the worker thread
(`--profile profiling`, `--unstable-presymbolicate`):

| bucket                                                     | % all samples |
| ---------------------------------------------------------- | ------------- |
| **`chain_binomial::step_one`** (free-particle propagation) | **82.4**      |
| ├ `rng::StatefulRng::binomial`                             | 38.9          |
| ├ `resolved_expr::eval_resolved`                           | 18.7          |
| ├ ChaCha8 `BlockRng::generate_and_set`                     | 8.6           |
| ├ `step_one` self                                          | 6.9           |
| └ `__powidf2` (BINV's `q^n` setup)                         | 2.6           |
| `pgas::log_transition_density_substep` (AS weights)        | 5.8           |
| observation likelihood (`lgamma` + densities)              | 5.2           |
| `__psynch_cvwait` (thread park)                            | 5.7           |
| **allocation / memcpy**                                    | **< 0.5**     |

Cost per particle-substep ≈ **2.8 µs**. Scaling in particle count is **affine,
not linear** — the per-particle cost falls (318 / 281 / 266 / 259 µs per
particle-sweep at 1200 / 2400 / 4800 / 9600), which is a fixed cost being
amortised. Fitted: slope 250.6 µs/particle, intercept ~81 ms/sweep, predicting
318.0 / 284.3 / 267.4 / 259.0 against those four. The intercept is **3.25% of
the sweep at N = 9600**, so the conclusion stands — there is no per-sweep fixed
cost worth attacking — but "linear" was the wrong word for the evidence.

## Finding 2 — two plausible hypotheses died on contact

Recording these because both are the kind of thing that reads as obviously true
from the source and is worth nothing.

**(a) Allocation churn: not a factor.** `csmc_as` clones `counts`, `cum_flows`
and `acc` at the top of _every_ substep (`pgas.rs:2199–2210`) but reads them
only when `did_resample` — 78 of 92 substeps clone 3 × 9600 vectors for nothing
— and pushes four `Vec<Vec<_>>` histories per substep (`pgas.rs:2488–2491`).
That is ~6 M allocations per sweep. It is **invisible in the profile**
(`_platform_memmove` 0.24%, `RawVecInner::finish_grow` 0.13%).

The history is **most of, but not all of**, the 6.9 GB RSS, and the arithmetic
that first appeared here was the 8-vs-16 chain error this note elsewhere claims
to have caught: `9600 × 92 × ~512 B ≈ 452 MB/chain`, but **8 chains ran**, so
the history accounts for ~3.6 GB, not ~7 GB. Multiplying by 16 to reach
"matching the measurement" was reading a 16-chain multiplier onto an 8-chain
measurement. Refusal happens at initialisation, before the history allocates, so
the skipped chains cannot own the other ~3.3 GB — either the
per-particle-substep record is nearer 1 KB than 512 B, or something else holds
half the RSS. **Unresolved, and it needs a heap profile rather than more
arithmetic.** A leaner per-particle history is still a real **memory** lever,
and not a speed lever.

(An earlier version of this paragraph cited gh#207 for that lever. Wrong issue:
gh#207 is CLOSED and is an external report about `simulate` RSS on a 244-patch
metapop — 9.3 GB/sim, agent-based, memory-bound before CPU-bound — not a PGAS
per-particle-history task. No open issue tracks this.)

**(b) The per-call θ finite-check: worth 0×.** `eval_propensities` iterates
`model.param_index` (a `HashMap`, 26 entries) checking every parameter is finite
— once per particle per substep, ~1.8e11 entry visits over the fit. Patched out
behind `if false` and measured: **13.85 / 13.24 s vs 13.34 / 13.25 s baseline.
No effect.** The bucket array is hot in cache and the loop is free. Reverted.

## Finding 3 — the ceiling probe: the binomial is 50% of the fit

Throwaway env-gated patch replacing `StatefulRng::binomial` with
`(n as f64 * p) as u64` (wrong numerics; reverted):

| binomial | wall (2 reps)   | ratio     |
| -------- | --------------- | --------- |
| real     | 13.25 / 14.12 s | 1.00×     |
| free     | 6.56 / 6.61 s   | **2.01×** |

So the sampler plus the RNG bytes it consumes is **half the fit**. That single
number bounds every other lever below.

## Why BTPE is slow here: the mode-walk

`StatefulRng::binomial` (`rng.rs`, `pub fn binomial`) routes
`n·min(p,1-p) < BINV_THRESHOLD` to its own hand-owned BINV (correct; owns the
gh#510 unbounded-loop and gh#525 large-`n` panic fixes — **do not touch it**)
and everything else to `rand_distr::Binomial`, which is BINV/BTPE per
Kachitvichyanukul & Schmeiser (1988), CACM 31(2):216–222.

At this model's rates almost every total-exit draw lands in BTPE. Sample
regimes: `S` group `np ≈ 192` on `n ≈ 6.3e6`; `E` stages `np ≈ 190`; `I`
`np ≈ 150`; `C` `np ≈ 87`. `__powidf2` at 2.6% of the fit bounds the BINV share
to roughly 17% of binomial time, leaving **BTPE at ~32% of the whole fit**
(inferred from that ratio, not directly instrumented).

Three costs per BTPE draw, read off
`~/.cargo/registry/src/*/rand_distr-0.4.3/src/binomial.rs`:

1. **Full setup, every call** (lines 134–161): `Binomial::new` then ~10
   constants — `npq`, `p1` (1 `sqrt` + `floor`), `x_m`, `x_l`, `x_r`, `c`,
   `lambda_l`, `lambda_r`, `p2`, `p3`, `p4` — ~30 flops with 4 divisions. None
   of it is reusable: `(n, p)` differ per particle per group.
2. **Two `Uniform::new` constructions per draw** (lines 166–167), each computing
   a scale. `UniformFloat::<f64>::new` shows at 0.59% self-time on its own.
3. **The mode-walk (step 5.1, lines 214–239) — the real cost.** The acceptance
   test takes the "recursive relationship" branch whenever
   `!(k > 20 && k < 0.5·npq - 1)` where `k = |y - m|`. Most draws land near the
   mode, so this fires almost always, and it is a loop from `m` to `y` with
   **one f64 division per step**. At `npq ≈ 190` the sd is ~13.8, so mean
   `|k| ≈ 11` → ~11 serially-dependent divisions ≈ 130 cycles ≈ 35 ns, on top of
   the ~30 ns setup. Cost per draw, three ways: the profile share gives **46-59
   ns** (`0.389 × 2810 / 24` through `0.501 × 2810 / 24`); the flop count above
   gives ~65; and the bench below MEASURES 49-64 ns at the total-exit regimes
   and 63-89 at the splits. An earlier draft asserted "the ~90–100 ns/draw the
   profile shows", which none of the three supports — the profile-derived figure
   was right and the assertion was not.

**The mode-walk is not the whole story, and the bench falsifies it at half the
cells.** Walk length scales with `|k|`, so this mechanism predicts the splits
(`npq ≈ 5..75`, a walk of 2–9 steps) should be the CHEAP regime. The bench
measures them 40% more expensive than the total-exit cells (75 vs 52 ns for
BTPE). The driver at small `npq` is region selection, not the walk: BTPE's
immediate-accept triangle is `p1/p4` of the `u` range, and that collapses from
**0.63** at `(400, 0.476)` to **0.52** at `(150, 0.5)` to **0.21** at
`(20, 0.5)` — so a small-`npq` draw rejects and re-tests far more often before
it ever reaches a walk. Keep this in view before extrapolating the mechanism to
any regime the bench does not cover.

## BTRS: what would replace it

**BTRS** = _Binomial, Transformed Rejection with Squeeze_, Hörmann (1993), _The
generation of binomial random variates_, J. Statist. Comput. Simul.
46(1–2):101–110. Same exactness (it samples Binomial(n,p) exactly; it is a
rejection method, not an approximation), different hat.

BTPE composites four regions — triangle, two parallelograms, two exponential
tails — hence 10 setup constants and a 4-way branch. BTRS uses **one**
transformed-rejection hat, so:

- **Setup is ~6 constants** (`stddev = sqrt(npq)`, and `b`, `a`, `c`, `v_r`
  derived from it by affine expressions), no `floor`, one `sqrt`.
- **The dominant path has no log, no division chain, and no mode-walk.** Draw
  `u ∈ (-0.5, 0.5)` and `v ∈ (0,1)`; form `us = 0.5 - |u|` and
  `k = floor((2a/us + b)·u + c)`; if `us` is not too small and `v ≤ v_r`,
  **accept immediately** — ~10 flops. Only the residual minority falls through
  to the `log`-based test with a Stirling correction.
- Two uniforms per attempt, same as BTPE, but far fewer rejected attempts reach
  an expensive test.

That immediate-accept branch is why JAX and TensorFlow both use BTRS for
`binomial` rather than BTPE.

~~Expected gain here: BTPE ~32% of the fit → ~12–15%, i.e. **~1.25–1.35× on the
whole fit**.~~ **Superseded — this flop-count prediction was wrong.** The bench
measures **1.48× on the sampler**, implying ~1.15× on the fit. Applying the
measured sampler ratio to this section's own ~32% BTPE share would give 1.13×,
which is the low end of the same band; the prediction above overshot by using
flop counts instead. Left visible rather than deleted, because over-predicting
from flop counts is the specific mistake this repo has now made twice — see
[the flat-evaluator note](2026-06-14-flat-bytecode-evaluator.md).

**The exact constants must be transcribed from the paper**, or from a
permissively-licensed reference implementation — TensorFlow and JAX are both
Apache-2.0, same as camdl (`LICENSE`), so either is citable with attribution.
They are not to be written from memory.

### Libraries: none needed

- **`rand_distr` version bump: no.** `crates/sim/Cargo.toml` declares
  `rand_distr = "0.4"` and `Cargo.lock` resolves it to 0.4.3
  (`crates/sim/Cargo.toml`); the BINV/BTPE structure is the crate's
  long-standing design. _(Unverified: whether 0.5.x changed it — a 10-minute
  check before writing code, but do not plan around it.)_
- **A new crate: no.** BTRS is ~150–200 lines of scalar f64 and needs only
  `sqrt`, `ln`, and a Stirling correction. `crates/numerics` already owns an
  inline `lgamma` ("no external dependencies — lgamma implemented inline for
  stability"). It belongs in `rng.rs` beside the hand-owned BINV, which is
  already the precedent for owning a sampler branch rather than delegating it.
- **PRNG replacement (separate change, see below):** `rand_xoshiro` or
  `rand_pcg` (both `rand_core` ecosystem, same maintainers, drop into `RngCore`)
  or a hand-rolled counter-based Philox/Threefry. `rand_pcg` and `rand_xorshift`
  are already in the local registry cache; neither is currently a `sim`
  dependency.

## Verified A/Bs

Serial, interleaved reps, result cache cleared per run (`--force`):

| lever                                                 | wall            | ratio     | byte-identical?                                     |
| ----------------------------------------------------- | --------------- | --------- | --------------------------------------------------- |
| baseline                                              | 14.92 / 14.89 s | 1.00×     | —                                                   |
| `CAMDL_EVAL_FLAT=1`                                   | 13.75 / 14.03 s | **1.07×** | **yes** — `draws.tsv` + `trace.tsv` `cmp`-identical |
| + `lto="fat"`, `codegen-units=1`, `target-cpu=native` | 13.34 / 13.25 s | **1.12×** | `best ll` unchanged; needs the full gate            |
| `lto`/`native` alone (no flat)                        | 15.38 / 14.14 s | ~1.01×    | —                                                   |

`lto`/`codegen-units` are unset in `rust/Cargo.toml` (only `[profile.profiling]`
exists), so the flags cost nothing to try. **But 1.12× is not an available win,
and this note previously said it was.** Three problems with that row, all
visible in the table above it:

- The two estimates of the same lever disagree. Measured in isolation the flags
  give ~1.01× with one of two reps (15.38 s) **slower than baseline**; measured
  incrementally on top of flat they give 1.045×. At n=2 against a ~5% noise
  floor, neither resolves.
- The 1.12× row **includes `target-cpu=native`**, which the "Next" list below
  then declines to enable. So the configuration actually proposed for landing
  (`lto` + `codegen-units`, no `native`) was never measured at all.
- Calling that row "byte-identical" contradicts both its own weaker note in the
  table (`best ll` unchanged) and the FP warning attached to `native`.

Byte-identity is verified for `CAMDL_EVAL_FLAT` only, at **1.07×**. Also drop
`target-cpu=native` from the proposal entirely: it makes the binary vary by
machine with nothing recording which machine, which contradicts the run-identity
argument the rest of this work rests on.

**Core utilisation is already optimal.** 16 chains, 6 sweeps:

| `--parallel` | wall        | user    | cores_eff |
| ------------ | ----------- | ------- | --------- |
| 10 (P-cores) | 16.04 s     | 138.8 s | 8.65      |
| **14**       | **15.15 s** | 156.4 s | **10.32** |
| 20           | 16.50 s     | 160.4 s | 9.72      |

10.3 of 14 on a 10P+4E part is effectively saturated. There is **no scheduling
win** available; the nested chains × particles rayon structure already works.

## What is actually available, and on what evidence

Read the third column as "if the factor held", not as a prediction. Only the
first row is byte-identity-verified end to end; the BTRS row is Amdahl applied
to a measured sampler ratio; the rest is inference.

| scope                                                       | factor                    | basis                       |
| ----------------------------------------------------------- | ------------------------- | --------------------------- |
| `CAMDL_EVAL_FLAT` alone                                     | **1.07×**                 | measured, byte-identical    |
| \+ `lto`/`codegen-units` (`target-cpu=native` NOT included) | unresolved                | within noise at n=2         |
| \+ BTRS binomial — sampler **1.48×**, so the fit            | ~1.11–1.19×, centre ~1.15 | **implied**, not observed   |
| \+ non-cryptographic PRNG                                   | ~1.06–1.09×               | inferred from an 8.6% slice |
| **hard ceiling** (binomial _and_ rate eval both free)       | ~3.2×                     | two disjoint profile shares |

**The sampler ratio is measured; the whole-fit factor is not, and cannot be
yet.** `cargo bench -p sim --bench binomial_ab` gives **1.48×** blended over the
24 draws a particle-substep actually makes — 1.45× at the total-exit regimes,
1.55× at the arm-routed splits, 1.00× at the three splits whose `np ≈ 0.1` sends
them to BINV in both arms. Pushing 1.48× through Amdahl at the profiled ≈39%
share gives **1.145×**; across a 30–50% share it spans 1.11–1.19×. BTRS is not
reachable from production, so no end-to-end number exists to check that against.

Two earlier figures in this family were wrong and are recorded rather than
dropped:

- **1.25–1.35× whole-fit, from flop counting.** Wrong by about the margin this
  repo's [flat-evaluator note](2026-06-14-flat-bytecode-evaluator.md) records
  for exactly this mistake (synthetic 2.5×, real corpus 1.27×).
- **1.58× on the sampler, and the ~1.17× it implied.** That came from a blend
  that weighted the five total-exit cells equally — the `E` regime is 6 of 12
  draws and was getting 1/5 of the weight, and it is the cell with the lowest
  ratio — and that omitted the BINV-routed splits entirely, so it asserted 24/24
  draws reach the arm under test. Under its own superseded accounting this
  machine now reports 1.51–1.52×, so roughly half the gap to 1.58× is the
  methodology fix and half is between-session variation at identical accounting.
  **Two significant figures is the precision this measurement supports.**

The ceiling is `1/(1 − 0.501 − 0.187)` = **3.21×** — both the sampler and rate
eval free. The 2.9× an earlier draft carried reproduced from no construction I
can rebuild; it required a multiplier of 1.44 corresponding to nothing measured.

The wall is set by
`chains × 8000 sweeps × 9600 particles × 92 substeps × 24 draws`, and BTRS
touches only the last factor. **Two caveats on every share in this note, both
found by adversarial review rather than by me:**

- **38.9% is a lower bound with ±0.8 pp of sampling noise** — that is the 95%
  interval; the bench prints the same fact as `SE ~0.4pp` (14,331 samples at 999
  Hz). `rand_distr::Binomial::sample` is generic and monomorphises into `sim`'s
  codegen unit under a profile with no LTO, so it partially inlines — proven by
  `UniformFloat::<f64>::new` appearing as its own 0.59% leaf. Write "≈39%", not
  38.9%.
- **NUTS is effectively absent from the profiled configuration.** Verified from
  the trace: `n_leapfrog = 1`, `tree_depth = 1`, `n_divergent = 1` on every
  sweep — the chain started infeasible and diverged on the first doubling. A
  healthy chain at the default `max_tree_depth = 10` can take up to 1023
  gradient evaluations per sweep. At 9600 particles that gradient is ~100× less
  significant than at the 100 particles the
  [2026-06-14 healthy-fit profile](2026-06-14-pgas-trajectory-io-bottleneck.md)
  measured it at ~1%, so csmc should still dominate — but that is an inference,
  **not a measurement**, and it is the one open threat to the ≈39% share. Read
  `n_leapfrog` off a chain with non-zero acceptance before trusting it.

## The lever that is larger than all of the above, and is not ours

`particles = 9600` is justified in `fit_province_nu_carec_16c.toml` by
bootstrap-filter loglik **estimator spread** — 18.18 / 10.05 / 6.72 nats at 1200
/ 4800 / 9600, "centre still climbing at 9600, so the bias has not plateaued."

That is the **PMMH** criterion: PMMH needs a low-variance marginal-likelihood
estimate because the estimate enters the MH ratio. PGAS never forms that
estimator. Its θ-move is NUTS on the _complete-data_ posterior given the sampled
trajectory; `N` buys **trajectory mixing**. The PGAS-appropriate diagnostics are
already written to every chain's `trace.tsv`: `trajectory_renewal`,
`renewal_b0..b9`, `as_accept`, `as_proposed`, `as_opportunity`.

If renewal holds at 2400 particles, that is **4× on its own** — more than every
engineering lever in this note combined.

**This is not evidence yet.** 6-sweep probes showed renewal in 0.3–1.0 at every
N from 1200 to 9600, but NUTS acceptance was 0% from an infeasible start, so θ
was frozen and all four runs were sampling at the same point. That is a reason
to run the experiment properly, and nothing more. Choosing `N` on a mixing
criterion is a statistical call for the maintainer.

## Design: this must not ship behind a Cargo feature

A `sim` Cargo feature gating "fast sampler" vs "old sampler" is the wrong shape,
for four reasons — the first decisive:

1. **A Cargo feature is invisible to the run identity.** `run.json` records
   `engine_version = 0.1.0+<git-hash>` and `provenance.camdl_version` — crate
   version plus git hash, and nothing about the build configuration. Two
   binaries from the _same commit_ with the feature on and off would produce
   different posteriors, write them under the same `engine_version`, and be
   indistinguishable in the stored artifact. That institutionalises a provenance
   hole rather than closing one.
2. **The repo's alpha posture forbids the dual path.** `VERSIONING.md` /
   CLAUDE.md: no compatibility shims, no fallback paths, "rename atomically and
   update all goldens." `rust-conventions.md`: "'v1' alongside 'v2' is dead
   code. When a rewrite lands, the old path is deleted in the same commit."
3. **It doubles the test matrix** that `sim-and-inference.md` requires to be
   dense (every backend × method cell tested), and CI would either run both or
   silently leave one untested — the exact failure mode that rule exists for.
4. **`sim`'s existing features scope _dependencies_, not numerics**
   (`lineage-parquet` → `arrow`/`parquet`; `ode` → `nlopt`). A numerics-forking
   feature would be a new and worse category on top of **five** existing
   _runtime_ toggles (`CAMDL_EVAL_FLAT`, `CAMDL_EVAL_UNRESOLVED`,
   `CAMDL_NO_BINDING_CACHE`, `CAMDL_NO_LICM`, `CAMDL_NO_CONSTANT_FOLD` — all
   five verified present in `rust/crates/*/src`). A sixth, `CAMDL_SERIAL`, is a
   **ghost**: it appears only in prose — in
   [`2026-06-14-pgas-trajectory-io-bottleneck.md`](2026-06-14-pgas-trajectory-io-bottleneck.md)
   and in an earlier draft of this note, which copied it from there. It is not
   in the tree.

**What to do instead** — separate the migration seam from a shipped feature:

- **During development:** a _temporary_ A/B seam, used to prove distributional
  equivalence and to bench, **deleted in the commit that makes BTRS the only
  sampler.** It must be togglable inside one process without a rebuild, so that
  an interleaved bench is possible at all.

  This note originally said "**env**, not Cargo" here. That is **superseded**:
  what shipped is a thread-local override, and the proposal
  ([`2026-08-24-faster-binomial-sampler.md`](../proposals/2026-08-24-faster-binomial-sampler.md))
  rules out an environment variable entirely, on the same run-identity grounds
  argument 1 above uses against a Cargo feature — an env var that changed draws
  without entering the run address would serve one sampler's posterior from the
  other's cache leaf, which is gh#241's already-decided case. Read this bullet
  as "not a Cargo feature", not as an endorsement of env vars.
- **Shipped:** one sampler. One atomic commit, CAS store invalidated once,
  goldens moved. `ir/VERSION` is untouched — this is runtime numerics, not IR.
- **Permanent, not temporary:** the equivalence test. BTPE and BTRS both sample
  Binomial(n,p) _exactly_, so the test is **distributional, not byte-wise** —
  exact-PMF χ² at small `n`, moment/quantile agreement at the `(n,p)` regimes
  this model actually visits (`np ≈ 87–192`, on `n` from 400 to 8.75e6 as the
  bench's cells actually span it), plus the gh#510/gh#525 pathological inputs.

  Necessary, and **provably not sufficient**: the distributional suite cannot
  see a k-dependent error in the acceptance density, which is why the sampler
  also carries a deterministic domination sweep and a proportionality test. See
  the proposal's §2.

## Sizing the two changes

**Binomial → BTRS: 1–2 days, low architectural risk.** One file
(`crates/sim/src/rng.rs`), one new function, keep BINV untouched, keep the
`p > 0.5` reflection at the existing call boundary. Care needed on: the Stirling
correction term, `i64`/`u64` edges at large `n`, and `k` outside `[0, n]`
rejection. Most of the calendar time is the equivalence suite, not the sampler.

**PRNG swap: ~1 day of work, but a broad _verification_ surface, for
~1.06–1.09×.** Cheaper than it looks in one respect — `ChainResumeState` stores
params, trajectory and NUTS adaptation but **not** RNG state, and PGAS
re-derives per-particle streams from `(seed, particle_index)` each sweep
(`init_particle_rngs`), so the resume format is not coupled to the generator.
The costs are: `StatefulRng` is a newtype over `ChaCha8Rng` whose `inner_mut()`
leaks the concrete type to 8 call sites (`cli/main.rs` prior sampling,
`obs_model.rs` synthetic draws); `ChaCha8Rng::set_stream` is the
stream-splitting contract that 9600 per-particle streams depend on, and a
replacement needs an equivalent with the same independence guarantee (xoshiro
`long_jump`, PCG stream increments, or counter-based `(key, counter)`); and
every determinism gate — including `gate_pgas_thread_invariance.rs` — has to be
re-baselined against the new generator.

**Recommendation: do the binomial first, then re-profile before deciding on the
PRNG.** When BTPE's rejection loop goes away, ChaCha8's _share_ may rise while
its absolute cost falls; the 8.6% figure is measured against today's draw
pattern and will not survive the first change.

## Repro

```bash
# fresh camdlc — ~/.local/bin/camdlc predates the `init { ~ poisson(…) }`
# seed law this model needs; do NOT `make install`
make build-ocaml
export CAMDLC=$PWD/ocaml/_build/default/bin/camdlc.exe CAMDL_SKIP_VERSION_CHECK=1

# 1-chain / 6-sweep serial probe (per-sweep cost); 16-chain for the wall
camdl fit run perf6.toml --parallel 1  --seed 7 --force   # chains=1,  sweeps=6
camdl fit run c16.toml   --parallel 14 --seed 7 --force   # chains=16, sweeps=6

# leaf profile (thread 1 is the worker; resolve rva via the .syms.json sidecar)
samply record --save-only --unstable-presymbolicate -o prof.json.gz -r 999 -- \
  target/profiling/camdl fit run p_9600.toml --parallel 1 --seed 7 --force

# byte-identity of the flat evaluator on this model
camdl fit run p_9600.toml --parallel 1 --seed 7 --force   # then cp draws.tsv/trace.tsv
CAMDL_EVAL_FLAT=1 camdl fit run p_9600.toml --parallel 1 --seed 7 --force
cmp flat_off_draws.tsv flat_on_draws.tsv && cmp flat_off_trace.tsv flat_on_trace.tsv
```

## Next

Items 3 and 4 of the original list are **done** — the bench landed
(`benches/binomial_ab.rs`) and so did BTRS behind a dormant seam
(`crates/sim/src/rng.rs`), with the distributional suite, a deterministic
domination sweep and a proportionality test. What remains:

1. Turn on `CAMDL_EVAL_FLAT` for this fit family now (1.07×, byte-identity
   verified on this model).
2. `lto = "fat"` + `codegen-units = 1` in `[profile.release]`. **Not**
   `target-cpu=native`: it makes the build vary by machine with nothing
   recording which, which contradicts the run-identity argument this work rests
   on. Expect no measurable win — see the A/B table's caveat.
3. The typed `binomial` field on `Stage::PGAS`, so the sampler becomes
   selectable and enters the run address. Note the transport constraint the
   proposal originally got wrong: the draws happen on rayon workers inside
   `pgas.rs`'s nested `par_iter`, so a thread-local set once per chain worker
   reaches almost none of them — the resolved value has to be threaded to
   `step_one`.
4. Re-profile on a **healthy** chain before deciding anything else. The ≈39%
   share is measured against a run where NUTS did nothing, and the `(n, p)` cell
   set the 1.48× rests on was read off that same frozen θ.
5. Only then decide on the PRNG.
6. Independent of all of it: measure ESS(θ) per wall-second vs `N` on a healthy
   chain (feasible start, non-zero NUTS acceptance). This is the 4×.

## Loose end found while profiling (different defect, different trigger)

**`CAMDL_EVAL_FLAT` does not enter the run identity.** Two runs under different
evaluators wrote the same CAS leaf in the A/B above; confirmed in the tree — the
variable is read only in `sim` (`flat_eval.rs`, `is_on()`) and appears nowhere
in `resolve.rs` or `cas.rs`. An unrecorded input decides the contents of a
stored artifact, which is the same provenance hole argument 1 above rejects a
Cargo feature for. Tracked as gh#746, which needs re-scoping (its body still
describes the killed env-var registry).

**On how far byte-identity is actually pinned, state it precisely.**
`crates/sim/tests/flat_eval_byte_identity.rs` exists and runs in `make test`,
asserting `to_bits()` equality of `eval_flat` against `eval_resolved` for every
rate of every golden model at 5 times × 3 state variants, plus hand-built edge
cases, behind a `checked_models >= 10` non-vacuity floor. What is missing is a
**run-level** A/B of the shape `gate_licm_ab.rs` has — same trajectory hash with
the variable on and off — and that gap is wider than the evaluator itself:
`propensity.rs` documents the flat path as using its own `FlatCache` and **not**
entering `CacheScope`, a second behavioural difference no expression-level test
can see. Do not write "there is no gate"; write "there is no run-level gate".
