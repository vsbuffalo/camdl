# A faster binomial sampler, and the env-var identity class it needs first

Date: 2026-08-24 Status: proposed Note ref:
`docs/dev/notes/2026-08-24-pgas-binomial-sampler-is-half-the-fit.md`

Re-keying is authorised by the maintainer (2026-08-24) for the sampler change.

## Problem

**Part 1 — half a PGAS fit is one binomial sampler.** Profiling
`bvd_province_nu_carec.camdl` at its production settings (16 chains × 8000
sweeps × 9600 particles ≈ **5.6 h**, ≈ 87 CPU-h) puts **82.4%** of samples in
`chain_binomial::step_one` and **38.9%** in `StatefulRng::binomial` alone. A
ceiling probe that made the draw free measured **2.01× on the whole fit**, so
the sampler plus the RNG bytes it consumes is half the run. The mechanism is
`rand_distr`'s BTPE recomputing ~10 setup constants per call, constructing two
`Uniform`s per call, and walking from the mode to the sampled value with one f64
division per step. Full evidence in the note; it is not repeated here.

**Part 2 — the toggles needed to measure a replacement are invisible to the
store.** `camdl fit` is content-addressed: it hashes the run's inputs into an
address and serves a stored result if one exists. **Environment variables are
not in that hash.** Demonstrated 2026-08-24: with the flat-evaluator result in
the store, the same config with `CAMDL_EVAL_FLAT` _unset_ — a different
evaluator — returned

```
cache hit — reusing fits/p_9600-bff129a0/01-posterior-24683bc9/seed_7-d0aef62a
engine_version = 0.1.0+b3666ba6
```

It did not run. Both variants share one address and one `engine_version`, so the
artifact cannot say which evaluator produced it. This is harmless _today_ only
because the flat evaluator happens to be byte-identical.

It is **not** harmless for a sampler swap, which is byte-different by design:

- Old sampler runs, stores at address X. New sampler, same config → hashes to X
  → cache hit → the A/B reports the new sampler as infinitely fast, having never
  run it.
- `--force` does not rescue it: the run executes but **overwrites X**, so the
  store holds A _or_ B, never both. The 2×2 in Phase 3 needs four results
  resident and would have one slot.

So the experiment cannot be run until env vars can enter run identity. That is
Phase 0.

This is the fifth wave of the defect class
`docs/dev/proposals/2026-08-23-run-identity-and-store-contract.md` was written
for — "a knob that changes what a run computes or stores is absent from its
`run_id`" — after gh#514 (5 flags), gh#540 (13), the 2026-08-23 fit flags (6)
and that audit (~10 more). Every prior wave was a CLI flag or config field; env
vars appear in neither that proposal's design, its exclusions, nor its
follow-ups. Phase 0 extends that contract to a new input kind rather than
inventing a parallel rule.

## What this does NOT do

- **No IR change and no `ir/VERSION` bump.** This is runtime numerics; no golden
  IR moves.
- **Does not migrate all 20 env-var read sites at once.** Phase 0 lands the seam
  and classifies the vars; the mechanical migration of the ~20 production
  `env::var` call sites may land incrementally behind it.
- **Does not touch BINV.** `StatefulRng::binomial`'s hand-owned small-`np`
  branch carries the gh#510 unbounded-loop and gh#525 large-`n` panic fixes and
  is out of scope.
- **Does not change `particles`, `sweeps` or `chains`.** The statistical lever —
  choosing `N` on a PGAS mixing criterion rather than a PMMH estimator-variance
  criterion, plausibly worth 4× — is larger than everything here and is a
  separate maintainer decision. Named as a follow-up.

## Design

### Phase 0 — the env-var identity class (prerequisite) — gh#745, gh#746

Every camdl environment variable is exactly one of:

- **Identity** — it changes what the run computes. It **must** enter the run
  address.
- **Provenance** — it does not change results. It is recorded in `run.json` and
  must **not** enter the address (or every log-level change would invalidate the
  cache).

**A registry, not two lists.** Two hand-maintained `Vec`s would work until
someone forgets to update one — the same shape that produced four waves, which
that proposal diagnosed as "a shape problem, not a vigilance problem." Instead,
one declaration site:

- An `enum EnvToggle` with one variant per variable and a
  `fn class(&self) ->
  EnvClass` whose match is **exhaustive with no
  catch-all**, so adding a variable without classifying it is a compile error
  (the same discipline `flat_eval::emit` and `ir_hash.rs` already use).
- The registry accessor is the **only** way to read one, and each is read once
  into a `OnceLock` — never `env::var` on a hot path (`flat_eval.rs:55` is the
  pattern; a `env::var_os`-per-draw probe was measurably wrong during this
  investigation).
- A test asserts no `env::var*("CAMDL_…")` occurs outside the registry module,
  so the seam cannot be bypassed. ~20 production sites today (10 in `sim`, 10 in
  `cli`), so this is tractable.

**The classification rule is asymmetric: Identity is the default; Provenance
must be earned by a byte-identity A/B gate in CI.** Misfiling as Provenance
yields a cache hit serving different numbers — the silent wrong answer this
store exists to prevent. Misfiling as Identity yields a redundant recompute.
Those costs are not comparable, so the default follows the cheap error.

This rule has immediate teeth. `gate_licm_ab.rs`, `gate_constant_fold_ab.rs`,
`gate_binding_cache_ab.rs` and `gate_pgas_thread_invariance.rs` already exist
and are exactly the required proof — `gate_licm_ab` compiles two IR fixtures
from one source with the pass on and off and asserts trajectory preservation.
**There is no such gate for `CAMDL_EVAL_FLAT`**, so under this rule it is
_Identity_ until someone writes `gate_eval_flat_ab.rs`, at which point it may be
demoted.

**Mechanism for Identity vars: fold them into the engine-version string**, which
is already a hashed run input everywhere — `EngineVersion(pub String)`
(`runid/src/inputs.rs:168`) is a field of three input structs and of `FitDigest`
(`cli/src/fit/cas.rs:499`). So:

```
0.1.0+b3666ba6   →   0.1.0+b3666ba6+binom=btrs
```

One change reaches every command's identity at once, both halves of the problem
close together (distinct addresses **and** a self-describing `run.json`), and no
new field is threaded through the hash. `VERSION_SHORT` is a compile-time
`concat!` const (`cli/src/version.rs:12`) and becomes a `OnceLock<String>`; the
handful of sites feeding it into run inputs switch to the accessor.

Initial classification. **This table is a starting point, not a finding** —
Phase 0 must classify all 20, and two are flagged because I have not read them:

| variable                                                                                                                                                                                                                                            | class      | basis                                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CAMDL_BINOMIAL` (new)                                                                                                                                                                                                                              | Identity   | changes draws by design                                                                                                                                                                     |
| `CAMDL_PRNG` (new)                                                                                                                                                                                                                                  | Identity   | changes draws by design                                                                                                                                                                     |
| `CAMDL_EVAL_FLAT`                                                                                                                                                                                                                                   | Identity   | **no CI gate** — ad-hoc byte-identity only; demote when gated                                                                                                                               |
| `CAMDL_EVAL_UNRESOLVED`                                                                                                                                                                                                                             | Identity   | no gate found                                                                                                                                                                               |
| `CAMDL_NO_BINDING_CACHE`                                                                                                                                                                                                                            | Provenance | earned — `gate_binding_cache_ab.rs`                                                                                                                                                         |
| `CAMDL_NO_LICM`, `CAMDL_NO_CONSTANT_FOLD`                                                                                                                                                                                                           | Provenance | **already captured transitively**: compiler toggles change the emitted IR, and `FitDigest` hashes `ModelDigest::from_model(<IR>)`. Gated by `gate_licm_ab.rs` / `gate_constant_fold_ab.rs`. |
| `CAMDL_PARALLEL`, `RAYON_NUM_THREADS`                                                                                                                                                                                                               | Provenance | earned — documented bit-identical; `gate_pgas_thread_invariance.rs`                                                                                                                         |
| `CAMDL_TRACE_STEPS`, `CAMDL_SKIP_VERSION_CHECK`, `CAMDL_CAPTURE_BASELINE`, `CAMDL_OUTPUT`, `CAMDL_OUTPUT_DIR`, `CAMDL_IR_CACHE_DIR`, `CAMDL_NO_IR_CACHE`, `CAMDL_GIT_HASH`, `CAMDL_BUILD_DATE`, `CAMDL_EXTERNAL_USE_DOCKER`, `CAMDL_REGEN_EXTERNAL` | Provenance | diagnostics, paths, build stamping, harness — none reaches numerics                                                                                                                         |
| **`CAMDL_SEED`**                                                                                                                                                                                                                                    | **audit**  | if it can override a run seed and is unhashed, that is a live instance of this bug independent of this proposal — check first                                                               |
| **`CAMDL_UNLABELED_THRESHOLD`**                                                                                                                                                                                                                     | **audit**  | not read; classify before landing                                                                                                                                                           |

### Phase 1 — BTRS — gh#747

Replace the `rand_distr::Binomial` (BTPE) branch of `StatefulRng::binomial` with
an in-house BTRS — _Binomial, Transformed Rejection with Squeeze_, Hörmann
(1993), _J. Statist. Comput. Simul._ 46(1–2):101–110. Exact, not an
approximation. One transformed-rejection hat instead of BTPE's four-region
composite, so ~6 setup constants and a dominant immediate-accept path with no
log, no division chain and no mode-walk; only a minority of attempts reach the
log-based Stirling test.

- **No new dependency.** ~150–200 lines of scalar f64 needing `sqrt`, `ln` and a
  Stirling correction; `crates/numerics` already owns an inline `lgamma` ("no
  external dependencies — lgamma implemented inline for stability"). It lives in
  `rng.rs` beside BINV, which is already the precedent for owning a branch
  rather than delegating. Do **not** bump `rand_distr` — 0.4.3's BINV/BTPE
  structure is the crate's long-standing design.
- **Transcribe the constants from the paper**, or from a permissively-licensed
  reference — TensorFlow and JAX both implement BTRS and are Apache-2.0, same as
  camdl, so either is citable with attribution. Not from memory.
- **Toggle:** `CAMDL_BINOMIAL=btpe|btrs`, value-based rather than the
  presence-based house style, deliberately — the default flips mid-arc and a
  value grammar keeps the measurement commands identical before and after.
- **Bench before integrating:** `cargo bench -p sim` A/B at this model's actual
  regimes (`np ≈ 87–192`, `n` from 10² to 6.3e6). The flat-evaluator note is the
  cautionary tale: a synthetic bench said 2.5× where the real corpus said 1.27×.

**Equivalence testing — distributional, not byte-wise, and permanent.** BTPE and
BTRS both sample Binomial(n,p) _exactly_, so there is no byte-identity to assert
and `cmp` is the wrong instrument. The suite outlives the toggle:

1. exact-PMF χ² at small `n` across a `p` grid including `p > 0.5` (the
   reflection path) and `p` near 0 and 1;
2. moment and quantile agreement at the regimes above, at a stated tolerance
   with a fixed seed so it cannot flake;
3. the gh#510 / gh#525 pathological inputs — large `n` with small `n·p`, and the
   underflow tail that used to spin — asserting termination and support;
4. `k ∈ [0, n]` for every draw, and the `n·p` threshold boundary where BINV and
   BTRS meet.

### Phase 2 — the PRNG — gh#748

`CAMDL_PRNG=chacha8|<candidate>`; candidate is `rand_xoshiro`, `rand_pcg`, or a
hand-rolled counter-based Philox. Worth ~1.06× measured against _today's_ draw
pattern, which Phase 1 will change — hence Phase 3 before any default flip.

Cheaper than it looks in one respect: `ChainResumeState` stores params,
trajectory and NUTS adaptation but **not** RNG state, and PGAS re-derives
per-particle streams from `(seed, particle_index)` each sweep
(`init_particle_rngs`), so the resume format is not coupled to the generator.

The real costs: `StatefulRng` is a newtype over `ChaCha8Rng` whose `inner_mut()`
leaks the concrete type to 8 call sites (`cli/main.rs` prior sampling,
`obs_model.rs` synthetic draws); `ChaCha8Rng::set_stream` is the
stream-splitting contract 9600 per-particle streams rest on, and a replacement
needs an equivalent with the same independence guarantee (xoshiro `long_jump`,
PCG stream increments, or `(key, counter)`); and every determinism gate
re-baselines.

### Phase 3 — the 2×2, then one commit that deletes both toggles — gh#748

With both toggles live, run the full factorial — {BTPE, BTRS} × {ChaCha8,
candidate} — end-to-end on this fit, not as a microbench.

**The interaction is expected and directional, so state the prediction before
measuring:** BTRS reaches an expensive test far less often than BTPE and
therefore consumes fewer uniforms per draw, so the PRNG's win measured on top of
BTPE will **overstate** its value in the final configuration. If the 2×2 shows
the two gains simply multiplying, that prediction was wrong and is worth writing
down as such.

Then **one commit flips both defaults and deletes both toggles**, their registry
entries, and their engine-version suffixes. Under `rust-conventions.md` the
toggles are permissible across ≥2 commits only as "a named step of a committed
arc"; this is that name. The equivalence suite and the A/B gates stay.

## Sequencing and re-key inventory

| step | lands                                                          | re-keys                                                        |
| ---- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| 0a   | gh#745 — audit `CAMDL_SEED` / `CAMDL_UNLABELED_THRESHOLD`      | nothing (may surface a separate bug)                           |
| 0b   | gh#746 — registry + classification + engine-version folding    | nothing while every var is default                             |
| 1    | gh#747 — BTRS behind `CAMDL_BINOMIAL`, default `btpe`          | nothing at default; `btrs` runs get their own addresses        |
| 2    | gh#748 — candidate PRNG behind `CAMDL_PRNG`, default `chacha8` | same                                                           |
| 3    | gh#748 — flip both defaults, delete both toggles               | **the whole CAS store; every pinned-number test re-baselines** |

Only step 3 invalidates anything, which is the point of the ordering: everything
before it is additive and reversible. Independent of all of it, and landable
today: `CAMDL_EVAL_FLAT=1` for this fit family (1.07×, byte-identity verified on
this model) and `lto = "fat"` + `codegen-units = 1` in `[profile.release]`
(gh#749) — with `target-cpu=native` gated on a green full `make test`, being the
one item that could perturb FP.

## Decisions

1. **Env vars split Identity / Provenance, enforced by a registry rather than a
   list.** Maintainer's design (2026-08-24); the registry-over-list and the
   asymmetric default are this proposal's amendments.
2. **Identity is the default class; Provenance requires a CI byte-identity
   gate.** Consequence: `CAMDL_EVAL_FLAT` is Identity until gated.
3. **Identity vars ride the engine-version string** rather than gaining their
   own hashed field.
4. **Re-keying is authorised** (maintainer, 2026-08-24) and is concentrated in
   step 3.
5. **Toggles are temporary**, deleted by the step-3 commit named above.

## Named follow-ups

- **`gate_eval_flat_ab.rs`** — the missing byte-identity gate that would let
  `CAMDL_EVAL_FLAT` be demoted to Provenance. Independent of this arc; the
  toggle's classification is correct either way.
- **Mechanical migration of the remaining `env::var` sites** behind the Phase 0b
  registry — additive, no re-key, can trail.
- **Choosing `N` on a PGAS mixing criterion.** `fit_province_nu_carec_16c.toml`
  justifies `particles = 9600` by bootstrap-filter loglik estimator spread,
  which is the PMMH criterion — PGAS forms no such estimator, and `N` there buys
  trajectory mixing, already instrumented per chain as `trajectory_renewal`,
  `renewal_b0..b9`, `as_accept`, `as_proposed`. If renewal holds at 2400 that is
  4×, larger than this entire proposal. Requires a _healthy_ chain (feasible
  start, non-zero NUTS acceptance); the 6-sweep probes in the note had θ frozen
  at 0% acceptance and are not evidence. Maintainer's call, not an engineering
  one.
- **gh#207 leaner per-particle history** — 6.9 GB RSS at these settings is the
  four `history_*` arrays. A memory lever, **not** a speed lever: allocation and
  memcpy are together under 0.5% of the profile. Recorded so it is not
  re-derived as a performance idea.
