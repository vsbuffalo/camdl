# LICM kernel benchmark: fittable in-model kernel reaches fixed-kernel speed

Date: 2026-06-22 Project: camdl Tags: gh#272, licm, performance, pgas,
benchmark, inference

## Context / question

gh#272: computing a gravity coupling kernel **in-model** (so its decay exponent
γ is a fitted parameter) costs ~3.77× more per likelihood eval than reading a
**precomputed** `kernel.tsv`, because the loop-invariant kernel is rebuilt every
integration substep. The loop-invariant code-motion (LICM) compiler pass hoists
that rebuild out of the per-step evaluation. This note records the measured
speedups now that the runtime threading is complete on every path, and pins the
on/off value-preservation as a standing CI gate.

Two questions to answer with numbers:

1. Does LICM close the in-model-vs-precomputed gap, and by how much?
2. Does it close it on the **stochastic** path (PGAS), not just MH-ODE? Phase 2
   threaded the staged scratch through the PGAS/IF2/PF producers, so this is the
   path that work was for.

## Setup

Real model: the SLE-14 (Sierra Leone, 14-district) metapopulation SEIRD from the
gh#272 MRE (`ebola_camdl/mre/stage/`, a private playpen — not vendored here).
Two models identical except for the kernel:

- `model_precomputed.camdl` — `read("kernel.tsv")`, γ baked (the fixed-kernel
  baseline; 1.35 MB IR).
- `model_inmodel.camdl` — `W[p,q] ∝ N0[q]·exp(−γ·log(dratio[p,q]))`,
  row-normalized inline, γ_k a fitted parameter (17.9 MB IR inlined).

LICM is **on by default** (since the gh#272 flip); `--no-licm` / `CAMDL_NO_LICM`
forces the inlined variant for the off measurements below. It is a compile-time
pass, so the two variants produce distinct (hoisted vs inlined) IR with distinct
IR-cache keys + run identity; see the wiring in `rust/crates/cli/src/util.rs`
(`ir_cache_key` / `run_camdlc_compile`). All timings below are with the IR cache
warm (compile excluded), `--progress none`, single chain, `dt = 1`. (The off
measurements were taken with `CAMDL_NO_LICM=1` / before the flip; the commands
shown predate the flip's `--licm` → `--no-licm` rename — substitute accordingly.)

Reproduction (shortened iteration counts; the upstream perf configs run 20k MH
iters):

```
# MH-ODE: ms/eval reported directly by the fit
camdl       fit run bench_precomputed.toml --seed 1 --force   # fixed kernel
camdl       fit run bench_inmodel.toml     --seed 1 --force   # equation, LICM off
camdl --licm fit run bench_inmodel.toml    --seed 1 --force   # equation, LICM on

# PGAS (chain_binomial backend + NUTS gradient): 12 sweeps, 6 burn-in, 64 particles
camdl       fit run bench_pgas.toml --seed 1 --force          # LICM off
camdl --licm fit run bench_pgas.toml --seed 1 --force          # LICM on
```

(`bench_*.toml` add `prior = { flat = {} }` to each `[estimate]` param — the MH
posterior needs a resolved prior — and shorten `iterations`/`sweeps`.)

## Measurements

MH-ODE (`algorithm = "mh"`, `backend = "ode"`), ms/eval reported by the fit, 3
reps, IR cache warm:

| kernel              | LICM | IR size | ms/eval (reps)   | mean |
| ------------------- | ---- | ------- | ---------------- | ---- |
| precomputed (fixed) | —    | 1.35 MB | 9.7, 9.6, 9.7    | 9.7  |
| in-model (equation) | off  | 17.9 MB | 36.8, 36.3, 36.6 | 36.6 |
| in-model (equation) | on   | 3.9 MB  | 9.6, 9.4, 9.5    | 9.5  |

PGAS (`algorithm = "pgas"`, `backend = "chain_binomial"`, NUTS enabled), wall
time for 12 sweeps + 6 burn-in:

| kernel              | LICM | wall    | best complete-data ll |
| ------------------- | ---- | ------- | --------------------- |
| in-model (equation) | off  | 322.1 s | −25771.0              |
| in-model (equation) | on   | 54.8 s  | −25771.0              |

## Observations

- **MH-ODE: 3.85× faster** (36.6 → 9.5 ms/eval). The fittable in-model kernel
  now runs at **parity with the precomputed fixed matrix** (9.5 vs 9.7 ms/eval)
  — the 3.77× penalty is fully erased.
- **PGAS: 5.9× faster** (322.1 → 54.8 s). PGAS gains _more_ than MH-ODE because
  it rebuilds the kernel on **two** surfaces per substep — the CSMC producer
  (`step_one`) and the NUTS gradient (`complete_data_loglik_grad`) — and Phase 2
  hoists both.
- **IR shrinks 17.9 MB → 3.9 MB** (4.6×), which also cuts the (national-scale)
  compile RSS.
- **Value-preserving.** MH-ODE MAP loglik is identical off vs on (−2025.3); PGAS
  best complete-data loglik is identical to the last digit (−25771.0) at the
  same seed. LICM does not move the posterior — it makes the same answer ~4–6×
  cheaper.
- The IR cache confirms the `--licm` run used the hoisted IR (a distinct cache
  key, `per_eval_bindings = 1`) and a non-`--licm` run used the inlined IR
  (`per_eval_bindings = 0`), so the flag re-keys as designed and an old
  (pre-LICM) CAS entry stays valid (a non-`--licm` run is byte-identical to
  pre-feature IR).

## Interpretation

The motivating use case — fit the gravity distance-decay exponent γ as a free
parameter with a posterior, instead of profiling over rebuilt `kernel.tsv` files
— now costs the **same per eval as the precomputed matrix**, on both the MH-ODE
production path and the exact PGAS path. The "in-model kernels are too slow to
fit" objection is removed.

The MH-ODE number reflects Phase 1 (the `run_ode` staging); the PGAS number
reflects Phase 2 (the stochastic producer + gradient staging). The two together
cover the deterministic and exact-Bayesian inference paths.

## Regression gates (standing, in CI)

Flipping `--licm` on must never change results. Pinned by `gate_licm_ab.rs`
(runs in the inference test tier, self-contained `licm_ab` fixture — no MRE
data):

- `gate_licm_is_byte_identical` — forward trajectory byte-identity off vs on
  across gillespie / chain_binomial / ode (§2), plus rate_grad eval-equality
  (§3) and a staged-scratch-vs-on-demand expression A/B (§4).
- `gate_licm_inference_producer_byte_identical` — the shared inference producer
  (`ProcessModel::step` → `step_one`, used by PF / IF2 / PGAS / PMMH): byte-
  identical particle counts + flow accumulators off vs on.
- `gate_licm_pgas_loglik_byte_identical` — the result-level guard: the PGAS
  complete-data log-likelihood AND its full NUTS gradient are byte-identical off
  vs on (drives `simulate_reference_on_grid`, `complete_data_loglik`, and
  `complete_data_loglik_grad`). This is the in-tree analogue of the SLE-14
  −25771.0-both measurement above.

## Next

- **LICM is now default-on** (`--no-licm` / `CAMDL_NO_LICM` to disable). The flip
  was confirmed golden-neutral: no golden model has hoistable structure, so
  `make update-golden` under LICM-on changed zero files, and run identity re-keys
  only for models that actually hoist (a user in-model kernel) — a non-hoisting
  model's IR is byte-identical to pre-flip, so existing CAS entries stay valid.
- A self-contained perf fixture (small, committed) would let a perf-acceptance
  number live in CI without the private MRE; not built yet.
- Follow-ons: the flat-eval per-eval tape and the strength-reduction peephole.
