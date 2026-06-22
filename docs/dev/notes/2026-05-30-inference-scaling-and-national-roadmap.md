# Inference scaling, and the path to national-scale fits

Date: 2026-05-29 Project: camdl Tags: inference, pmmh, pgas, if2, scaling,
spatial-coupling, profiling, roadmap

## Context / question

Forward-`simulate` is now fast and memory-safe at Kano scale after Fix E/D/B
([`2026-05-29-foi-scaling-bench.md`](2026-05-29-foi-scaling-bench.md)). The open
question is **inference**: can we get _fits_ (PMMH / IF2 / PGAS) to run at
**national scale (P ≈ 774 LGAs)** in a usable timeframe — ideally **< 5 days**,
the rough envelope a Kano-size fit occupies today? This note measures how
inference scales, identifies the ROI-ranked levers from profiling, and lays out
a path. It is the umbrella investigation; the load-bearing pieces (sparse
coupling DSL, binding cache) get their own proposals.

## Measured scaling (this is data, not a model)

Harness: `scripts/gen_scaling_models.py --observe` →
`camdl profile --algorithm
pmmh` (flamegraph) and `camdl fit run` per method
(cross-method timing). Figures: `assets/scaling/flamegraph_pmmh.svg`,
`assets/scaling/pmmh_scale.png`, `assets/scaling/method_scale.png`. The
per-step-eval finding is method- independent — every sampler runs the same
particle filter / propensity eval.

- **Per-step-eval-bound — verified.** Flamegraph of a PMMH run (P=16, A=7):
  **~72% in `sim::resolved_expr::eval_resolved`** (rate-tree walking), ~9%
  chain-binomial step, ~7% rayon mutex/scheduler, obs-likelihood/resample ≈ 0%.
  The whole cost _is_ the propensity evaluation.
- **Patches: ~O(P²)** — measured log-log slope 1.7 over P=16–44, _steepening_
  with P and with particle count (the small-P points are deflated by fixed
  per-run overhead; the cleaner high-N line slopes 1.4 and rising). The O(P²) is
  the dense spatial-coupling sum.
- **Ages: ~linear** (slope ≈ 1.1–1.3) — ages multiply cells but don't enter the
  coupling.
- **Particles: linear once compute dominates** — `wall(400)/wall(100)` climbs
  2.2× (P=4, overhead-bound) → 3.5× (P=32, compute-bound; 4.0 = pure linear). At
  national scale compute utterly dominates, so particles are firmly linear.
- **Iterations / MCMC steps: linear** by construction (each is one more PF
  sweep).
- **CPU parallelism is already exhausted — verified.** `particle_filter.rs:200`
  and `if2.rs:376` already `par_iter_mut()` across particles. More cores is
  _not_ an available lever; the ~7% mutex in the flamegraph is that
  coordination.

**So: `wall ≈ (iterations) × (particles) × O(P²·A)`, on a fixed core count.**

## Cross-method bench + big-run estimates

A direct IF2-vs-PGAS sweep (`assets/scaling/method_scale.png`; `fit run` per
method, A=7, 50 particles, 10 iters/sweeps, P=4–32) **corrected a wrong
assumption in an earlier draft of this note** — that PGAS would be _heavier_ per
iteration because of gradients. It is the opposite:

| P  | PGAS (per sweep) | IF2-refine (per iter) |
| -- | ---------------- | --------------------- |
| 4  | 0.4 s            | 4.5 s                 |
| 16 | 2.9 s            | 39 s                  |
| 32 | 9.3 s            | 134 s                 |

- **PGAS benches cleanly** (fixed sweep count → deterministic), slope **1.51**,
  and is _cheap_ per sweep. Particle-Gibbs + one NUTS-on-θ step is lighter than
  a bare perturb-filter pass, and gradient-informed proposals mix better → fewer
  sweeps. It wins on _both_ axes — matching the tool's own verdict
  (`fit
  methods`: PGAS is `[stable]` "production Bayesian path"; PMMH is
  `[experimental]`, "degrades for T > 500 observations").
- **The IF2 number here is confounded** — run via a `refine` stage whose
  dt-convergence machinery does a _data-dependent_ number of extra filter passes
  (a near-identical P=4 config took 1.0 s in one run and 45 s here). Discard the
  IF2 _absolutes_; a bare-IF2 bench is a TODO.
- **Both scale ~O(P^1.5→2)** — model-size scaling is method-independent (the
  whole point; see the lever ranking).

National posterior estimate, from the _measured_ PGAS per-sweep (×(774/32)^1.5
in P, ×3 for A=7→21; 50 particles, ~500 post-burn-in sweeps):

| coupling                | national PGAS per-sweep | national posterior                 |
| ----------------------- | ----------------------- | ---------------------------------- |
| dense (current default) | ~3400 s                 | **~20 days** — over the <5-day bar |
| sparse (÷~50)           | ~70 s                   | **~hours–a few days** — feasible   |

So **national PGAS in <5 days is reachable, but only with sparse coupling** —
the same conclusion the PMMH analysis reached, now confirmed on the production
method. Choosing PGAS buys a healthy constant factor and is the right engine; it
does **not** beat the quadratic. (For reference, the PMMH extrapolation that
opened this investigation put national PMMH at ~1 year — `pmmh_scale.png`.)

Caveats so "days" isn't over-trusted: the sweep _count_ to a converged national
posterior is unmeasured (PGAS default burn-in is 2000), 50 particles is
optimistic (bigger national state → more particles to avoid filter degeneracy),
and the P-slope ~1.5 is overhead-deflated toward 2 at scale. The robust claim is
the _direction_ — sparse coupling moves national PGAS from weeks-to-months down
to days — not the exact figure. A fixed-ESS study (below) is needed to rank
methods on _total_ fit time.

## ROI-ranked levers

| # | lever                                                                                                    | type                            | est. gain                                        | cost                                                                                    | status                                                                    |
| - | -------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 1 | **Sparse coupling** (sparse `W` literal + a constant-fold pass dropping zero-weight terms): O(P²)→O(P·k) | algorithmic, **byte-identical** | **~50×** @ national                              | small–medium (a fold pass + compile-time kernel fns; **no sparse IR/eval** — see below) | deferred non-goal of the bindings proposal — **now promoted & de-risked** |
| 2 | **Per-step binding cache** (compute `N[l]`,`I_agg[l]` once/step)                                         | constant, byte-identical        | ~2–4×                                            | small                                                                                   | the preamble Fix B skipped; scoped                                        |
| 3 | **SIMD / flattened eval** (vectorise the propensity walk)                                                | constant                        | ~2–8×                                            | medium                                                                                  | `eval_resolved` is an interpreted tree-walk                               |
| 4 | **Right method = PGAS** (cheap per-sweep + iteration-efficient; IF2 for MLE, PMMH only as fallback)      | statistical + constant          | fewer iters **and** cheaper per-sweep (measured) | low (PGAS is already production)                                                        | bench: PGAS ≪ IF2-refine per iter, both O(P^1.5)                          |
| 5 | **GPU particle batching**                                                                                | hardware                        | ~100–1000×                                       | large                                                                                   | future, costed (below)                                                    |
| — | ~~more CPU threads~~                                                                                     | —                               | already spent                                    | —                                                                                       | rayon per-particle is live                                                |

### Minimal viable stack for < 5-day national fits

**PGAS (the right method) on sparse coupling** is the path: measured PGAS at
national scale is ~20 days dense → **~hours–days with sparse coupling (÷~50)**.
The binding cache (~3×) and SIMD (lever 3) are headroom on top. Sparse coupling
_also_ shrinks the IR (fewer FOI terms), fixing the national-scale **compile**
(which otherwise won't fit in memory). **Sparse coupling is the one
non-negotiable; PGAS is the vehicle; binding cache / SIMD / GPU are headroom.**

## Coupling structure as a first-class, comparable model

The key reframing (per maintainer): the spatial coupling is a **scientific
hypothesis**, not just a perf knob. The dense all-to-all `W` is _one_ model
(sum-everything). The national-scale program wants **all of**:

- **local neighbour mixing** — kNN / gravity / radiation kernels (the O(P·k)
  bulk),
- **a few long-range links** — aviation, transport corridors (a sparse overlay,
  +m),
- and **model comparison across them** — which coupling structure the data
  actually supports, via the existing `camdl compare` (elpd / CRPS / PIT).

So this is "the middle layer is the program": make coupling a swappable
component. And — verified against the compiler — **most of the lift is the
compiler's, not new surface.**

**The mechanism (existing primitives + one fold pass).** `expander.ml:1706`
already lowers `sum(q in patch, W[l,q]·…)` to an n-ary `Reduce` of P terms, each
with `W[l,q]` a `TableLookup` over _constant literal indices_ into the literal
`W` table — but `normalize_expr` (866–894) does no constant-folding, so the
zeros survive. Add a fold/peephole pass that resolves a constant-indexed
`TableLookup` of a literal table to its scalar, folds `0·x→0`, and **drops
`Const 0.0` terms from a `Reduce`**. A dense P-term row collapses to a k-term
row → **O(P²)→O(P·k)**, with _no_ new IR node and _no_ sparse runtime evaluator
(the runtime already evals `Reduce`, just with fewer terms). It is
**byte-identical** — dropping a `+0.0` from the left-fold is the additive
identity, and the FOI's existing `Cond` div-guard keeps the dropped term exactly
0 (`0·Cond(N>0, I/N, 0)=0`) — so it ships under the trajectory gate, same as Fix
B/D. The earlier "needs a sparse IR representation + sparse-FOI eval" framing
was therefore **overstated**: it doesn't.

**Compile-time `W` is the enabling design rule.** The fold only fires when `W`
is a **compile-time literal** — so the coupling matrix should be supplied _at
compilation_, not loaded from a runtime data file. (A runtime-loaded `W` falls
back to the dense O(P²) sum; that real-sparse-eval case is optional and
separate.) This is a clean rule to surface to users: **pass the coupling
structure to the compiler and you get the O(P·k) IR for free.**

**The DSL ergonomics functions (wanted — they read clearer).** Layer the
human-first surface on top as **compile-time table-producing functions**:
`W : patch × patch = knn(coords, 8) + corridors(air_links)` reads exactly the
mixing assumed, evaluates to a literal `W` at compile time, and the fold pass
then prunes it. That's a constant-position function evaluator (`gravity_kernel`,
`radiation_kernel`, `knn`, `corridors`) — a _small_ addition, not a sparse-IR
rewrite. (The same matrices can also be emitted offline as a literal table via a
Makefile data step — committed, with provenance — but the in-DSL form is
preferred for readability and is the maintainer's call.)

The cross-method bench confirms this is **method-independent** — IF2, PGAS and
PMMH all inherit the O(P²) coupling — so it is the single highest-leverage lift
_regardless of sampler_, and now also the **lowest-risk** (byte-identical,
gate-tested). It still warrants a short proposal, but a far smaller one: (1) the
constant-fold pass + its gate, (2) the compile-time kernel functions, (3) the
compile-time-`W` design rule. No sparse IR type, no sparse runtime eval.

## Method: PGAS is the vehicle (measured, corrected)

The cross-method bench corrected an earlier assumption that PGAS would be
_heavier_ (gradients). It is the opposite:

- **PGAS** — the production posterior engine, and _measured cheapest_: ~9
  s/sweep at P=32 (A=7, 50p), cleanly O(P^1.5). Particle-Gibbs + one NUTS-on-θ
  step is lighter than a perturb-filter pass, and gradient-informed proposals
  mix better → fewer sweeps. Wins on per-sweep cost _and_ iterations. Push
  **this** to national scale.
- **IF2** — fast MLE / point estimates; the gradient-free perturb-filter loop.
  Its cost here was confounded by the `refine` stage's convergence machinery
  (see above); a bare-IF2 bench is needed before ranking it against PGAS.
- **PMMH** — robust gradient-free fallback, but `[experimental]` and degrades
  for T > 500 observations (`fit methods`) — _not_ the method for national
  posteriors. It was the right place to _start_ the scaling investigation
  (gradient- free, simplest), but not the destination.

So: center the national-scale path on **PGAS**, use **IF2** for cheap point
estimates, keep **PMMH** as a fallback. The remaining unmeasured axis is
_iterations-to-converge_ per method (per-iter cost × iters = total fit time);
the fixed-ESS study (below) is the disciplined way to settle the ranking.

## GPU (future, costed)

The particle filter is embarrassingly parallel across particles (already rayon
on CPU; GPU is the natural next substrate). Batching particles on GPU is
~100–1000× and the only lever buying headroom _beyond_ national scale
(continental, fine-grained age × space, large ensembles). Cost: a parallel eval
path (the `ResolvedExpr` interpreter → a GPU kernel or a flattened bytecode the
GPU runs), RNG-on-device, and resampling-on-device — a multi-month project
touching the eval core. Not a near-term prerequisite; revisit once 1+2 land and
the profiler says the remaining bottleneck is raw eval throughput.

## Profiling-driven sequencing

1. **Land the binding cache** (lever 2) — cheap, byte-identical, immediate.
   Re-run `make profile-pmmh`; confirm `eval_resolved` share drops.
2. **Prototype sparse coupling** (lever 1) on a national-scale synthetic (kNN
   `W`); measure the actual P-slope flip (O(P²)→O(P·k)).
3. **Re-profile** — the bottleneck _will move_. Likely candidates next: RNG
   draws, resample, obs-likelihood, or memory bandwidth. Let the flamegraph pick
   the next target rather than guessing.
4. **SIMD** the eval (lever 3) if it still dominates.
5. **GPU** (lever 5) as the beyond-national bet.

Each step is gated by a re-profile — no speculative optimisation.

## Next

- Particle linearity — **done** (`pmmh_scale.png`); IF2-vs-PGAS bench — **done**
  (`method_scale.png`).
- A **fixed-ESS cross-method study**: run IF2/PGAS/PMMH to the same effective
  sample size and compare _total_ wall — the only fair ranking (the
  iterations-to-converge axis, currently unmeasured). Includes a **bare-IF2
  bench** (the refine-stage number here is confounded).
- A (now _small_) proposal for **sparse coupling**: a byte-identical
  constant-fold pass that drops zero-weight `Reduce` terms + compile-time kernel
  functions (`knn`/`gravity_kernel`/`corridors`) + the compile-time-`W` rule.
  **No sparse IR type, no sparse runtime eval** (verified: `expander.ml:1706`
  already emits the `Reduce`; only the fold is missing). The one thing that
  makes national scale tractable on _any_ method.
- Land the **binding cache** (small, scoped) and re-profile.
- Settle **particles-vs-state-dimension** (does national need proportionally
  more particles? — re-run with a degeneracy diagnostic).
- A clean, committed `bench-inference-scale` harness (the sweeps here were
  ad-hoc in `/tmp`) so the curves are reproducible and trackable as levers land.
