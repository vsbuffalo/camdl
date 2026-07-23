# Mean-field model build: two O(H²) setup costs in the runtime

Date: 2026-07-22 Project: camdl Tags: perf, compiled-model, mean-field,
gillespie, scaling

## Context

A household-structured TB model — household-as-stratum, with a two-scale force
of infection (a strong within-household term plus a weak between-household
mean-field `sum` over all `H` households) — forward-simulated with wall time and
RSS that grew _superlinearly_ in `H`, even though the compiled IR is O(H)
(commit `68d3385b` already fixed the IR-size blowup by hoisting the shared
aggregate).

A single 10-year daily chain-binomial run cost ~52 s and 9.4 GB at H=6400. A
60-year run cost about the same as a 10-year one, which localised the cost to
one-time model **setup** (building `CompiledModel`), not per-step integration:

```
# new binary, H=3200, --stdout (no store write)
10-yr sim: 8.47s ; 60-yr sim: 40.99s   # setup ≈ intercept ≈ 2s; the rest scales with steps
```

## What the profiler showed

`sample` of `camdl simulate <ir>` at H=3200/6400: `CompiledModel::new`
dominated, and inside it the time was a HashMap-insert / SipHash storm plus
`collect_int_comp_deps` — building per-transition dependency structures. Root
cause: every household's FOI rate references the hoisted aggregate
`I_glob = sum over all H households`, and the build walkers _see through_ that
binding, re-expanding its O(H) dependency set into each of the ~2H coupled
transitions → **O(H²)**. Two independent instances of the same pattern:

1. **`comp_to_transitions`** — the compartment → dependent-transitions graph.
   O(H²) to build and O(H²) to _store_ (the 9.4 GB). Verified single consumer:

   ```
   $ rg -n comp_to_transitions rust/crates/sim/src
   compiled_model.rs: (field + build)
   gillespie.rs:453:  for &tr_idx in &model.comp_to_transitions[local]   # only reader
   ```

   Only Gillespie's sparse propensity-invalidation reads it; chain-binomial /
   ODE / PGAS / PMMH recompute all propensities each step and never touch it.

2. **`expr_contains_dt` / `expr_is_time_dependent`** — the per-transition scans
   that build `time_dep_transitions` and the `RUNTIME_DT` capability check. Each
   re-descends the O(H) mean-field binding once per transition → O(H²) in _time_
   only (traversal, no allocation → time-superlinear, RAM-linear, which is
   exactly what the post-fix-1 benchmark showed).

## The fixes

1. **Lazy `comp_to_transitions`** (commit `f62d6d07`). Move it behind
   `CompiledModel::comp_to_transitions()`, populated on first Gillespie use via
   a `OnceLock`. Non-Gillespie backends never build it. Behaviour unchanged —
   Gillespie gets the identical graph — the O(H²) cost simply leaves every other
   backend.

2. **Memoize the binding scans** (commit `cd5ec70e`). `expr_contains_dt_memo` /
   `expr_is_time_dependent_memo` cache, per binding, whether its body contains
   `Dt` / is time-dependent, so a reused `BindingRef` is descended once rather
   than once per transition. Sharing one memo across a transition scan makes it
   O(1) in the binding and O(H) overall.

Both are behaviour-preserving: byte-identical trajectories, identical
classifications feeding Gillespie. `cargo test -p sim`: green (the lone
`expr_eval` failure is a pre-existing parallel race on the process-global
`allow_degenerate_rates`, unrelated — passes alone / single-threaded).

## Measured (forward sim, 10-year daily chain-binomial, one run)

`old` = installed pre-fix binary (has `68d3385b` but neither fix here).
`bench_sim.sh`.

|    H | old s | old GB | +fix1 s | +fix1 GB | +both s | +both GB | total speedup |
| ---: | ----: | -----: | ------: | -------: | ------: | -------: | ------------: |
| 1600 |  11.0 |   2.11 |     5.2 |     1.53 |     5.3 |     1.53 |          2.1× |
| 3200 |  36.1 |   5.32 |    11.9 |     3.08 |     9.4 |     2.97 |          3.8× |
| 6400 | 133.7 |  13.15 |    28.8 |     5.85 |    19.8 |     5.85 |          6.8× |

Fix 1 (lazy graph) does the heavy lifting and **all** the RAM saving: peak RSS
goes from superlinear to linear in H (13.2 → 5.9 GB at H=6400). Fix 2 (memoized
binding scans) removes the residual time-only O(H²): a further ~1.45× at H=6400
(28.8 → 19.8 s). With both, forward-sim wall time is ~linear in H (5.3 → 19.8 s
from H=1600 → 6400), so a 10k-household run extrapolates to ~30 s (vs ~5 min
unfixed). Numbers are ±measurement variance (single runs).

## Why it matters, and what it does not touch

The O(H²) was one-time model _setup_. It inflated single forward simulations and
their RAM at large H; the per-step integration was always cheap. So this speeds
up forward simulation and scenario sweeps substantially at large H, and drops
peak RSS from superlinear to linear.

It does **not** change per-iteration fitting cost. A fit builds the model once
(`Arc<CompiledModel>`) and reuses it across every particle and MCMC iteration,
so the dominant fit cost (particles × integration) is unchanged. Fitting large
`H` remains gated by that product, not by model build.

## Follow-ups

- The TB whitepaper's "compiles and simulates in time and memory linear in H"
  was accurate for the IR but overclaimed the runtime. With these fixes the
  setup is ~O(H); forward-sim time is now floored by integration and trajectory
  I/O.
- Pre-existing flaky test:
  `expr_eval::test_log_nonpositive_rate_errors_via_eval_propensities` races on
  the process-global `allow_degenerate_rates` toggle under parallel test
  execution. Worth scoping that global (a serialized guard or a per-call param).
