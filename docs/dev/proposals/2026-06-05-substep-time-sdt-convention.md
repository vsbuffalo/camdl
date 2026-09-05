# Substep-time convention: `t_start + s·dt` everywhere

Status: accepted (maintainer, 2026-06-05) — adopt the robust convention, accept
the result changes for time-inhomogeneous models. Implemented: chain_binomial
(the whole-run drifter, now matches PGAS) — 5aa4bf2. Deferred (low priority,
task #14): the EXACT steppers (ode, tau_leap, particle_filter, correlated_pf,
if2) clip to every boundary/obs, so their drift is already bounded to one window
(~1e-15); s*dt there is near-zero-benefit consistency (byte-identical except
ode's continuous output). Supersedes: nothing. Part of the
unified-timeline-effect architecture
(`2026-06-05-unified-timeline-effect-architecture.md`).

## Decision

Every fixed-step integrator computes the substep start time — the `t` passed to
rate / forcing evaluation — as `window_start + s·dt` (s = within-window substep
index), **not** by accumulation (`t += dt`). This is the only value `t` that
reaches a time-dependent rate (`propensity.rs:87` `Expr::Time → ctx.t`, `:186`
`eval_time_func(…, ctx.t)`), so it is the only thing that changes.

## Why (the robustness argument)

`s·dt` is one multiply + one add → bounded **O(1)** rounding error. Accumulation
is `s` separate roundings → error grows **O(s)**. Measured: summing `0.1` out to
1100 days drifts the clock to `1100.1000000000095` (vs exact `1100`), a peak
`|accumulated − s·dt|` of **1.6e-10 days**, which for annual seasonal forcing is
a **~1e-12 relative** shift in the rate. Integer counts are insensitive to that
(forward trajectories byte-identical), but the **continuous** PGAS transition
density (`gamma`, `shape = dt/σ²`) is not — so today the forward simulator and
the PGAS likelihood sample seasonal forcing at _different_ times for the same
model. That latent forward/inference disagreement is what this removes. Runtime
cost: nil (multiply vs add). It is purely an accuracy + consistency improvement.

## Blast radius

Sharper than first thought: the `s·dt` rate shift is ~1e-12 relative, and a
~1e-12 shift in a rate **never flips an integer draw**. So the only things that
move are **continuous** quantities of time-inhomogeneous models at fractional
`dt`:

- **Observably changes:** **ODE forward** trajectories (continuous state). That
  is the _only_ observable forward change in the whole effort (verified:
  chain_binomial on `seasonal_drift@dt=0.1` is byte-identical before/after —
  integer draws are insensitive).
- **Byte-identical (consistency-only):** chain_binomial, tau_leap, gillespie
  forward (integer counts); the bootstrap PF / IF2 / correlated-PF / PMMH
  likelihoods (functions of the integer trajectory + obs density on integer
  projections). These adopt `s·dt` for robustness + to agree with PGAS, but the
  output does not move.
- **Already correct, hence unchanged:** PGAS / pgas_grad (use `s·dt` today).
  PGAS's transition density _is_ continuous and _would_ move under accumulation
  — which is exactly why the forward backends adopt `s·dt`: to remove the latent
  (and, in integer trajectories, invisible) forward/PGAS disagreement, not to
  change PGAS.
- **N/A:** time-homogeneous models (`t` never enters a rate); integer-`dt` runs
  (`accumulation == s·dt` exactly); gillespie (absolute event times); NUTS
  leapfrog (its `dt` is the HMC step size, not a sim grid).

## Exhaustive site inventory (verified by grep + read, 2026-06-05)

| site                           | role          | current                                  | anchor after fix                         |
| ------------------------------ | ------------- | ---------------------------------------- | ---------------------------------------- |
| `chain_binomial.rs:211,304`    | forward SNAP  | `t += dt` (whole run)                    | **global** `t_start + s·dt` (match PGAS) |
| `tau_leap.rs:150,301`          | forward EXACT | `t += dt` per clipped window             | per-window `window_start + s·dt`         |
| `ode.rs:254,261`               | forward EXACT | `t += dt` per clipped window             | per-window `window_start + s·dt`         |
| `particle_filter.rs:243,251`   | bootstrap PF  | `t_local += step_dt` per obs window      | per-obs-window                           |
| `correlated_pf.rs:342,347,354` | correlated PF | `t_local += step_dt`                     | per-obs-window                           |
| `if2.rs:409,415`               | IF2           | `t_local += step_dt`                     | per-obs-window                           |
| `pmmh.rs`                      | PMMH          | runs a PF for L̂(θ); no own stepping loop | inherits PF / correlated_pf fix          |
| `pgas.rs:568,605,716`          | PGAS          | `t_start + s·dt` ✓                       | unchanged (canonical)                    |
| `pgas_grad.rs:397`             | PGAS gradient | `t_start + s·dt` ✓                       | unchanged                                |
| `gillespie.rs`                 | forward SSA   | absolute `t = iv_t/boundary/t_next`      | unaffected                               |

The inference EXACT steppers (PF/correlated_pf/if2) all advance the kernel via
`ChainBinomialProcess::step` (`chain_binomial_process.rs:92`), which forwards
the caller's `t` to `step_one`. So fixing the `t_local` each one computes is the
fix; the kernel needs no change.

## Anchoring

- **SNAP** (chain_binomial forward, PGAS): global grid,
  `window_start = t_start`, step count `= interval_steps(t_start, t_end, dt)`.
  chain_binomial adopts PGAS's exact convention so the same model samples
  forcing identically in sim and fit.
- **EXACT** (tau_leap, ode, PF, correlated_pf, if2): the grid re-anchors at each
  clip — `window_start` = the boundary (output / intervention) or obs time the
  stepper just landed on; `s` resets to 0 there. This bounds the drift to a
  single inter-boundary window (≤ `window/dt` substeps) instead of the whole
  run.

## Implementation

A single robust helper on `Schedule` (one source of truth):

```rust
/// Substep start time: window_start + s*dt, bit-exact regardless of s.
pub fn substep_time(&self, window_start: f64, s: u64) -> f64 {
    window_start + s as f64 * self.dt
}
```

Each stepper tracks `(window_start, s)`, passes
`schedule.substep_time(window_start, s)` to the kernel, and re-anchors
`(window_start = boundary, s = 0)` at each clip (EXACT) or never (SNAP, global).
Step size still comes from `Schedule::substep` (the bit-exact
`dt.min(boundary - t)` already landed in 16a61c8).

## Baselines that regenerate

- Forward seasonal goldens: `seir_vaccine_seasonal`, `seir_seasonal_patch`,
  `seir_spatial_5_inference` (any with `forcing {}` at fractional `dt`).
  `make update-golden && make update-expected`; review the diff is ONLY the
  seasonal/fractional-dt models.
- `gate_trajectory_baseline` / `gate_corner_case_baseline`: recapture; the new
  `seasonal_drift` corner case is added here as the permanent pin.
- `gate_inference_baseline`: the `sir` references are `dt=1` (integer) and
  homogeneous → **unchanged** (regression check: they must not move).

## Verification

1. Time-homogeneous + integer-`dt` corpus: byte-identical (regression — must not
   move). This is the proof the change is scoped to what we intend.
2. `seasonal_drift` (chain_binomial, `dt=0.1`): forward trajectory + a PGAS
   loglik baseline now AGREE on forcing-sample times (the consistency this
   buys).
3. The `schedule::tests::substep_is_bit_exact_*` style pin for `substep_time`.

## Open item

Confirm PMMH's stepping path (does it call `bootstrap_filter`, inheriting the
fix, or duplicate a stepping loop?). Resolve during implementation.
