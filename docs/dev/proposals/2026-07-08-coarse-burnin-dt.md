# Coarse burn-in step for ODE gradient fits — cheap, faithful transient integration

Status: Phase 0 implemented (this branch); Phase 1 (general `ForcingTimes`
version) is spec. Follows: gh#396 (the periodic-equilibrium warm-start, shelved
for densely-coupled models — see PR #399). High-risk in Phase 1 only: adds a
boundary source to the shared scheduling spine — read `schedule.rs` +
`boundary_times.rs` and gate on golden-neutrality. Rust-only (a fit-time runtime
option); no IR/schema change. Knots are extracted from the existing forcing IR.

Phase 0 — landed as `burnin_dt` on `[stages.<nuts>]`, wired into
`nuts`-on-`ode`, with three deviations/findings recorded during implementation:

- **Split at `first_obs`, no `condition_from` required** (narrower than the
  proposal's "requires a `condition_from`"). The unscored warm-up is
  `[t_start, first_obs)` regardless, so for a **prevalence** (state-scored)
  stream `first_obs` is exactly the split `condition_from` would name.
  **Incidence** (interval) streams are refused: their first scored bin
  accumulates flow from `t_start`, which coarsening would bias, and the upstream
  wide-gap guard only catches windows > 5 cadences — so the safe rule (refuse
  incidence in Phase 0) is gated on stream temporal kind. The multi-stream split
  is the global `min(first_obs)` (coarse only where no stream is scored).
- **Clamp-refusal guard added first** (separate commit). The augmented
  `rk4_step` clamps the state to `>= 0` but not the signed sensitivity `S`; when
  the clamp fires, value and gradient diverge. The comment/capability-gate long
  _claimed_ this was refused under `nuts` — it was not (a doc-vs-code lie).
  Coarse steps make the overshoot materially more likely, so the latent
  silent-wrong hole is now a hard error (propagated to the NUTS target as `-inf`
  / an up-front probe error).
- **Non-finite-gradient guard** in the ODE-NUTS target: a stiff/coarse step that
  blows `S` to `inf`/`NaN` while the value stays finite now falls to `-inf`
  (clean rejection) instead of poisoning the leapfrog trajectory.

## Problem

A seasonally-forced model with an origin decades before the data (garki
`mv_age`, origin 1910, data 1970) re-integrates the whole ~63-yr transient on
every gradient. The periodic-equilibrium warm-start (gh#396) solves this in
principle by jumping to the limit cycle, but forming the monodromy is
`O(n_comp)` per Newton step — one monodromy ≈ the entire transient at garki's
196 compartments, and its dense force-of-infection coupling makes even a
matrix-free rework expensive. It is also a _model change_ when the system isn't
actually settled at the data-window start (garki's ~61-yr burn-in ≈ one measured
~56-yr relaxation time).

The pragmatic alternative: integrate the transient **faithfully but cheaply** —
big time steps where the dynamics are smooth, exact `dt` where the data is
scored. This recomputes the state (and its gradient) at the _actual_ parameters
each eval, so — unlike a frozen checkpoint — the NUTS gradient is correct; the
only cost is a small, controllable discretization bias.

## Empirical basis (measured, not assumed)

The garki testing agent measured plain coarse `dt` on the burn-in (no-obs
simulate): **rel-L2 state drift 0.33% at dt=3.5, 1.27% at dt=7, 0.66% at
dt=14**, for total speedups of **~5.4× (dt=7) to ~8.5× (dt=14)** (burn-in ≈ 95%
of the trajectory). The mechanism works; what Phase 0 confirms is that the
speedup and accuracy carry through the _augmented-gradient fit path_, not just a
forward solve.

## Goal

A per-fit resolution knob, `burnin_dt`: coarse RK4 steps on the unscored warm-up
`[t_start, cond_from]`, exact `dt` on the scored window `[cond_from, t_end]`,
with steps **snapped to forcing knots** so a big step never straddles a forcing
change. General across forcing kinds; correct gradient (the augmented
sensitivity flows through the coarse region); no monodromy, no dense-coupling
penalty.

## Why this reuses the scheduling spine (not a new mechanism)

The ODE stepper already takes `h = min(dt, h_max)` where `h_max` is the distance
to the next output/effect boundary (`ode.rs` `Rk4Fixed`, `Schedule::next_stop`)
— "big step, clipped so it lands on the next boundary" is the _existing_
snap/clip logic. Two additions:

1. **Region-aware step.** Use `burnin_dt` while `t < cond_from`, `dt` after —
   the stepper's nominal step becomes region-aware. `cond_from` is already the
   boundary that splits unscored warm-up from scored data (it's an existing
   schedule stop), so the split needs no new time.
2. **`ForcingTimes` boundary source.** A sibling to `OutputTimes`/`EffectTimes`,
   built by `ForcingTimes::from_model`. It contributes the forcing knots so
   `next_stop`/`clip` truncate coarse steps at them — the same pattern, so
   nothing in the step loop changes.

## Generality across forcing kinds (self-adapting)

`ForcingTimes` extracts knots per kind; smooth kinds contribute none, so plain
coarse steps just work there:

| forcing kind     | knots                            |
| ---------------- | -------------------------------- |
| `Sinusoidal`     | none (smooth)                    |
| `Fourier`        | none (smooth)                    |
| `Piecewise`      | segment breakpoints              |
| `Interpolated`   | table `times` (garki's `C_v(t)`) |
| `Periodic` step  | step times within each period    |
| `PeriodicSpline` | implicit uniform knots `k·(P/n)` |

So one general mechanism: coarse-step everywhere; knot-truncation kicks in
exactly where the forcing has structure a big step could smear (garki's
piecewise-constant table jumps 0.25→3.52, a 14× step, at ~Jun 21 / Nov 8 each
year).

## Golden-neutrality (the Phase-1 gate)

Adding a boundary source to the `Schedule` used by all four backends must not
change any existing trajectory. It doesn't, because **forcing knots are calendar
times on the `dt=1` grid**: at `dt=1` the walk already lands on every integer
day, so adding those days as boundaries is a no-op. The change only clips steps
when `step > knot spacing`, i.e. only under `burnin_dt > dt` (the new path).
Phase 1 must _verify_ this across all goldens (`make update-golden` produces an
empty diff) — any forcing knot off the `dt` grid would clip a `dt=1` step and
shift a trajectory, which is the one thing to catch before merge.

## Surface

```toml
[stages.posterior]
algorithm = "nuts"
backend = "ode"
burnin_dt = 7.0 # coarse RK4 step on [t_start, cond_from]; default = dt (off)
```

A single knob on the ODE gradient stage (consistent with the sibling stage
knobs; identity-defining — it changes the scored trajectory, so it re-keys the
run). Requires a `condition_from` (the warm-up/scored split); a hard error names
the fix if `burnin_dt > dt` without one.

## Plan — trial-first

**Phase 0 (trial, this branch, ~a day).** The _minimal_ version: region-aware
`burnin_dt` in the ODE stepper only (`ode.rs` — swap the nominal step at
`cond_from`), **no `ForcingTimes`** (plain coarse steps, garki's measured ~1%).
Wire into `nuts`-on-`ode`. Validate on garki mv_age end-to-end:

- **speed:** per-gradient time + time-to-first-sample vs the current build;
- **correctness:** posterior parity vs a short-burn-in (or frozen-1961)
  reference.

If Phase 0 shows the fit-path speedup and acceptable bias → Phase 1. If the ~1%
smearing bias is too large → Phase 1's knot-truncation is the fix (and we know
it's needed). Either outcome is decision-useful and cheap.

**Phase 1 (general, if Phase 0 confirms).** `ForcingTimes::from_model` + thread
through the ~6 `Schedule` constructor sites; golden-neutrality verification; the
`burnin_dt` config surface; tests (coarse-vs-fine within tol, knot-snapping,
golden-neutral, end-to-end parity). This is the general, all-forcings version.

## Scope

- ODE gradient path (`nuts`) first; the same coarse-transient idea extends to
  `mh`-on-`ode` and (with hazard-correct discrete steps) the chain-binomial
  transient — named follow-ups, not v1.
- `cond_from`-keyed split (coarse warm-up / fine scored). Multi-region or
  per-stream coarse schedules are a follow-up.

## Resolved decisions

- **`burnin_dt` keyed to `cond_from`**, reusing the existing warm-up/scored
  boundary — no new anchor time.
- **`ForcingTimes` always-active** (Phase 1), relying on the `dt=1` no-op for
  golden-neutrality rather than a region-scoped boundary set (simpler, and the
  neutrality is verified, not assumed).
- **Trial before spine change** — Phase 0 validates the fit-path value on garki
  before touching the shared scheduling spine.

## Tests

Coarse-transient + fine-scored trajectory matches all-fine within a stated
tolerance; a step never straddles a forcing knot (knot-snapping);
`make
update-golden` is an empty diff (golden-neutrality); an end-to-end
`nuts`-on-`ode` fit with `burnin_dt` recovers the reference posterior at a
fraction of wall-clock.

## Follow-ups

`mh`/chain-binomial transient coarsening (hazard-correct for large steps);
adaptive `burnin_dt`; the equilibrium warm-start (PR #399) for sparsely-coupled
seasonal models genuinely at their cycle.
