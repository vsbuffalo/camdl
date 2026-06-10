# Implementation plan: phases and gates for the time + observation system

- **Status:** The single runbook coding agents work from to land the time +
  observation redesign. The
  [overview](2026-06-09-time-and-observation-overview.md) is the **what** (the
  system, the types, what exists); this is the **how/when** (phases, gates,
  dependency order, agent scoping). Each phase names the proposal section it
  implements from.
- **The three specs it sequences:** `2026-06-06-scheduling-effect-topology.md`
  (spine), `2026-06-06-observation-system.md` (data), and
  `2026-06-09-time-interval-model.md` (interval).
- **Hard rule for every phase:** TDD red→green with the failing test pasted in
  the commit; `make test` (the full gate, incl. integration) green before merge;
  goldens move only as a deliberate, human-reviewed act; no `ir/VERSION` /
  schema change without the atomic OCaml+Rust+golden update. Every backend ×
  inference cell must be supported-and-tested or hard-error — no silent third
  option.

## The dependency picture

```
P1 (P0 bug fixes)  ── independent, unblocked NOW ──────────────┐
                                                               │
P0 reconcile-with-main ── do alongside P1 ─────────────────────┤
                                                               ▼
P2 SPINE  (TemporalKind, per-observer ResetWindow, Stage)  ── the long pole
      │  unblocks ▼
P3 DATA early (loader unify, bind/BoundObs, dense parity) ── partly parallel w/ P2
      │
P4 DATA correctness tier + conditioning window  ── GATED on P2's ResetWindow
      │
P5 INTERVAL reconciliation + forecast  ── GATED on P2; reconcile w/ main's forcing v3
```

Read it bottom-up: the **spine is the foundation** and the long pole — its
per-observer `ResetWindow` + `TemporalKind` are 0%-built and the correctness
tiers of both other layers wait on them. The P0 bug fixes and the
main-reconciliation are independent and start now.

## Phase 0 — reconcile with main (do alongside P1)

`main` moved ~20 commits while this design was written, and several **overlap**.
Before building, an agent must check what each already does and fold/align:

| main commit                                                                                       | overlaps                                                             | action                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `852827d` const‖parametric forcing v3 + `2455c27` plan (gh#119) + `b34ca3d` frozen-param incident | F2 forcing-domain OOB; the gh#186 TimeFunc-frozen-param finding      | **the forcing-domain work (P5) folds into / extends forcing v3 — do not write a competing forcing design.** Confirm whether gh#186 is already addressed there. |
| `0b63599` + `27693cb` latent-trajectory consolidation                                             | `ForecastOrigin` / forecast-from-trajectories                        | align the forecast design (P5) with the existing latent-trajectory output; reuse, don't reinvent.                                                              |
| `389e339` capabilities-system.md (three compatibility axes)                                       | the "every cell supported or hard-error" gate; the conditioning gate | cite it; route the conditioning gate (P4) through that axis framing.                                                                                           |
| `c11f0a7` method caveat banner from the registry                                                  | F1's false "conditioning on y₁" banner                               | the P0 agent verifies whether this already fixed the banner before fixing F1.                                                                                  |
| `7d92a4d` gh#191 real-compartment support across the inference stack                              | the REAL_COMPARTMENTS capability gap                                 | confirm it closed; touched the inference stack — re-verify F1/F4 against it.                                                                                   |

Phase 0 does **not** require standalone incident docs for F1/F4: they are
well-scoped fixes whose reproduction + red→green proof live in their fix-commit
messages (`b0b044e`, `4106d6d`). An incident doc is for a serious bug warranting
an engineering response beyond the fix — not every silent-wrong finding clears
that bar.

## Phase 1 — P0 silent-wrong bug fixes (unblocked, in progress)

From the interval-model §5 / §9-P0. Loud-not-silent, TDD red→green, each its own
commit. **In progress** (agent running):

- **F1** — `ic_free`/conditioning silently ignored on PGAS / ODE-MLE /
  correlated-PF → hard-error at the `(algorithm)` dispatch gate (`methods.rs`),
  not the IR capability flags. (Honoring it on ODE is deferred design,
  P5-adjacent.)
- **F4** — an observation before `t_start` is loaded then scored unpropagated →
  reject at load with a located error; promote the release-only `debug_assert!`.
- **F2 (forcing flat-extrapolation) is NOT in P1** — its loud-error needs the
  per-kind policy (error for `interpolated`/`CubicSpline`, `constant`-outside
  for step kinds) and must reconcile with forcing v3; it lands in P5.

**Gate:** `make test` green; red+green pasted in each fix commit.

## Phase 2 — the spine (the foundation; the long pole)

From `scheduling-effect-topology.md` + its `lifecycle-consolidation-todo.md`.
Builds `TemporalKind{Interval,Instant}`, the per-**observer** accumulator + its
reset (`Stage::Reset`), `StepPolicy{Snap,Exact}` threading, the sub-`dt`
collision guard.

**Gates before starting P2:**

1. **Spec readiness.** The topology doc is a "design map" with a v2 promised
   "when the timeline-tightening tier begins." Confirm it is
   implementation-ready, or write the v2 first. _Agents implement from a spec,
   not a map._
2. **`TemporalKind` ownership decision.** The spine defines it; the data layer
   imports it. This must be recorded before P3's typed step can compile.
3. **Per-observer, not per-flow.** The accumulator/reset is keyed
   per-`(observer,
   flow)` (two streams can read one flow at different
   cadences). Scope note: it touches only the observation projection, not the
   transition density.

**Gate to finish:** the existing dense/homogeneous behavior is byte-identical
(goldens don't move); the cross-backend lifecycle property is tested _including_
PGAS (the gh#187-class gap).

## Phase 3 — data layer, early (partly parallel with P2)

From `observation-system.md` migration steps 1–2. No behavior change:

- delete the positional value-column fallbacks (`pfilter.rs:147-149`,
  `:671-675`); bind by name only;
- the NaN/finiteness guard (`pfilter.rs:692,722`);
- `bind` / `BoundObs` reproducing today's dense semantics, routing the ~5
  scattered load sites through it; **`MultiStreamObsModel::new` consumes a
  `BoundObs`** and the raw `StreamSpec` path is privatized (so the
  validated-ctor invariant is real); `bind` returns `Result` (no `BoundObs` on a
  fatal error); derived severity + structured findings.

**Decision needed in P3:** dense vs **sparse** `BoundObs` storage. Sparse
(per-stream present cells; union axis as a derived view) is required for
national/polio scale — confirmed. Logical model stays union-axis-with-holes;
physical storage is sparse.

**Gate:** goldens don't move (dense parity); `TemporalKind` exists (P2 dep for
the typed `BoundObs`).

## Phase 4 — data correctness tier + conditioning window (GATED on P2)

From `observation-system.md` step 3 + `time-interval-model.md` §7.2. **Blocked
until the spine's per-observer `ResetWindow` ships.**

- relax the shared-grid assertions to the **present-cell union axis** (all-hole
  times excluded by construction);
- **fixed-bin incidence** (pomp-grounded): reset on the stream's cadence, a
  missing value suppresses only the score; per-observer accumulators;
- the **conditioning window C** = `[cond_from, cond_to]` as a reset at a
  scheduled-but-unscored boundary (= pomp's fictitious-NA-observation idiom);
  define the `C` that reproduces today's `ic_free=true` bit-for-bit on IF2/PF,
  or declare the break; **`ic_free` stays as its own orthogonal flag**;
- `Counted{value,denom}` k-of-n binding with `value ≤ denom` checked at the
  binder.

**Gate:** FD/likelihood parity on the dense case (goldens don't move); the
sparse-interval per-observer-reset correctness test; the conditioning window
gated per cell (re-uses P1's F1 dispatch gate); correlated-PMMH's
identical-substep-count constraint honored or hard-errored.

## Phase 5 — interval reconciliation + forecast (GATED on P2; reconcile w/ main)

From `time-interval-model.md` §7 + §9 P1/P3.

- **`RunWindows`** authority + **route every
  `simulation.t_end`/`output.times.end` read through it and delete the direct
  reads** (`util.rs:2029`, the three backend configs) — only then is
  "unconstructible" real. Reconcile the three "end" fields (F3/F5; resolves
  gh#143).
- **Forcing-domain OOB (F2)** — fold into / extend main's forcing v3 (gh#119);
  the per-kind policy. Note SIAs are interventions, not forcings.
- **Forecast as an operation** — `ForecastOrigin` keyed by artifact capability
  (align with main's latent-trajectory consolidation), interval-scoped CAS
  hashing for forecast covariates (with the global-spline caveat), forecast-`D`
  per fit method.

**Gate:** extending a forecast horizon does **not** re-key the fit; forecasting
past a covariate domain hard-errors with a named message.

## Declined / deferred (so agents don't pick them up)

- The external review's four-layer
  `RunPlan`/`TemporalPlan`/`BackendExecutionPlan` stack,
  `ObservationBoundaryKind`, the `ModelTime`/`CalendarTime` newtype hierarchy,
  and the `TemporalEvent` enum — **declined** (see the overview §9). Use the
  single `RunWindows`, the existing `Schedule`×`Stage`, and the per-observer
  accumulator. (Typed-time: fix the two concrete `f64` bugs — the release-only
  negative-cast and the scattered tolerances — do not newtype the codebase.)
- A future `DifferentiableObjective` trait (value+gradient) for ODE→NUTS is a
  later lift; do not foreclose it (keep scoring behind a clean seam), do not
  build it now.
