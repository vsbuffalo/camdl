# Implementation plan: phases and gates for the time + observation system

- **Status:** The single runbook coding agents work from to land the time +
  observation redesign. The
  [overview](2026-06-09-time-and-observation-overview.md) is the **what** (the
  system, the types, what exists); this is the **how/when** (phases, gates,
  dependency order, agent scoping). Each phase names the proposal section it
  implements from.
- **The three specs it sequences:** `2026-06-07-scheduling-spine-v2.md` (spine —
  the implementable spec; it supersedes the
  `2026-06-06-scheduling-effect-topology.md` design map, and most of it has
  already landed in code), `2026-06-06-observation-system.md` (data), and
  `2026-06-09-time-interval-model.md` (interval).
- **Hard rule for every phase:** TDD red→green with the failing test pasted in
  the commit; `make test` (the full gate, incl. integration) green before merge;
  goldens move only as a deliberate, human-reviewed act; no `ir/VERSION` /
  schema change without the atomic OCaml+Rust+golden update. Every backend ×
  inference cell must be supported-and-tested or hard-error — no silent third
  option.

## The dependency picture

```
P1   P0 bug fixes ───────────────── independent, unblocked NOW ────────┐
P1.5 TemporalKind enum + owning crate ── tiny standalone PR ────────────┤
P0   reconcile-with-main ── recurring, re-run at every phase boundary ──┤
                                                                        ▼
P3 DATA early (loader unify, bind/BoundObs, dense parity) ── follows P1.5
      │
P4 DATA correctness tier + conditioning window
      │   needs the per-observer reset — a LOCALIZED 6-call-site change
      │   P4 owns, NOT a wait on a multi-week spine lift
      │
P5 INTERVAL reconciliation + forecast ── reconcile w/ main's forcing v3
```

Read it top-down. The framing this plan first carried — "the spine is the long
pole everything gates on" — is wrong on two counts the review surfaced and the
code confirms:

1. **The spine v2 already exists and mostly landed.**
   `2026-06-07-scheduling-spine-v2.md` (commit `2db81ba`) is the implementable
   spec, supersedes the topology design map, and specifies concrete types
   (`StepClock`, `TimelineStop`, `EffectBatch`); the `Schedule` substrate +
   `StepPolicy{Snap,Exact}` are in the code today. There is **no "write the
   spine first" gate** to clear.
2. **The two obs-facing primitives are small, and the data track owns them.**
   `TemporalKind` is a one-line enum that does not exist yet (the _only_ hard
   compile-time dependency the data layer has on the spine) — land it as a tiny
   standalone PR (P1.5). The per-observer `ResetWindow` is a localized
   generalization of `reset_accumulators` (6 call sites, all in the inference
   projection path, none in the transition density; an in-tree canary comment
   already anticipates it) — P4 builds it directly. Neither is a multi-week
   spine lift.

So the real sequence is: P1 + P1.5 unblock now; P3 (data, no behaviour change)
follows P1.5; P4 (correctness tier) and P5 (interval/forecast) layer on top,
each owning its own small spine-adjacent change rather than waiting on a
foundation rewrite.

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

**Reconcile is recurring, not one-time.** `main` keeps moving while this
multi-phase lift proceeds; a single front-loaded fold goes stale by P4. Re-run
this overlap check — rebase onto current `main`, re-verify the table — as a gate
at **every phase boundary**, not just here.

## Phase 1 — P0 silent-wrong bug fixes (COMPLETE)

From the interval-model §5 / §9-P0. Loud-not-silent, TDD red→green, each its own
commit. **Landed** (`b0b044e` F1, `4106d6d` F4; full `make test` green):

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

## Phase 1.5 — `TemporalKind` (COMPLETE — `f2581dc`)

The name for the incidence-vs-prevalence distinction the data layer and the
reset / missing-data logic branch on. **Landed as a _derived_ classification,
not a stored field**, in the `ir` crate:

- `ir::observation::TemporalKind { Interval, Instant }` +
  `Projection::temporal_kind()` — `CumulativeFlow*` ⇒ `Interval` (incidence,
  resets on cadence); `CurrentPop*` / `DerivedExpr` ⇒ `Instant` (prevalence, no
  reset).
- `StreamProjection::temporal_kind()` delegates to the same classification, and
  `resets_after_observation()` now reads `== Interval` — behaviour identical
  (`FlowSum` is still the only resetting kind).
- **Design (refines the earlier "sim owns it"):** it is _derived_ from the
  projection, never a stored `StreamCells.kind` — a stored kind could only ever
  _disagree_ with the projection (an `Interval` stream on an `IntCompSum`
  projection is an illegal state you would then have to validate). And it lives
  in `ir` because it classifies `ir::observation::Projection`, which the data
  layer has at bind time _before_ runtime resolution. Pure-Rust method, no
  serde, no schema / golden change. `TemporalKind` is an obs/ir concept the
  spine _consumes_, not one the spine _defines_.

**Gate met:** all 5 IR projection variants + all 3 runtime variants pinned, the
`resets == (kind == Interval)` invariant pinned, full `make test` green (DRIFT
0). P3's typed `BoundObs` imports it.

## Phase 2 — the spine: already landed; remaining reshapes out of scope

The first framing — "build the spine, the long pole everything gates on" — was
wrong. `2026-06-07-scheduling-spine-v2.md` (commit `2db81ba`) is the
implementable spec, supersedes the topology design map, and most of it has
landed: `StepClock`, `TimelineStop`/`EffectBatch`, `StepPolicy{Snap,Exact}`, the
`Schedule` substrate, tau-leap removal (see `lifecycle-consolidation-todo.md`
for landed/dropped/deferred per step). The remaining spine reshapes
(`Target=Parameter`, the closure-driver question) are the **topology owners'
territory and are NOT a prerequisite for the observation work** — do not start
them as part of this lift.

What the obs layers need from this layer is exactly two things, both small and
both pulled out of "the spine": `TemporalKind` (P1.5, above) and the
per-observer reset (a localized change P4 owns — see P4). There is **no spine
gate** to clear before P3 / P4 / P5.

The cross-backend lifecycle property (byte-identical dense behaviour; the
gh#187-class "every cell, _including PGAS_" coverage) is a standing test
obligation, not a P2 deliverable — it lives wherever the behaviour it guards
does (P4 for the reset, the existing suite for the landed spine).

## Phase 3 — data layer, early (partly parallel with P2)

From `observation-system.md` migration steps 1–2. No behavior change (for
well-formed inputs).

- **DONE — item 1 (`1a20bcf`):** deleted the positional value-column fallbacks
  (the outer `.or_else(load_data_tsv)` and the inner 2-column fallback) in
  `pfilter.rs`, and the same fallback in `profile.rs`/`survey.rs`; bind strictly
  by name with a located error; added the NaN/finiteness guard. (Exposed and
  fixed a latent instance of the same G1 bug in the `he2010_pfilter_loglik`
  fixture — model stream `weekly_cases` vs data column `cases`.) Malformed
  inputs now error; well-formed unchanged; goldens did not move.
- **DONE — item 2 (`6d2eed8`):** `BoundObs` validated-ctor seam.
  `MultiStreamObsModel::new` now consumes a `BoundObs`; the four construction
  checks (empty streams, empty/non-increasing times, heterogeneous schedules)
  moved into `BoundObs::bind`, which returns
  `Result<(BoundObs, BindReport),
  BindReport>` (verdict DERIVED, not stored).
  `StreamSpec` stays public as bind's input; all ~17 caller sites (4 prod + 13
  tests) migrated. Dense (`values: Vec<f64>`), reproducing today exactly —
  goldens did not move.

P3 (the no-behavior-change loader + bind seam) is **complete**. What is
deliberately NOT here, deferred to P4 (the correctness tier): typed
`Option<ObsCell>` cells (holes), `Counted{value,denom}`, the per-stream
`ResetWindow`, the present-cell **union axis**, and the dense-vs-**sparse**
storage decision (sparse per-stream cells are required for national/polio scale
— confirmed — but the logical model stays union-axis-with-holes; the physical
representation change belongs with the correctness tier, not the no-op seam).

**Gate met:** goldens did not move (dense parity); `TemporalKind` exists (P1.5).

## Phase 4 — data correctness tier + conditioning window

From `observation-system.md` step 3 + `time-interval-model.md` §7.2. **Not
blocked on the spine.**

**DONE — hole-scoring slice (`188a9b0`):** `ObsCell{Scalar}` + per-observation
`Vec<Option<ObsCell>>` (None = hole); the scoring seam skips a `None` (omits the
factor — marginalization), the loader reads `NA` as a hole keeping its time in
the grid, and the existing per-obs-index reset still fires at holes (a missing
week closes its bin). Hole ≠ observed-zero, in the type. A review-found leak
(the dense placeholder reaching `--save-prequential`/`--trace`) is hard-errored
(`check_holes_output_compat`). **Verified camdl-vs-pomp on a sparse/holes
He-2010 series: -4726.55 vs oracle -4723.42 (3.13 nats, inside the 35-nat
band)** — the oracle is in `tests/external/sparse_oracle_wip/` (untracked, NA→1
dmeasure + weekly accumvar reset, independently reproduced). Dense parity intact
(goldens DRIFT 0). For a **single** stream the per-obs-index reset is already
correct (its cadence is the grid) — so this slice greens the He-2010 gate
WITHOUT the per-observer reset below.

**DONE since:**

- **Formal sparse gate wired (`b06e71d`).**
  `tests/external/cases/he2010_pfilter_loglik_sparse/` runs in every `make test`
  (5-case `run_all`; camdl −4726.55 vs pomp fixture −4716.65, inside the 35-nat
  band). The −4723-class invariant is now permanent.
- **Holes through `camdl fit` (`4f1a509`).** IF2/PGAS/PMMH/ODE-MLE now accept
  `NA` (the loader was the only blocker; the algorithms were already
  hole-correct). Proven: fit PFilter-stage loglik == standalone `pfilter` on the
  same holed data, exactly. Added the `ic_free` + missing-first-obs guard
  (hard-error: no y₁ to condition on). `profile`/`survey` stay dense-only
  (separate paths; reject `NA` loudly) — follow-up if needed.

**NEXT in P4 (remaining):**

- **Per-observer / per-cadence reset** (needed only for MULTI-stream different
  cadences — polio ES+AFP — NOT the He-2010 gate): the localized generalization
  of the reset (`state.reset_flows()` at `particle_filter.rs:415`,
  `if2.rs:319,559`, `correlated_pf.rs:521`; PGAS `cum_flows` at
  `pgas.rs:843,1250`) to "reset only the flows whose bin closes at this
  obs_idx", obs-model-driven via `reset_due_flows(state, obs_idx)`. Dense ⇒
  resets all ⇒ parity. Its own multi-cadence test. The reviewer owns this seam
  (inference math).

- relax the shared-grid assertions to the **present-cell union axis** (all-hole
  times excluded by construction);
- **fixed-bin incidence** (pomp-grounded): reset on the stream's cadence, a
  missing value suppresses only the score; per-observer accumulators;
- **accumulator scaling — confirm, don't assume.** The per-observer accumulators
  are a _separate_ allocation from the (confirmed-sparse) cells:
  `O(observers × flows_per_observer × n_particles)`. Only _present_ observers
  allocate (not a dense N×F grid) — state that invariant explicitly and confirm
  the footprint is bounded at the 774-LGA × multi-stream cVDPV2 target. "Cells
  are sparse" does **not** by itself bound the accumulators.
- the **conditioning window C** = `[cond_from, cond_to]` as a reset at a
  scheduled-but-unscored boundary (= pomp's fictitious-NA-observation idiom);
  define the `C` that reproduces today's `ic_free=true` bit-for-bit on IF2/PF,
  or declare the break; **`ic_free` stays as its own orthogonal flag**;
- `Counted{value,denom}` k-of-n binding with `value ≤ denom` checked at the
  binder.

**Gate:** FD/likelihood parity on the dense case (goldens don't move); **a
camdl-vs-pomp likelihood-agreement test on a _sparse/irregular_ series** —
extend the existing `tests/external/cases/he2010_pfilter_loglik/` harness (it
already compares `camdl pfilter` to pomp's `pfilter()` on the He et al. 2010
London measles series to a tolerance band) with a thinned, ragged,
multi-cadence, holes-included variant and assert the log-likelihood still agrees
(pomp scores NA rows with `dmeasure = 1`). This converts the §8 pomp-equivalence
claim from a provenance table into a _tested_ invariant on the new sparse path —
which the goldens-don't-move regression guards structurally cannot do (no golden
exists for a new feature). Plus: the internal sparse-interval per-observer-reset
correctness test; the conditioning window gated per cell (re-uses P1's F1
dispatch gate); correlated-PMMH's identical-substep-count constraint honored or
hard-errored.

## Phase 5 — interval reconciliation + forecast (reconcile w/ main's forcing v3)

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
  accumulator. (Typed-time: fix the two concrete `f64` bugs **as single
  chokepoints, not scattered inline patches** — one named tolerance constant
  (replacing the scattered `1e-9`/`1e-10`/exact comparisons) and one checked
  `time→step` function (replacing the release-only saturating negative-cast) —
  so the eventual newtype migration has a single seam to convert. Do not newtype
  the codebase now.)
- A future `DifferentiableObjective` trait (value+gradient) for ODE→NUTS is a
  later lift; do not foreclose it (keep scoring behind a clean seam), do not
  build it now.
