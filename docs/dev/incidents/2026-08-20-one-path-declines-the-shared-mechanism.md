# Three silent wrong answers in one day, all the same shape: a correct shared mechanism that one path declines to call

Date: 2026-08-20 Issues: gh#607 (fixed 5b04dba4), gh#680 (fixed), gh#681
(guarded) Severity: incorrect inference on shipped backend × method cells; none
user-detectable

## Why this is one incident and not three

Three defects surfaced within a day, in three different subsystems, found by
three different routes. They share a structure precise enough to be a search
pattern:

> A mechanism is written once, correctly, and used by one path. A sibling path
> needs the same behaviour, does not call it, and reimplements or omits it. Both
> paths keep producing plausible numbers.

|        | the shared mechanism                                                       | the path that calls it                                 | the path that does not                                                               |
| ------ | -------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| gh#607 | continuity of the spliced trajectory                                       | `complete_data_loglik`'s support                       | the CSMC-AS traceback, which recorded the reference's own pre-state                  |
| gh#680 | `fold_into_acc_real` / `reset_due_acc_real` — per-stream incidence binning | the ODE **value** path (`ode_loglik.rs:98,113`)        | the ODE **gradient** path, which blanket-zeroed at every union index (`ode.rs:1181`) |
| gh#681 | threading real compartment state into evaluation                           | the ODE **gradient** path (`multi_stream_obs.rs:1402`) | the ODE **value** path, which passes a permanently-zero buffer (`:868`)              |

gh#680 and gh#681 are exact mirrors of each other. Whichever path is right, the
other one is wrong, and no one noticed because the two are never compared.

## What each cost

**gh#680.** Two incidence streams at 7 d and 14 d, truth `mu = 0.05`, posterior
sd ~0.0008. `nuts` returned `mu_a` 0.0414 / `mu_b` 0.0674 where `mh` on the same
model returned 0.0482 / 0.0522 — **~170 sd**, reproducing across seeds and
scaling monotonically with the cadence ratio.

**gh#681.** `projected = I + W` with `W` a real reservoir. Every emitted draw
tracked `I` alone; `W` reached twice `I`'s size and contributed exactly zero.
Poisson(896) returning 314.

**gh#607.** Measured separately: 510 `-inf` draws in a retained posterior, 218
posterior trajectories silently missing from output, R̂(β) 10.55 → 2.72 after the
fix.

## Why the tests did not catch them

Each was tested — for the _other_ cell.

- gh#680: multi-cadence is tested for chain-binomial
  (`sim/tests/per_stream_reset.rs:227` carries a mutation guard for this exact
  hazard — it asserts each AFP bin tallies its full 30-day window, 300, where a
  blanket reset would give the 2-day remainder, 20) and not for `ode × nuts`.
- gh#681: no file under `sim/tests/` has an observation referencing a real
  compartment. The two real-compartment goldens carry no `observations` block.
- gh#607: ancestor sampling was tested for its WEIGHT (`pgas_ancestor_weight.rs`
  pins Eq. 17) and never for the OBJECT it returns.

The gap is not coverage of a feature. It is coverage of a **feature × cell**
combination — the cross product the dense-matrix rule in
`.claude/rules/sim-and-inference.md` already requires and nothing mechanically
enforces.

## The oracle that works, and the two that do not

**Parameter recovery does not work here, and gh#681 proves it.** The forward
emitter (`cli/src/main.rs:3161`) and the value-path scorer
(`multi_stream_obs.rs:868`) pass the _same_ zero real state. Simulate from known
θ, fit, and θ recovers cleanly — data generated with `W` dropped, scored with
`W` dropped. **A recovery test is green while the model is wrong.** Any oracle
whose data generator shares the defect is blind to it.

**Finite-differencing the gradient does not work either**, which was the
intuition and it is wrong. The existing FD oracles difference `det_grad`'s _own_
value. A mis-binned value is still perfectly smooth in θ, with its forward
sensitivity as its exact derivative — the bug is self-consistent under FD.

**What works is a cross-path differential.** Two paths that are supposed to
compute the same quantity are compared directly:

```
ll_grad = det_grad(...).0          // the nuts path
ll_value = compute_ode_loglik(...) // the mh path
assert!((ll_grad - ll_value).abs() < 1e-6);
```

Against the pre-fix code this reports `-2920.81` vs `-1242.30`. It needs no
statistics, no tolerance judgement, and no second implementation — the sibling
path _is_ the oracle. It is only available where two paths genuinely target one
quantity, which is exactly the situation this bug class arises in.

## What changed

- gh#680: the ODE gradient path now owns `acc` / `acc_sens` and folds and resets
  through the same seam as the value path. `fold_into_acc_real` became the
  `width == 1` case of a block-strided sibling and delegates to it, so a
  sensitivity bin cannot diverge from the value bin it differentiates — the two
  share one slot map and one due-predicate by construction rather than by
  agreement.
- gh#681: observation-side references to real compartments are refused outright.
  The ODE `REAL_COMPARTMENTS` grant is deliberate and stays; only scoring
  against a real compartment is refused, because the seam cannot carry it.
- A stale claim removed from `per_stream_reset.rs`'s header ("the CLI still
  loud-rejects heterogeneous cadences end-to-end" — it does not). Leaving a
  false safety claim in the canonical multi-cadence test file invites exactly
  the reasoning that let gh#680 sit.

## What should change next

**The half-applied lesson is the tell.** Immediately above the site where gh#681
reads its zero real state, a comment describes this _same hazard_ being found
and fixed for integer state: _"not a zero scratch — the zero scratch silently
turned PopSum-valued denominators into 0 → NaN."_ The fix was applied to one
half. The other half sat beside it. Whenever a comment records a lesson, the
question "where else does this apply" is part of the fix, not a follow-up.

**Prefer deleting the fork to fixing its symptom.** There is one `score_streams`
(`multi_stream_obs.rs:1223`), called by three value-path scorers (`:1280`,
`:1296`, `:1311`) — and its per-stream walk is _re-implemented_ by the ODE
gradient path and by `pgas_grad`, neither of which call it. The gradient
re-implementation matches on `s.projection` directly and never touches
`stream_to_slot`, which made the per-stream seam _unreachable_ from that path —
which is why `ode.rs` had to invent its own reset. gh#680's fix closes the
accumulator seam so the two cannot diverge numerically, but the walk is still
written three times, and that fork is the next instance waiting to happen.

**A helper with one caller is a signal.** `grep`ping for
`fold_into_acc_real|reset_due_acc_real` returning exactly one call site each is
what located gh#680's mechanism. `resets_after_observation`
(`multi_stream_obs.rs:212`) currently has zero production callers — every
reference to it is inside the test module.

**Enforce the matrix mechanically.** The dense-matrix rule is a convention
followed by hand. A test that enumerates backend × method cells and fails when a
cell has no coverage would convert three silent failures into three reds.

**A second manifestation of gh#680, found only on review.** `--condition-from`
inserts a per-stream leading reset-only hole (`cli/src/fit/runner.rs:1391-1397`)
whose time joins the canonical union axis. Pre-fix, the recorder's blanket zero
closed EVERY stream's tally at that time; post-fix only the conditioned stream's
bin closes. So conditioning silently truncated the other streams' first bins on
the gradient path, and nobody had connected that to this bug. The same fix
repairs it.

## Follow-ups

- gh#682: `CompGradMap` is a `HashMap`, and `resolve_comp_grad_map` iterates it
  to fix a summation order — so a seeded `nuts` × `ode` run is not
  bit-reproducible across processes (~1e-15, but NUTS's U-turn and divergence
  tests are threshold comparisons).
- Prevalence rounding: `multi_stream_obs.rs:1428` projects `IntCompSum` from
  unrounded `f64` counts while `ode_loglik.rs:100` passes rounded `i64`, so `mh`
  and `nuts` optimise slightly different objectives on prevalence data.
  Documented as smoothness; the consequence was never stated, and it means the
  cross-path oracle above must be scoped to incidence until it is decided.
