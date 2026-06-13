# Revisit: how events and interventions differ (firing model)

Date: 2026-06-13 Project: camdl Tags: scheduling, interventions, events, spine,
gh#216

## Reminder / question

The gh#216 spine fix (`ae346eb5`) made **events** and **scheduled
interventions** fire on two different clocks, and we should come back to whether
that split is the right model or an artifact worth unifying / re-specifying.

Current state (verified against `ae346eb5`):

- **Always-active events** (`is_event`) fire on the **nominal `grid_dt`** every
  substep boundary, decided by `crate::effects::due_events` keyed on
  `round(t/grid_dt)` against `fire_steps`. They re-tile nothing.
- **Scheduled interventions** (`!is_event`) fire **cursor-keyed** off the
  registered `effect_times` timeline (the `ScheduledEffects{times,batches}` +
  `Schedule::substeps` cursor), so an off-grid observation that re-tiles the
  Exact substep grid does NOT move an on-grid intervention's firing instant (the
  bug gh#216 fixed). `effect_times` carries `!is_event` interventions only —
  registering event times would re-tile the Exact grid for events-only models
  and break byte-identity.

## Why revisit

The two constructs are sisters (`events {}` vs `interventions {}` in the DSL;
both apply discrete state modifications through the shared `step_one` effects
seam) yet they now fire on different keys. Open questions for a later pass:

- Is "events fire every substep on `grid_dt`, interventions fire cursor-keyed on
  their own timeline" a principled distinction (always-active vs scheduled), or
  an implementation artifact of how each was wired?
- Should events ALSO be cursor-keyed for consistency (and what would that cost
  in byte-identity / re-tiling for events-only models)?
- Does the multi-cadence union-axis work (per-stream reset) interact with either
  firing clock? (The union axis is the OBS cursor; effects are a separate cursor
  — but worth checking they compose cleanly when both are live.)

Not blocking any current work; park here and return after the multi-cadence
Phase 2 lands.
