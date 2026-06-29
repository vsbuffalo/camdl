# Scenario-aware `fit predict` (Layer 1: prospective scenario overlay)

Status: proposed Relates-to:
`docs/dev/proposals/2026-06-25-counterfactual-contrasts.md` (this promotes that
proposal's deferred free-forward/prospective path to a built feature, scoped
overlay-only) Relates-to:
`docs/dev/proposals/2026-06-22-predictive-ergonomics.md` (the `fit predict` verb
and its horizon/treatment column stacking)

## Problem

`fit predict` replays a fit's posterior forward and bands the result, stacking
two axes into one tidy file — the **horizon** and the **parameter treatment** —
as columns. It does **not** vary the model's scenario: it builds a single
hardcoded baseline (`predict.rs`: _"a single no-op baseline scenario, matching
simulate's no-`--scenario` path"_). To get a counterfactual's predictive bands
(e.g. with-SIA vs no-SIA) a user must run a separate `simulate --draws` per
scenario and join by hand.

Downstream wants what `simulate` already has: repeatable `--scenario` (with
`--enable`/`--disable`), the forward replay looped over scenarios, and a
`scenario` column in the outputs — so **one `fit predict` produces every
scenario's bands**, stacked the same way horizon and treatment already are.

## Why this is small: the seam already exists

- The engine **already loops over `Vec<ScenarioRef>`** (`engine.rs`:
  `for scenario in scenarios.iter()`), and `ScenarioRef` +
  `resolve_scenario_ref` are **shared with `simulate --scenario`**
  (`sim_job.rs`).
- `fit predict` already passes a `scenarios` Vec into `SimulateJob` — it is just
  pinned to `[baseline]`.
- `simulate` already parses repeatable `--scenario`/`--enable`/`--disable` into
  that type.

So Layer 1 is wiring the user's scenarios into a loop that already runs, plus a
new output column — not a new mechanism. (Reach for the existing seam.)

## Scope (Layer 1 = overlay only)

In: each requested scenario's **predictive bands and quantity bands**, in one
file, tagged by a `scenario` column/field.

Out (deferred to the counterfactual-contrasts proposal): any **difference**
between arms ("cases averted"), and the **retrospective/conditioned** fork. This
boundary is load-bearing — see Guard 1.

## Design

### CLI — reuse simulate's scenario parsing

`FitPredictArgs` gains the same repeatable scenario surface `simulate` has:

```
camdl fit predict --fit fit.toml \
    --scenario no_sia --disable sia \
    --scenario with_sia --enable sia
```

`--scenario NAME` selects a `scenarios {}` preset;
`--enable`/`--disable`/`--set` form inline overlays — all parsed into
`Vec<ScenarioRef>` by the **existing** `simulate` arg path, not a new parser. No
`--scenario` given → `[baseline]` (today's behaviour, unchanged).

### Engine — pass the parsed scenarios through

Replace the hardcoded `scenarios: vec![baseline]` in the predict job with the
parsed `Vec<ScenarioRef>`. The engine's existing scenario loop produces per-draw
outputs per scenario; the predict **sink** is made scenario-aware (it currently
assumes one scenario) so it partitions `quant_draws` / predictive draws by
scenario name.

### Output schema — a `scenario` column, stacked like horizon/treatment

`predictive/<stream>.tsv` and `quantities/<name>.tsv` gain a leading `scenario`
column; `quantities.json` gains a `scenario` field per entry.

```
predictive/<stream>.tsv:
  scenario | time | <dims...> | horizon | treatment | rhat_max | q05..q95
```

**Decision (resolved): the `scenario` column is ALWAYS present**, value `fitted`
when no `--scenario` is given — exactly as `horizon`/`treatment` are always
present. Rationale: tidy-data stability (a consumer's join/group key doesn't
appear/disappear with arity) and consistency with the sibling axes. This is a
one-time predictive/quantities output-format change (alpha; goldens for these
artifacts updated in the same commit).

**Why `fitted`, not `baseline`.** `baseline` is the `simulate` convention for
the default-parameter scenario; in `fit predict` the parameters come from the
**fit**, not a preset, so reusing `baseline` would mislead. `fitted` reads as
"the fitted model, no overlay" and sits in parallel with the overlay names
(`fitted` / `no_sia` / `with_sia`). The fit's own identity is **not** put in
this column — it is already captured losslessly in the output path
(`results/fits/<run-id>/`) and `run.json`; this column is purely the _overlay_
axis (the fit is common to every row, overlaid or not).

**`fitted` is a reserved scenario name.** A `scenarios {}` entry named `fitted`
is rejected with a clear diagnostic (it would collide with the no-overlay row's
reserved value, making rows ambiguous). The diagnostic names the reservation and
the fix (rename the scenario).

### Paired-seed CRN (free property)

`fit predict` replays under a single seed; running the scenarios under that same
seed makes their pre-divergence trajectories **coupled** (paired-seed CRN —
`ARCHITECTURE`/RNG section), so cross-scenario comparison at fixed draw is
low-variance. No work required; worth documenting for the user.

## Guards (the silent-wrong fences)

1. **Overlay, never "averted."** Layer 1 emits per-scenario forecast bands only.
   It must not compute or label a between-scenario difference as "cases averted"
   — a forward-only prospective difference read as the realized averted count is
   a silent-wrong on a policy headline (the counterfactual-contrasts proposal's
   core reason for deferring the prospective contrast). Differences stay in that
   proposal's conditioned fork.
2. **`set`/`scale`-param scenarios first; intervention-toggling gated.**
   Scenarios that only `set`/`scale` parameters replay cleanly. Scenarios that
   `enable`/`disable` an intervention need schedule/cursor re-seating (gh#216) —
   higher-risk; Layer 1 routes those through the existing capability/validation
   path (loud error if not yet supported) rather than silently mis-replaying.

## Decisions recorded

- `scenario` column ALWAYS present (no-overlay value `fitted`, a reserved
  scenario name). NOT `baseline` (a `simulate` convention) and NOT the run-id
  (already captured in the path + `run.json`; this column is the overlay axis).
  (§Output schema.)
- Overlay only; no difference/averted surface in Layer 1. (Guard 1.)
- Reuse `ScenarioRef` + simulate's parser; no new scenario surface. (§CLI.)
- Param-overlay scenarios are the supported set; intervention-toggling is
  gated/loud-errored, not silently replayed. (Guard 2.)

## Test plan

- CLI: repeated `--scenario` parses to the expected `Vec<ScenarioRef>` on
  `fit predict` (mirrors the simulate parser test).
- Output: a two-scenario `fit predict` writes both scenarios' rows to one
  `predictive/<stream>.tsv` and one `quantities/<name>.tsv`, each tagged; the
  manifest carries the `scenario` field; the no-`--scenario` path still emits a
  single `fitted`-tagged file (byte-compatible except the new column).
- Reserved name: a `scenarios {}` entry named `fitted` is a clear compile-time
  error naming the reservation and the fix.
- Coupling: two scenarios at the same seed share pre-divergence draws (CRN).
- Guard: an intervention-toggling scenario without schedule re-seating support
  is a loud error, not a silent baseline replay.
