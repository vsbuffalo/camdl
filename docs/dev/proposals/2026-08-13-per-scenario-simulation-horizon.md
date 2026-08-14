# Per-scenario simulation horizon (`scenarios { … simulate { to } }`)

- Date: 2026-08-13
- Status: implemented (all four increments; §10)
- Fixes: gh#561
- Builds on: `2026-08-11-scenario-banding-in-simulate.md` (landed; it names this
  work as the thing it unblocks)
- Related: gh#573 (scenario-level identity omits `scale`/`compose` — same class,
  independent fix), gh#576/gh#577 (the fabricated `baseline` member)

## Summary

A scenario may declare its own end time:

```camdl
scenarios {
  endemic { simulate { to = 3650 'days } }
}
```

The DSL accepts it, the spec documents it (§17.1), the expander resolves it, and
both IR types carry it. The Rust runtime never reads it, so every scenario runs
to the model-level `t_end` with no diagnostic (gh#561).

This proposal makes the field live: the declared horizon becomes the cell's
simulation window and enters that cell's run identity. It also settles the
design question the drop was papering over — a scenario is sometimes an _arm_ of
a comparison and sometimes an _entry in a menu_, and the horizon is legal for
one and not the other — and fixes two grammar defects that are harmless only
while the field is dead.

Scope: `ocaml/lib/compiler` (two parser fixes, one lint), `rust/crates/cli`
(threading, identity, two guards). **No IR schema change and no `ir/VERSION`
bump** — `Preset::t_end` already exists as `Option<f64>`
(`rust/crates/ir/src/model.rs:122`) and `SimConfig::t_end` is already a hashed
identity input (`rust/crates/runid/src/inputs.rs:152`). No golden IR churn.

## 1. Two readings of `scenarios {}`, and why only one admits a horizon

A preset is used for two different jobs, and camdl has never named the
difference:

- **Arms of a comparison.** `contrasts {}` differences two presets; a
  scenario-tagged `fit predict` stacks their predictive bands; "cases averted"
  subtracts one from the other. Differencing reductions taken over _unequal
  windows_ is meaningless — `final(I)` at 100 days minus `final(I)` at 3650 days
  is not a counterfactual contrast, it is an artifact of the windows.
- **A menu of related runs.** `simulate --scenario a --scenario b` runs each and
  labels the rows. The arms are not being differenced; each answers its own
  question. An epidemic arm that burns out in 100 days sitting beside an endemic
  arm that needs a decade of turnover to stabilise is a legitimate, well-posed
  pair — and it is exactly what `ocaml/golden/sir_demography.camdl` encodes
  today (`baseline` asks for 100 days, `endemic` for 3650, model horizon 365;
  all three run to 365).

**The rule this proposal adopts:** a per-scenario horizon is legal in the menu
reading and refused in the arms reading. That is why `contrasts.rs:254`
hardcodes `let run_end = model.simulation.t_end` — the arms reading is correct
to insist on one window — and why making the field live must add an explicit
refusal there rather than silently inheriting it (§6).

## 2. What a scenario may overlay: the prefix-safety line

The spec already restricts the block to `to`, and the parser enforces it
(`parser.mly:1507`, E106: "`dt` is not a per-scenario override"). The
restriction is right, and the reason is sharper than "`dt` is a model knob":

**A scenario may overlay anything that leaves the trajectory prefix intact.**

`to` qualifies. Extending or truncating the horizon never re-tiles
`[t_start, old_end]`: the RNG stream is consumed per substep in state order, and
`output_times` enumerates up to `t_end`, so a longer horizon only appends
snapshots. The prefix stays byte-identical, which is what preserves the
paired-seed property the run spec relies on (§3.1: the scenario index is
deliberately absent from the seed mix, "which is what makes their pre-divergence
trajectories byte-identical").

`dt` does not qualify. Two scenarios at different steps consume different
numbers of draws over the same interval, so their paths diverge from `t_start`
for purely numerical reasons; common random numbers are destroyed and any
between-arm difference becomes a mixture of the counterfactual and
discretization error. `integrator` is the same argument, plus the spec's
structural one: on the inference path the integrator is part of the model's
content identity.

`from` does not qualify either, and is currently _accepted and ignored_ (§3).
`init {}` is evaluated at `t_start`, so shifting it per scenario changes both
the initial condition's timing and the draw sequence — it breaks CRN pairing the
way `dt` does.

**Decision:** the accepted key set for a scenario's `simulate {}` block is
exactly `{ to }`, and `to` is required. `from`, `dt`, and `integrator` are all
E106, as is a block that omits `to` (§3.1).

## 3. Two grammar defects, inert only while the field is dead

Both must land before or with the threading.

**3.1 An absent `to` compiles to `0.0`, not `None`.** The scenario rule searches
its key-value list for the end-time key and, when there is none, falls back to a
literal zero instead of propagating absence:

```ocaml
(* parser.mly:1523 *)
let e = match List.find_map (function `To e -> Some e | _ -> None) kvs with
        | Some e -> e | None -> EConst 0.0 in
Ast.ScTEnd e
```

Verified against the compiler at `29445fd4`:

```camdl
simulate { from = 0.0  to = 100.0 }
scenarios {
  only_from { simulate { from = 10.0 } }
  empty_sim { simulate { } }
}
```

```
$ camdlc sc_fields.camdl | jq '.model.scenarios[] | {name, t_end}'
{"name": "only_from", "t_end": 0.0}
{"name": "empty_sim", "t_end": 0.0}
```

Honour that field and both scenarios run to `t = 0` — an empty trajectory, no
diagnostic.

**Decision:** an absent `to` is **E106**, not a defaulted value. Once `from`,
`dt`, and `integrator` are also rejected (§2, §3.2), `to` is the block's only
legal key, so a scenario's `simulate {}` that does not set it — including an
empty one — is a block that cannot mean anything. Erroring keeps the AST's
end-time constructor total: every parsed block carries a real end time, and
`Option` survives only where it belongs, on `Preset::t_end`, meaning "this
scenario declared no `simulate {}` block at all."

**3.2 `from` is accepted and silently discarded.** Only `Dt` and `Integrator`
are trapped; `from` parses, contributes nothing, and (via 3.1) drags `to` to
zero. This violates the no-loose-semantics rule directly.

**3.3 A `to` beyond an `at [...]` output list is a silent no-op.** With
`output { trajectories { at = [...] } }`, `output_times` emits exactly the
listed entries `≤ t_end` (`sim/src/output.rs:31`), and `quantities {}` reduces
over those snapshots (`quantity.rs:339`, `eval_series` maps `traj.snapshots`).
So a scenario's `to` is **inert** precisely when it selects the same entries the
model horizon already would.

**Decision:** emit **W106** at compile time when no listed time falls between
the model horizon and the scenario's `to` — in either direction, so a `to` that
truncates without dropping a listed time is caught too. Not an error: the run is
well-defined, it just doesn't do what the author wrote.

The predicate is deliberately NOT "is `to` past the last listed entry". With
`simulate { to = 80 }`, `at = [0, 30, 60, 90]` and a scenario `to = 200`, the
override selects the entry at 90 that 80 excludes — it changes the trajectory
and every reduction over it — so warning there, and telling the author to drop
the `to`, would silently change their answer.

## 4. Where it is dropped today

The compiler side is complete. `expander.ml:9921` sets `Ir.preset_t_end`;
`serde.ml:1547` round-trips it; `rust/crates/ir/src/model.rs:122` carries
`pub t_end: Option<f64>`.

The loss is at `main.rs:1811`, where a matched preset becomes a
`batch::ResolvedEntry`:

```rust
pub struct ResolvedEntry {          // batch.rs:305
    pub name: String,
    pub route: Option<String>,
    pub enable: Vec<String>,
    pub disable: Vec<String>,
    pub params: HashMap<String, f64>,
}
```

There is no horizon field, so the value has nowhere to go. `cell_resolve` then
hardcodes the base model's horizon into the identity context (`batch.rs:925`:
`t_end: self.base_model.simulation.t_end`).

## 5. The change

**5.1 Carry it.** Add `t_end: Option<f64>` to `ResolvedEntry`, populated from
the preset at `main.rs:1811` and from the named route in
`resolve_batch_scenarios` (`batch.rs:317`). Inline ad-hoc scenarios carry `None`
— a horizon is a model-file declaration, not a CLI patch, and there is no `--to`
flag (§9).

**5.2 Apply it.** `util::resolve_run_model` is the sole builder of a cell's
model and already receives `run.scenario_name`. When the named preset declares a
horizon, set `model.simulation.t_end` from it before `CompiledModel::new`.
Because `simulation.t_end` is the sole horizon authority since gh#143 and
`output_times(sched, t_end)` (`sim/src/output.rs:20`) is the single seam every
backend builds its schedule through, nothing else needs threading: output times,
boundary times, and the integrator loop all follow.

A `to` _shorter_ than the model horizon is the cell's window, full stop — fewer
rows for that scenario in the long-format TSV. Ragged output across scenarios is
honest and already representable; the wide TSV carries a `scenario` column and
`--quantities-out` keys per scenario with its own validated time axis
(`main.rs:1690`, landed with gh#562).

**5.3 Key it.** `cell_resolve` passes the cell's effective horizon instead of
the base model's. This is a one-value change to a field that is _already_ an
identity input — `SimConfig::t_end` (`inputs.rs:152`), documented there as
carrying the collapsed output horizon.

Config level is the right home, not the scenario level. Hashing is structural
via the derived `RunInput`/`ContentAddressed` composition (`hash.rs:298`), not
serde with `skip_serializing_if`, so adding a field to `ResolvedScenario` would
change its digest for **every value** and re-key every scenario'd run in every
existing store. Routing through the existing `SimConfig::t_end` costs nothing
and re-keys nothing: a preset with no `to`, or with a `to` equal to the model
horizon, resolves to the same number and the same hash it does today.

This is load-bearing, not bookkeeping. The purest form of the feature is a
horizon menu with no parameter changes at all:

```camdl
scenarios {
  short  { simulate { to = 90 'days } }
  medium { simulate { to = 365 'days } }
  long   { simulate { to = 3650 'days } }
}
```

Those three cells share a model, params, seed, and an identical scenario-level
digest — `resolve_scenario` hashes `enabled`/`disabled`/`patch`, all empty for
all three — and the scenario _label_ is provenance, not identity. Without 5.3
they collide on one `run_id` and the store serves one trajectory for three
different questions.

## 6. The arms guard

Making the field live must not let it leak into the differencing paths.

**6.1 `contrasts {}`.** Contrast arms are presets (`contrasts.rs:436`,
`resolve_arm` maps the run name to `scenario`), and both arms are replayed over
`run_end = model.simulation.t_end` (`contrasts.rs:254`). **Decision:** refuse at
`Arm::build` when two arms of one contrast have different effective horizons,
naming both. Refuse only on a genuine difference — a preset whose `to` equals
the model horizon is a no-op and must keep working, which is what the 12
redundant golden fixtures (§7) rely on.

**6.2 The data-windowed commands.** `fit predict`, `pfilter` and `profile` all
take their time axis from the DATA, not the model horizon, so a per-scenario
`simulate { to }` is inert for all three — the same silent drop this proposal
exists to remove, on three more paths.

- `fit predict`'s free-forward replay emits at `leaf_times`, the observed data's
  time column (`predict.rs:745` via `load_leaf_obs`).
- `pfilter` and `profile` score through the particle filter, whose schedule is
  built from `obs_model.obs_time(i)` (`particle_filter.rs:185`); `SMCConfig`
  carries a `t_start` and **no** `t_end`, and neither module reads
  `simulation.t_end` anywhere. `--save-paths` is written over that same
  observation-time axis, so it is a filtered path across the data window, not a
  projection past it.

**Decision:** all three hard-error when a named scenario's effective horizon
differs from the model's, through one shared `util::refuse_scenario_horizon` so
the wording and the "only a genuine difference" rule cannot drift between them.
Each names the reason its window is data-bound and points at
`camdl simulate --scenario`, which does run the scenario's own window. Extending
predict's forecast window past the last observation is a real and separate
defect, filed as its own issue (§9).

## 7. Fixture and golden impact

No golden `.ir.json` changes: the parser fixes affect only inputs that no golden
contains (no golden preset carries `t_end: 0.0`), and the schema is untouched.

13 golden `.camdl` files declare a preset horizon across 20 presets. In 12 of
them every value equals the model's own `simulation.t_end` — the March idiom of
restating the run horizon per UI button — so behaviour there is unchanged by
construction. `sir_demography` is the exception and the motivating case:
`baseline` 100, `endemic` 3650, model 365. After this change each scenario runs
its declared window, which is what the fixture has always claimed.

**Decision on `compose`.** `extends` inherits a parent's horizon (the expander
merges `rs_t_end`, `expander.ml:9638`); `compose = [...]` did not, so
`combined { compose = [endemic] }` silently ran to the model horizon while
`endemic` ran to its own. Latent while the field was dead — every scenario ran
to the model horizon — and live the moment it is honoured.

The horizon composes, walking the chain exactly as `resolve_preset_params` and
`composed_preset_scale` already do for `set` and `scale`: composed members in
list order, the parent's own value last and winning. A horizon is a preset field
like any other, and making it the one field that cannot compose would be its own
surprise. (Refusing a horizon-bearing `compose` member was the alternative; it
was rejected because the last-wins rule it objects to is the rule `set` has had
since the beginning, and no committed model uses `compose` at all, so neither
choice breaks anything today.)

**Decision on the trajectory baseline gate.** `load_and_apply_baseline`
(`rust/crates/sim/tests/gate_trajectory_baseline.rs:49`) applies
`presets.first()`'s _params_ but not its horizon — half a preset. Leave it
as-is: that gate exists to pin engine determinism against the model-level
horizon, and changing it would re-capture three `sir_demography` hashes
(gillespie/chain_binomial/ode) for a fixture-hygiene reason rather than a
correctness one. Add a comment there recording that the helper is deliberately
params-only, and cover scenario horizons with the dedicated tests in §8.

## 8. Tests

Red first, each failing for the stated reason before the corresponding change:

1. **Compiler.** A scenario `simulate { from = 10.0 }` and an empty
   `simulate { }` are both E106 — not a silent `t_end = 0.0` (§3.1, §3.2).
2. **Compiler.** `to` past the largest `at [...]` entry emits W106 (§3.3).
3. **Runtime.** The gh#561 fixture: `simulate { from = 0 to = 50 }` with
   `ctrl_50 { simulate { to = 120.0 } }` — the scenario cell's last `t` is 120.
4. **Runtime.** A shorter `to` truncates that scenario only; the other
   scenarios' rows are unaffected.
5. **CRN.** Two scenarios differing only in horizon are byte-identical over the
   shared prefix (the paired-seed property, §2).
6. **Identity.** The three-entry horizon menu of §5.3 resolves to three distinct
   `run_id`s (the collision test — red before 5.3).
7. **Identity, negative control.** A preset with no `to`, and one whose `to`
   equals the model horizon, both produce the `run_id` they produce today. No
   silent re-key.
8. **Guard.** A contrast whose two arms declare different horizons is refused;
   one whose arms declare the same horizon (or none) still runs.
9. **Guard.** `fit predict --scenario X` with a differing horizon on `X` is
   refused with the follow-up named.

Mutation-check each fix: revert the source change, keep the test, confirm red.

## 9. Deliberately not in scope

- **A CLI `--to` / horizon flag.** Rejected. The horizon belongs in the model
  file precisely so it travels with it: `simulation.t_end` is hashed into model
  identity, and `camdl simulate demog.camdl --scenario endemic` should reproduce
  that run from the file in git without anyone's shell history. A CLI override
  would be recorded in the run record (as `--dt` is) but would stop the model
  file from being self-describing, which is the property this proposal is
  arguing from.
- **`fit predict`'s forecast horizon.** Its observation-space bands stop at the
  last observed time regardless of the model horizon, so no scenario setting can
  extend them. Separate defect, separate issue, referenced by the §6.2 guard.
- **gh#573** — the scenario-level identity omitting `scale` and `compose`. Same
  class as §5.3 and already filed; independent, because this change routes
  through the config level and touches no scenario-level input.
- **Removing the surface.** Considered and rejected. The field's only historical
  consumer was the browser editor removed in `b2fc4f8f`, which made "orphan
  primitive, delete it" the tempting read — but what vanished was a _consumer_,
  not the concept. The surface is deliberate (it arrived with `scenarios {}`
  itself in `146fc556`), specified, and used by 13 fixtures; deleting a language
  feature because the runtime never plumbed it is fixing the wrong layer, and it
  leaves modellers with no correct spelling: post-hoc row truncation cannot
  recover a `quantities {}` reduction already taken over the wrong window, and
  the alternative is maintaining one model file per horizon.

## 10. Increments

Each lands green on its own.

1. **Grammar** — E106 for `from` and for an absent `to`; W106 lint. Tests 1–2.
2. **Thread + apply** — `ResolvedEntry.t_end`, `resolve_run_model` sets the
   cell's horizon. Tests 3–5.
3. **Identity** — `cell_resolve` passes the effective horizon. Tests 6–7.
   Increments 2 and 3 land together or 3 first; 2 without 3 is a live cache
   collision.
4. **Guards** — contrasts and `fit predict`. Tests 8–9.

### As implemented

Increments 2 and 3 landed as one commit, per the note above.

One deviation:

- **The `fit predict` test uses a contrast-free model.** With a contrast
  present, the §6.1 arm guard fires first and masks the §6.2 guard under test.

One refinement inside §3.1's decision: the missing-`to` error is suppressed when
a rejected key was already reported, so `simulate { from = 10 }` yields one
error naming the root cause instead of two.

### What a three-agent review changed

Three reviewers (correctness / test quality / DSL-and-docs) read the branch. The
substantive findings, all reproduced before acting:

- **§3.3's W106 predicate was wrong**, and its advice would have silently
  changed answers. "Past the last `at` entry" is not "inert" — §3.3 now carries
  the corrected predicate and the counter-example. The fixture was rewritten to
  a genuinely-inert model and gains a `clean_` sibling pinning the case the old
  predicate got backwards.
- **The observation time axis was left behind.** `ObsRegular.end` is baked from
  the model horizon at compile time (`expander.ml:7692`), so a shortened
  scenario kept emitting observations past the end of its own trajectory, where
  every reader clamps — fabricating rows on four paths (`simulate --obs`, the
  CAS `obs/` subtree, `[synthetic]` datasets, obs-sourced `quantities {}`). For
  a `prevalence` stream the tail is a frozen compartment wearing fresh
  observation noise, which reads as a plausible plateau. Emission now follows
  the run's horizon, exactly as `output_times` does for the trajectory axis; the
  observation axis is a second, independent axis and §5.2's "nothing else needs
  threading" was wrong about it.
- **§6.1's guard compared arms to each other rather than to the horizon they
  replay at**, so two arms declaring the same non-model horizon — the natural
  "fit window in `simulate {}`, projection window on the arms" pattern — passed
  and were silently replayed at the model horizon. Now each arm is compared
  against `model.simulation.t_end`, which also stops it over-refusing across
  independent contrasts.
- **An unknown key inside a scenario's `simulate {}` reported a spurious `dt`
  rejection first**, because the parser's recovery placeholder was
  indistinguishable from a real `dt`. It now has its own variant.
- **"What horizon does this scenario run to" had four independent answers** that
  agreed only by coincidence. They now route through one accessor,
  `params_resolver::composed_preset_t_end` / `effective_horizon`, per
  `.claude/rules/rust-conventions.md`. That also settled the `compose` question
  below.
- The §2 prefix test ran on the ODE backend, which consumes no random numbers —
  it could not have detected a paired-seed break. It now runs on
  `chain_binomial` and `gillespie`.
- **`pfilter` and `profile` accepted `--scenario` and ignored a declared
  horizon**, with no guard on either. Both were originally left as a follow-up;
  they are folded in here (§6.2) because "silently ignored on two commands" is
  the same defect this proposal is about, and the guard is three lines once the
  shared authority exists.

Verified: `make test` green end to end (exit 0, 227 suites, integration
included); no golden, IR, or `ir/VERSION` movement. The identity change was
mutation-checked in isolation — reverting only the `cell_resolve` line leaves
the window tests green and fails exactly the `run_id` test — and each guard
likewise turns exactly its own test red when disabled.

`ocaml/golden/sir_demography.camdl` now behaves as written: `baseline` runs to
its declared 100 days and `endemic` to its declared 3650, where both previously
took the model's 365. Per §7 the trajectory baseline gate is unchanged, so no
baseline hash moved.
