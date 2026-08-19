# DSL observation anchors: `last_obs` (± offset) in `to`, `breakpoints`, `value_at`

Status: proposed (revised after a 3-agent adversarial review; the first draft's
placeholder and run-identity claims were refuted and are replaced) Issue: gh#616
(the DSL half of F15; gh#626 shipped the CLI half) Scope: OCaml expander + IR
schema (ir/VERSION 0.31 → 0.32, atomic golden update) + Rust runtime resolution.

## Problem

gh#626 gave the CLI `--to "last_obs + 8 weeks"`, removing the horizon retype.
The model file still hand-types data-dependent literals: the forcing fork
(`breakpoints = [date("2026-08-15")]` = the last observed day, and a four-knot
ramp beside it), and `value_at` accepts a bare `last_obs` but rejects
arithmetic. Downstream pins these with regex tests that re-fail on every data
release.

## Grammar

In exactly three positions, `last_obs` / `first_obs` become symbolic observation
anchors with optional constant duration arithmetic:

```camdl
simulate { to = last_obs + 4 'weeks }
breakpoints = [last_obs, last_obs + 1 'weeks, last_obs + 2 'weeks]
outbreak_size = value_at(N0 - S, last_obs - 1 'weeks)
```

- Shape: `ANCHOR`, or `ANCHOR ± <constant duration>`. The offset folds to model
  units at COMPILE time (`unit_to_model_time`); only the anchor's value binds
  later, so the compiled model stays data-independent.
- Units are the DSL's native tick spelling and are **plural only** (`1 'weeks`,
  not `1 'week` — the lexer has no singular). A numeric literal followed by a
  bare duration WORD in an anchor position gets a dedicated error naming the
  tick form, mirroring the CLI's existing tick rejection — the asymmetry (CLI
  words, DSL ticks) is deliberate but must be self-correcting on both sides.
- The offset must carry a duration unit (no bare numbers — mirror W324) and must
  dimension-check to `(0,1)`; the `simulate` block is not visited by dimcheck,
  so reuse the `tol_dim` helper pattern.
- Under an `origin`, a `'months`/`'years` offset on an anchor is refused the
  same way `date(...) + 6 'months` already is (E321): an anchor IS an instant,
  and the fixed-span semantics would silently contradict Rule 1.
- Elsewhere these identifiers stay ordinary (E100), unchanged.

## Semantics

Per-run, not a release constant: `last_obs` = max observation time over the
run's **bound** streams; `first_obs` = min. Two consequences to state loudly
because they are behaviour changes for a model that used a literal:

- A model file's fork becomes **fit-config-dependent**: two fit tomls binding
  different stream sets resolve different anchors from the same model. The
  resolved value is therefore recorded in the run record and printed on stderr
  (`simulate: last_obs → t = 91 (2026-08-15)`), as `--to` already does.
- Anchors resolved from a `--fit <run dir>` are checked against the fit's
  recorded `data_hashes` (`run_meta.rs`); a mismatch refuses by default, naming
  resolved-vs-fitted times. Re-vendored data must not silently move a structural
  feature of a model whose posterior never saw it.

`value_at` anchors resolve from the SAME pair wherever the horizon/knot anchors
do — including `simulate --fit`, which today refuses them (ebola F23, two
documented downstream workarounds). "One number per run" must be true.

## IR changes (0.31 → 0.32)

```
type obs_anchor = First | Last
type anchored_time = { anchor : obs_anchor; offset : float (* model units *) }
```

1. `simulation_config` gains `t_end_anchor : anchored_time option`.
2. `preset` gains `t_end_anchor : anchored_time option`.
3. `value_at`: `ALastObs` generalizes to `AObs of anchored_time`. Wire: the bare
   string `"last_obs"` stays the canonical zero-offset emission;
   `{"anchor": …, "offset": …}` is the offset form — and its decode arm must be
   inserted ABOVE the existing order-sensitive `("value_at", v)` catch-all in
   `value_reduce_of_json`.
4. `piecewise.breakpoints` needs no schema change (already `expr list`); an
   anchored entry lowers to a new `Expr::ObsAnchor(anchored_time)`. Dimension:
   `(0,1)`. Interception must be scoped to the `breakpoints` key — the
   `get_kw_list` closure is shared by six keys (`values`, `coefs`, …), which
   must NOT accept anchors.

Serde must use the append-when-present idiom (the `integrator` style, not the
adjacent `null`-emitting fields), pinned by a round-trip test asserting a
no-anchor model's JSON is byte-identical modulo the version string. Churn: 108
committed JSON files carry `ir_version`; `ir/golden/` (17) and the sim test
fixtures (18) are hand-updated sets (gh#384). `ir/schema.json` needs three edits
(simulation_config, preset, the `time_anchor` oneOf).

## Unresolved-anchor safety: no sentinel

The first draft baked `t_end = t_start` as a placeholder and claimed it could
not run. **False**: `ir::validate` never reads `t_end`, `t_end ==
t_start`
yields a one-row TSV at exit 0, and — worse — two placeholders compare EQUAL,
which defeats `refuse_scenario_horizon`, the `--to` conflict rule,
`check_baked_recurring_ends`, and the `--obs` preflight. That is the gh#561
class reintroduced. Replaced by:

- **The anchor field IS the unresolved marker.** The resolver CLEARS
  `t_end_anchor` when it substitutes. `CompiledModel::new` REFUSES any model
  with `t_end_anchor.is_some()` — one guard at the single choke point every path
  goes through, instead of one guard per entry point.
- The baked `t_end` value is `f64::NAN`, so every equality-based horizon guard
  fails-closed (refuses) rather than passing on a coincidence.
- A real ordering check (`t_end > t_start`, finite) lands in
  `ir::validate::validate` — it reads `t_end` nowhere today, which is its own
  latent bug (an inverted horizon is a silent header-only TSV).
- Horizon guards that compare a scenario's declared horizon to the model's must
  resolve anchors FIRST, or compare `(anchor, offset)` structurally. Red test:
  model anchored +4 'weeks, scenario anchored +8 'weeks — `fit predict` must
  refuse, not silently drop the scenario's window.

## Compile-time lowering (OCaml)

`sim_to` is re-resolved at FIVE sites; each classifies the raw AST itself (a
pure function of `ctx.simulate` — never state set by `expand_simulate`, whose
evaluation order relative to sibling record fields is unspecified):

- `expand_simulate` (6015) → `t_end_anchor` + NaN bake.
- Recurring default ends (7193/7260): a recurring schedule with no explicit `to`
  cannot bake its end under an anchored horizon → new E-code, which must
  PRE-EMPT the E241 (`from <= to`) that otherwise fires first and blames the
  intervention. Note `until` was renamed to `to` (gh#423) — the message says
  `to`.
- `SEveryAtDay` (7288-7290) bakes an end and its grammar has NO `to` key at all
  → its own message ("this schedule form cannot be used with an anchored
  horizon"), since the general advice is unachievable there.
- `ObsRegular.end_` (7694): baked NaN; the runtime already ignores baked emit
  ends (gh#561). But REACTIVE policies read that window, and the existing
  runtime guard matches `r.end == old_end` — which goes quiet once the two
  differ — so reactive + anchored horizon is refused at COMPILE time.
- W106 (9913/9995): skipped when either side is anchored.

`value_at` (8602-8636): the `mentions_last_obs` rejection becomes the offset
parser. Breakpoints (via the `Piecewise` arm, not the shared closure).
`Expr::ObsAnchor` forces ~23 exhaustive arms across `lib/ir` + `pp_expr`
(warning 8 is an error there); the two `*_of_json` decoders are key-matched, so
they need explicit arms and tests.

## Runtime resolution (Rust)

One resolver, called on the model that is **both hashed and run**:

- **Identity (the first draft was wrong here).** `deps` are documented
  non-identity; `--fit` does not populate `fit_dep` on this path; there is no
  observation-data digest on the Sim `run_id`. Resolving inside
  `resolve_run_model` (where `--to` lands) would leave `CasSink.base_model`
  unresolved, so two data vintages would share a `run_id` and the CAS would
  serve a stale trajectory. Therefore: substitute at the `base_model` load,
  before `CasSink` is built, AND add the resolved anchor pair to a hashed config
  level (`Some` only when the model is anchored, so unanchored models re-key
  nothing). No-collision test: `breakpoints = [last_obs]` with a LITERAL `to`
  (config level byte-identical), two data vintages → distinct run_ids.
- **Breakpoint ordering must be validated by the resolver** — the first draft
  cited runtime validation that does not exist (`compiled_model.rs`'s Piecewise
  arm only evaluates; `piecewise_value` is an order-dependent scan that silently
  returns a wrong step for unsorted knots). Check: non-decreasing knots,
  `values.len() == breakpoints.len() + 1`, no knot ≤ `t_start`.
- **`fit run`**: resolve before compile. `CompiledModel::new` is called at
  `runner.rs:242`, data loads at `:320` — the times-only load has no
  `CompiledModel` dependency (as `resolve_simulate_obs_anchors` shows), so MOVE
  it earlier; do not compile twice. Resolving in fit is required, not optional:
  `ode_grad.rs:72` takes the integration window from `simulation.t_end`, so an
  unresolved horizon would integrate nothing. Cost to measure before shipping:
  an anchored `+4 'weeks` horizon extends every gradient evaluation 28 days past
  the last scored observation.
- **`fit predict`**: must RESOLVE model-level anchors (it loads the archived IR
  and has the fit's data). It cannot inherit an unresolved horizon — today's
  `t_end_override: None` plus a placeholder would emit a single-snapshot
  "forecast" that reads as a plausible plateau at exit 0.
- **pfilter / profile / survey**: keep the gh#561 refusals for scenario
  horizons; model-level anchors resolve from their own bound data.
- Data-free tooling: nothing resolves; `CompiledModel::new` refuses, naming the
  construct and the flag.
- Gates: `references_last_obs` / `last_obs_quantity_names` are hardcoded to
  `LastObs` — rename and match every anchor form, and replace the
  release-compiled-out `debug_assert!` with a hard error, or a `first_obs`
  quantity returns NaN and is reported as _censored_ rather than refused.

## Downstream migration (must ship with the feature)

- A multi-knot fork must anchor EVERY knot; the ebola `ramp_control` has four.
- Their contract test's fork check is `if re.search(date-literal): assert …` —
  after migration the regex misses and the test passes VACUOUSLY. The
  replacement contract is the inverse: assert the fork is anchored and that no
  `date("…")` literal remains in `forcing {}` / `simulate {}`.
- Scenario `label = "…15 August…"` strings are not anchored and will rot; remove
  the dates or pin them separately.

## Non-goals

- Anchors in arbitrary expressions (rejected by construction).
- Calendar-aware month/year offsets.
- `simulate { from = first_obs }` — named follow-up; `t_start` participates in
  identity, calendar conversion, and every baked schedule. (`origin` itself can
  never be anchored: observation times are converted to model time USING
  origin.)
- `condition_from` DSL surface; per-stream anchors in these positions.
