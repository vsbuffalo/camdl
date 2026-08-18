# `value_at`: read a quantity series at a time, including the end of data

Status: proposed Issues: gh#538 (the reducer), gh#602 (design context), gh#616
(the fit-time anchor vocabulary this deliberately does NOT enter)

## Problem

The `quantities {}` reducers are whole-run (`final`, `max`, `time_of_*`,
`integral`) or nothing. "Cumulative infections at the end of data" — the
headline estimand of an outbreak fit — cannot be a scalar when the simulation
horizon extends past the data: `final(N0 - S)` reports the projection at
`simulate.to`, not the outbreak size. This produced a real misreading in the
field (a Sep-30 extrapolation quoted as outbreak size; ebola-bdbv-camdl friction
log F14). The workaround — declare the series and read it off a ribbon at a date
— is exactly the error-prone step a scalar-with-band exists to remove.

Two reads are needed:

1. at a **stated time**: `value_at(N0 - S, date("2026-08-10"))`;
2. at the **end of observed data**: `value_at(N0 - S, last_obs)` — the headline
   form, which must track the data across releases without editing the model.

## Surface

```camdl
quantities {
  #' Outbreak size by the stated date.
  cum_by_aug10 = value_at(N0 - S, date("2026-08-10"))

  #' Outbreak size at the last observed data point.
  outbreak_size = value_at(N0 - S, last_obs)
}
```

`value_at(SERIES, TIME)` is a two-argument temporal reduction, joining
`count_above(SERIES, THRESH)` in the classifier's dispatch set (`expander.ml`,
`temporal_reduction_names`). The name is `value_at`, not gh#538's `at`: `at` is
one indexing generalization away from ambiguity, and the existing reducers
already pay for self-description (`time_of_max`).

`TIME` is either

- an expression of dimension time, constant-foldable at compile (`date(...)`
  under a declared `origin`, a literal in model time units, or arithmetic over
  those); or
- the bare identifier `last_obs`.

Anchor arithmetic (`last_obs - 1 week`) is **rejected in v1** with a hint
pointing at the literal-date form and the fit-time override vocabulary (gh#616).
`first_obs` is deferred with it — same machinery, no use case yet.

## Semantics

**The read is last-observation-carried-forward on the output grid**: the value
at the last output snapshot with `t ≤ TIME`. Rationale: "cumulative X by T"
reads the state as of T; linear interpolation invents sub-grid values for jump
processes; and clamping to the window edge silently answers a different question
— which is the original misreading. A `TIME` exactly on a snapshot reads that
snapshot.

**Out-of-window is censored, not clamped**: `TIME` earlier than the first
snapshot or later than the last yields `Censored` for that draw — the same
per-draw contract as a non-firing `TimeReduce` crossing, so banding excludes it
and reports `n_censored` with no new machinery. A scenario whose horizon ends
before `last_obs` therefore reports a censored quantity loudly rather than a
truncated number silently.

**Sources**: `value_at` applies uniformly to state series (read on the snapshot
grid) and `observations.<stream>` series (read on that stream's own observation
times), like every other reducer.

**Dimension**: the result carries the series' dimension (like `final`/`max`);
`TIME` must check as dimension T (dimcheck: sibling rule to the `CountAbove`
threshold rule at `dimcheck.ml:793`, but requiring T rather than the series
dimension).

## `last_obs` and the model/data seam

The compiled model stays data-independent. `last_obs` is a **symbolic anchor**
in the IR, resolved at evaluation time — the same shape as
`observations.<stream>` sources, which already make a quantity's _value_
data-dependent while its _compiled form_ is not (`QSource::Observation` resolves
against an `ObsSeriesSet` supplied per draw, and a data-free context yields
nothing).

- Resolution: `last_obs` = the maximum observation time over the run's bound
  streams — the same canonical union of observation times the fit already
  builds. The **caller** resolves it once (one `f64`) and passes it to
  `eval_draw`; the evaluator never digs it out of a possibly-partial
  `ObsSeriesSet`.
- Contexts with no observation data (`simulate` without `--obs`,
  `--quantities-out` on a forward run): constructing the evaluator for a model
  whose quantities reference `last_obs` **hard-errors naming the quantity** —
  the capability-gap convention, expressed in code, not an empty column.
  `QuantityEvaluator` grows `references_last_obs()`, the sibling of
  `references_observations()`.
- `last_obs` is intercepted only as `value_at`'s second argument. In a rate, a
  binding, or anywhere else it remains an unknown identifier — it cannot leak
  into the model dynamics by construction (the `TriggerQuantity` precedent: a
  different language with different leaves).

This is why `last_obs` is admissible here but stays refused in `simulate { to }`
and forcing breakpoints (gh#616): a quantity is evaluated where the data is in
hand; a horizon is not.

## IR and identity

OCaml (`ir.ml`): `value_reduce` gains `VValueAt of time_anchor` with
`time_anchor = ATime of expr | ALastObs`. Rust (`ir/src/quantity.rs`):
`ValueReduce::ValueAt(TimeAnchor)` with
`enum TimeAnchor { Time(Expr), LastObs }`. Serde: externally tagged like its
siblings — `{"value_at": {"time": <expr>}}` / `{"value_at": "last_obs"}`.

This is an IR schema change: **`ir/VERSION` 0.30 → 0.31**, with the atomic
OCaml + Rust + golden update and a new golden fixture exercising both anchor
forms (with first-scenario params and trajectory baselines, per the golden
coverage requirements). The spec section is edited by hand
(`camdl-language-spec.md` carries doctest preambles; no `mdfmt`).

## Evaluation (Rust)

`fold_reduce` (`sim/src/quantity.rs`) gains the `ValueAt` arm: binary-search the
series' time axis for the last `t ≤ anchor`; `Censored` outside the window.
`eval_draw` gains `last_obs: Option<f64>`; a `ValueAt(LastObs)` program with
`None` is unreachable behind the construction-time gate, and `debug_assert!`s
so.

Callers updated: `fit predict` (resolves `last_obs` from the bound data it
already loads), `simulate --quantities-out` (passes `None`; the gate errors
before it for `last_obs` models). The streamed-quantities work (separate
proposal) inherits the same evaluator unchanged.

## Tests

Red first, then green:

- **OCaml**: serde round-trip for both anchors; dimcheck accepts a T-typed time
  and rejects a person-typed one; expander goldens for both forms; arity and
  anchor-arithmetic rejections with their hints.
- **Rust unit** (`sim/src/quantity.rs`): LOCF read on and between snapshots;
  censoring on both window edges; `last_obs` resolution; observation-source
  `value_at` on the stream's own time axis.
- **CLI e2e**: `fit predict` on a fixture whose
  `outbreak_size =
  value_at(cum, last_obs)` equals the series value at the
  last data time (cross-checked against the series TSV);
  `simulate --quantities-out` on the same model errors naming `outbreak_size`.
- **Mutation check** per repo convention: revert the LOCF arm to clamping,
  confirm the censoring e2e goes red.

## Deviations found at implementation

Two, both documented here per the follow-the-proposal rule:

1. **The TIME dimension check is deferred, not implemented.** The proposal said
   `TIME` "must check as dimension T" via a dimcheck sibling rule. The dimcheck
   quantity pass is deliberately read-only (it computes result dimensions and
   "never surfaces a new error on an existing quantity-bearing model" —
   `dimcheck.ml`), and the existing `CountAbove` threshold expressions are not
   dimension-enforced either. `value_at`'s time argument gets the same
   treatment: the result-dimension rule is implemented (preserves the series
   dimension); argument-dimension enforcement waits for whichever increment
   dimension-checks reduction arguments as a class.
2. **Censorability needed real plumbing, not "no new machinery".** The banding
   layer classifies censorable scalars by reduction kind (`QShape::of`) and
   propagates censorability through `Derived` arithmetic via an explicit set —
   both keyed on `TemporalReduce::Time(_)` only. A censored `value_at` draw
   would have gone down the plain-scalar path (dropped silently, no
   `n_censored`). Both sites now include `ValueReduce::ValueAt(_)`; the e2e test
   pins the censoring trio in the emitted TSV.

## Explicitly out of scope

Anchor arithmetic and `first_obs` (deferred above); per-stream
`last_obs(stream)`; `total`/`sum` (flow-source work); any fit-toml/predict
relative-time override (gh#616 — a different resolution site with the
`condition_from` grammar); excluding `quantities {}` from the fit identity
(gh#618 — adding a quantity still re-keys the fit until that design lands).
