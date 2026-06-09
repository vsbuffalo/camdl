# A unified time-interval model for simulate, fit, and forecast

- **Status:** Draft — design from a verified code + reproduction surface audit
  (2026-06-09). Proposes the abstraction; the per-area fixes are phased in §9.
- **Supersedes:** `2026-06-09-burnin-conditioning-window.md` (the burn-in /
  conditioning-boundary design, now ON HOLD). Its inference math — the
  flow-accumulator reset at the conditioning boundary, the faithful stochastic
  warm-up, the per-filter reset sites — is **preserved** here as the
  conditioning window's mechanism (§7.2); only its _surface_ (a lone
  `condition_from` fit.toml scalar) is replaced by the interval model.
- **Issues:** gh#134 (incidence over-accumulation), gh#142 (CAS horizon keying —
  closed), gh#143 (output vs dynamics end — open), plus two silent-wrong leads
  this audit surfaced (§5, F1/F4) that need a maintainer reproduction before
  filing.
- **Required reading before implementing:** `ir/schema.json`
  (`simulation_config`, `output_config`, `output_schedule`); the inference
  modules
  `rust/crates/sim/src/inference/{particle_filter,if2,pgas,correlated_pf}.rs`
  and the fit dispatch `rust/crates/cli/src/fit/{methods,runner}.rs`; the
  forward backends + `rust/crates/sim/src/schedule.rs`;
  `docs/camdl-language-spec.md` §2.1 (units/time) and §7 (forcings);
  `docs/dates.md`. This touches inference math and the IR contract — high-risk;
  read the full function before editing.

## 1. The problem: there is no single time-axis authority

A camdl run juggles several distinct notions of time, and today **no single
component owns the time axis**. The dynamics span, the output window, and the
observation extent are three independent fields, read by different code, kept
consistent only by convention — and several genuinely-distinct intervals are
either _faked_ (per-stream observation windows), _unguarded_ (covariate-table
domains), or _conflated_ (the forecast horizon is the model's dynamics end). The
gaps between these uncoordinated notions are precisely where camdl produces
silent wrong answers — the failure class this software exists to prevent.

Concretely, three separate "end" concepts exist with no reconciliation:

| Concept      | Field / source                       | Read by                               |
| ------------ | ------------------------------------ | ------------------------------------- |
| Dynamics end | `simulation.t_end`                   | forward backends (loop terminator)    |
| Output end   | `output.times.end` (compiler-pinned) | `output::output_times` → emitted rows |
| Data end     | `obs_times.last()`                   | stochastic inference filters          |

In forward simulation the dynamics end and output end are distinct IR fields
kept equal **only by the OCaml compiler** (`expand_output`, `expander.ml:3586`)
and read independently by Rust — a hand-edited or tool-generated IR desyncs them
(this is the gh#142/#143 fault line). In inference, the dynamics end is
_discarded_ entirely: the filters set their terminal boundary to
`obs_times.last()` (`if2.rs:241`, `particle_filter.rs:151`,
`correlated_pf.rs:266`), while dynamics _start_ is honored
(`simulation.t_start`). Start and end are sourced from different places by the
same code path.

This proposal names the intervals explicitly, makes one component reconcile
them, and routes every "interval doesn't line up" case through a loud error or a
capability gate — so the silent configurations in §4 become uniform and
diagnosable across every backend and inference algorithm.

## 2. The cast of intervals

The design is types-first: a run is described by a small set of named intervals,
and each operation (simulate / fit / forecast) is defined by which intervals it
reads and the constraints among them.

| Interval                                       | Meaning                                                    | Where it should live         |
| ---------------------------------------------- | ---------------------------------------------------------- | ---------------------------- |
| **D** — dynamics `[t_start, t_end]`            | the span over which the process is integrated              | derived per operation (§7)   |
| **Wₖ** — observation window, per stream        | when stream _k_ carries data (fit) / is emitted (simulate) | data (fit) / output schedule |
| **C** — conditioning `[cond_from, cond_to]`    | the sub-span over which the likelihood actually scores     | fit (data-relative)          |
| **Eⱼ** — covariate/forcing domain              | where a time-indexed forcing has real data                 | model (with the forcing)     |
| **O** — output/emission `[out_start, out_end]` | when output rows are written                               | model default / operation    |
| **F** — forecast horizon                       | how far to project beyond the data                         | **operation parameter**      |

The reframe that dissolves the earlier "where does `condition_from` live"
debate: `condition_from` is simply **C.start**, one boundary of one named
interval. Each interval has a natural home, and the operations compose them
rather than one field meaning five things at once.

## 3. Master diagram (the clean target)

```
       t_start     cond_from          data…          cond_to     horizon
          |            |                                |            |
burn-in   |············|                                |            |
condition              |================================|            |
forecast                                                |············|
dynamics  |=====================================================…F…==|
stream A       [obs·········obs]
stream B               [obs···············obs]
forcing E [knots······································]   ← must cover the run
```

The dynamics span D is the _union_ of everything an operation must integrate
(earliest burn-in start → forecast horizon). The conditioning window C and the
observation windows Wₖ are sub-intervals used by fitting. The forcing domain Eⱼ
must cover whatever D the operation integrates. The forecast horizon F extends D
to the right for projection, with no conditioning past the data.

## 4. The overlap configurations — and what each does today

These are the distinct ways the observation interval(s) and the simulation
interval can relate, plus the forecast and covariate-domain cases. Each is
tagged with current behavior; §5 gives the evidence.

```
1. BURN-IN  (W starts after D starts)            gh#134 — SILENT over-accumulation
   D |==================================|         of the first incidence window
   W        [obs··········obs]                     (W329 warns, never errors)

2. FIT-TO-PAST + FORECAST TAIL (W ends before D)  inference IGNORES t_end; forecast
   D |==================================|         needs a separate run; the horizon
   W   [obs······obs]      forecast →              is baked into model identity

3. DATA PAST DYNAMICS END (W ⊅ D, right)          INCONSISTENT across backends:
   D |==============|                             silent on chain-binomial/PGAS
   W   [obs················obs]                    (uses last_obs), HARD-ERROR on ODE

4. DATA BEFORE DYNAMICS START (W ⊅ D, left)       SILENT-WRONG in release: obs skipped
        D |==============|                        at load, particle scored without
   W [obs····obs]                                 ever propagating to it

5. PER-STREAM STAGGER                             HARD-ERROR ("heterogeneous schedules
   W_A [obs······obs]                             not supported"); per-stream windows
   W_B      [obs········obs]                       are faked (one shared vector)

6. FORCING DOMAIN SHORTER THAN RUN (E ⊊ D/F)      SILENT flat-extrapolation (forcings)
   D/F |================================|         — but the sibling table mechanism
   E   [knots········]                            HARD-ERRORS. Same codebase, opposite law

7. OUTPUT vs DYNAMICS (O.end ≠ D.end)             FROZEN TAIL (gh#143): emits frozen
   D |==============|                             terminal state past dynamics; the
   O |=============================|              reverse truncates output but runs
                                                   full dynamics (wasted compute)
```

## 5. Current behavior, with evidence

Findings are tagged `[repro]` (reproduced with a command), `[code]` (read from
the implementation at the cited site), or `[lead]` (strong code evidence but
needs an independent maintainer reproduction before being filed as an incident).
Ranked by blast radius.

**F1 — Conditioning is silently ignored on the PGAS and ODE inference cells.**
`[lead]` `skip_first_obs_from_loglik` (the `ic_free` mechanism) is honored only
by IF2 (`if2.rs:501`) and the bootstrap PF (`particle_filter.rs:355`; PMMH
inherits it via its PF closure). `compute_ode_loglik` (`runner.rs:519`) does not
read it and scores every observation including `y₁`; `pgas.rs` has no first-obs
/ conditioning handling at all. So `ic_free = true` on an `nl-sbplx`/`nl-bobyqa`
or PGAS stage silently computes the _unconditional_ likelihood — and the startup
banner and the `ic_free && !ivp` precondition (`runner.rs:427`, `mod.rs:466`)
fire regardless of stage type, _falsely signaling_ that conditioning is active.
No capability flag catches it. (Config 1/2 × the wrong cell.) **Silent-wrong,
live today.** Repro pending: a one-stage `ic_free` fit on each of ODE and PGAS,
asserting the returned loglik equals the _conditional_ value, not the full sum.

**F2 — Forcings silently flat-extrapolate past their data.** `[repro]` A
data-driven `interpolated`/`Constant` forcing clamps to its first/last knot
outside its domain (`propensity.rs:357-380`) with no warning. Reproduced from
this worktree's binary: a `clim` forcing whose last knot is `t=750`, simulated
to `t=1100`, pins at the last value for 350 days, exit 0, no stderr:

```
CAMDL_TRACE_STEPS=1 camdl simulate clim_probe.camdl --backend chain_binomial \
  --dt 1 --seed 1 -o traj.tsv 2> trace.tsv
# t=751..1100 → clim ≡ 1.37770 (last knot); exit 0; no warn/error text
```

The sibling stratum-indexed `TableLookup` mechanism does the opposite — it
_hard-errors_ out of range (`resolved_expr.rs:540`), and its code comment
(`table.rs:4-6`) records that `Clamp`/`Wrap` were **built and then removed**
because "silent extrapolation masks modeling errors." The principle is already
in the codebase; it is applied to the stratum path and not the time path — which
is the one prone to the forecast-past-data error. `TimeFunction` carries no
out-of-bounds field (`time_func.rs:76`), so there is currently no surface to
even request an error. **Silent-wrong; the original forecasting worry.**

**F3 — `sim.t_end` is backend-inconsistent in inference.** `[code]`
Chain-binomial / PGAS inference set the terminal boundary to `obs_times.last()`
and never consult `simulation.t_end` (`particle_filter.rs:151`,
`pgas.rs:368/375`), so observations beyond `simulate.to` are silently included;
the ODE-loglik path uses `model_sim.t_end` and _hard-errors_ if it precedes the
last obs (`runner.rs:612`). The same model + data file gets opposite treatment
by backend. (Config 3.)

**F4 — Data before `t_start` is silently mis-scored in release.** `[lead]` The
loader skips observations with `t < t_start` (`caltime_load.rs:250`); the only
downstream lower-bound guard is a `debug_assert!` in `interval_steps`
(`time.rs:108`), compiled out of `--release`. Rust's saturating float→int cast
turns the resulting negative step count to `0`, the substep iterator yields no
steps (`schedule.rs:426`), and the particle is then scored against that datum
_without having propagated to it_. No panic, no error. **Silent-wrong.** Repro
pending: a fit with one obs at `t < t_start`, asserting it is either rejected or
correctly handled rather than scored against the unpropagated state.

**F5 — Frozen tail / output truncation.** `[repro]` `output.times.end` and
`simulation.t_end` are independent fields with no reconciliation. With
`simulation.t_end=20`, `output.end=30` the final flush
`drain_outputs(cursor, f64::INFINITY, …)` (`chain_binomial.rs:285`,
`gillespie.rs:409`, `ode.rs:313`) emits the remaining output times as frozen
copies of the terminal state (flows zeroed) — a flat-lined pseudo-forecast. The
reverse (`simulation.t_end=800`, `output.end=80`) emits 82 rows but runs the
full 800 substeps (~10× wasted compute). This resolves gh#143's open question:
the row cap is `output.times.end`, not a reconciliation against
`simulation.t_end`. (Config 7.)

**F6 — Per-stream observation windows are faked.** `[code]` `StreamSpec` carries
a per-stream `obs_times`, but every construction site forces them identical —
validated _twice_ with two different tolerances (exact `!=` at
`multi_stream_obs.rs:337`, `1e-9` at `runner.rs:314`) — and the model collapses
to one shared `obs_times` vector; heterogeneous schedules hard-error ("not
supported yet"). The flow-reset carries a canary comment
(`particle_filter.rs:401-413`) marking exactly where true per-stream windows
would require a per-flow, per-stream-indexed reset. (Config 5.)

**Healthy, for contrast.** The (algorithm × backend) _pairing_ gate is loud and
well-formed: the registry (`methods.rs:67`), `validate_combo`, and per-pair
`rejection_reason` reject unsupported pairs with a clear message; there are no
`if backend == "…"` string branches in the sim layer; the `Capabilities`
bitflags hard-error capability mismatches (overdispersion-on-ODE,
real-compartments-on- chain-binomial-inference, etc.). Every silent gap above is
on the **window/conditioning axis**, which was never routed through that
machinery.

## 6. The leak, stated once

All six findings are symptoms of the same missing abstraction. There is no
object that owns "the time axis of this run," so:

- three "end" fields drift (F3, F5);
- intervals that should be first-class are faked (F6) or absent (F: the forecast
  horizon, the conditioning end `cond_to`);
- domains that should bound integration are unguarded (F2, F4);
- and the conditioning window is invisible to the capability system, so cells
  that cannot honor it degrade silently instead of failing loudly (F1).

## 7. The proposed model

One component reconciles the time axis. The model declares the _process_;
operations declare the _windows_ they need; the integration span is _derived_,
not a field that means five things.

### 7.1 Six design points

1. **One integration span D per operation**, computed as the union of the
   windows that operation needs, explicit and validated (`t_end > t_start`,
   `dt > 0`, `dt ≤ (t_end − t_start)` — none of which is checked today). This
   kills the three-independent-ends problem: there is one span, and output /
   conditioning / data windows are sub-intervals validated against it.
2. **Observation windows Wₖ become first-class and per-stream.** `first_obs`,
   `last_obs`, and C are _derived_ quantities computed once, not recomputed ad
   hoc at the ~6 sites that do so today. The flow-reset becomes
   per-stream-indexed (the canary already marks the site). This is the
   unified-observation-data lift (gh#134/#139); the interval model is its
   time-axis half.
3. **Conditioning window C = `[cond_from, cond_to]` is explicit and routed
   through the capability/dispatch gate.** A cell that cannot condition (today:
   ODE and PGAS for `ic_free`) **hard-errors** at dispatch with a message naming
   the limitation, instead of silently ignoring it. This is the "every backend ×
   inference cell supported or fail loudly" rule (CLAUDE.md) applied to the
   window axis; F1 becomes loud. The reset mechanism is the one from the
   superseded burn-in proposal (§7.2).
4. **Forecast horizon F is a per-run/operation parameter, not model identity.**
   A `--until`/`--horizon` knob (or a `forecast` operation) projects from the
   fitted posterior to F, reusing the fit — so you forecast further _without
   recompiling and re-fitting_. Today the horizon is baked into `simulate.to`,
   so extending it re-keys the model hash (`hashing.rs:60-78`) and invalidates
   the fit cache; that conflation is the most likely source of the user-reported
   forecasting friction (confirm against the colleague's report — §10).
5. **Forcing/covariate domains Eⱼ get explicit bounds and an out-of-bounds
   policy**, checked against the integration span — uniform with the table
   mechanism that already does this. Default `error` for data-driven series
   (`interpolated`); `constant`-outside permitted explicitly for genuine step
   functions (`piecewise`/`Constant`), where a flat tail is intended rather than
   a ran-out-of-data accident. F2 becomes loud (or an explicit opt-in).
6. **One reconciliation pass over the time axis.** Every window is validated
   against D in a single place, every mismatch is a loud diagnostic, and the
   behavior is identical across backends. F3, F4, F5 become uniform and
   diagnosable.

### 7.2 Conditioning mechanism (preserved from the superseded proposal)

The conditioning window's _operative_ behavior is unchanged from the burn-in
design: over the burn-in span `[t_start, cond_from)` the dynamics run faithfully
(full process noise, interventions, forcings — the covariates that make the
warm-up informative), and **the incidence flow accumulator is reset at
`cond_from`** so the first scored window is one cadence `(cond_from, first_obs]`
rather than the whole leading gap. This is purely an incidence
(flow-accumulator) concern; prevalence (state-snapshot) observations are already
scored correctly regardless of where dynamics start. The reset lands in each
filter's idiom — the PF/IF2 propagate-to-`cond_from` prelude, and the PGAS
`cum_flows` reset at the boundary substep in `complete_data_loglik`, `csmc_as`,
**and** `complete_data_loglik_grad` (missing the gradient mirror is a silent
NUTS bug). The default is no warm-up (`cond_from = t_start`), bit-identical to
today. See the superseded `2026-06-09-burnin-conditioning-window.md` §2/§6 for
the full inference math and per-filter reset sites, which this proposal carries
forward intact.

The change versus that proposal is the _surface_: instead of a lone
`condition_from` scalar in fit.toml, the conditioning window is one of the run's
named intervals (§2), with `cond_to` (the lag-truncation back edge) as its
symmetric partner, and the whole window routed through the capability gate.

## 8. Operations — which intervals each reads

| Operation                  | Integration span D                                   | Conditions over | Forecast | Forcing constraint |
| -------------------------- | ---------------------------------------------------- | --------------- | -------- | ------------------ |
| **simulate** (forward)     | `[t_start, out_end or --until]`                      | —               | optional | Eⱼ ⊇ D             |
| **simulate --draws** (PPC) | same as the fit's D, optionally + F                  | —               | optional | Eⱼ ⊇ D             |
| **fit**                    | `[t_start, max_k Wₖ.end]` (burn-in start → last obs) | C ∩ Wₖ          | no       | Eⱼ ⊇ D             |
| **forecast**               | `[last data, F]`, continuing the posterior state     | none            | yes      | Eⱼ ⊇ [.., F]       |

Placement of the intervals — the resolution of the model-vs-fit question:

- **Model (`.camdl`)** declares the _process_ and the natural origin
  `simulate.from` (= `t_start`, the dynamics/burn-in start and the calendar
  anchor), the forcing domains Eⱼ (declared with each forcing), and a _default_
  span. The model is self-contained for everything intrinsic to the process.
- **Fit (`fit.toml`/CLI)** declares data wiring (streams → Wₖ) and the
  conditioning window C = `[cond_from, cond_to]` — data-relative, fit-time. The
  fit's D is derived; the fit identity is keyed by data + C + model, **not** by
  a forecast horizon.
- **Forecast (operation/CLI)** declares the horizon F as a per-run parameter,
  reuses the posterior, and does not re-key the fit.

This honors the `.camdl`-self-containment goal at the level that matters (the
_process_ is fully specified by the model, and a model with an early origin
cannot silently mis-score because the conditioning window is loud through the
capability gate) while keeping the data-relative and how-far-to-look choices
where the data and the question live.

## 9. Capability/dispatch integration and phasing

Every (backend × inference algorithm × window-feature) combination must be
supported-and-tested or hard-error through the capability system — no silent
third option (CLAUDE.md). Concretely, the conditioning window and the
forcing-domain check join the existing dispatch gates (`required_capabilities()`
/ `Simulate::capabilities()` / `resolve_obs_alignment`), so an unsupporting cell
fails loudly with a message naming the limitation.

Suggested phasing (each phase ends green, none batches semantic changes):

- **P0 — Loud the silent-wrong cases (independent of the abstraction).**
  Reproduce and fix F1 (route `ic_free`/conditioning through the capability gate
  so ODE/PGAS hard-error or honor it), F2 (forcing-domain OOB policy, default
  error for data series), F4 (reject or correctly handle data before `t_start`).
  These are bug-fixes; do them first, TDD red→green, and file the leads (F1, F4)
  as incidents once reproduced.
- **P1 — Reconcile the ends (F3, F5).** Make the integration span explicit and
  validated; reconcile `output.times.end` vs `simulation.t_end`; make inference
  honor or explicitly reject data beyond the declared span uniformly across
  backends. Resolves gh#143.
- **P2 — The conditioning window surface.** Replace the burn-in proposal's
  `condition_from` scalar with the explicit C interval; wire `cond_from` (the
  preserved reset mechanism, §7.2) and `cond_to` (lag truncation). Gate per
  cell. Note the correlated-PMMH constraint: it requires every obs window to
  span an identical substep count, so a non-trivial leading window must match
  the modal cadence or hard-error on that cell (`correlated_pf.rs:166`).
- **P3 — Forecast as an operation.** The `--until`/`forecast` horizon parameter,
  posterior reuse, and CAS keying that does not invalidate the fit. Needs the
  colleague's friction report to pin the surface (§10).
- **P4 — Per-stream windows.** First-class Wₖ and the per-stream-indexed flow
  reset (F6) — folds into the unified-observation-data surface (gh#134/#139).

IR changes are confined to the time-axis structs (`simulation_config`,
`output_config`, forcing domains) and follow the atomic OCaml+Rust+golden update
procedure (CLAUDE.md "Changing the IR schema").

## 10. Open questions

- **The colleague's forecasting friction (blocking P3).** The design assumes the
  friction is the horizon-baked-into-model conflation (F: extending
  `simulate.to` re-keys the model and forces a re-fit). Confirm against the
  actual report before committing the P3 surface — it may instead (or also) be
  the frozen-tail (F5) or the inference-ignores-`t_end` (F3) behavior.
- **Default span placement.** Does the model declare a default `t_end` at all,
  or is the integration end always derived (output window for simulate, last obs
  for fit, F for forecast)? Leaning derived, with `simulate.to` as an optional
  model default for bare forward runs.
- **Forcing OOB default.** `error` for `interpolated`/data series is proposed;
  confirm `piecewise`/`Constant` step functions keep `constant`-outside as their
  intended semantics (a flat tail is correct for a step policy, a hazard for a
  data series).
- **`cond_to` / lag truncation.** Whether the back edge is a user interval,
  derived from a stated reporting-lag, or both.

## 11. References

- gh#134 — incidence over-accumulation when dynamics start far before the data
  (the burn-in motivation; §4 config 1).
- gh#142 (closed) — CAS model hash now folds `output.times` + `simulation` span
  (`hashing.rs:60-78`); confirms the forecast-horizon-in-model-identity
  conflation.
- gh#143 (open) — output end vs dynamics end; resolved by §5 F5 (cap is
  `output.times.end`).
- `2026-06-09-burnin-conditioning-window.md` (superseded) — the conditioning
  inference math (§2/§6) carried forward here (§7.2).
- `2026-05-30-conditioning-boundary-tcond.md` (superseded, transitively) — the
  original `t_cond` inference-math note.
- Surface evidence (file:line) is inline in §5; the forcing-extrapolation (F2)
  and frozen-tail (F5) findings carry their reproduction commands; F1 and F4 are
  leads pending an independent maintainer reproduction (§9 P0).
