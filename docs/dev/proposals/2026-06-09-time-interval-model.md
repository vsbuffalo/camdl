# A unified time-interval model for simulate, fit, and forecast

- **Status:** Draft — design from a verified code + reproduction surface audit
  (2026-06-09); the interval abstraction (`RunWindows`, `cond_to`, the
  `forecast` operation) is **unbuilt**. The per-area fixes are phased in §9.
- **Generalizes (future direction, not yet superseding):**
  `2026-06-09-burnin-conditioning-window.md`. That design's `condition_from`
  fit.toml scalar **shipped** (per-stream, in the sparse/multi-cadence
  observation work) and is the current reality; this interval model would
  generalize the lone scalar to an interval, but until `RunWindows` is built it
  does **not** replace it. The conditioning inference math — the flow-accumulator
  reset at the boundary, the faithful stochastic warm-up, the per-filter reset
  sites — is unchanged and lives in the shipped scalar (§7.2 documents it).
- **Issues:** gh#134 (incidence over-accumulation), gh#143 (output vs dynamics
  end — open), and the CAS horizon-keying fix (the proposal cites gh#142; the
  code comment at `hashing.rs` labels the same fix gh#147 — a citation drift to
  reconcile). The two silent-wrong findings F1/F4 (§5) have since been
  **reproduced** (no longer leads) and are filable incidents.
- **New here? Read the overview first:**
  [`2026-06-09-time-and-observation-overview.md`](2026-06-09-time-and-observation-overview.md)
  — the system-level map of how time and observations work across all three
  proposals, with diagrams and the type designs.
- **Sibling proposals — this is the top of a three-layer stack; read §1.5:**
  `2026-06-06-observation-system.md` (the data layer — `bind`/`BoundObs`, holes,
  missing data) and `2026-06-06-scheduling-effect-topology.md` (the temporal
  spine — `Observe`, `TemporalKind`, the per-stream `ResetWindow`,
  `StepPolicy`). This proposal sits above both and must not re-own what they own
  (§1.5).
- **Required reading before implementing:** the two sibling proposals above;
  `ir/schema.json` (`simulation_config`, `output_config`, `output_schedule`);
  the inference modules
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

## 1.5 Where this sits: the three-layer stack

This proposal is the **top** of a three-layer stack, and it owns only the top
layer. A reader must hold the other two, because this proposal deliberately does
**not** re-specify what they own:

| Layer                | Proposal                                   | Owns                                                                                                                                  |
| -------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Data**             | `2026-06-06-observation-system.md`         | rows → typed per-stream cells, holes (missing values), `bind`/`BoundObs`                                                              |
| **Temporal spine**   | `2026-06-06-scheduling-effect-topology.md` | `Schedule`, `TemporalKind{Interval,Instant}`, per-stream `ResetWindow`, `StepPolicy`, the sub-`dt` collision guard                    |
| **Intervals (this)** | this proposal                              | the run's time-axis _windows_ and their reconciliation; the conditioning window C; the forecast horizon F; covariate-domain bounds Eⱼ |

The seams: the **incidence-vs-prevalence** distinction is the spine's
`TemporalKind` (this proposal uses it, does not redefine it); the **per-stream
flow reset** is the spine's `ResetWindow`, _placed_ at `cond_from` (a non-obs
boundary) rather than at an obs — a placement-generalization of today's
obs-keyed reset, not a re-key of an existing effect (the spine's reset is itself
unbuilt, so this is genuine per-cell work, enumerated in §7.2); **per-stream
observation windows** are the data layer's `BoundObs` cells + the spine's
`ResetWindow` (this proposal references them, §7.1 pt 2, and does not re-own
them). **Dependency order:** the spine's `ResetWindow`/`TemporalKind` are **0%
built today** (the topology doc is a design map), so this proposal's
conditioning and per-stream work (P2, P4) is gated on the spine shipping them —
the same gate the data layer's heavy tier waits on. What this proposal adds and
nobody else owns: the **forecast horizon F**, the **covariate-domain bound Eⱼ**,
and the **reconciliation of the three "end" fields** into one authority (§7).

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

**Make the illegal states unrepresentable — one `RunWindows` authority type.**
The findings in §5 (F3, F4, F5) are all the same shape: independent `f64` fields
that can silently disagree (`simulation.t_end` vs `output.times.end` vs
`obs_times.last()`), or an unchecked ordering (an obs before `t_start`). The fix
is not "add a validation pass" — it is a type whose _construction_ forbids the
bad state. A single `RunWindows` value owns the axis:

```rust
struct RunWindows {
    dynamics:    Interval,          // D = [t_start, t_end]; t_end > t_start enforced
    conditioning: Option<Interval>, // C = [cond_from, cond_to] ⊆ D; None ⇒ no warm-up
    output:      OutputSchedule,    // O, validated ⊆ D
    forecast:    Option<f64>,       // F: horizon ≥ D.end, an operation parameter
    // per-stream observation windows Wₖ are NOT here — they live in the data
    // layer's BoundObs; this type only carries the global axis they derive against.
}
impl RunWindows { fn new(...) -> Result<Self, Error> { /* enforces every ordering */ } }
```

Built once per operation, it makes F3/F5 (disagreeing ends) and F4 (obs before
`t_start`) **unconstructible**, not merely caught — the orderings
`t_start ≤ cond_from < cond_to ≤ D.end ≤ F` and "every window ⊆ D" are checked
in one constructor, and nothing downstream can hold a contradictory pair of
ends. The covariate-domain bound Eⱼ is the one cross-cutting constraint a pure
ordering type cannot encode (it depends on the forcing data), so it stays a
checked constructor argument (§7.1 pt 5). The per-stream windows Wₖ and the
incidence flow reset are **not** fields here — they belong to the data layer and
the spine (§1.5); `RunWindows` carries only the single global axis those derive
against.

**Caveat, stated as bluntly as the data layer states its own:**
"unconstructible" holds only once **every backend read of `simulation.t_end` /
`output.times.end` is routed through `RunWindows` and the direct IR-field reads
are deleted** (today the forward backends read `model.simulation.t_end` straight
into their configs — `util.rs:2029`, the three backend configs). Until then
`RunWindows` is a validation pass wearing a type's clothing. This is the exact
obligation the observation system states for `BoundObs` ("a private constructor
guards a door next to an open window" — `MultiStreamObsModel::new` must consume
the validated type and the raw path be privatized); the interval layer owes the
same.

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

**F1 — Conditioning is silently ignored on the PGAS, ODE, _and_ correlated-PMMH
inference cells.** `[repro]` `skip_first_obs_from_loglik` (the `ic_free`
mechanism) is honored only by IF2 (`if2.rs:501`) and the bootstrap PF
(`particle_filter.rs:355`; plain PMMH inherits it via its PF closure). Three
cells ignore it: `compute_ode_loglik` (`runner.rs:519`) scores every observation
including `y₁`; `pgas.rs` has no first-obs/conditioning handling at all; and
**correlated-PMMH** (`correlated_pf.rs:480`) adds every increment
unconditionally — a fourth cell the earlier draft missed. **Reproduced:** a
one-stage `nl-sbplx`/ODE fit returns byte-identical `best_loglik` (−58.714… to
15 digits) whether `ic_free` is true or false, _while printing_ "ic-free
inference: conditioning on y₁"; the PGAS config struct (`pgas.rs:477`) has no
conditioning field at all. So `ic_free = true` silently computes the
_unconditional_ likelihood while the banner and the `ic_free && !ivp`
precondition (`runner.rs:427`, `mod.rs:466`) falsely signal conditioning is
active. No gate catches it. (Config 1/2 × the wrong cell.) **Silent-wrong, live,
filable.** The fix is **not** "presume these cells can't condition and gate them
off" — it is to route conditioning through the dispatch path so each cell either
honors it or hard-errors (see §7.1 pt 3; this matches the observation system's
"unify first, gate only the floor" for the same ODE cell).

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

**F4 — Data before `t_start` is silently mis-scored in release.** `[repro]`
**Reproduced** (model `from = 21`, obs at `t = 0, 7, 14`): `camdl pfilter` loads
all 12 observations and returns a stored loglik with **exit 0** (−inf when the
early obs are zero, `EssCollapsed` when large) — no diagnostic ever names "obs
precedes `t_start`." The mechanism, corrected from the earlier draft: the early
observations are **not** dropped at load (the `caltime_load.rs:250` `continue`
is inside the distinct-substep collision check only, not the loader) — they are
**loaded and then scored without the particle ever propagating to them**, which
is strictly worse than "skipped." The substep iterator yields zero steps for an
obs at `t < t_start` (`schedule.rs:426`); the only lower-bound guard is a
`debug_assert!` in `interval_steps` (`time.rs:108`), compiled out of
`--release`, and the saturating float→int cast turns the negative step count to
`0`. The particle is then scored unpropagated (`particle_filter.rs:288`). No
panic, no error. **Silent-wrong, live, filable.**

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
would require a per-flow, per-stream-indexed reset. (Config 5.) This is the
_same_ finding the data layer (`observation-system.md`) and the spine
(`scheduling-effect-topology.md`, its `M3`) record; the **fix is owned there**
(the spine's per-stream `ResetWindow` + the data layer's `BoundObs` cells), not
here. This proposal only consumes it — see §1.5 and §7.1 pt 2.

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

1. **One integration span D per operation — the `RunWindows` authority type
   (§2).** D is computed as the union of the windows that operation needs, and
   the orderings (`t_end > t_start`, every window ⊆ D,
   `t_start ≤ cond_from <
   cond_to`) are enforced in the type's constructor —
   so the three-independent-ends problem and the obs-before-`t_start` case are
   _unconstructible_, not merely validated after the fact. None of these
   orderings is checked today.
2. **Observation windows Wₖ are first-class and per-stream — owned by the data
   layer and the spine, not here.** `first_obs`, `last_obs`, and C are _derived_
   from the bound data, computed once rather than recomputed ad hoc at the ~6
   sites that do so today. The per-stream flow reset is the spine's
   `ResetWindow`; the per-stream cells are the data layer's `BoundObs` (§1.5).
   This proposal **consumes** them and must not re-specify them; the work itself
   is the data-layer + spine lift (gh#134/#139).
3. **Conditioning window C = `[cond_from, cond_to]` is explicit and gated at the
   right seam.** A cell that does not honor conditioning hard-errors with a
   message naming the limitation instead of silently ignoring it (F1). Mechanism
   detail to pin: conditioning is **per-stage inference config**, so the gate is
   the `(algorithm)` dispatch registry (`methods.rs::validate_combo` /
   `resolve_obs_alignment`), **not** the IR-scanning `required_capabilities()` /
   `Simulate::capabilities()` flags (those scan the _model_, which knows nothing
   about `ic_free`). And the resolution is **unify first, gate only the floor**
   (matching `observation-system.md`'s ODE step): route conditioning through the
   shared path so each cell honors it where it can; gate only a cell that
   genuinely cannot, with the bar "show it cannot be unified before you gate
   it." The reset mechanism is §7.2. **`ic_free` is orthogonal and stays.**
   Verified: `ic_free` (`skip_first_obs_from_loglik`, honored only by IF2 and
   the bootstrap PF) means precisely "drop the _first_ likelihood term while
   still reweighting/resampling at it" — it does **not** warm up dynamics or
   reset a flow accumulator. So `C` does not replace `ic_free`; they **compose**
   (a fit can both warm up over `[t_start, cond_from)` and drop the first scored
   term). The migration must define the `C` that reproduces today's
   `ic_free=true` bit-for-bit on the IF2/PF cells, or declare the break and
   re-baseline — `cond_from = t_start` reproduces today's _no-conditioning_
   default, not the `ic_free=true` case.
4. **Forecast horizon F is a per-run/operation parameter — _up to the covariate
   horizon_.** A `--until`/`--horizon` knob projects from the fitted posterior
   to F, reusing the fit, so you forecast further _without recompiling and
   re-fitting_ — the CAS premise holds (extending `simulate.to` today re-keys
   the model hash, `hashing.rs`, and invalidates the fit cache). **But the
   headline is bounded:** a covariate-driven model reads forcing tables that
   only have data to some `last_knot`, and design point 5 makes reading past
   that a hard error — so the maximum reachable F is capped by where the
   covariate data ends, a _model-input_ property, not a free dial. Forecasting
   beyond it requires extending the covariate data (a model-input change), not
   just a larger F. Since the motivating models _are_ covariate-driven (the Kano
   SIA/births burn-in), state this limit, do not paper over it. Also unresolved
   and method-dependent: forecast's integration span D. The origin is keyed on
   _what the fit persisted_, not the method name: with a stored filtered
   end-state (PGAS) D = `[last_data, F]`, and a bootstrap-PF fit that saved its
   paths likewise continues from the filtered cloud; with an MLE fit (IF2/ODE,
   no stored end-state) — **or a bootstrap-PF fit that did not persist paths**
   (paths are optional output) — the forecast falls through to re-filtering from
   `t_start`, so D = `[t_start, F]`. The absent-paths case degrades to the
   re-filter path, it does not error. The forcing-domain and conditioning checks
   key on D, so this must be pinned per fit method (§10).
5. **Forcing/covariate domains Eⱼ get explicit bounds and an out-of-bounds
   policy**, checked against the integration span — uniform with the table
   mechanism that already does this. Per forcing kind: **`error` for data-driven
   series** (`interpolated`) and **for `CubicSpline`** — note both currently
   _clamp flat_ at the last knot (`compiled_model.rs:100`, "Clamps to boundary
   values"), the **same** silent-extrapolation hazard as `interpolated`; they
   error for that reason, not because a polynomial blows up past the knots (it
   does not — an earlier draft of this point mis-stated the mechanism).
   **`constant`-outside** is permitted explicitly only for genuine step
   functions (`piecewise`/`Constant`), where a held level past the domain is the
   intended semantics; **`Periodic`/`Fourier`/`PeriodicSpline` are exempt**
   (they wrap by construction, there is no domain to exceed). Note also that
   scheduled campaigns (SIAs) are **`interventions`**, not forcings — they fire
   `transfer()` `at [dates]` and simply do not fire past them, so this
   forcing-OOB policy never touches them. F2 becomes loud (or an explicit opt-in
   for the step kinds).
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
scored correctly regardless of where dynamics start.

**The reset is the spine's `ResetWindow`, keyed at `cond_from` instead of at an
obs — not a new per-filter mechanism.** Framing it as N hand-rolled per-filter
edits would re-introduce the very drift the topology's `Stage::Reset` exists to
remove. Each cell still expresses it in its idiom (PF/IF2: a
propagate-to-`cond_from` prelude; the chain-binomial-stepping cells: a
`cum_flows` reset at the `cond_from` substep), and it must land in **every cell
that accumulates incidence flow** — `complete_data_loglik`, `csmc_as`,
`complete_data_loglik_grad` (the gradient mirror; missing it is a silent NUTS
bug), **and `correlated_pf`** (`correlated_pf.rs:521`, the cell the earlier
draft omitted — matching F1's omission). Clarity caveat: the "one long
propagation, no resample" mental model is **PF-specific** — `csmc_as` resamples
and ancestor-tracks every substep, so the per-particle `cum_flows[j]` reset
applies across the resampled set, not to a single path. The default is no
warm-up (`cond_from = t_start`), bit-identical to today. See the superseded
`2026-06-09-burnin-conditioning-window.md` §2/§6 for the full inference math.

The change versus that proposal is the _surface_: instead of a lone
`condition_from` scalar in fit.toml, the conditioning window is one of the run's
named intervals (§2), with `cond_to` (the lag-truncation back edge) as its
symmetric partner, and the whole window routed through the capability gate.

## 8. Operations — which intervals each reads

| Operation                   | Integration span D                                              | Conditions over | Forecast | Forcing constraint |
| --------------------------- | --------------------------------------------------------------- | --------------- | -------- | ------------------ |
| **simulate** (forward)      | `[t_start, out_end or --until]`                                 | —               | optional | Eⱼ ⊇ D             |
| **simulate --draws** (PPC)  | same as the fit's D, optionally + F                             | —               | optional | Eⱼ ⊇ D             |
| **fit**                     | `[t_start, max_k Wₖ.end]` (burn-in start → last obs)            | C ∩ Wₖ          | no       | Eⱼ ⊇ D             |
| **forecast** (PGAS)         | `[last data, F]` — continue from the stored latent end-state    | none            | yes      | Eⱼ ⊇ [.., F]       |
| **forecast** (MLE: IF2/ODE) | `[t_start, F]` — no stored end-state, so re-filter then project | none            | yes      | Eⱼ ⊇ [.., F]       |

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
supported-and-tested or hard-error — no silent third option (CLAUDE.md). The two
new checks attach at **different** seams, and conflating them is a mistake to
avoid: **conditioning** is per-stage inference config, so it gates at the
`(algorithm)` dispatch registry (`methods.rs::validate_combo` /
`resolve_obs_alignment`), _not_ the IR-scanning capability flags; the
**forcing-domain** check (Eⱼ) is on model data, so it _can_ ride the
`required_capabilities()` / `Simulate::capabilities()` path that already scans
the IR. Either way an unsupporting cell fails loudly with a message naming the
limitation, and the conditioning fix is unify-first, gate-only-the-floor (§7.1
pt 3).

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
- **P4 — Per-stream windows (NOT this proposal's to schedule freely).**
  First-class Wₖ and the per-stream-indexed flow reset (F6) are the data
  layer's + spine's work; **this phase is blocked on the spine shipping the
  per-stream `ResetWindow` and `TemporalKind`, which are 0% built today** — the
  same gate `observation-system.md`'s heavy migration tier waits on. It is
  listed here only for completeness; the owning staging is in those two
  proposals, and this proposal must not present it as free-standing.

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
