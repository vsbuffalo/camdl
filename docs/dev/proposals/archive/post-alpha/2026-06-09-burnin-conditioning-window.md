# The conditioning boundary: a covariate-informed burn-in

- **Status:** SUPERSEDED (archived 2026-06-09) — replaced as a standalone design
  by a wider time-interval model. The placement of `condition_from` (model vs
  fit) turned out to depend on a larger question being mapped first: how the
  simulation interval, per-stream observation intervals, covariate-table
  domains, and the forecast horizon compose across the fit and
  simulate-into-the-future operations. Retained for its problem statement (§1,
  gh#134) and inference math (§2, §6); **do not implement the §4 surface
  as-is**. Superseded by `2026-06-09-time-interval-model.md`, which folds this
  conditioning inference math (§2, §6) into the conditioning window of the
  interval model and replaces the lone `condition_from` surface. Start from
  `2026-06-09-time-and-observation-overview.md`.
- **Supersedes:** `2026-05-30-conditioning-boundary-tcond.md` (the
  inference-math half; now marked superseded). This document is self-contained
  and owns the inference math, the surface, and the UX.
- **Issues:** gh#134 (closed; its diagnostics shipped — see §1). This is the
  remaining substantive feature it pointed at.
- **Required reading before implementing:** the IC-free inference proposal
  `archive/pre-alpha/2026-04-18-ic-free-inference.md` (the adjacent
  propagate-without-scoring mechanism);
  `rust/crates/sim/src/inference/particle_filter.rs` (the obs loop, ll. 218–420)
  and `pgas.rs` (`complete_data_loglik`, `csmc_as`, and the `cum_flows` reset
  sites); `docs/camdl-language-spec.md` §2.1 (units/time) and §7 (forcings);
  `docs/dates.md`. This touches inference math (`particle_filter.rs` / `if2.rs`
  / `pgas.rs` / `pgas_grad.rs`) — high-risk regardless of how mechanical it
  looks.

## 1. What exists today

A stochastic compartmental fit has two clocks that need not coincide: where the
**dynamics begin** and where the **data begins**. camdl already decouples them.
`simulate.from` sets the model origin `t_start` (`expander.ml:3576`,
`Ir.t_start`); the observation schedule is the data file's own time column. A
modeler can legitimately start dynamics in 2011 and have case data begin in
2014.

The particle filter conditions observation _k_ over the half-open window
`(t_{k-1}, t_k]`, with the first window's left edge being `t_start` by
convention. Concretely (`particle_filter.rs`): `t` initializes to
`config.t_start` (l. 141); the obs loop (l. 218) propagates every particle from
`t` to `obs_time(k)`, scores `y_k` against the propagated cloud, resamples, and
**resets the flow accumulators** for the next window (l. 415). So the first
scored window is `(t_start, first_obs]`, and for an **incidence** observation —
one projected from a flow accumulator, `incidence(S→I)` — that window's score is
the flow integrated over the _entire_ span from `t_start` to the first datum.

When `t_start` sits far behind the first datum, this is gh#134. Same data, fixed
parameters, only `simulate.from` moved (Kano measles, weekly notifications):

| `simulate.from`                              | first window       | loglik   |
| -------------------------------------------- | ------------------ | -------- |
| `date("2011-12-26")` (origin)                | `[0, 980]` = 980 d | −3416.31 |
| `date("2014-08-25")` (1 wk before first obs) | `[973, 980]` = 7 d | −3202.18 |

Two failures, both silent: the model **free-runs unconditioned** for ~3 years
(no observation pulls the cloud toward anything), and the **first incidence
window accumulates 980 days of flow** scored against a single weekly count, so
the opening likelihood term is off by orders of magnitude. The modeler's
workaround was to abandon the early origin and instead estimate a free
`initial_susceptible_fraction` — a real loss of rigor, because the covariates
(births accumulating susceptibles, MCV/SIA campaigns depleting them over
2011–2014) are genuinely _informative_ about `S(2014)` and that information is
thrown away.

This is purely an **incidence** problem. A **prevalence** observation (a state
snapshot at `first_obs`) is already scored correctly today regardless of how far
back `t_start` sits: `y_1 ~ g(state(first_obs))` reads the instantaneous state,
not a flow integral, so window length does not enter. The bug lives entirely in
the flow-accumulation path (`FlowSum`/`CumulativeFlow` projections).

Three pieces of the eventual fix are already in the tree:

- **W329** (`util.rs:970`, `check_first_interval_window`) — a soft warning that
  fires when the first inter-observation interval exceeds `K=5×` the modal
  cadence. It names the gap, the cadence, and the fix. This catches the
  _accidental_ version of gh#134; it never rejects a model.
- **`skip_first_obs_from_loglik`** (`SMCConfig`, l. 182) — the IC-free toggle
  (`archive/pre-alpha/2026-04-18-ic-free-inference.md`). The filter still
  weights and resamples at the first observation (that pins `x_0` given `y_1`)
  but does not accumulate that term into the returned log-likelihood. This is
  the existing precedent for "let the cloud pass through a window without that
  window contributing to the score."
- **IVP parameters** (`ivp = true`) — the current way to express "I don't know
  the initial state": PGAS draws `Binomial(N, frac)` per particle at the initial
  state and estimates `frac` as a free parameter. This is the machinery the
  gh#134 workaround fell back on.

## 2. The right behavior is singular

The covariate-informed burn-in the modeler wanted is not a new kind of model and
needs no new modeling language. It is one statement: **the leading span over
which there is no data is a warm-up — it simulates, but its accumulated
incidence flow is not scored.** Over `[t_start, t_cond)` the dynamics run
faithfully — births accumulate, campaigns deplete, seasonality forces, full
process noise — because that is the prior the covariates are shaping. What
changes at `t_cond` is exactly one thing:

> **Reset the incidence flow accumulator at `t_cond`.** The first scored window
> becomes `(t_cond, first_obs]` — one normal cadence — instead of
> `(t_start, first_obs]` — the whole gap. The leading span's flow is discarded
> because there is no incidence observation to score it against; everything else
> about the leading span (its state evolution, its process variance, in PGAS its
> transition-density contribution to the path prior) is retained.

This single operative change — a flow-accumulator reset at the boundary — is why
the design space has one coherent point rather than a menu. The leading span
_must_ run as the same stochastic process the conditioned phase uses, because
that is what makes its endpoint the correct Bayesian prior
`p(x_{t_cond} | θ, covariates)`. The tempting alternative — seed every particle
from a deterministic ODE-skeleton mean — is **incoherent** here and is worth
stating so review does not re-propose it. The chain-binomial process has no
real-valued skeleton beside it (`ChainBinomialProcess::initial_state` returns
integer counts; the PF path owns no `REAL_COMPARTMENTS` state). A single
deterministic trajectory seeds every particle identically → **zero process
variance** at `t_cond` → an overconfident cloud → near-degenerate effective
sample size at the very first weight. That is a silent statistical error, the
exact failure camdl exists to prevent. So the faithful stochastic warm-up is not
_a_ choice; it is _the_ choice.

Each filter expresses the same reset in its own idiom — the substrate is shared,
the algorithms stay distinct:

- **Bootstrap PF / IF2** (`particle_filter.rs`, `if2.rs`): there are no
  observations in `[t_start, t_cond)`, so there is nothing to weight or resample
  there anyway. Activating a warm-up means propagating the swarm from `t_start`
  to `t_cond` as a prelude, resetting flows, then entering the **unchanged** obs
  loop with `t` at `t_cond`. The loop's first window is then
  `(t_cond, first_obs]` automatically.
- **PGAS** (`pgas.rs`): there is no obs-loop preamble to add — the complete-data
  likelihood already propagates every substep from `t_start`, and the warm-up
  substeps' transition densities `Σ_s log p(x_s | x_{s-1}, θ)` _legitimately_
  enter the path prior (they always did, and should). The `t_cond`-specific
  change is to add a `cum_flows` reset at the substep whose end is `t_cond`,
  alongside the existing obs-keyed resets — in `complete_data_loglik`,
  `csmc_as`, and the gradient mirror `complete_data_loglik_grad`. PF and PGAS
  therefore do **not** compute the same scalar log-likelihood (they never did —
  PGAS scores the path prior, PF the predictive density); "parity" here means
  the same first _scored incidence window_, not a matching number.

**The default is no warm-up.** `t_cond` defaults to `t_start`, making the
warm-up span empty and every filter bit-for-bit identical to today. The burn-in
is opt-in. A default of `first_obs` (rather than `t_start`) would make the first
scored incidence window zero-width and break every no-gap model — the He,
Ionides & King (2010) London measles vignette, for instance, has `from = 0` with
weekly data whose first row is at `t = 7`, so its first window `(0, 7]` is
_already_ exactly one cadence and must stay scored. The bit-identical-default
test (§7) is the guard, and the run-identity must not change when the field is
unset (§4).

## 3. Why this is localized, not a big lift

The operative change is one flow-accumulator reset at a boundary, and most of
the plumbing already exists:

- **`config.t_start`** is on `SMCConfig` (l. 169) and the IF2/PGAS configs — the
  warm-up's start point is already threaded.
- **The PF/IF2 propagation** is the same parallel per-particle walk the obs loop
  already runs; the warm-up is that walk stopped at `t_cond` with a flow reset.
- **The PGAS substep loop** already runs from `t_start` and already resets
  `cum_flows` at obs substeps (`pgas.rs` ~ll. 819–843 in `complete_data_loglik`;
  ~ll. 1236–1251 in `csmc_as`). The change is one more gated reset at the
  `t_cond` substep, mirrored in the gradient path.
- **`t_cond` joins the schedule's boundary set** — a third "reason" alongside
  observation times and intervention/event effects — so every backend's substep
  walk lands on it exactly under both Snap and Exact obs-alignment. The modeler
  is not asked to align `t_cond` to the `dt` grid; the schedule does it (the
  same way it already lands exactly on obs times).

The new state is a **single optional field** — `t_cond: Option<f64>` (default
`None` ≡ `t_start`) — on `SMCConfig`, `IF2Config`, and the PGAS config. No IR or
schema change: this is inference config, not model IR; no `ir/VERSION` bump.
There are ~17 struct-literal construction sites for these configs in production
(`pfilter.rs`, `survey.rs`, `profile.rs`, `fit/mod.rs`, `fit/runner.rs`,
`fit/pgas.rs`, and `sim/inference/{pgas,if2,traits}.rs`) and ~29 more in tests —
~46 across the workspace (`rg 'SMCConfig \{|IF2Config \{|PGASConfig \{'`); each
takes `t_cond: None` (compiler-enforced, so an omission is a build error, not a
silent default). The resolution `condition_from → t_cond` happens once in the
fit-config layer, where `t_start` and the loaded obs times are both in hand.

PF/IF2 warm-up sketch (PGAS adds the boundary reset, not a preamble):

```rust
let mut t = config.t_start;

// Warm-up: propagate over [t_start, t_cond), no obs to score. Empty when
// t_cond == t_start (the default) → a no-op on existing models: zero RNG
// draws, schedule unchanged, t enters the loop at t_start → byte-identical.
if let Some(t_cond) = config.t_cond.filter(|&tc| tc > config.t_start) {
    propagate_swarm(&mut swarm, /* from */ t, /* to */ t_cond, dt, &schedule, ...)?;
    for s in &mut swarm.states { s.reset_flows(); }
    t = t_cond;                          // obs loop's first window is now (t_cond, first_obs]
    // wall-clock watchdog timer starts here, not at t_start (§6.4)
}

for obs_idx in 0..n_obs { /* unchanged */ }
```

## 4. The surface

`condition_from` is an **inference** choice — _where the likelihood starts_ —
not a property of the model. The same `.camdl` can be fit conditioning from 2011
or from 2014 depending on the question; baking the boundary into the model would
force two model files that differ only in inference setup. This is the IC-free
placement argument, and it lands `condition_from` in the same home: a
**top-level fit.toml key**, beside `ic_free` (not a `[fit]` table — there is
none; `FitConfigV2` is `deny_unknown_fields` and `ic_free` is top-level — and
not the `[config]` table, which holds grid/backend knobs like `dt`, `backend`,
`obs_alignment`; `condition_from` is a likelihood-factorization choice, not a
grid knob).

```toml
# fit.toml — top level, beside ic_free
model = { camdl = "kano_measles.camdl" }
condition_from = first_obs - 1 'week     # idiomatic: condition one cadence before the data

[data.observations]
cases = "data/kano_weekly.tsv"

[estimate]
beta  = { bounds = [5, 40] }
# ... no free initial_susceptible_fraction needed — the warm-up derives S(t_cond)
```

**Forms.** `condition_from` accepts three forms, resolved in the fit-config
layer against the loaded data:

- `condition_from = first_obs - 1 'week` — **the idiomatic form.** `first_obs`
  is a reserved read-only `Instant` in fit-config constant position (the
  earliest observation time across streams), analogous to `origin` in `.camdl`
  constant positions. The subtraction is an _Exact_ duration (`'days`/`'weeks`
  are translation-invariant; no `E321` Calendar-duration hazard, unlike
  `'months`/ `'years` — `docs/dates.md`). This expresses the intent ("condition
  one cadence before the data") without the modeler hand-computing a date — the
  same reason `add_calendar_months(d, 1)` beats hand arithmetic everywhere else.
- `condition_from = date("2014-08-18")` — the absolute escape hatch, resolving
  through `origin` exactly as `simulate.from` does.
- Omitted — default `simulate.from` (`t_cond = t_start`), no warm-up.

**CLI.** `--condition-from <DATE|first_obs-DUR>` overrides the fit.toml key, on
every subcommand that runs the filter (`camdl fit`, `camdl pfilter`,
`camdl profile`). Every behavior is expressible as a flag; fit.toml bundles, it
does not gate. (This is a deliberate departure from the IC-free proposal's "no
CLI flag" line, in keeping with the project's CLI-first ethos in CLAUDE.md;
inference knobs like `--seed`, `--particles`, `--stage`, `--init` are already
CLI-exposed.)

**Domain.** `condition_from ∈ (t_start, first_obs)`. A value `≥ first_obs` is
the discard-leading-observations case, which we deliberately do not support here
(§5). A value `≤ t_start` is meaningless. Both are hard errors (§5).

**Run identity (CAS).** `condition_from` changes the stored log-likelihood and
trajectory, so it **must** re-key the fit's `run_id` when set, and leave it
byte-identical when unset. Mirror `obs_alignment` precisely:
`#[serde(default, skip_serializing_if = "Option::is_none")]` on the fit-config
field, so the canonicalized fit-identity JSON is unchanged for existing fits and
two fits that differ only in `condition_from` get distinct `run_id`s. (A bare
`serde(default)` would serialize `null` and risk re-keying every existing fit —
the wrong attribute.) §7 pins this with a red→green no-collision test.

**Header echo (the legibility that carries the two-`from` split).** Whenever a
warm-up is active, the fit/`--dry-run` startup block echoes both origins, their
source files, and the resulting windows in calendar terms — this is the one
place `simulate.from` (in the `.camdl`) and `condition_from` (in fit.toml)
appear together, so it is what makes reasoning across the two files safe, not
optional polish:

```
conditioning:
  model origin       (kano_measles.camdl):  simulate.from  = 2011-12-26
  conditioning bound (kano.fit.toml):        condition_from = 2014-08-18
  → warm-up  [2011-12-26, 2014-08-18)  = 973 d ≈ 2.66 yr   unweighted (no score)
    first scored window  (2014-08-18, 2014-08-25] = 7 d    (modal cadence 7 d) ✓
```

Brackets follow §1's convention (warm-up half-open `[…)`, scored window
half-open `(…]`) so a careful modeler can verify the boundary instant is _not_
double-counted. The decoherence diagnostic (§6.5) is surfaced here only when it
is large — a bare ensemble-CV number on every run is noise a non-specialist
cannot act on; a `[warn]` above a CV threshold is actionable.

We do **not** auto-infer the boundary from the modal cadence. Silently choosing
`first_obs − cadence` would change the first-window semantics of any incidence
model whose first window is legitimately not one cadence — the no-silent-change
line. The `first_obs - 1 'week` form is the opposite of that: the modeler typed
the boundary _and_ the offset, so the cadence is their stated number, not a
guess. W329 should additionally print the paste-ready boundary date for the
modeler who prefers the absolute form.

## 5. What we deliberately drop: skipping over observations

The one thing `simulate.from` + a warm-up cannot express is starting
conditioning _after_ the first datum — deliberately discarding the first K
observations (e.g. "the first season's reporting is unreliable"). Earlier drafts
carried this as a `condition_from > first_obs` mode. **We drop it.** Discarding
leading observations is upstream data preparation, not an inference knob: the
modeler removes those rows from the data, which is transparent, auditable, and
re-runnable — the project's standard for data steps. Encoding "ignore rows 1..K"
inside the fit config hides a data decision in an inference setting, where it is
easy to forget why a fit's likelihood silently excludes the opening weeks.

So `condition_from` is constrained to `(t_start, first_obs)`, and a value at or
past the first datum is rejected with the valid range and the concrete remedy:

```
error: condition_from (2015-01-01) is at or after the first observation
       (2014-08-25). condition_from sets where conditioning *begins* over the
       no-data span before the first datum; it must lie in
       (2011-12-26, 2014-08-25) — strictly between the model origin and the
       first datum. To drop the first weeks of data, remove those rows upstream
       (`camdl data split <data> --at-time date("2014-09-01")`), so the decision
       is explicit and auditable.
```

The exact-equality case `condition_from = first_obs` gets a tailored hint rather
than the generic range message — that value _is_ the no-warm-up default, so the
fix is to omit the key, not to split the data:

```
error: condition_from (2014-08-25) equals the first observation. That is the
       default (no warm-up) — omit `condition_from` to get it. To condition one
       cadence before the data, write `condition_from = first_obs - 1 'week`.
```

and a value at or before the origin is rejected pointing at the file to fix:

```
error: condition_from (2010-06-01) is before the model origin / simulate.from
       (2011-12-26). Conditioning cannot begin before the dynamics exist.
       condition_from must lie in (2011-12-26, 2014-08-25). To start dynamics
       earlier, move simulate.from in the model file.
```

`camdl data split --at-time` currently takes internal time (`f64`); making it
accept a `date(...)` (auto-detected, as the data loader already does —
`docs/dates.md`) is a small follow-up so the §5 remedy is date-native rather
than asking the modeler to convert to internal time. Tracked separately; not a
blocker for this feature.

## 6. Things not to miss

The checklist that separates a correct warm-up from a plausible-looking wrong
one. The implementation should treat each as a gate.

1. **Interventions and events fire during the warm-up — including under PGAS.**
   The Kano SIA/MCV campaigns _are_ scheduled interventions in 2011–2014;
   depleting susceptibles is the entire point. The warm-up simulates, it does
   not skip — `at [...]` scheduled interventions and every-substep events fire
   exactly as they do in the conditioned phase, on every forward backend _and_
   in the PGAS producer/CSMC path. The shared `chain_binomial::step_one` that
   PGAS steps with (`pgas.rs:918` in `simulate_reference`, `:1146` in `csmc_as`)
   resolves fire-steps for **all** interventions (`resolve_fire_times`, no
   `always_active` filter — `compiled_model.rs`) and applies the scheduled ones
   through `due_effects → apply_post_advance` (`effects.rs:271–274`, `:457`).

   This corrects a stale belief worth stating so it is not re-inherited: gh#187
   ("PGAS silently ignores scheduled (non-`always_active`) interventions")
   described _pre-refactor_ code — its named mechanism `inject_event_deltas` has
   since been deleted and replaced by the unified `due_effects` seam, which
   fixed the gap. Verified empirically on the current tree:

   ```
   cargo test -p sim --test gh187_pgas_scheduled_intervention
   ```

   drives the PGAS producer on a fixture with a coincident always-active event
   `add(A, 100)` and a scheduled `transfer(0.5·A, A→B)` at `t=5` (init `A=50`,
   drain rate `k=0` so counts are RNG-independent). The producer's latent
   trajectory reads **`A=75, B=75`** after `t=5` — the scheduled transfer of
   `floor(150·0.5)=75` fired on the post-event `A=150`. The bug signature would
   be **`A=150, B=0`** (scheduled transfer skipped); a discrimination probe that
   flips the assertion to demand that signature fails, so the test is not
   vacuous. The producer is the path gh#187 named; `csmc_as` runs the identical
   `resolve_fire_steps` + `step_one`, so it is code-identical. **Action:**
   re-verify and close gh#187, and correct the now-false parenthetical comment
   at `pgas.rs:1521–1522` that still asserts the gap.

   The one genuinely-remaining PGAS intervention restriction is narrow and
   _loud_, not silent: `obs_alignment = "exact"` combined with **always-active
   events** is rejected with a hard error (`pgas.rs:1523–1532`), because such
   events key their firing on `round(t/dt)` and a shortened exact substep shifts
   that key. Scheduled `at [...]` interventions are unaffected under both Snap
   and Exact. So a covariate-informed warm-up driven by scheduled SIA campaigns
   works on PGAS today; only a warm-up relying on always-active _events_ under
   exact obs-alignment meets the (loud) restriction.
2. **Forcings and covariates apply in the warm-up.** Forcing tables index on
   absolute time `t`, and the warm-up propagates over the same absolute-`t`
   substeps; no re-anchoring to `t_start`/`t_cond` is needed. The covariates are
   what _make_ the warm-up informative.
3. **IVP composition — stated correctly per algorithm.** The per-particle
   `Binomial(N, frac)` initial draw exists only in **PGAS** (`csmc_as`,
   `pgas.rs` ~ll. 1002–1006); the stochastic warm-up transports that spread to
   `t_cond` (a deterministic skeleton would collapse it — another reason the
   warm-up must be stochastic). In the **bootstrap PF and IF2**, `initial_state`
   is deterministic (every particle gets identical counts at `t_start`); their
   warm-up spread comes from accumulated chain-binomial process noise, and IVP
   there is a spread in the _parameter_ across particles, not in initial counts.
   The `warmup_ensemble_has_nonzero_variance_at_tcond` test (§7) checks the
   right thing in each case, but the rationale differs. Guidance, not a hard
   rule: prefer a covariate-informed burn-in over a free initial-susceptible IVP
   for the _susceptible pool_ (covariates carry real information about it); keep
   IVP for genuinely unobserved initial compartments (`e0`/`i0`). A working
   burn-in _reduces_ the IVP burden — it derives the boundary state instead of
   estimating it.
4. **The degeneracy watchdog over the warm-up.** The warm-up is one long
   propagation with no resample, which collides with the gh#110/gh#133 watchdog:
   the per-call wall-clock budget (`degeneracy.rs`, floor
   `WALLCLOCK_FLOOR_S =
   120` + per-particle) scales with particle count but
   not interval length, and the warm-up pushes no ESS entry. For the **PF**, the
   timer is a single local `Instant` started just before the obs loop
   (`particle_filter.rs:168`) — start it at `t_cond` instead (clean). For
   **IF2**, the timer spans all iterations by design (`if2.rs:292`) and
   `Instant` cannot pause; the warm-up runs inside each iteration, so warm-up
   wall-clock simply counts against the run budget — and note that production
   fits disable the wall-clock watchdog anyway (`pf_wallclock_disabled = true`,
   set in `fit/runner.rs:420` and `:493`; the watchdog it gates is
   `if2.rs:295`), so this only bites ad-hoc IF2, for which sizing the budget is
   sufficient. The **deterministic** substep budget (`iters`/`ITER_BUDGET`, the
   compute-blowup guard) still counts warm-up substeps — it is
   interval-length-aware and parallel-invariant, so it correctly bounds a
   pathological warm-up. The warm-up contributes no ESS-trace entry, so it must
   not advance the `ESS_COLLAPSE_WINDOWS = 3` count.
5. **Decoherence on non-stationary spans (diagnostic, not a cap).** A long
   warm-up over a span with directional covariate drift and no restoring
   dynamics can let the ensemble spread without bound; the seasonal-endemic
   attractor bounds it, the general case does not. We do not cap it (a cap would
   distort the prior); we surface it as a `[warn]` when the ensemble
   coefficient-of-variation at `t_cond` exceeds a threshold, with a band so the
   number is actionable ("CV 1.4 at the boundary suggests an unbounded warm-up;
   check for directional covariate drift with no restoring dynamics"). A hard
   cap is out of scope.
6. **PGAS gradient path and the `cum_flows` reset sites (highest-risk).** The
   `t_cond` flow reset must land in `complete_data_loglik`, `csmc_as`, **and**
   `complete_data_loglik_grad` — miss the gradient mirror and the posterior
   gradient is inconsistent with the value it differentiates, a silent NUTS bug.
   Ancestor sampling must not index into the warm-up span (no resampling
   happened there). This is where the code review spent its time and where the
   implementation should too.
7. **Composition with `ic_free`.** Both may be set: the warm-up runs
   `[t_start, t_cond)`, then the first scored obs at/after `t_cond` is
   reweighted-but-not-accumulated (IC-free). The IC-free precondition (an
   estimated `ivp` param for `x_0` spread) **still applies** — the warm-up
   supplies _process-noise_ spread, but IC-free pins `x_0 | y_1` and still wants
   an estimated initial-state parameter; they are complementary, not
   substitutes. State this in validation so a modeler does not assume the
   warm-up satisfies the IC-free precondition.
8. **W329 stays a soft warning.** The earlier draft proposed escalating the
   first-interval guard to a hard error + opt-out, on the grounds that "there is
   no way to express the intentional case." There now is one — `condition_from`.
   So the end state is: soft W329 warn for the **accidental** case (early
   `simulate.from`, no `condition_from`), and `condition_from` making the
   **intentional** case correct. The W329 message is repointed (§8).
9. **The symmetric end question is out of scope.** This proposal decouples where
   conditioning _starts_. Whether conditioning can _stop_ before `simulate.to`
   (a forecast-only tail) is the mirror question, named here only so it is not
   silently assumed solved.
10. **Per-stream boundaries are deferred.** Heterogeneous streams may each begin
    at a different time; a single global `t_cond` is too coarse. Deferred to the
    unified-observation-data surface, where streams are first-class.
    (`first_obs` is defined as the earliest obs time across streams.)

## 7. Implementation plan and tests

One push for PF/IF2 + surface; a second for PGAS (incl. the gradient path) if it
wants isolation. Order:

1. **Config field.** `t_cond: Option<f64>` on `SMCConfig` / `IF2Config` / PGAS
   config; `t_cond: None` at all ~46 struct-literal sites (~17 production, the
   rest in tests). No IR/schema change.
2. **Warm-up mechanics.** PF/IF2: propagate-to-`t_cond` prelude + flow reset,
   obs loop untouched. PGAS: `cum_flows` reset at the `t_cond` substep in
   `complete_data_loglik`, `csmc_as`, `complete_data_loglik_grad`. `t_cond`
   added to the schedule boundary set. Wall-clock per §6.4.
3. **Surface + validation.** Parse `condition_from` (`first_obs - DUR` /
   `date()` / offset) → `t_cond` in the fit-config layer and the
   `--condition-from` path; resolve through `origin` and the loaded `first_obs`;
   reject `≥ first_obs` and `≤ t_start` with the §5 messages.
   `#[serde(default, skip_serializing_if]` for run-identity.
4. **Header echo + decoherence warn** (§4, §6.5).
5. **Docs** (§8).

Tests (red → green; paste red-then-green output in the commit per the TDD gate —
this is inference math, each is load-bearing):

- **`condition_from_default_is_bit_identical`** — a data-from-`t_start` model
  (no warm-up) gives byte-identical loglik and trajectory with `t_cond = None`
  vs the absence of the field. Guards the "default `first_obs`" trap (§2).
- **`condition_from_changes_run_id`** /
  **`condition_from_none_run_id_unchanged`** — the CAS no-collision pair: two
  fits differing only in `condition_from` get distinct `run_id`s; the field's
  absence leaves the existing `run_id` byte-identical (the `skip_serializing_if`
  guard).
- **`burnin_first_window_is_one_cadence`** — the Kano-shaped repro: early
  `simulate.from` + `condition_from = first_obs - 1 'week` yields a first scored
  window equal to the modal cadence and a loglik near the `−3202` neighborhood
  (the workaround's value), _without_ moving `simulate.from`.
- **`warmup_ensemble_has_nonzero_variance_at_tcond`** — the negative control
  against the degenerate-seed failure: the spread of compartment counts across
  particles at `t_cond` is strictly positive (PF/IF2: from process noise; PGAS:
  the per-particle `Binomial` init is transported, not collapsed).
- **`pgas_cum_flows_reset_at_tcond`** — the first scored incidence window in
  PGAS (value path) integrates flow over `(t_cond, first_obs]`, not
  `(t_start, first_obs]`; and `complete_data_loglik_grad` resets at the same
  substep (gradient consistent with value via finite-difference check).
- **`condition_from_domain_errors`** — `≥ first_obs` and `≤ t_start` rejected
  with the §5 messages and the valid range.

Note: there is **no** test asserting PF and PGAS produce the same scalar loglik
under a warm-up — they score different objects (predictive density vs path
prior); such a test would be vacuous-or-wrong.

## 8. Documentation

The doc edits land **with** the implementation (they describe the shipped
behavior). Integration points and drafted text:

- **`docs/camdl-inference-spec.md`** — new section adjacent to IC-free, e.g.
  _§3.9 The conditioning boundary (covariate-informed burn-in)_, with the
  mechanism (faithful warm-up; flow reset at `t_cond`; first scored window one
  cadence), the top-level `condition_from` surface, and the `first_obs - DUR`
  form. **While here, correct the §3.8 IC-free example's `[fit]` table to a
  top-level key** (doc-vs-code drift — `FitConfigV2` is `deny_unknown_fields`,
  `ic_free` is top-level; the `[fit]` header would fail to deserialize).
- **`docs/camdl-run-spec.md` §6.2** — add `condition_from: Option<Date|Offset>`
  to the `FitConfig` type. Note: §6.2 is independently stale (missing `ic_free`,
  `config`, `scenario`, `fit_starts`, `simplex_groups` relative to the real
  `FitConfigV2`); this adds one field but a full §6.2 resync is a separate
  doc-vs-code fix, flagged not silently appended-to.
- **`docs/book/.../inference/guide.md`** — the "Incidence observations and the
  model origin" subsection already teaches the gh#134 failure mode and three
  remedies (drop the row / shift times / move the origin). `condition_from` is
  the **fourth remedy** and belongs right there, where the modeler is already
  reading about the problem.
- **`camdl docs` topic** — add `condition_from` to the `fit-toml` topic (or a
  short `conditioning` topic) so an agent on any binary, and a modeler who just
  saw W329 fire, can discover it without the dev-proposal path.
- **`cli/src/util.rs` (W329 message, `check_first_interval_window` at l. 970)**
  — the message text is built at ll. 1020–1030 and its final sentence dangles a
  pointer at l. 1030 to the now-retired
  `2026-05-30-conditioning-boundary-tcond.md`. Repoint it to a binary-portable
  target: "… if the long pre-data burn-in is intentional, set `condition_from`
  to begin conditioning at the first datum (`camdl docs fit-toml`)." End-users
  do not have the `docs/dev/proposals/...` tree, so the current pointer dangles
  for them. The unit test at l. 1079 asserts `msg.contains("tcond.md")`, so it
  must be updated in the same change or it fails the moment the message is
  repointed.
- **`docs/dev/warning-catalog.md`** — repoint its W329 entry's proposal link
  likewise.
- **`docs/language-changes.md`** — not applicable; this adds a fit-config field
  and a CLI flag, not a breaking DSL change.

## 9. References

- gh#134 (Kano measles repro, §1; closed — diagnostics W324/W325/W329 shipped).
- He, Ionides & King (2010), _J. R. Soc. Interface_ 7:271–283 — the London
  measles vignette (`he2010_*` test cases); grounds the bit-identical-default
  case (§2).
- `archive/pre-alpha/2026-04-18-ic-free-inference.md` —
  `skip_first_obs_from_loglik`, the adjacent propagate-without-scoring
  mechanism, and the fit.toml-not-CLI placement argument this feature adapts.
- gh#133 / gh#110 — the PF degeneracy / wall-clock watchdog split (§6.4).
- gh#187 — "PGAS silently ignores scheduled interventions"; describes
  pre-refactor code and no longer reproduces on the current tree — candidate for
  closure (§6.1, with repro evidence).
- `docs/dates.md` — Exact vs Calendar durations (the `first_obs - 1 'week` form
  is the well-behaved Exact case); `origin` as a reserved identifier (the
  precedent for `first_obs` in fit-config constant position).
