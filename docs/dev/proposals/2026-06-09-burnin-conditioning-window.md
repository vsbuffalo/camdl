# The conditioning boundary: a covariate-informed burn-in

- **Status:** Active. A fit often has covariate history (births, SIA/MCV
  campaigns, seasonal forcing) for a span _before_ the first case datum. A
  modeler should be able to run the model through that span so the boundary
  state is _derived_ from the covariates rather than estimated as a free
  parameter — that is the burn-in this proposal makes expressible. The
  `condition_from` surface and warm-up mechanics (§2–§4) are implemented on the
  `camdl fit` path (588a40e); they ride the shared hole/reset seam, so PF / IF2
  / PGAS / PMMH all receive the boundary reset through `BoundObs`.
  **Remaining:** the §6.8 guard escalation (W329 soft-warn → hard error on the
  incidence case), the doc updates (§8), and per-stream boundaries (§6.10,
  deferred).
- **Supersedes:** `2026-06-09-burnin-conditioning-dsl-surface.md` (the "no new
  keyword — early `simulate.from` + leading-window unweighting" surface) and
  `2026-05-30-conditioning-boundary-tcond.md` (the `t_cond` inference-math
  half). Both fix the bug by placing the conditioning boundary _at_ `first_obs`
  and rule out `condition_from < first_obs` as "meaningless (no data to
  condition on)." **That rule is wrong for incidence:** placing the boundary at
  `first_obs` resets the flow accumulator there, leaving the first scored bin
  empty, so the first observation is dropped. The boundary that _keeps_ it is
  exactly `first_obs − one cadence` — strictly _before_ `first_obs`. §2 and §4
  carry that correction; it is the load-bearing reason this proposal stands as a
  separate document rather than a patch to those two.
- **Issues:** gh#134 (its diagnostics W324/W325/W329 shipped — see §1; this is
  the substantive feature it pointed at).
- **Required reading before implementing:** the IC-free inference proposal
  `archive/pre-alpha/2026-04-18-ic-free-inference.md` (the adjacent
  propagate-without-scoring mechanism);
  `rust/crates/sim/src/inference/particle_filter.rs` (the obs loop) and
  `pgas.rs` (`complete_data_loglik`, `csmc_as`, and the `cum_flows` reset
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

- **W329** (`check_first_interval_window`, `util.rs:1029`) — fires when the
  leading gap `first_obs − t_start` exceeds `K = 5×` the modal observation
  cadence. (The doc-comment calls it the "first inter-observation interval," but
  the code compares the _gap_ to the cadence — `util.rs:1042` — so it already
  targets the gh#134 span, not the data spacing.) Today it is a pure soft
  warning that never rejects. §6.8 **escalates it to a hard error** for
  incidence streams — where the wide gap is the −3416 wrong-number — while
  keeping the soft warn for prevalence, where a wide gap is only free-running
  drift that the first datum corrects. Its message still points at the
  now-retired tcond proposal (`util.rs:1089`); §8 repoints it.
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
incidence flow is not scored.** Over `[t_start, cond_from)` the dynamics run
faithfully — births accumulate, campaigns deplete, seasonality forces, full
process noise — because that is the prior the covariates are shaping. What
changes at `cond_from` is exactly one thing:

> **Reset the incidence flow accumulator at `cond_from`.** The first scored
> window becomes `(cond_from, first_obs]` — one normal cadence — instead of
> `(t_start, first_obs]` — the whole gap. The leading span's flow is discarded
> because there is no incidence observation to score it against; everything else
> about the leading span (its state evolution, its process variance, in PGAS its
> transition-density contribution to the path prior) is retained.

This single operative change — a flow-accumulator reset at the boundary — is why
the design space has one coherent point rather than a menu. The leading span
_must_ run as the same stochastic process the conditioned phase uses, because
that is what makes its endpoint the correct Bayesian prior
`p(x_{cond_from} | θ, covariates)`. The tempting alternative — seed every
particle from a deterministic ODE-skeleton mean — is **incoherent** here and is
worth stating so review does not re-propose it. The chain-binomial process has
no real-valued skeleton beside it (`ChainBinomialProcess::initial_state` returns
integer counts; the PF path owns no `REAL_COMPARTMENTS` state). A single
deterministic trajectory seeds every particle identically → **zero process
variance** at `cond_from` → an overconfident cloud → near-degenerate effective
sample size at the very first weight. That is a silent statistical error, the
exact failure camdl exists to prevent. So the faithful stochastic warm-up is not
_a_ choice; it is _the_ choice.

**One mechanism, all algorithms — the boundary is a leading reset-only hole.**
The implementation adds no per-algorithm warm-up prelude. It reuses the
sparse-observation hole/reset seam: `condition_from` resolves to a model-time
`cond_from`, and a single **reset-only hole** is prepended to the shared
observation grid at `cond_from` — a row with a `None` cell (`BoundObs`). A hole
already has exactly the two properties the boundary needs: its grid time fires
the per-observation-index accumulator reset (the bin resets at `cond_from`),
while its `None` cell contributes no likelihood term (the warm-up flow is
discarded, not scored). Every algorithm iterates that grid through `BoundObs`,
so all of them — bootstrap PF, IF2, correlated-PF, PGAS, PMMH — receive the
boundary reset from the one prepended hole, with no algorithm-specific code:

- **Bootstrap PF / IF2 / correlated-PF** propagate from `t_start` to `cond_from`
  as the first window, score nothing (the hole), reset flows, then score the
  first real datum over `(cond_from, first_obs]` — the existing obs loop with
  one extra leading row.
- **PGAS** already propagates every substep from `t_start` and already resets
  `cum_flows` at obs-grid substeps; the leading hole adds one more such reset at
  `cond_from` and scores no term there (the `None` cell), in
  `complete_data_loglik`, `csmc_as`, and the gradient mirror. The warm-up
  substeps' transition densities `Σ_s log p(x_s | x_{s-1}, θ)` still enter the
  path prior, as they should.

PF and PGAS do **not** compute the same scalar log-likelihood (they never did —
PGAS scores the path prior, PF the predictive density); "parity" here means the
same first _scored incidence window_, not a matching number. The win of the
leading-hole approach is that it consolidates the boundary onto the seam every
cell already routes through, so the burn-in cannot be live in one algorithm and
silently absent in another — the "no silent gaps" matrix rule.

**The default is no warm-up.** When `condition_from` is unset, or resolves to
`t_start`, no hole is prepended — the grid is untouched and every filter is
bit-for-bit identical to today. The burn-in is opt-in. A default of `first_obs`
(the superseded proposals' `t_cond = first_obs`) would prepend the hole at
`first_obs` itself, making the first scored incidence window zero-width and
dropping the first datum on every no-gap model — the He, Ionides & King (2010)
London measles vignette, for instance, has `from = 0` with weekly data whose
first row is at `t = 7`, so its first window `(0, 7]` is _already_ exactly one
cadence and must stay scored. The bit-identical-default test (§7) is the guard,
and the run-identity must not change when the key is unset (§4).

## 3. Why this is localized, not a big lift

The boundary is a leading reset-only hole, and the hole/reset seam it rides
already exists (built for sparse/irregular observations). So the feature adds
**no new mechanism**:

- **The resolution `condition_from → cond_from`** happens once, in the
  fit-config layer (`fit/runner.rs`), where `t_start`, the model `origin`, the
  `dt` grid, and the loaded `first_obs` are all in hand
  (`resolve_condition_from`).
- **Prepending the hole** is a few inserts, gated on
  `cond_from ∈ (t_start, first_obs)`: one row on the canonical observation times
  every algorithm reads, and a `None` cell (with a dense-placeholder row whose
  value is unread) on each stream.
- **The reset and the no-score** are inherited from the hole seam — no change to
  `particle_filter.rs` / `if2.rs` / `pgas.rs` scoring or reset logic. The
  per-obs-index reset fires at the hole's grid time; the `None` cell scores no
  term. PGAS's `cum_flows` reset at obs-grid substeps already covers the leading
  hole's substep.
- **Grid alignment** is handled by `resolve_condition_from`, which rejects a
  `cond_from` off the `dt` step grid (§5) so the boundary lands on a substep
  boundary exactly, the same way obs times do.

There is **no new inference-config field** and no IR/schema change:
`condition_from` lives in `FitConfigV2` (top-level, §4), resolves to a leading
hole at the fit-config layer, and never reaches `SMCConfig` / `IF2Config` / the
PGAS config as a threaded `t_cond`. `cond_from == t_start` (or an unset key)
inserts nothing — bit-identical to today.

The shipped insertion (`fit/runner.rs`), after `resolve_condition_from` returns
`Some(cond_from)` with `cond_from ∈ (t_start, first_obs)`:

```rust
// Prepend the leading reset-only hole to the canonical times every
// algorithm reads. `cond_from == t_start` (or unset) inserts NOTHING →
// byte-identical to today.
observations.insert(0, Observation { time: cond_from, value: 0.0 });
// And to every stream: a hole cell (`None`) at cond_from, with a
// dense-placeholder Observation row whose value is unread (the cells, not
// `data`, are authoritative for scoring).
for s in &mut streams {
    s.data.insert(0, Observation { time: cond_from, value: 0.0 });
    s.cells.insert(0, None);
}
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
# ... no free initial_susceptible_fraction needed — the warm-up derives S(cond_from)
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
- Omitted — default `simulate.from` (`cond_from = t_start`), no warm-up.

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

**Why the boundary lies _strictly before_ `first_obs` — the correction to the
superseded proposals.** For a flow-accumulator (incidence) observation the
filter resets the accumulator _at_ the boundary and scores the first datum
against the flow accumulated from the boundary to `first_obs`. Place the
boundary _at_ `first_obs` — the default in both superseded proposals
(`t_cond = first_obs`) — and the accumulator resets at the same instant the
first datum is scored: the first bin is empty, the first observation cannot be
scored, and the first _scored_ datum silently becomes the second observation.
The boundary that _keeps_ `y₁`, scored against exactly one cadence `Δ` of flow,
is `first_obs − Δ`. So `condition_from < first_obs` is not "meaningless (no data
to condition on)" — it is the _only_ placement that retains the first
observation. Both superseded proposals rule out exactly the boundary that keeps
`y₁`. (For a _prevalence_ observation the question is moot:
`y₁ ~
g(state(first_obs))` reads the instantaneous state, with no accumulation,
so the boundary position does not change whether `y₁` is scored — §1.)

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
   substeps; no re-anchoring to `t_start`/`cond_from` is needed. The covariates
   are what _make_ the warm-up informative.
3. **IVP composition — stated correctly per algorithm.** The per-particle
   `Binomial(N, frac)` initial draw exists only in **PGAS** (`csmc_as`,
   `pgas.rs` ~ll. 1002–1006); the stochastic warm-up transports that spread to
   `cond_from` (a deterministic skeleton would collapse it — another reason the
   warm-up must be stochastic). In the **bootstrap PF and IF2**, `initial_state`
   is deterministic (every particle gets identical counts at `t_start`); their
   warm-up spread comes from accumulated chain-binomial process noise, and IVP
   there is a spread in the _parameter_ across particles, not in initial counts.
   The boundary-ensemble-variance test (§7) checks the right thing in each case,
   but the rationale differs. Guidance, not a hard rule: prefer a
   covariate-informed burn-in over a free initial-susceptible IVP for the
   _susceptible pool_ (covariates carry real information about it); keep IVP for
   genuinely unobserved initial compartments (`e0`/`i0`). A working burn-in
   _reduces_ the IVP burden — it derives the boundary state instead of
   estimating it.
4. **The degeneracy watchdog over the warm-up.** Because the warm-up is the
   first obs-loop window — the propagation `[t_start, cond_from)` that ends at
   the leading hole, not a separate prelude — the wall-clock watchdog (a single
   `Instant` started before the obs loop) already covers it; there is no special
   prelude to time. Two interactions remain. (a) That first window can be long
   (the whole pre-data gap), and the per-call wall-clock budget
   (`degeneracy.rs`, floor `WALLCLOCK_FLOOR_S = 120` + per-particle) scales with
   particle count, not interval length — so a long warm-up adds compute to the
   call budget and could trip a false `PFWallclockTimeout`. Production fits
   disable this watchdog (`pf_wallclock_disabled = true`, `fit/runner.rs`; the
   watchdog it gates is in `if2.rs`), so it only bites ad-hoc runs, where sizing
   the budget suffices. (b) The leading hole scores no term, so the filter
   passes through it at (near-)uniform weights (systematic resampling ≈
   identity), pushing one healthy ESS entry — it must not falsely advance the
   `ESS_COLLAPSE_WINDOWS = 3` collapse count (verify in §7). The
   **deterministic** substep budget (`iters`/`ITER_BUDGET`) is
   interval-length-aware and still bounds a pathological warm-up correctly.
5. **Decoherence on non-stationary spans (diagnostic, not a cap).** A long
   warm-up over a span with directional covariate drift and no restoring
   dynamics can let the ensemble spread without bound; the seasonal-endemic
   attractor bounds it, the general case does not. We do not cap it (a cap would
   distort the prior); we surface it as a `[warn]` when the ensemble
   coefficient-of-variation at `cond_from` exceeds a threshold, with a band so
   the number is actionable ("CV 1.4 at the boundary suggests an unbounded
   warm-up; check for directional covariate drift with no restoring dynamics").
   A hard cap is out of scope.
6. **PGAS gradient path and the `cum_flows` reset sites (highest-risk).** The
   leading hole's reset must reach `complete_data_loglik`, `csmc_as`, **and**
   `complete_data_loglik_grad`. All three iterate the `BoundObs` grid and reset
   `cum_flows` at obs-grid substeps, so the prepended hole is covered — but
   verify the gradient mirror reads the same cells: miss it and the posterior
   gradient is inconsistent with the value it differentiates, a silent NUTS bug.
   Ancestor sampling must not index into the warm-up span (no resampling
   happened there). This is the highest-risk surface; the §7 finite-difference
   test pins it.
7. **Composition with `ic_free` is rejected — loudly.** The shipped code does
   **not** compose the two: `condition_from` + `ic_free` is a hard error,
   "nothing to condition on" (`runner.rs`; test
   `condition_from_with_ic_free_errors_loudly`). The reason is mechanical and
   correct as a guard: `condition_from` makes obs-index 0 the leading reset-only
   hole, while `ic_free` conditions on the first observation (obs-index 0) to
   pin `x_0 | y₁` without accumulating its term — and a hole is not an
   observation, so there is nothing for `ic_free` to condition on. Rather than
   silently skipping an obs-index-0 that is already empty, the fit rejects. This
   is the same missing-first-obs (F1-class) guard that rejects `ic_free` when
   the real first datum is itself a hole. Whether a future variant should
   compose them — warm up to `cond_from`, then apply `ic_free` to the first
   _real_ datum at `first_obs` (obs-index 1) — is plausible but unbuilt; today
   the combination is a loud error, not a silent compose. The two address
   overlapping uncertainty anyway (§5): a covariate-informed burn-in _derives_
   the boundary state, reducing the need for the `ic_free` initial-state
   estimate.
8. **W329 escalates to a hard error for incidence (the no-silent-wrong-answer
   line).** When `simulate.from` sits a wide gap before `first_obs`, an
   incidence stream's first bin accumulates the whole gap and the opening
   log-likelihood term is the −3416 garbage of §1. A soft warning leaves that
   wrong number in the fit — and per CLAUDE.md, "warnings are noise an agent
   will suppress and a non-specialist will skim," while a silent wrong answer is
   the exact failure this software exists to prevent. So when the first
   incidence (`Interval`) window exceeds `K ×` the modal cadence **and
   `condition_from` is unset**, the fit is **rejected** with a hard error naming
   the gap, the cadence, and the two fixes:

   - set `condition_from = first_obs - 1 'week` to run the covariate-informed
     warm-up and score the first datum against one cadence (the right answer
     when the early origin is intentional); or
   - move `simulate.from` closer to the first datum (the right answer when the
     early origin was accidental).

   The **opt-out** is `condition_from` itself: setting it at all suppresses the
   guard, because the modeler has then engaged with the boundary and
   `resolve_condition_from` validates the value (§5). A modeler who genuinely
   wants the whole gap scored sets `condition_from` to a value resolving to
   `t_start` (no warm-up, gap scored) — making that rare choice explicit and
   auditable rather than an accident.

   The guard is **incidence-specific** via the `TemporalKind` classifier
   (`StreamProjection::temporal_kind() == Interval`): for a **prevalence**
   (`Instant`) stream a wide gap is not a wrong number — `y₁` reads the
   instantaneous state regardless of how far back `t_start` sits (§1), only
   free-running drift the first datum corrects — so it **remains a soft W329
   warn**, not an error. This resolves the pinned proposal's §4 "error +
   opt-out" decision (the opt-out is `condition_from`) and supersedes the
   `dsl-surface` proposal's "W329 stays a soft warn" stance for the incidence
   case. The message is repointed (§8).
9. **The symmetric end question is out of scope.** This proposal decouples where
   conditioning _starts_. Whether conditioning can _stop_ before `simulate.to`
   (a forecast-only tail) is the mirror question, named here only so it is not
   silently assumed solved.
10. **Per-stream boundaries are deferred.** Heterogeneous streams may each begin
    at a different time; a single global boundary is too coarse. Deferred to the
    observation-system surface, where streams are first-class. (`first_obs` is
    defined as the earliest obs time across streams.)

## 7. Implementation plan and tests

Most of the surface and mechanics landed with 588a40e; the remaining items are
step 4 (the guard escalation) and step 6 (docs). Order:

1. **Surface + resolution (landed).** `condition_from` on `FitConfigV2`
   (top-level, `#[serde(default, skip_serializing_if = "Option::is_none")]` for
   run-identity); `resolve_condition_from` (`runner.rs`) parses
   `first_obs - DUR` / `date()` / offset → `cond_from`, resolving through
   `origin`, the loaded `first_obs`, and the `dt` grid, and rejects
   `≥ first_obs`, `≤ t_start`, and off-grid values with the §5 messages.
   `--condition-from` CLI override.
2. **Boundary mechanics (landed).** Prepend the leading reset-only hole to the
   canonical times and each stream's `BoundObs` (`runner.rs`), gated on
   `cond_from ∈ (t_start, first_obs)`. No new inference-config field; no IR
   change. The hole/reset seam carries it to PF / IF2 / correlated-PF / PGAS /
   PMMH.
3. **Header echo (landed).** The conditioning-window line printed when a warm-up
   is active (§4).
4. **W329 escalation (remaining).** Gate `check_first_interval_window` on
   `condition_from.is_none()` and the canonical stream's `TemporalKind`: an
   `Interval` (incidence) stream over the threshold returns an `Err` (hard
   error, §6.8) instead of the soft-warn `eprintln`; an `Instant` stream keeps
   the soft warn. Repoint the message (§8).
5. **Decoherence diagnostic (remaining, optional).** Surface the boundary
   ensemble CV as a `[warn]` above a threshold (§6.5). Diagnostic only; lands
   separately.
6. **Docs (remaining).** §8.

Tests (red → green; paste red-then-green output in the commit per the TDD gate —
this is inference math, each is load-bearing). **Landed** (verified by name):

- **`unset_condition_from_is_unchanged`** +
  **`condition_from_at_t_start_inserts_nothing`** (`runner.rs`) — an unset key,
  or one resolving to `t_start`, inserts no hole; the grid is untouched. Guards
  the "default `first_obs`" trap (§2).
- **`interior_condition_from_inserts_leading_hole`** (`runner.rs`) — an interior
  `cond_from ∈ (t_start, first_obs)` prepends the reset-only hole to the times
  and each stream's cells.
- **`condition_from_changes_the_fit_identity_when_set`** +
  **`unset_condition_from_does_not_change_the_fit_identity`** (`cas.rs`) — the
  CAS no-collision pair: two fits differing only in `condition_from` get
  distinct `run_id`s; the key's absence leaves the existing `run_id`
  byte-identical (the `skip_serializing_if` guard).
- **`conditioning_boundary_resets_leading_incidence`** (`sparse_holes_reset.rs`)
  — the deterministic reset proof: a leading hole at `cond_from` makes the first
  scored incidence bin `(cond_from, first_obs]` one cadence, not the whole gap;
  mutation-verified (without the reset the bin is double-width).
- **`condition_from_with_ic_free_errors_loudly`** (`runner.rs`) — the §6.7 loud
  rejection: `condition_from` + `ic_free` trips the missing-y₁ guard.

**Remaining:**

- **W329 hard error** — an early `simulate.from`, an `Interval` stream, and no
  `condition_from` is rejected (red against today's soft-warn); the same model
  with `condition_from = first_obs - 1 'week` passes; a prevalence (`Instant`)
  stream with the same gap only warns. The `TemporalKind` gate is the
  discriminator.
- **PGAS gradient consistency (recommended)** — the leading hole's `cum_flows`
  reset reaches `complete_data_loglik_grad` (gradient consistent with value via
  finite-difference), pinning §6.6. The deterministic reset test covers the
  value path; this extends it to the gradient mirror.

Note: there is **no** test asserting PF and PGAS produce the same scalar loglik
under a warm-up — they score different objects (predictive density vs path
prior); such a test would be vacuous-or-wrong.

## 8. Documentation

The doc edits land **with** the implementation (they describe the shipped
behavior). Integration points and drafted text:

- **`docs/camdl-inference-spec.md`** — new section adjacent to IC-free, e.g.
  _§3.9 The conditioning boundary (covariate-informed burn-in)_, with the
  mechanism (faithful warm-up; a leading reset-only hole at the boundary
  `cond_from`; first scored window one cadence), the top-level `condition_from`
  surface, and the `first_obs - DUR` form. **While here, correct the §3.8
  IC-free example's `[fit]` table to a top-level key** (doc-vs-code drift —
  `FitConfigV2` is `deny_unknown_fields`, `ic_free` is top-level; the `[fit]`
  header would fail to deserialize).
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
- **`cli/src/util.rs` (W329, `check_first_interval_window` at l. 1029)** — two
  changes. (1) The §6.8 escalation: for an `Interval` (incidence) stream over
  the threshold with no `condition_from`, the caller (`fit/runner.rs`) raises
  this as a hard error rather than `eprintln`-ing it; the `Instant` case stays a
  soft warn. (2) The message's final sentence (l. 1089) dangles a pointer to the
  now-retired `2026-05-30-conditioning-boundary-tcond.md` — end-users do not
  have the `docs/dev/proposals/...` tree. Repoint it to a binary-portable
  target: "… if the long pre-data burn-in is intentional, set `condition_from`
  (`camdl
  docs fit-toml`)." The unit test at l. 1138 asserts
  `msg.contains("tcond.md")`, so it must change in the same commit or it fails
  the moment the message is repointed.
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
