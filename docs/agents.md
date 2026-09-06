# Agents using camdl

Orientation for an AI coding agent helping someone build and fit models with
camdl. Read it once at the start of a session — it's denser than a tutorial and
covers what you can't infer from a `.camdl` file alone. Surfaced offline and
version-matched via `camdl docs agents`.

This is about _using_ camdl — writing models, fitting them, reading diagnostics.
It is **not** about developing camdl itself (the OCaml compiler, the Rust
runtime); that lives in the source tree's `CLAUDE.md` and you do not need it to
build models.

---

## What camdl is

A **DSL plus runtime for stochastic compartmental epidemic models**. You write
the math (compartments, transitions, rate laws, observations); a compiler
expands it into a flat IR; a runtime simulates and fits.

Think of it as the lineage of:

| Tool      | Compiler-DSL? | Stochastic? | Inference?           | Closest analogue                              |
| --------- | ------------- | ----------- | -------------------- | --------------------------------------------- |
| **Stan**  | Yes           | Latent      | NUTS HMC             | Probabilistic-programming DSL with autodiff   |
| **odin**  | Yes           | ODE-only    | External fitting     | Compartmental ODE DSL in R                    |
| **pomp**  | No (R+C)      | Yes         | IF2, PMMH, particle  | Hand-coded SSA + obs models in R              |
| **camdl** | Yes (OCaml)   | Yes         | IF2, PGAS+NUTS, PMMH | DSL + stochastic runtime + autodiff inference |

Models in pretraining data: **lots of Stan and pomp, very little camdl.** When
in doubt, analogize from pomp (closest in problem domain) or Stan (closest in
DSL philosophy), then verify against the camdl spec.

---

## Mental model in one paragraph

A camdl model is a flat declaration: `compartments { ... }`,
`transitions { ... }`, `observations { ... }`, optionally
`dimensions { ... } + stratify(...)` for expansion,
`interventions { ... } / events { ... }` for scheduled state changes. The OCaml
compiler (`camdlc`) reads the `.camdl` file, dim-checks every expression,
expands stratification at compile time, emits source-to-source gradients for
every rate, and serialises the result as a versioned IR JSON envelope. The Rust
runtime (`camdl`) consumes that IR and runs simulation backends (Gillespie,
chain-binomial, ODE) plus inference algorithms (particle filter, IF2, PGAS+NUTS,
PMMH). Parameter values are supplied at runtime — the model file is
parameter-free.

---

## Canonical workflow

`camdl docs workflow` is the authoritative, command-verified runbook — read it
for the full sequence, the diagnostics table, and the guardrails. In one line:

```
check -> simulate (sanity) -> survey -> write fit.toml -> fit run
      -> fit summary -> read diagnostics -> refine priors
      -> fit predict (predicted-vs-observed) -> validate
```

The predicted-vs-observed step is **one verb, not a hand-rolled pipeline**:
`camdl fit predict --fit fit.toml --stream <name>` resolves the fit's posterior
draws and writes a tidy `predictive/<stream>.tsv` (`q05…q95` ribbon + typed
`horizon`/`treatment` columns, the fit's `fit_rhat_max`/`fit_ess_min` stamp, and
the per-row `rhat_mean`/`rhat_pred` convergence channels — act on `rhat_mean`,
which observation noise cannot dilute) and `observed/<stream>.tsv` under the run
directory. Join the two on `(time, <dims>)` and plot — do not glob the run store
for `trace.tsv`, re-inject fixed params, shell `simulate --draws`, and
pivot/quantile by hand; that reconstruction is exactly what this verb owns. Omit
`--horizon` for all applicable horizons (chain-binomial → `free_forward` +
`one_step`; ODE → `free_forward`).

Two reflexes it drills (and you should not import elsewhere): a failing
convergence gate is _information_, not a thing to tune away; and `Â` (IF2
chain-agreement) is not `R̂` (posterior mixing). The _why_ behind the procedure —
identifiability, what priors are for — is `camdl docs concepts`.

---

## When to stop and ask the human

camdl outputs feed real public-health decisions. The asymmetry matters: a fit
that takes an extra day because the agent paused for confirmation costs roughly
nothing; a posterior that's silently miscalibrated because the agent bypassed an
error costs much more. Default to pausing.

**Always pause and ask before:**

- **Reaching for an escape-hatch flag** (`--allow-degenerate-rates`,
  `CAMDL_SKIP_VERSION_CHECK=1`, `--no-nuts`, `--force` on a fit re-run). Each of
  these bypasses a check that exists for a reason. If a flag is the obvious fix
  to make an error go away, that's the signal to stop.
- **Loosening a convergence gate** because scout failed it. The gate exists to
  fail loudly rather than pass a bad fit through. The right move when scout's
  gate fires is to diagnose _why_ (widen bounds? more chains? more iterations?),
  not lower the threshold.
- **Choosing prior shape for a parameter you don't have domain context for.**
  Picking `Normal(0, 1)` "to make PGAS run" is the worst-case communication
  failure the audit's C4 fix was designed to prevent. Priors show up in the
  posterior. If the model author's prior intent isn't documented, ask.
- **Anything that publishes / shares a fit hash** as a result. Before a fit's
  output goes into a paper, brief, or policy artefact, a human should sign off
  on the diagnostics, the priors, and the model assumptions.

**Flag and proceed (don't block, but surface):**

- **Diagnostics fired in `camdl fit summary`.** Report which fired (R̂,
  ParamNearBound, DivergentTransitions, etc.), with the interpretation from the
  diagnostics table below. Don't decide unilaterally that "R̂ = 1.12 is fine."
- **`degenerate_step_count > 0`** in the eval-stats summary. Even with
  per-particle recovery handling it, the user should see the count and decide
  whether the model needs a `Cond` guard.
- **Profile likelihood non-monotonicity** or wide CIs from `camdl profile`.
  Identifiability problems are model-design issues, not fitting bugs.
- **`camdl survey` results.** The HTML pair-plot is a visual artefact
  (parameter-pair scatter coloured by loglik). Agent vision is unreliable on
  scatter geometry — what looks like a "clear basin" or a "ridge" or "bound
  pinning" to an agent is often partially wrong about location, extent, or
  whether multiple basins are present. Surface the rendered HTML path and a
  one-line summary ("survey rendered to `survey.html`; my read is X but please
  confirm before I seed scout"), don't act on the survey unilaterally. The
  numerical TSV next to the HTML is reliable for argmax-loglik points; the
  geometry interpretation is not.
- **External-oracle disagreements**: if a model is meant to reproduce a
  published result (He et al. 2010 measles, K-McK final-size, etc.) and the
  numbers don't match within the expected tolerance, surface immediately rather
  than tweaking until they match.

When a bug looks like a camdl defect (not a modeling mistake) and you can't
resolve it, **package a minimal reproducible example** for the maintainer rather
than guessing: `camdl mre fit fit.toml` bundles the model, its compile-time
`read()` files, the data, and the config into one tarball. See `camdl docs mre`
(includes `--no-data` for sensitive data).

**Safe to do autonomously:**

- Run `camdl check`, `camdl simulate`, `camdl survey`, `camdl pfilter`,
  `camdl fit run` (single stage, not committing the fit dir),
  `camdl fit
  summary`, `camdl fit predict`, `camdl list`, `camdl show`,
  `camdl fit diff`.
- Edit a `.camdl` model file in response to a compile error from the error table
  (typo fix, missing declaration, dim-correction).
- Widen the search range in response to `ParamNearBound`: relax a _narrowed_
  `fit.toml` bound back toward the model's declared range, or widen the declared
  bounds in the **model** itself — a `fit.toml` bound may only narrow the model,
  never widen past it (see the idiom below). Add a missing `[estimate.X.prior]`
  block (asking what shape the user wants if not obvious).
- Add `Cond` guards to rate expressions in response to
  `NumericalCollapse{DivByZero}` (this is the _correct_ fix, not a bypass).
- Run `make build && make install` after pulling.

**The general principle:** agents are good at running the workflow; humans are
needed for _modeling decisions_ and _interpreting calibration_. Calibration is
the half of compartmental modelling that's actually identifiability and
prior-belief judgement, not engineering. Don't pretend otherwise to "make
progress."

---

## Error → cause → fix table

Compile-time errors (from `camdlc`):

> If a construct looks correct but still errors — especially a bare `E001`
> syntax error on a forcing, a block keyword, or a likelihood — the **DSL may
> have changed since the model (or doc) was written**. Check
> `camdl docs
> language-changes` for the migration (old → new) before "fixing"
> the model.

| Code   | What it says                                                  | What it usually means                                                 | What to do                                                                                                                   |
| ------ | ------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `E100` | undeclared name 'X'                                           | Typo, or use of a name not declared in compartments/parameters/let    | Add the declaration. Don't introduce a new symbol just to make the error go away.                                            |
| `E107` | ambiguous unit literal after '/'                              | `20 / 100_000 'per_year` — unit binds to the adjacent number          | Parenthesise: `(20 / 100_000) 'per_year`, or pre-compute a single literal.                                                   |
| `E300` | transition rate has wrong dimension                           | Per-capita rate where population-level was needed (or vice versa)     | The rate must have dim `P·T⁻¹` (population per time). If you have a per-capita rate `T⁻¹`, multiply by `S` (the source pop). |
| `E302` | dimension mismatch in addition                                | Adding incompatible quantities                                        | Check units of both sides; usually a missing `* N` or `/ N`.                                                                 |
| `E303` | conflicting dimensions for parameter X                        | Same parameter inferred to be different dims in different transitions | Pick the right dim for the parameter and fix the transition that's wrong.                                                    |
| `L401` | rate expression `(1 - exp(-rate * 1 'days))` not dt-invariant | Discretization-correction shape that's only correct at dt=1 day       | Use the `dt` primitive: `(1 - exp(-rate * dt))/dt` — invariant across integrator steps.                                      |

Run-time errors from `camdl simulate` / `pfilter` / `fit`:

| Error                                                               | What it usually means                                                                                           | What to do                                                                                                                                                                                                                       |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `IR version mismatch`                                               | Stale `camdlc` binary vs `camdl` binary. The IR envelope's `ir_version` doesn't match what the runtime expects. | `make build && make install`. The runtime checks the on-PATH `camdlc` hash against its own.                                                                                                                                      |
| `SimError::NumericalCollapse { kind: DivByZero }`                   | A rate expression hit `0/0` or similar (e.g. `beta * I[a] / N_local[a]` when stratum `a` is empty)              | Add a `Cond` guard: `cond(N_local[a] > 0, beta * I[a] / N_local[a], 0)`. **Do not** reach for `--allow-degenerate-rates` unless you've decided the silent-zero is the modeling intent.                                           |
| `SimError::NumericalCollapse { kind: PowNanInf / SqrtNegative }`    | Negative base raised to fractional power, or sqrt of negative                                                   | Domain bug in the rate expression. Add a guard or fix the formula.                                                                                                                                                               |
| `SimError::NegativeCount { cause: BinomialOvershoot }`              | Binomial split overshot (rate × dt → 1 for some particle). Common in inference exploration                      | If during `simulate`: reduce `--dt`. If during `fit`: per-particle recovery handles it (the offending particle gets `−Inf` log-likelihood and is killed in resampling). Watch the `eval-stats` summary for how often this fires. |
| `SimError::NegativeCount { cause: InterventionAddNegative }`        | An `Action::Add` expression resolved to a negative value                                                        | Config bug. There's no inference scenario where `Add` should remove individuals. Fix the expression or use `transfer` instead of `add`.                                                                                          |
| `requires capabilities: BALANCE` (on gillespie/ode)                 | Model uses `balance { ... }`; only chain-binomial supports it                                                   | Use `--backend chain_binomial`. Don't try to translate `balance` to a manual transition — its semantics are chain-binomial-specific (the residual-compartment fix).                                                              |
| `--record-prequential requires --stage <pfilter-stage>`             | Flag used with a non-PFilter stage                                                                              | Pass `--stage` with a PFilter stage from your fit.toml. The error message lists available PFilter stages.                                                                                                                        |
| `pgas refuses to run with implicit improper-uniform priors`         | `[estimate.X]` block exists with no `[estimate.X.prior]`                                                        | Add an explicit prior. For uniform-on-bounds: `prior = { uniform = { lower = ..., upper = ... } }`. **Do not** add a wide normal "to make it shut up" — the prior shows up in the posterior.                                     |
| `PGAS gradient does not yet include obs-likelihood ... derivatives` | Estimating `rho`, `psi`, `k`, or any param appearing in the obs-likelihood / overdispersion expression          | Move that param to fixed (`[fixed.rho] value = ...`) and either grid-search it, or fit it with IF2 first (gradient-free). Full obs-likelihood gradient threading is on the roadmap (audit C1 follow-up).                         |
| `IR JSON parse error: missing field 'ir_version'`                   | Loading a bare-Model JSON; runtime requires the envelope wrapper                                                | Re-emit with `camdlc` (it always wraps). For hand-curated JSON: wrap with `jq '{ir_version: "<match ir/VERSION>", validated_by: "manual", model: .}' in.json > out.json` — the version must match your binary's schema.          |

---

## Diagnostics → interpretation table

After `camdl fit run`, `camdl fit summary <fit-dir>` shows diagnostics. They're
not all "the fit failed"; they're typed signals.

| Diagnostic                   | Threshold                           | What it means                                                                           | What to do                                                                                                                                         |
| ---------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RhatHigh`                   | R̂ > 1.1 (warn), > 1.5 (error)       | Chains haven't agreed on this parameter's posterior                                     | More sweeps; check for multimodality with `survey`. R̂ > 1.5 → almost certainly a real basin problem.                                               |
| `LowESSAtMLE`                | ESS < 5% × n_particles              | Particle filter is struggling at the point estimate. Loglik estimate has wide variance. | Increase `n_particles` in the validate stage, or check for model misspecification at the MLE.                                                      |
| `ParamNearBound`             | within 1% of natural-scale bound    | Posterior pile-up at a bound                                                            | Almost always: widen the bound. The data is telling you the parameter wants to live outside your prior support.                                    |
| `DivergentTransitions`       | any post-burn-in divergence         | NUTS hit a divergent trajectory (high curvature in posterior geometry)                  | Reparameterise (log/logit transforms), shrink the step size, or check for funnel geometry. Stan-canonical: any post-burn divergence is suspicious. |
| `MaxTreeDepthHits`           | > 5% of post-burn-in sweeps         | NUTS trees not finishing — step size too small or posterior too elongated               | Increase `max_tree_depth`, or reparameterise.                                                                                                      |
| `LowSwapRate`                | adjacent-rung pair < 10%            | Tempering ladder too sparse — chains don't mix across rungs                             | Add intermediate β values to the ladder.                                                                                                           |
| `DegenerateAncestorSampling` | > 10% of post-burn-in CSMC substeps | Reference trajectory too far from particle cloud                                        | More particles, or smaller PGAS proposal SDs (let scout run longer first).                                                                         |
| `LowTrajectoryRenewal`       | mean post-burn renewal < 10%        | PGAS reference trajectory not getting refreshed — possibly stuck                        | More particles. Check that CSMC is actually proposing diverse trajectories (run with `RUST_LOG=camdl_sim::inference::pgas=debug`).                 |
| `MultimodalLikelihood`       | ll spread > 50 nats with R̂ > 1.5    | Different chains are in different basins                                                | Run `camdl survey` to map the landscape. Likely need more chains or different initialisation.                                                      |
| `ConvergenceIncomplete`      | max R̂ > 1.1 with finite agreements  | Some parameters haven't converged                                                       | More sweeps; check the per-parameter R̂ table to see which.                                                                                         |
| `AcceptanceRateUnhealthy`    | < 10% or > 50%                      | MH proposal SD is too big (low accept) or too small (high accept)                       | Let burn-in run longer; the Robbins-Monro adapter usually fixes this. If not, fix the proposal SD manually in fit.toml.                            |

`eval-stats` summary at end of run (separate from diagnostics) shows counter
increments: `div_by_zero`, `pow_nan_inf`, `binomial_fallback`, etc. Non-zero
counts mean the model hit a degenerate path during this run. Cross-reference
with `camdl fit summary` for context.

---

## Idioms / anti-idioms

**If the data is dated, anchor the model.** Declare `origin` at the top level
whenever you are fitting real surveillance data. This is the first decision in
the file and the hardest to notice you got wrong:

```camdl
time_unit = 'days
origin    = date("2020-03-01")   # t = 0 is 1 March 2020
```

The argument is a quoted ISO date and nothing else — `date(2020, 3, 1)` is
**E101**.

Anchored, camdl reads a dated `time_col` directly, and every time it reports
back — an outbreak peak, an intervention date, a forecast window, `time_of_max`
in a `quantities {}` block — is a calendar date you can check against the
record. Unanchored, all of those are bare numbers on an axis whose zero means
nothing, and only you know what day 84 was.

**The anti-idiom is converting dates to integers before camdl sees them.** It
looks like helpfulness — the data has `2020-03-01`, the model wants a number, so
you map dates to day indices in your loader and hand camdl the integers. It
compiles, it fits, the numbers are right, and the calendar is gone: nothing
downstream can map an estimate back to a date, and the loss is invisible in
every artifact, so it can survive a long way into an analysis before anyone
notices.

**If a scheduled day is unusable, write `NA` — never drop the row.** This is the
one that costs the most time to diagnose, because the symptom does not point at
the cause.

For a stream whose `projected` is an `incidence(...)`, camdl accumulates
modelled flow over the interval **since that stream's previous row** — not over
a window you name. So the two ways of handling a bad day mean different things:

| you write         | the model sees                                              |
| ----------------- | ----------------------------------------------------------- |
| `2026-07-14  NA`  | scheduled, unobserved — no likelihood term, interval CLOSES |
| (no row on 07-14) | not scheduled — the NEXT row's interval spans the gap       |

Filtering unusable days out therefore _silently widens_ the next row's exposure
window while its own count still covers one day. A real case: after filtering,
the retained rows sat up to 13 days apart, so `projected` was compared against
13 days of modelled flow against one day of specimens — a 13x inflation that
**refused 23 of 24 chains at initialisation**, because a modelled flow above the
specimen count is impossible rather than merely a poor fit. Nothing in that
error mentions the filtering.

So: **emit every scheduled time, and make the unusable ones `NA`.** The days you
cannot use then contribute no observation at all — which is what they are,
rather than a fabricated zero or an invisible widening.

```python
# polars: reindex onto the full span, then left-join, so gaps become nulls
span = pl.date_range(out["time"].min(), out["time"].max(),
                     interval="1d", eager=True).alias("time")
full = pl.DataFrame({"time": span}).join(out, on="time", how="left")
full.write_csv(path, separator="\t", null_value="NA")
```

Two traps that follow:

- **A proportion stream needs a non-zero denominator even on unscored rows.** If
  the stream is `k / n`, fill `n` with `1` (or any positive value) on the `NA`
  rows. The observation is `NA` so the denominator is never read, but a `0`
  there can still reach a division depending on how the projection is written.
- **`NA` is not a null token in every tool.** polars does not parse it as null
  by default, so one `NA` turns a numeric column into a string column, and a
  plotting library will then draw those strings at the axis baseline — holes
  rendering as _observed zeros_ in a figure that looks entirely plausible. Read
  with `null_values="NA"`.

Full semantics, including how a hole differs from an observed `0`:
[`camdl-data-spec.md`](camdl-data-spec.md), "Missing observations".

camdl refuses the dated file rather than guessing, and the refusal names the
fix:

```
error: data has dated time cells but the model declares no `origin`.
       Add `origin = date("YYYY-MM-DD")` to the model, or supply numeric times.
```

Take the first branch. "Supply numeric times" is there for data that genuinely
has no calendar; reaching for it to silence the error gets you past the guard
rather than past the problem.

Anchoring is not always right. A textbook SIR, an SBC run, a simulation study on
synthetic indexed time — none of these have a calendar, and `origin` would be a
fiction. The rule is about provenance, not preference: **if a date exists in the
source data, it should still exist in the output.** One constraint to know
before you choose: `time_unit = 'months` and `'years` are refused in anchored
mode (**E320**), because a calendar month is not a constant number of days —
anchored models step on `'days` or `'weeks`. Read `camdl docs dates` before
anchoring a monthly model; it has the migration.

**Annotate your parameters with `#'` doc comments.** A name and kind
(`beta : rate`) tells the next reader almost nothing about what the parameter
_is_ — and parameters are the easiest part of a model to leave unexplained. Put
a one-line `#'` doc comment directly above each one:

```camdl
parameters {
  #' per-capita transmission rate (contact rate × per-contact prob)
  beta : rate
  #' mean infectious period is 1/gamma
  gamma : rate
}
```

It surfaces in `camdl inspect --parameters`, travels with the model, and costs
one line. This is a strong recommendation, not a requirement — an unannotated
model compiles fine — but **when you author or edit a model, default to
documenting at least its parameters.** They carry the largest information
deficit (an `S`/`I`/`R` usually explains itself; `beta`, `rho`, `k` do not) and
are the easiest to forget. Annotation is a graded style: **parameters always** →
then the non-obvious **compartments** (an environmental reservoir `W`, distinct
latent stages) → then any **`let`** whose meaning is not evident from its
arithmetic (`let N = S + I + R` — total population, or the force-of-infection
denominator?) → broader declarations by request. `#'` attaches to a
_declaration_, so inside a block it goes above the member, not above the `{`;
the sites that take one are compartments, parameters, dimension entries,
transitions, observation streams, quantities, and `let`. The `#'` says what a
parameter _means_; its _value_ still belongs in a `--params` TOML, never the
model — `#'` is the right home for the intent that otherwise leaks into a
`# FIXED = 0.3` comment (which does nothing, drifts from the real value, and is
exactly what `#'` replaces).

**Open the file with a `#'` block saying what the model is.** A `#'` block at
the very top — above `time_unit`, with only blank lines or `#` comments before
it — documents the model itself rather than any one declaration:

```camdl
#' National SEIR with a facility-death delay: cases and deaths come from one
#' confirmation flow, deaths lagged through an isolation compartment.
#' Fitted to weekly confirmed cases and weekly confirmed deaths.
#' @base bvd_national_twocfr.camdl
#' @adds nothing
#' @changes f_cfr_unret becomes free with a beta(2,2) prior

time_unit = 'days
```

Say what the compartments mean, what observation streams it is fitted to, and —
when the model is a variant of another — what it branches from and what that
variant changes. `@base` / `@adds` / `@changes` are free text the compiler keeps
verbatim; nothing validates them, and no other `@tag` is refused here. The block
reaches `camdl inspect`, both `camdl render` projections, and the fit sidecar,
so "what is this model?" is answerable without opening the file. It is envelope
metadata, so correcting it never re-keys a fit — write real content, and fix it
when it is wrong.

**Backend choice for fits.** Use `chain_binomial` — the production fit backend.
Gillespie is for forward-simulation sanity checks, not fits (too slow).

**Always `camdl survey` before `camdl fit run`.** Surveying is the cheapest hour
of compute in the pipeline; fitting a model you haven't surveyed is the single
most common way to spend a week producing a wrong answer.

**One stage at a time when iterating.** `camdl fit run fit.toml --stage scout`
gives you one stage's output to inspect before committing to refine + validate.
Run all stages only when the fit.toml is stable.

**Explicit priors, always.** PGAS now refuses implicit-Flat priors. A wide
uniform is fine if that's actually your belief; a wide normal is fine for log
parameters; but the choice has to be in the file. "No prior" is no longer an
option.

**Cond guards on rate expressions with potentially-zero divisors.** Spatial and
stratified models are the common case (an empty patch's force-of-infection is
`0 / 0`). Write `cond(N > 0, beta * I / N, 0)` rather than relying on
silent-zero (the runtime no longer silently zeros — it errors).

**Reparameterise to natural support.** Use `transform = "log"` for rates and
positives, `transform = "logit"` for probabilities. The MCMC moves on the
transformed scale; bounds are enforced by construction.

**A `fit.toml` `bounds` may only _narrow_ the model, never widen it.** The
model's `parameters { p : rate in [lo, hi] }` range is the scientific claim
about where `p` can plausibly live — the source of truth. A `[estimate.p]`
`bounds` override is for a _tighter_ experiment-specific search (restrict a
scout to a sub-range, pin a sensitivity sweep), so it must be a **subset** of
the declared range. camdl enforces this: a fit whose bounds fall outside the
model's — `p : rate in [0.001, 1.0]` in the model, `bounds = [0.01, 2.0]` in the
fit.toml — is **rejected** at config resolution
(`estimate.p: fit.toml bounds …
lie outside model bounds …; a fit can tighten bounds but not loosen them`).
To search a wider range the answer is not to loosen it in the config: **edit the
declared bounds in the model.** That change is visible in the model file,
travels with it, and re-keys the run identity; the same widening buried in a
fit.toml would not, which is exactly why it's disallowed.

**Reach for `camdl fit summary`, not eyeballed traces.** The summary already
extracts R̂, ESS, the MLE table, and any fired diagnostics. Eyeballing trace TSVs
is for debugging the summary, not for routine inspection.

It leads with the verdict and the estimates; the diagnostics follow, and the
posterior table is sorted worst-R̂ first so the problems are at the top rather
than alphabetized among the healthy parameters. Two flags add material that is
off by default:

```bash
camdl fit summary <fit-dir>                # verdict, estimates, diagnostics
camdl fit summary <fit-dir> --explain      # + what each column means
camdl fit summary <fit-dir> --parameters   # + the model's parameter legend
```

`--explain` appends a short prose block after each section defining that
section's terms — R̂ and its bulk/folded halves, the per-chain log-likelihood
columns, what a forkable draw is, what a frozen latent cell is. Each term is
defined once, under the section it is read in. **Use it when you are unsure what
a column means rather than guessing from the name**; several of these (`mod-z`,
`frozen-disagree`, `chains frozen`) mean something narrower than they sound. It
interleaves prose with the tables, so under `--explain` the text output is no
longer cleanly machine-readable — use `--format json` for that.

`--parameters` prints the model's parameter legend: each parameter's symbol,
prior, citations and caveats, from the `#'` doc comments. It is off by default
because on a well-documented model it runs to dozens of lines before the first
number. Reach for it when you need to know what a parameter _means_ or what its
prior asserts, not when you need its value.

**Derived channels belong in `quantities {}`, not a downstream script.** When
you want a time-varying quantity the model computes but doesn't carry as a
compartment — force of infection `λ(t)`, effective reproduction number `Rₑ(t)`,
cumulative incidence, EIR, prevalence `I/N` — declare it in a `quantities {}`
block as an **unreduced** expression. camdl evaluates it at every output time,
writes `quantities/<name>.tsv`, and `fit predict` bands it over the posterior
alongside the observed streams. Do **not** reconstruct it in a pandas/matplotlib
script from `traj.tsv`: a hand-rolled recomputation drifts from the model's own
arithmetic (a subtly different `N`, a dropped `cond` guard), is not banded over
the posterior, and has to be rewritten for every run. A reduction (`max`,
`final`, `time_of_max`, …) collapses the same expression to a scalar summary
(peak, attack rate, time-to-peak); omit the reduction to keep the full series.
See `camdl docs user-features` ("Reporting derived quantities").

**A shared reporting vocabulary is a file, not a copy-paste.** When several
models want the same summaries, put the `quantities {}` block in its own
`.camdl` file — a file containing nothing else — and apply it at the point of
use:

```
camdl simulate model.camdl --quantities reporting/national.camdl --quantities-out out/
camdl fit predict @jigawa-baseline --quantities reporting/national.camdl
```

The file REPLACES the model's own block; it never merges. It is compiled against
the model it is applied to, so a name that model does not declare is an error
naming both the name and the file — which is the signal that this family of
models needs its own vocabulary (an Erlang-staged model's `reff` formula is not
an exponential-dwell model's). The emitted tables land in `quantities-<key>/`,
keyed by the file's contents, so applying two vocabularies to one run gives you
two tables rather than overwriting one, and correcting a formula in place
produces a new table instead of a stale cache hit. Neither the simulation nor
the fit's identity moves — quantities are derived reports, not inputs, so
nothing re-runs.

This is also the answer to "I corrected a `quantities {}` formula and
`fit predict` still reports the old number": `fit predict` reads the model IR
archived inside the fit, so editing the source does nothing to a fit that
already ran. Pass the corrected block with `--quantities` instead. It is refused
if the model source has drifted from the one the fit ran on.

**Don't reach for these escape hatches without understanding them:**

- `--allow-degenerate-rates` — restores legacy silent-zero on `Div by zero` etc.
  Use only when the model legitimately means "rate is 0 when divisor is 0" (e.g.
  force of infection in a patch with no people). Default is hard error, which is
  correct for almost every model.
- `CAMDL_SKIP_VERSION_CHECK=1` — bypasses the camdl/camdlc version handshake.
  Almost always means you should `make install` instead.
- `--no-nuts` (PGAS) — falls back to MH-within-Gibbs. For posterior geometries
  where NUTS struggles, but verify with a small run before committing.

---

## fit.toml shape

The full, verified schema — `[model]`, `[data.observations]`, `[estimate]`
(bounds, prior, transform), `[fixed]`, and user-named `[stages.<name>]` blocks
with `algorithm = if2|pgas|pmmh|pfilter` — is `camdl docs fit-toml`. Two
load-bearing rules: every estimated parameter needs an explicit prior for a
Bayesian stage (in the model's `~` declaration or `[estimate].prior`; PGAS
refuses _silent_ flat), and fits run the `chain_binomial` backend.

---

## Reproducibility primitives — use them

Every fit run is content-addressed: hash of
`(model IR, params, seed, data,
algorithm config, tool version)`. Same inputs →
same hash → cache hit (no re-run).

```bash
camdl list                        # all cached runs (content-addressed leaves)
camdl show <run>                  # full metadata for one run
camdl cat  <run>                  # emit its trajectory or observations
camdl fit summary <fit-dir>       # convergence, gate verdict, MLE table for a fit
camdl fit summary <fit-dir> --explain      # ... plus what each column means
camdl fit summary <fit-dir> --parameters   # ... plus the model's parameter legend
camdl fit table   results/fits/   # one row per fit across a results tree
camdl fit diff <a.toml> <b.toml>  # diff two fit.toml *configs* (not run hashes)
```

The iterative model-building loop:

```bash
cp fit.toml fit_prev.toml               # snapshot the config you're about to change
# edit fit.toml — say, widen a prior
camdl fit diff fit_prev.toml fit.toml   # what changed in the config
camdl fit run fit.toml                  # output lands in a new content-addressed leaf
camdl list                              # the new run appears alongside the previous one
```

Cite a fit hash in writeups — paste it into a methods section and any reader
with the source can reproduce the result bit-for-bit.

---

## Project layout, the run store, and tooling

### The layout the tools expect

| Path        | What                                                                                          |
| ----------- | --------------------------------------------------------------------------------------------- |
| `models/`   | `.camdl` model files and their `fit.toml` configs, side by side                               |
| `data/`     | committed input streams plus the lock/manifest pinning their provenance                       |
| `results/`  | the content-addressed run store — `results/fits/`, `results/sims/`, `results/pfilters/`, …    |
| `scripts/`  | everything the Makefile or workflow calls; no logic inline in a rule                          |
| `workflow/` | the fit DAG (`*.smk`), once the project has enough fan-out to need one                        |
| `tests/`    | offline contract tests — do the data's columns match what the model's `observations` declare? |
| `notes/`    | dated working notes                                                                           |
| `Makefile`  | the named entry points, including the networked/offline split                                 |

`results/` is the one that is not a matter of taste. Every run is
content-addressed on the tuple above, so keeping them all under one root is what
makes a run **addressable**: `camdl list`, `camdl show <run>` and
`camdl fit table results/fits/` work off it, an identical re-run is a cache hit
rather than a recompute, and a refit on updated data forks a new leaf beside the
old one instead of overwriting it. Scatter fits into per-experiment directories
and all three are lost. A leaf belongs to camdl — never write a derived summary
into one.

The reasoning behind each directory, and the house style for a model file's
header, is `docs/camdl-style.md` in the camdl repo.

### camdl 'scope — read a fit store in a browser

**camdl 'scope** is a browser-based viewer and live monitor for a fit store. Per
fit it shows a doc-labelled forest of marginal posteriors (median [90%], R̂ /
ESS), a pair/corner plot with a prior overlay, posterior-predictive ribbons
against the observed series, generated quantities, per-parameter and
log-posterior traces, camdl's own convergence verdict, and the
syntax-highlighted `.camdl` + `fit.toml`. Across fits it runs the authoritative
`camdl compare` on any that carry a `prequential.json` (elpd, Δelpd ± paired SE,
CRPS, PIT — what those are and how to read them is `camdl compare --explain`).
It is read-only on the store, auto-discovers concurrent runs, and refreshes a
run that is still sampling — so diagnostics are readable _during_ a long fit
rather than only after it.

It is a separate package; install its `camdl-watch` command with `uv`. The
install builds the browser UI from source, so Node.js (for `npm`) must be on the
PATH alongside `uv` — without it the install fails rather than leaving a server
with no UI:

```sh
uv tool install git+https://github.com/vsbuffalo/camdl-scope
# or, without installing:
uvx --from git+https://github.com/vsbuffalo/camdl-scope camdl-watch --port 8800
```

Run it from the project root. `--store` defaults to `results/fits` under the
current directory — which is precisely why the layout above needs no
configuration:

```sh
camdl-watch --port 8800                  # http://127.0.0.1:8800
camdl-watch --port 8800 --host 0.0.0.0   # reachable from a phone over the LAN
```

Hand the human the URL and let them read the plots. Agent vision on scatter and
ribbon geometry is unreliable — the same caution as `camdl survey` above.

### Reading camdl outputs correctly

This matters more than any tool recommendation: the store is only as useful as
your ability to read what it wrote.

**Chain ids are 1-based everywhere you type them and 0-based inside
`draws.tsv`.** The `chain_N/` directories, the per-chain table in
`camdl fit summary`, and `--exclude-chains 3,5` are all **1-based**. The `chain`
column _inside_ `draws.tsv` is **0-based**, because it is the join key to
`trajectories.tsv`. User chain `k` is the rows whose `chain` field is `k - 1`.
Get this wrong and you drop the wrong chain or mislabel a trace, and **nothing
errors** — the numbers quietly describe a different chain. (Spec:
`docs/camdl-run-spec.md` §10.3; gh#666 is open on surfacing it to someone who
only ever opens the TSV.)

**Read `run.json`'s `output_schema` rather than reverse-engineering a header.**
Each leaf's `run.json` declares, for every tabular file that leaf wrote, which
column is the time or iteration axis, which are grouping keys (`chain`,
`replicate`, `scenario`), which are model quantities, and which are sampler
diagnostics. It is built by classifying each file's **actual** header, so it
cannot disagree with the file it describes. Consume the _role_, never the name —
the iteration column is spelled `sweep`, `step`, `draw` or `iteration` depending
on the method, and `time` (physical) and `iteration` (a sampler index) are
deliberately distinct roles. Not every producer declares one: `sim` leaves and
completed fit stages do; `pfilter`, `survey`, `profile` and the `fit predict`
outputs do not.

### Workflow tooling — the principle, not a mandated tool

**Every artifact is a named, re-runnable target.** Nothing that feeds a model, a
figure, or a fit is generated ad hoc from a shell one-liner or a notebook cell;
a file under a build directory with no rule that regenerates it is a bug, not
furniture. Two reasons specific to camdl, beyond ordinary hygiene:

- **The run store is already content-addressed caching, so a workflow that
  re-runs unconditionally fights it.** Declare the artifact you want and let
  camdl's hashing decide whether sampling has to happen at all. A rule that
  shells out into a fresh output directory every time throws the cache away and
  hides the fact that an input changed.
- **Target the thing you actually want to look at.** Asking the workflow for the
  fit alone stops at the end of sampling with no posterior predictive written,
  so the viewer has no ribbons; targeting the predictive runs
  `fit run → fit predict` as one chain.

`make` for light pipelines and for the networked/offline seam — a networked
target refreshes a **committed** reference copy of the data, every other target
is offline and deterministic, so a fit can never silently acquire different
inputs. Snakemake once the DAG, or the fan-out over models × configs, warrants
it. A real project uses both, split exactly that way: `make` for the data seam,
Snakemake for the fit DAG.

---

## Where the docs live

**Run `camdl docs`.** The guides are embedded in the binary — offline, and
version-matched to the `camdl` you're running. No checkout, no network:

| For                                                 | Run                                                 |
| --------------------------------------------------- | --------------------------------------------------- |
| Writing a model (the DSL by example)                | `camdl docs getting-started`, `camdl docs language` |
| The fit workflow, in depth                          | `camdl docs workflow`                               |
| The `fit.toml` schema                               | `camdl docs fit-toml`                               |
| The reasoning (identifiability, priors, the stance) | `camdl docs concepts`                               |
| Backends / data format / debugging                  | `camdl docs backends` / `data` / `debugging`        |
| Dated data and calendar time (`origin`, anchoring)  | `camdl docs dates`                                  |
| Packaging a bug report (minimal repro example)      | `camdl docs mre`                                    |
| Reading a `camdl compare` table (elpd, evidence)    | `camdl docs model-comparison`                       |
| Full topic list / search                            | `camdl docs` / `camdl docs --search <term>`         |

For sustained work you can also pin the source (working `.camdl` for every
language feature under `ocaml/golden/`):

```bash
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/vsbuffalo/camdl .camdl-source
(cd .camdl-source && git sparse-checkout set docs ocaml/golden)
```

Inside the camdl repo itself, the same content is the source `docs/*.md`, with
`docs/dev/` for design proposals and reviews.

---

## What the runtime can validate that you can't see in source

Things to remember the compiler is doing for you, so you can lean on them:

- **Unit conversion** between `'days`, `'weeks`, `'years`, `'per_year`, etc. to
  the model's `time_unit`. Don't pre-convert; the compiler handles it.
- **Dimensional analysis** on every expression. Per-capita vs population-level
  rate confusion → `E300` at compile time.
- **Source-to-source autodiff** on every rate expression. PGAS+NUTS gets exact
  gradients; you don't need to hand-write Jacobians.
- **Identifiability checks** in `camdl survey` (informally) and in
  `camdl profile` (formally — 1D and 2D profile likelihoods).
- **Backend-capability gate**: requesting a backend that can't run a model (e.g.
  `--backend gillespie` on a model with `overdispersed()`) errors at dispatch
  time with a hint.
- **Per-particle recovery** in PGAS / particle filter: a particle that hits
  `NumericalCollapse` is killed in resampling (its log-weight goes to `−Inf`).
  The chain continues. The `degenerate_step_count` in the eval-stats summary
  tells you how often this fired.

---

## When to write back

If the agent gets stuck in a way the tables above don't cover:

1. Run `camdl check` on the current model file — the compile error is almost
   always the right entry point.
2. Run `camdl --help` and `camdl <subcommand> --help` — the help text is
   maintained as part of CI.
3. Look at `docs/dev/reviews/` for recent design discussions.
4. Look at `ocaml/golden/*.camdl` for working examples of every language
   feature; corresponding `*.params.toml` shows runtime parameter values.
5. As a last resort, the source is in `ocaml/lib/compiler/` (parser, expander)
   and `rust/crates/sim/src/inference/` (inference math).
